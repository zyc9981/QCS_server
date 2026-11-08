#!/usr/bin/env python3
r"""Persistent HH acquisition prototype: Windows service and Linux coordinator.

Linux:   python3 QCS_HH_TRX.py coordinator --bind 0.0.0.0
Windows: python QCS_HH_TRX.py service --host 158.130.30.167 --source simulate
Linux prompt: record 10 /home/zyc/QCS/hh_trial01.bin
Windows prompt: exit

Real hardware mode is experimental. The SDK's documented raw-buffer overrun
warning is not exposed as a reliable Python status. Check the SDK logs; matching
network hashes do not prove that the acquisition lost no records.
See --help. Data port 50050 and control port 50051 are separate TCP listeners.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shlex
import socket
import struct
import sys
import threading
import time
import uuid
import zlib

LENGTH = struct.Struct('!I')
FRAME = struct.Struct('!QII')
FORMAT = 'hydraharp400_t2_raw_le_u32'
MAX_CONTROL = 65536

class Error(Exception):
    pass

def exact(sock, size):
    data = bytearray(size)
    view = memoryview(data)
    while view:
        n = sock.recv_into(view)
        if not n:
            raise Error('connection closed with %d bytes remaining' % len(view))
        view = view[n:]
    return data

def read_json(sock):
    n = LENGTH.unpack(exact(sock, 4))[0]
    if not 2 <= n <= MAX_CONTROL:
        raise Error('invalid JSON frame length')
    obj = json.loads(exact(sock, n).decode('utf8'))
    if not isinstance(obj, dict):
        raise Error('JSON message is not an object')
    return obj

def send_json(sock, obj):
    data = json.dumps(obj, separators=(',', ':'), allow_nan=False).encode('utf8')
    if len(data) > MAX_CONTROL:
        raise Error('JSON message too large')
    sock.sendall(LENGTH.pack(len(data)) + data)

def save_json(path, obj):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def console(q):
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                q.put('exit')
                return
            q.put(line.strip())
    except Exception:
        q.put('exit')

def start_console():
    q = queue.Queue()
    threading.Thread(target=console, args=(q,), daemon=True).start()
    return q

def valid_id(value):
    return isinstance(value, str) and len(value) == 32 and all(c in '0123456789abcdef' for c in value)

class Spool:
    def __init__(self, file):
        self.file = file
        self.cv = threading.Condition()
        self.size = 0
        self.done = False
        self.error = None
        self.digest = hashlib.sha256()
    def append(self, data):
        if len(data) % 4:
            raise Error('source returned a partial record')
        self.file.write(data)
        self.file.flush()
        with self.cv:
            self.digest.update(data)
            self.size += len(data)
            self.cv.notify_all()
    def finish(self, error=None):
        with self.cv:
            self.error = error
            self.done = True
            self.cv.notify_all()
    def chunks(self, filename, block):
        offset = 0
        with open(filename, 'rb') as source:
            while True:
                with self.cv:
                    while offset == self.size and not self.done:
                        self.cv.wait(0.2)
                    available = self.size - offset
                    done = self.done
                    error = self.error
                if available:
                    count = min(block, available)
                    data = source.read(count)
                    if len(data) != count:
                        raise Error('spool shorter than committed prefix')
                    offset += count
                    yield data
                elif done:
                    if error:
                        raise Error('acquisition failed: ' + error)
                    return

class Simulated:
    def __init__(self):
        self.metadata = {'device': 'simulated', 'mode': 'T2', 'base_resolution': None,
                         'device_loss_status': 'not applicable to simulated source'}
    def capture(self, duration, spool, stop):
        deadline = time.monotonic() + duration
        counter = 0
        while time.monotonic() < deadline and not stop.is_set():
            data = b''.join(struct.pack('<I', (counter + n) & 0xffffffff) for n in range(256))
            spool.append(data)
            counter += 256
            stop.wait(min(0.02, max(0, deadline - time.monotonic())))
        return 'stopped_by_operator' if stop.is_set() else 'completed'
    def close(self):
        pass

class Hardware:
    def __init__(self):
        # Use the repository's own SDK loader on Windows; no explicit exitAPI.
        from picoquant_snapi import create_hydraharp_api, MeasMode, RefSource
        self.api = None
        api = create_hydraharp_api()
        self.api = api
        self.opened = False
        try:
            if not api.getDevice():
                raise Error('HydraHarp unavailable or owned by another application')
            self.opened = True
            if not api.deviceConfig.get('ID'):
                raise Error('HydraHarp could not be opened: no device ID returned; check connection and whether another application is using it')
            if not api.initDevice(MeasMode.T2, refSrc=RefSource.External_10MHZ):
                raise Error('HydraHarp T2 initialization with external 10 MHz reference failed')
            config_file = Path(__file__).with_name('HH.ini')
            if config_file.is_file() and not api.loadIniConfig(str(config_file)):
                raise Error('could not load HH.ini')
            if not api.device.setSyncCFD(300, 10):
                raise Error('could not set sync CFD')
            for channel, level in ((0, 300), (1, 150)):
                if channel >= int(api.deviceConfig.get('NumChans', 0)):
                    continue
                if not api.device.setInputCFD(channel, level, 10):
                    raise Error('could not set input CFD on channel %d' % (channel + 1))
                if not api.device.setInputChannelOffset(channel, 0):
                    raise Error('could not set input offset on channel %d' % (channel + 1))
            if not api.getDeviceConfig():
                raise Error('could not read back HydraHarp configuration')
            config = dict(api.deviceConfig)
            self.metadata = {'device': config.get('Model', 'unavailable'),
                             'device_id': config.get('ID', 'unavailable'), 'mode': 'T2',
                             'base_resolution_ps': config.get('BaseResolution'),
                             'resolution_ps': config.get('Resolution'),
                             'device_config': config,
                             'device_loss_status': 'unknown; no automatic overrun detection; inspect snAPI console or log',
                             'reference_source': 'External_10MHZ',
                             'configuration_note': 'controller startup CFD/offset settings applied; optional HH.ini loaded first; values are SDK readback'}
            json.dumps(self.metadata, allow_nan=False)
        except BaseException:
            self.close()
            raise
    def capture(self, duration, spool, stop):
        api = self.api
        if not api.raw.startBlock(int(duration * 1000), size=1048576, savePTU=False):
            raise Error('raw.startBlock failed')
        stopping = False
        try:
            while True:
                if stop.is_set() and not stopping:
                    api.raw.stopMeasure()
                    stopping = True
                finished = api.raw.isFinished()
                block = api.raw.getBlock()
                if len(block):
                    spool.append(block.astype('<u4', copy=False).tobytes())
                if finished:
                    break
                time.sleep(0.01)
        except BaseException:
            # Do not leave the native acquisition running on a Python error.
            try:
                api.raw.stopMeasure()
            except Exception:
                pass
            raise
        return 'stopped_by_operator' if stopping else 'completed'
    def close(self):
        if self.api is not None and self.opened:
            self.api.closeDevice()
            self.opened = False
        self.api = None

def stream_send(host, port, session, spool, path, block, timeout, ready_event, state):
    try:
        with socket.create_connection((host, port), timeout) as sock:
            sock.settimeout(timeout)
            send_json(sock, {'type':'HELLO', 'version':2, 'format':FORMAT, 'record_bytes':4,
                             'block_size':block, 'metadata':{'request_id':session,
                                 'requested_duration_seconds':state['requested_duration_seconds'],
                                 'source':state['source']}})
            ready = read_json(sock)
            sid = ready.get('session')
            if (ready.get('type') != 'READY' or ready.get('version') != 2 or
                    ready.get('block_size') != block or not valid_id(sid)):
                raise Error('data receiver did not accept session: %s' % ready)
            send_json(sock, {'type':'START', 'session':sid})
            state['ready'] = True
            ready_event.set()
            digest = hashlib.sha256()
            total = count = 0
            for data in spool.chunks(path, block):
                sock.sendall(FRAME.pack(count, len(data), zlib.crc32(data) & 0xffffffff))
                sock.sendall(data)
                digest.update(data)
                count += 1
                total += len(data)
            sock.sendall(FRAME.pack(count, 0, 0))
            send_json(sock, {'type':'END', 'session':sid, 'bytes':total, 'blocks':count,
                             'records':total//4, 'sha256':digest.hexdigest()})
            saved = read_json(sock)
            if (saved.get('type') != 'SAVED' or saved.get('session') != sid or
                saved.get('bytes') != total or saved.get('blocks') != count or
                saved.get('records') != total//4 or saved.get('sha256') != digest.hexdigest()):
                raise Error('invalid SAVED acknowledgement: %s' % saved)
            state.update(transfer='complete', linux_saved='verified', sent_bytes=total,
                         sha256=digest.hexdigest())
    except Exception as exc:
        state.update(transfer='incomplete', linux_saved='unverified', transfer_error=str(exc))
        ready_event.set()

class Service:
    def __init__(self, args, source):
        self.args, self.source = args, source
        self.stop = threading.Event()
        self.capture_stop = threading.Event()
        self.lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.active = None
        self.conn = None
        self.seen = set()
    def notify(self, msg):
        print(json.dumps(msg, default=str), flush=True)
        with self.send_lock:
            try:
                if self.conn:
                    send_json(self.conn, msg)
            except (OSError, Error):
                pass
    def request(self, msg):
        sid, duration = msg.get('id'), msg.get('duration')
        if not valid_id(sid) or type(duration) not in (int, float) or not math.isfinite(duration) or not 0.001 <= duration <= 3600:
            self.notify({'type':'ERROR', 'id':sid, 'message':'invalid ID or duration (0.001..3600 seconds)'})
            return
        with self.lock:
            if sid in self.seen:
                response = {'type':'DUPLICATE', 'id':sid}
            elif self.active is not None:
                response = {'type':'BUSY', 'id':sid}
            else:
                try:
                    path = Path(self.args.spool_dir) / (sid + '.bin')
                    file = path.open('xb')
                    self.seen.add(sid)
                    self.capture_stop.clear()
                    self.active = sid
                    response = {'type':'ACCEPTED', 'id':sid}
                    threading.Thread(target=self.record, args=(sid, duration, path, file), daemon=True).start()
                except OSError as exc:
                    response = {'type':'ERROR', 'id':sid, 'message':str(exc)}
        self.notify(response)
    def record(self, sid, duration, path, file):
        state = {'id':sid, 'requested_duration_seconds':duration, 'local_path':str(path),
                 'capture':'not_started', 'transfer':'incomplete', 'linux_saved':'unverified',
                 'source':self.source.metadata, 'start_unix':None, 'stop_unix':None}
        spool = Spool(file)
        ready = threading.Event()
        worker = threading.Thread(target=stream_send, args=(self.args.host, self.args.data_port, sid, spool,
                                    path, self.args.block_size, self.args.timeout, ready, state), daemon=True)
        worker.start()
        try:
            if not ready.wait(self.args.timeout + 2) or not state.get('ready'):
                raise Error('data receiver not ready; capture was not started')
            state['start_unix'] = time.time()
            state['capture'] = self.source.capture(duration, spool, self.capture_stop)
            state['stop_unix'] = time.time()
        except Exception as exc:
            state['capture'] = 'failed' if state['start_unix'] else 'not_started'
            state['capture_error'] = str(exc)
            state['stop_unix'] = time.time()
            if isinstance(self.source, Hardware) and state['start_unix'] is not None:
                # Do not reuse a device after an unverified native acquisition failure.
                self.stop.set()
        finally:
            try:
                file.flush()
                os.fsync(file.fileno())
            except OSError as exc:
                state['capture'] = 'failed'
                state['capture_error'] = str(exc)
            spool.finish(state.get('capture_error'))
            file.close()
        worker.join(self.args.shutdown_timeout if self.stop.is_set() else None)
        if worker.is_alive():
            state.update(transfer='incomplete', linux_saved='unverified', transfer_error='shutdown timeout')
        state.update(bytes_captured=spool.size, records_captured=spool.size//4,
                     local_sha256=spool.digest.hexdigest())
        try:
            save_json(str(path)+'.json', state)
        except Exception as exc:
            state['metadata_error'] = str(exc)
        self.notify({'type':'RESULT', **state})
        with self.lock:
            self.active = None
        if not self.stop.is_set():
            self.notify({'type':'READY', 'device':self.source.metadata})
    def stop_capture(self, sid):
        with self.lock:
            if self.active == sid and self.active is not None:
                self.capture_stop.set()
                return True
        return False
    def run(self):
        q = start_console()
        while not self.stop.is_set():
            try:
                with socket.create_connection((self.args.host, self.args.control_port), self.args.timeout) as conn:
                    conn.settimeout(0.25)
                    with self.send_lock:
                        self.conn = conn
                    self.notify({'type':'READY' if self.active is None else 'BUSY', 'device':self.source.metadata})
                    # Select avoids using a timeout in the middle of a JSON frame.
                    import select
                    while not self.stop.is_set():
                        try:
                            if q.get_nowait().lower() == 'exit':
                                self.stop.set()
                                break
                        except queue.Empty:
                            pass
                        if select.select([conn], [], [], 0.1)[0]:
                            conn.settimeout(self.args.timeout)
                            msg = read_json(conn)
                            conn.settimeout(0.25)
                            if msg.get('type') == 'RECORD':
                                self.request(msg)
                            elif msg.get('type') == 'STOP':
                                stopped = self.stop_capture(msg.get('id'))
                                self.notify({'type':'STOPPING', 'id':msg.get('id'),
                                             'active_match':stopped})
                            else:
                                self.notify({'type':'ERROR', 'message':'unknown control command'})
            except (OSError, Error, ValueError) as exc:
                print('Control disconnected: %s; retrying' % exc, flush=True)
                with self.send_lock:
                    self.conn = None
                try:
                    if q.get(timeout=1).lower() == 'exit':
                        self.stop.set()
                except queue.Empty:
                    pass
        with self.lock:
            active = self.active
        if active:
            self.capture_stop.set()
            print('Stopping active recording; waiting for drain and transfer', flush=True)
            deadline = time.monotonic() + self.args.shutdown_timeout + 5
            while time.monotonic() < deadline:
                with self.lock:
                    if self.active is None:
                        break
                time.sleep(0.1)
            else:
                print('INCOMPLETE: shutdown timed out; waiting for acquisition to stop before closing device', flush=True)
                while True:
                    with self.lock:
                        if self.active is None:
                            break
                    time.sleep(0.1)
        self.source.close()
def receive_data(listener, output, part_file, manifest, sid, timeout, max_bytes, duration,
                 progress=None, connected=None, quiet=False):
    info = {'status':'incomplete', 'request_id':sid, 'bytes_saved':0, 'blocks_saved':0,
            'output':str(output), 'format':FORMAT}
    digest = hashlib.sha256()
    try:
        listener.settimeout(timeout)
        conn, peer = listener.accept()
        with conn, part_file:
            if connected:
                connected(conn)
            conn.settimeout(timeout)
            hello = read_json(conn)
            block = hello.get('block_size')
            if (hello.get('type') != 'HELLO' or hello.get('version') != 2 or
                hello.get('format') != FORMAT or hello.get('record_bytes') != 4 or
                type(block) is not int or not 4 <= block <= 1048576 or block % 4 or
                not isinstance(hello.get('metadata'), dict) or hello['metadata'].get('request_id') != sid):
                raise Error('invalid data handshake or request ID')
            stream_id = uuid.uuid4().hex
            info.update(peer=str(peer), stream_session=stream_id, block_size=block,
                        capture_metadata=hello['metadata'])
            send_json(conn, {'type':'READY', 'version':2, 'session':stream_id, 'block_size':block})
            start = read_json(conn)
            if start.get('type') != 'START' or start.get('session') != stream_id:
                raise Error('invalid data START')
            # Permit a quiet source to produce no records during its acquisition.
            conn.settimeout(max(timeout, duration + timeout))
            while True:
                seq, size, crc = FRAME.unpack(exact(conn, FRAME.size))
                if seq != info['blocks_saved']:
                    raise Error('unexpected frame sequence %d' % seq)
                if size == 0:
                    if crc:
                        raise Error('invalid end frame')
                    break
                if size > block or size % 4 or info['bytes_saved'] + size > max_bytes:
                    raise Error('invalid frame size or maximum file size exceeded')
                data = exact(conn, size)
                if zlib.crc32(data) & 0xffffffff != crc:
                    raise Error('CRC mismatch in block %d' % seq)
                part_file.write(data)
                digest.update(data)
                info['bytes_saved'] += size
                info['blocks_saved'] += 1
                if progress:
                    progress(info['bytes_saved'], info['blocks_saved'])
            end = read_json(conn)
            if (end.get('type') != 'END' or end.get('session') != stream_id or
                end.get('bytes') != info['bytes_saved'] or end.get('blocks') != info['blocks_saved'] or
                end.get('records') != info['bytes_saved']//4 or end.get('sha256') != digest.hexdigest()):
                raise Error('END counts or SHA-256 mismatch')
            part_file.flush()
            os.fsync(part_file.fileno())
            part_file.close()
            os.link(str(output)+'.part', output)
            os.unlink(str(output)+'.part')
            info.update(status='complete', records_saved=info['bytes_saved']//4, sha256=digest.hexdigest())
            save_json(manifest, info)
            send_json(conn, {'type':'SAVED', 'session':stream_id, 'bytes':info['bytes_saved'],
                             'blocks':info['blocks_saved'], 'records':info['records_saved'], 'sha256':info['sha256']})
    except Exception as exc:
        info['error'] = str(exc)
        if not part_file.closed:
            part_file.close()
        try:
            save_json(manifest, info)
        except OSError as error:
            print('Metadata write failed: %s' % error, flush=True)
    finally:
        listener.close()
        if not quiet:
            print('Data result: %s' % json.dumps(info), flush=True)
    return info


def coordinator(args):
    q = start_console()
    import select
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.bind, args.control_port))
        server.listen(1)
        server.setblocking(False)
        print('Control listening on %s:%d; data listener port %d' %
              (args.bind, args.control_port, args.data_port), flush=True)
        conn = None
        state = 'disconnected'
        while True:
            try:
                try:
                    command = q.get_nowait()
                except queue.Empty:
                    command = None
                if command is not None:
                    words = shlex.split(command)
                    if not words:
                        continue
                    if words[0].lower() == 'exit':
                        break
                    if words[0].lower() != 'record' or len(words) != 3:
                        print('Use: record SECONDS /absolute/path/output.bin, or exit', flush=True)
                        continue
                    try:
                        duration = float(words[1])
                        path = Path(words[2])
                        if not math.isfinite(duration) or not 0.001 <= duration <= 3600:
                            raise Error('duration must be 0.001..3600 seconds')
                        if not path.is_absolute():
                            raise Error('output must be an absolute Linux path')
                        if not conn or state != 'ready':
                            raise Error('Windows service is not ready')
                        if path.exists():
                            raise Error('output already exists')
                        manifest = Path(str(path)+'.json')
                        part = Path(str(path)+'.part')
                        meta_file = manifest.open('x', encoding='utf8')
                        meta_file.close()
                        try:
                            file = part.open('xb')
                        except Exception:
                            manifest.unlink()
                            raise
                        listener = socket.socket()
                        try:
                            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                            listener.bind((args.bind, args.data_port))
                            listener.listen(1)
                        except Exception:
                            listener.close()
                            file.close()
                            save_json(manifest, {'status':'incomplete', 'error':'data listener could not start'})
                            raise
                        sid = uuid.uuid4().hex
                        save_json(manifest, {'status':'waiting', 'request_id':sid, 'output':str(path)})
                        threading.Thread(target=receive_data, args=(listener,path,file,manifest,sid,
                                         args.timeout,args.max_bytes,duration), daemon=True).start()
                        state = 'busy'
                        send_json(conn, {'type':'RECORD', 'id':sid, 'duration':duration})
                        print('Requested %s' % sid, flush=True)
                    except (OSError, ValueError, Error) as exc:
                        print('Request failed: %s' % exc, flush=True)
                sockets = [server] + ([conn] if conn else [])
                for ready in select.select(sockets, [], [], 0.1)[0]:
                    if ready is server:
                        new, peer = server.accept()
                        if conn:
                            new.close()
                            continue
                        conn = new
                        conn.settimeout(args.timeout)
                        print('Windows connected: %s' % (peer,), flush=True)
                    else:
                        try:
                            msg = read_json(conn)
                            print('Windows: %s' % json.dumps(msg), flush=True)
                            if msg.get('type') == 'READY':
                                state = 'ready'
                            elif msg.get('type') in ('BUSY', 'ACCEPTED'):
                                state = 'busy'
                        except (OSError, Error, ValueError) as exc:
                            print('Windows disconnected: %s' % exc, flush=True)
                            conn.close()
                            conn = None
                            state = 'disconnected'
            except KeyboardInterrupt:
                break
        if conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='role', required=True)
    service = sub.add_parser('service')
    service.add_argument('--host', default='158.130.30.167')
    service.add_argument('--source', choices=('simulate','hh'), default='simulate')
    service.add_argument('--spool-dir', default='hh_spool')
    service.add_argument('--block-size', type=int, default=1024)
    service.add_argument('--shutdown-timeout', type=float, default=15)
    coord = sub.add_parser('coordinator')
    coord.add_argument('--bind', default='0.0.0.0')
    coord.add_argument('--max-bytes', type=int, default=16*1024**3)
    for p in (service,coord):
        p.add_argument('--control-port', type=int, default=50051)
        p.add_argument('--data-port', type=int, default=50050)
        p.add_argument('--timeout', type=float, default=30)
    a = parser.parse_args()
    if (not 1 <= a.control_port <= 65535 or not 1 <= a.data_port <= 65535 or
        a.control_port == a.data_port or not math.isfinite(a.timeout) or a.timeout <= 0):
        parser.error('invalid port or timeout')
    if a.role == 'service' and (not 4 <= a.block_size <= 1048576 or a.block_size % 4 or
                               not math.isfinite(a.shutdown_timeout) or a.shutdown_timeout <= 0):
        parser.error('invalid block size or shutdown timeout')
    if a.role == 'coordinator' and a.max_bytes < 4:
        parser.error('invalid maximum bytes')
    try:
        if a.role == 'coordinator':
            coordinator(a)
        else:
            Path(a.spool_dir).mkdir(parents=True, exist_ok=True)
            source = Simulated() if a.source == 'simulate' else Hardware()
            service_obj = Service(a, source)
            try:
                service_obj.run()
            except KeyboardInterrupt:
                service_obj.stop.set()
                service_obj.run()
        return 0
    except (OSError, ValueError, Error, ImportError) as exc:
        print('ERROR: %s' % exc, file=sys.stderr, flush=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
