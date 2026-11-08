"""Background coordinator for the Time Tag Internet panel; no local SDK loads."""
import math
from pathlib import Path
import queue
import select
import socket
import threading
import time
import uuid

from QCS_HH_TRX import read_json, send_json, receive_data


class InternetCoordinator:
    def __init__(self, bind='0.0.0.0', control_port=50051, data_port=50050, timeout=30):
        self.bind, self.control_port, self.data_port = bind, control_port, data_port
        self.timeout = timeout
        self.lock = threading.Lock()
        self.commands = queue.Queue()
        self.closed = threading.Event()
        self.sockets = set()
        self.ready = False
        self.active = False
        self.session = None
        self.result = None
        self.bytes_saved = 0
        self.blocks = 0
        self.started = None
        self.finished = None
        self.status = 'Starting Internet listener...'
        self.thread = threading.Thread(target=self._run, name='internet-control', daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return dict(ready=self.ready, active=self.active, result=self.result,
                        bytes=self.bytes_saved, blocks=self.blocks, started=self.started,
                        finished=self.finished, status=self.status)

    def record(self, output, duration=3600, max_bytes=16 * 1024**3):
        if not math.isfinite(duration) or not 0.001 <= duration <= 3600:
            raise ValueError('Internet duration must be 0.001..3600 seconds')
        output = Path(output).absolute()
        for path in (output, Path(str(output)+'.part'), Path(str(output)+'.json')):
            if path.exists():
                raise FileExistsError(f'Internet output already exists: {path}')
        with self.lock:
            if not self.ready or self.active:
                raise RuntimeError('Internet Windows service is not ready')
            self.session = uuid.uuid4().hex
            self.active, self.ready = True, False
            self.result = None
            self.bytes_saved = self.blocks = 0
            self.started = self.finished = None
            self.status = 'Requesting Windows recording...'
            self.commands.put(('record', (self.session, output, duration, max_bytes)))

    def stop_recording(self):
        with self.lock:
            if self.active:
                self.status = 'Stopping Internet acquisition; waiting for verified transfer...'
                self.commands.put(('stop', self.session))

    def close(self):
        self.closed.set()
        with self.lock:
            sockets = list(self.sockets)
            self.ready = False
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def _track(self, sock):
        with self.lock:
            if self.closed.is_set():
                sock.close()
                raise OSError('Internet listener closed')
            self.sockets.add(sock)
        return sock

    def _listen(self, port):
        sock = self._track(socket.socket())
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.bind, port))
            sock.listen(1)
            return sock
        except BaseException:
            sock.close()
            raise

    def _receive(self, listener, output, file, manifest, sid, duration, maximum):
        connection = [None]
        def progress(size, blocks):
            with self.lock:
                self.bytes_saved, self.blocks = size, blocks
        def connected(conn):
            self._track(conn)
            connection[0] = conn
            with self.lock:
                self.started = time.monotonic()
                self.status = 'Receiving Internet data...'
        result = receive_data(listener, output, file, manifest, sid, self.timeout,
                              maximum, duration, progress=progress,
                              connected=connected, quiet=True)
        with self.lock:
            self.result = result
            self.active = False
            self.finished = time.monotonic()
            self.status = ('Internet complete; SHA-256 verified' if result['status'] == 'complete'
                           else 'Internet incomplete: ' + result.get('error', 'transfer failed'))
            if result['status'] != 'complete':
                self.commands.put(('stop', sid))
            self.sockets.discard(listener)
            self.sockets.discard(connection[0])

    def _begin(self, conn, values):
        sid, output, duration, maximum = values
        listener = file = None
        worker = None
        manifest = Path(str(output)+'.json')
        reserved = False
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            listener = self._listen(self.data_port)
            with manifest.open('x'):
                reserved = True
            file = Path(str(output)+'.part').open('xb')
            worker = threading.Thread(target=self._receive,
                args=(listener, output, file, manifest, sid, duration, maximum),
                name='internet-data', daemon=True)
            worker.start()
            send_json(conn, dict(type='RECORD', id=sid, duration=duration))
        except Exception:
            if listener:
                listener.close()
            if file and (worker is None or not worker.is_alive()):
                file.close()
            if reserved:
                # The receiver writes details when it has already started.
                if not manifest.stat().st_size:
                    manifest.write_text('{"status":"incomplete","error":"request failed"}\n')
            raise

    def _run(self):
        conn = server = None
        try:
            server = self._listen(self.control_port)
            server.settimeout(0.2)
            with self.lock:
                self.status = f'Internet listening on {self.bind}:{self.control_port}; waiting for Windows'
            while not self.closed.is_set():
                if conn is None:
                    try:
                        conn, _ = server.accept()
                        self._track(conn)
                        conn.settimeout(self.timeout)
                    except socket.timeout:
                        continue
                try:
                    try:
                        kind, values = self.commands.get_nowait()
                    except queue.Empty:
                        kind = None
                    if kind == 'record':
                        self._begin(conn, values)
                    elif kind == 'stop':
                        send_json(conn, dict(type='STOP', id=values))
                    if not select.select([conn], [], [], 0.1)[0]:
                        continue
                    msg = read_json(conn)
                    with self.lock:
                        if msg.get('type') == 'READY':
                            self.ready = True
                            if not self.active and self.result is None:
                                self.status = 'Internet ready; Windows service connected'
                        elif msg.get('type') == 'BUSY':
                            self.ready = False
                        elif msg.get('type') in ('ERROR', 'DUPLICATE'):
                            raise RuntimeError(msg.get('message', str(msg)))
                except Exception as exc:
                    conn.close()
                    conn = None
                    with self.lock:
                        self.ready = False
                        self.status = f'Internet control error: {exc}'
                        if self.active:
                            self.result = dict(status='incomplete', error=str(exc))
                            # A receiver may still be draining; close it before reuse.
                            self.active = False
                    # Do not reuse a session whose control channel was lost.
                    break
        except Exception as exc:
            with self.lock:
                self.status = f'Internet listener error: {exc}'
        finally:
            self.close()
