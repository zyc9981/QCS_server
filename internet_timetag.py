"""Background coordinator for the Time Tag Internet panel; no local SDK loads."""
import math
from pathlib import Path
import queue
import select
import socket
import threading
import time
import uuid
from datetime import datetime, timezone

from QCS_HH_TRX import read_json, send_json, receive_data


class InternetCoordinator:
    def __init__(self, bind='0.0.0.0', control_port=50051, data_port=50050, timeout=30):
        self.bind, self.control_port, self.data_port = bind, control_port, data_port
        self.timeout = timeout
        self.lock = threading.Lock()
        self.commands = queue.Queue()
        self.closed = threading.Event()
        self.sockets = set()
        self.control_connected = False
        self.connection_generation = 0
        self.clock_adjustment_supported = False
        self.ready = False
        self.active = False
        self.session = None
        self.result = None
        self.bytes_saved = 0
        self.blocks = 0
        self.started = None
        self.finished = None
        self.remote_start_time_us = None
        self.remote_start_monotonic_us = None
        self.clock_commands = {}
        self.status = 'Starting Internet listener...'
        self.thread = threading.Thread(target=self._run, name='internet-control', daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return dict(ready=self.ready, active=self.active, result=self.result,
                        bytes=self.bytes_saved, blocks=self.blocks, started=self.started,
                        finished=self.finished, remote_start_time_us=self.remote_start_time_us,
                        remote_start_monotonic_us=self.remote_start_monotonic_us,
                        control_connected=self.control_connected,
                        clock_adjustment_supported=self.clock_adjustment_supported,
                        clock_commands={key: dict(value) for key, value in self.clock_commands.items()},
                        status=self.status)

    def set_clock_correction(self, correction_ps_per_s):
        """Queue a clock correction over the persistent Windows control connection.

        Returns a command ID. Check get_clock_command() for the asynchronous result.
        """
        try:
            correction = float(correction_ps_per_s)
        except (TypeError, ValueError) as exc:
            raise ValueError('Clock correction must be a finite number in ps/s') from exc
        if not math.isfinite(correction):
            raise ValueError('Clock correction must be a finite number in ps/s')

        command_id = uuid.uuid4().hex
        with self.lock:
            if not self.control_connected:
                raise RuntimeError('Windows control connection is not connected')
            if not self.clock_adjustment_supported:
                raise RuntimeError('Windows node has not advertised clock-adjustment support')
            # Retain a bounded history of recent command results.
            while len(self.clock_commands) >= 64:
                oldest_terminal = next((key for key, item in self.clock_commands.items()
                                        if item['status'] in ('applied', 'failed', 'timed_out',
                                                              'connection_lost')),
                                       None)
                if oldest_terminal is None:
                    raise RuntimeError('Too many clock commands are awaiting results')
                del self.clock_commands[oldest_terminal]
            generation = self.connection_generation
            self.clock_commands[command_id] = {
                'command_id': command_id,
                'correction_ps_per_s': correction,
                'status': 'queued',
                'created_monotonic': time.monotonic(),
                'connection_generation': generation,
                'result': None,
            }
            self.commands.put(('set_clock', (command_id, correction, generation)))
        return command_id

    def get_clock_command(self, command_id):
        """Return command state, decoded result fields, and the raw remote result."""
        with self.lock:
            command = self.clock_commands.get(command_id)
            if command is None:
                return None
            snapshot = dict(command)
            details = command.get('result')
            if isinstance(details, dict):
                snapshot['result'] = dict(details)
                snapshot['message'] = details.get('message') or command.get('message', '')
                snapshot['applied_correction_ps_per_s'] = details.get('applied_correction_ps_per_s')
                snapshot['no_change'] = (
                    command.get('status') == 'applied' and details.get('no_change') is True
                )
            return snapshot

    def _fail_clock_commands(self, generation, status, message):
        with self.lock:
            for command in self.clock_commands.values():
                if (command['connection_generation'] == generation
                        and command['status'] in ('queued', 'sent')):
                    command['status'] = status
                    command['message'] = message

    def _expire_clock_commands(self):
        now = time.monotonic()
        with self.lock:
            for command in self.clock_commands.values():
                if (command['status'] == 'sent'
                        and now - command['sent_monotonic'] >= self.timeout):
                    command['status'] = 'timed_out'
                    command['message'] = 'No CLOCK_RESULT received before the control timeout'

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
            self.remote_start_time_us = self.remote_start_monotonic_us = None
            self.status = 'Requesting Windows recording...'
            self.commands.put(('record', (self.session, output, duration, max_bytes)))

    def record_memory(self, on_data, on_metadata, duration=3600, max_bytes=2**63-1):
        """Receive verified frames through callbacks, without local file writes."""
        if not math.isfinite(duration) or not 0.001 <= duration <= 3600:
            raise ValueError('Internet duration must be 0.001..3600 seconds')
        with self.lock:
            if not self.ready or self.active:
                raise RuntimeError('Internet Windows service is not ready')
            self.session = uuid.uuid4().hex
            self.active, self.ready = True, False
            self.result = None
            self.bytes_saved = self.blocks = 0
            self.started = self.finished = None
            self.remote_start_time_us = self.remote_start_monotonic_us = None
            self.status = 'Requesting remote acquisition into local memory...'
            self.commands.put(('memory', (self.session, on_data, on_metadata, duration, max_bytes)))

    def _begin_memory(self, conn, values):
        sid, on_data, on_metadata, duration, maximum = values
        listener = self._listen(self.data_port)
        connections = []
        def receive():
            try:
                def connected(sock):
                    connections.append(sock)
                    self._track(sock)
                    with self.lock:
                        self.started = time.monotonic()
                        self.status = 'Receiving remote acquisition into local memory...'
                result = receive_data(listener, None, None, None, sid, self.timeout,
                                      maximum, duration, connected=connected, quiet=True,
                                      on_data=on_data, on_metadata=on_metadata,
                                      progress=self._memory_progress)
                with self.lock:
                    self.result = result
                    self.active = False
                    self.finished = time.monotonic()
                    self.status = ('Memory acquisition complete' if result['status'] == 'complete'
                                   else 'Memory acquisition failed: ' + result.get('error', 'unknown error'))
                if result['status'] != 'complete':
                    self.commands.put(('stop', sid))
            except Exception as exc:
                result = {'status':'incomplete', 'error':str(exc)}
                with self.lock:
                    self.result = result
                    self.active = False
                    self.finished = time.monotonic()
                    self.status = 'Memory acquisition failed: ' + str(exc)
                self.commands.put(('stop', sid))
            finally:
                listener.close()
                with self.lock:
                    self.sockets.discard(listener)
                    for sock in connections:
                        self.sockets.discard(sock)
        threading.Thread(target=receive, name='internet-memory', daemon=True).start()
        send_json(conn, dict(type='RECORD', id=sid, duration=duration))

    def _memory_progress(self, size, blocks):
        with self.lock:
            self.bytes_saved, self.blocks = size, blocks

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
                        with self.lock:
                            self.connection_generation += 1
                            self.control_connected = True
                            self.clock_adjustment_supported = False
                    except socket.timeout:
                        continue
                try:
                    try:
                        kind, values = self.commands.get_nowait()
                    except queue.Empty:
                        kind = None
                    if kind == 'record':
                        self._begin(conn, values)
                    elif kind == 'memory':
                        self._begin_memory(conn, values)
                    elif kind == 'stop':
                        send_json(conn, dict(type='STOP', id=values))
                    elif kind == 'set_clock':
                        command_id, correction, generation = values
                        with self.lock:
                            command = self.clock_commands.get(command_id)
                            can_send = (self.control_connected
                                        and generation == self.connection_generation
                                        and command is not None
                                        and command['status'] == 'queued')
                            if can_send:
                                command['status'] = 'sent'
                                command['sent_monotonic'] = time.monotonic()
                        if can_send:
                            send_json(conn, {
                                'type': 'SET_CLOCK',
                                'command_id': command_id,
                                'correction_ps_per_s': correction,
                            })
                        else:
                            self._fail_clock_commands(
                                generation, 'connection_lost',
                                'Control connection changed before command was sent')
                    self._expire_clock_commands()
                    if not select.select([conn], [], [], 0.1)[0]:
                        continue
                    msg = read_json(conn)
                    if msg.get('type') == 'CLOCK_RESULT':
                        command_id = msg.get('command_id')
                        with self.lock:
                            command = self.clock_commands.get(command_id)
                            if command is not None:
                                command['status'] = 'applied' if msg.get('success') is True else 'failed'
                                command['result'] = msg
                                command.pop('message', None)
                            else:
                                self.status = f'Received unmatched clock result: {command_id}'
                    elif msg.get('type') == 'ACQ_STARTED':
                        with self.lock:
                            if msg.get('id') == self.session:
                                self.remote_start_time_us = msg.get('start_time_us')
                                self.remote_start_monotonic_us = msg.get('start_monotonic_us')
                                start_time_us = self.remote_start_time_us
                            else:
                                start_time_us = None
                        if start_time_us is not None:
                            seconds, microseconds = divmod(int(start_time_us), 1_000_000)
                            utc = datetime.fromtimestamp(seconds, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
                            print(f'HH reported capture start: {utc}.{microseconds:06d}Z '
                                  f'(Unix us={start_time_us})', flush=True)
                    with self.lock:
                        if msg.get('type') == 'READY':
                            self.ready = True
                            capabilities = msg.get('capabilities', [])
                            self.clock_adjustment_supported = (
                                isinstance(capabilities, list)
                                and 'set_clock_correction_v1' in capabilities
                            )
                            if not self.active and self.result is None:
                                self.status = 'Internet ready; Windows service connected'
                        elif msg.get('type') == 'BUSY':
                            self.ready = False
                        elif msg.get('type') in ('ERROR', 'DUPLICATE'):
                            raise RuntimeError(msg.get('message', str(msg)))
                except Exception as exc:
                    with self.lock:
                        generation = self.connection_generation
                        self.control_connected = False
                        self.clock_adjustment_supported = False
                    conn.close()
                    conn = None
                    self._fail_clock_commands(
                        generation, 'connection_lost', f'Control connection failed: {exc}')
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
            with self.lock:
                generation = self.connection_generation
                self.control_connected = False
                self.clock_adjustment_supported = False
            self._fail_clock_commands(generation, 'connection_lost', 'Coordinator closed')
            self.close()
