"""Live acquisition into bounded memory for the Time Stamps QCS controls.

Analysis will reuse qcs_analyzer. No fitting or frequency command is run here.
The existing remote service may still spool its own files; this receiver never
creates timestamp files. Starting both streams is coordinated, not a hardware
synchronization of their clocks.
"""
from __future__ import annotations

from collections import deque
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np


DEFAULT_QCS_CONFIG = {
    'init_interval_s': 2.0,
    'acquisition_interval_s': 1.0,
    'sweep_range_ps_per_s': 1000.0,
    'init_time_offset_range_ms': 40.0,
    'coarse_steps': 50,
    'init_bin_number': 2000_000,
    'latency_ms': 250.0,
}


def check_timestamp_quality(t_ps, ch, *, allowed_channels, required_channels,
                            min_events_per_channel=2):
    """Check one source without sorting or repairing data; return reasons/counts."""
    t_ps, ch = np.asarray(t_ps), np.asarray(ch)
    errors, insufficient = [], []
    if t_ps.ndim != 1 or ch.ndim != 1 or t_ps.size != ch.size:
        return {'errors': ['Timestamp/channel arrays must be parallel and one-dimensional'],
                'insufficient': [], 'counts': {}}
    if not np.issubdtype(t_ps.dtype, np.integer) or not np.issubdtype(ch.dtype, np.integer):
        errors.append('Timestamps and channel IDs must be integers')
    if not np.all(np.isfinite(t_ps)) or not np.all(np.isfinite(ch)):
        errors.append('Non-finite timestamps or channel IDs')
    if len(t_ps) and (np.any(t_ps < 0) or np.any(t_ps[1:] < t_ps[:-1])):
        errors.append('Negative or nonchronological timestamps')
    invalid = np.unique(ch[~np.isin(ch, list(allowed_channels))])
    if len(invalid):
        errors.append(f'Unexpected channels: {invalid.tolist()}')
    counts = {int(channel): int(np.count_nonzero(ch == channel)) for channel in required_channels}
    for channel, count in counts.items():
        if count < min_events_per_channel:
            insufficient.append(f'ch {channel}: {count} events (need {min_events_per_channel})')
    return dict(errors=errors, insufficient=insufficient, counts=counts)


class RealtimeG2Controller:
    """Own a local TimeTagStream and remote in-memory recording session."""

    def __init__(self, app: Any, config: Mapping[str, Any] | None = None):
        self.app = app
        self.config = dict(config or {})
        self.init_interval_s = float(self.config.get(
            'init_interval_s', DEFAULT_QCS_CONFIG['init_interval_s']))
        if not np.isfinite(self.init_interval_s) or self.init_interval_s <= 0:
            raise ValueError('Init_interval must be a positive finite number of seconds')
        self.latency_ms = float(self.config.get('latency_ms', DEFAULT_QCS_CONFIG['latency_ms']))
        if not np.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError('latency_ms must be a nonnegative finite number')
        alignment_options = dict(self.config.get('global_alignment_kwargs', {}))
        alignment_options.setdefault(
            'sweep_range_ps_per_s', DEFAULT_QCS_CONFIG['sweep_range_ps_per_s'])
        alignment_options.setdefault(
            'init_time_offset_range_ms', DEFAULT_QCS_CONFIG['init_time_offset_range_ms'])
        alignment_options.setdefault('init_bin_number', DEFAULT_QCS_CONFIG['init_bin_number'])
        alignment_options.setdefault('coarse_steps', DEFAULT_QCS_CONFIG['coarse_steps'])
        steps = float(alignment_options['coarse_steps'])
        if not np.isfinite(steps) or not steps.is_integer() or steps < 3:
            raise ValueError('sweep_num must be an integer of at least 3')
        alignment_options['coarse_steps'] = int(steps)
        init_bins = float(alignment_options['init_bin_number'])
        if not np.isfinite(init_bins) or not init_bins.is_integer() or init_bins < 2:
            raise ValueError('init_bin_number must be an integer of at least 2')
        alignment_options['init_bin_number'] = int(init_bins)
        for name in ('sweep_range_ps_per_s', 'init_time_offset_range_ms'):
            value = float(alignment_options[name])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
            alignment_options[name] = value
        self.config['global_alignment_kwargs'] = alignment_options
        self.initialization_status = 'Waiting for initial acquisition'
        self.alignment_result = None
        self.alignment_plot_shown = False
        self.timestamp_origins = None
        self.initialization_worker = None
        legacy_limit = self.config.get('max_events_per_source')
        self.max_events = {
            'ttx': int(self.config.get('max_events_ttx', legacy_limit or 10_000_000)),
            'hh': int(self.config.get('max_events_hh', legacy_limit or 1_000_000)),
        }
        self.interval_s = float(self.config.get('interval_s', 0.1))
        self.acquisition_interval_s = float(self.config.get(
            'acquisition_interval_s', DEFAULT_QCS_CONFIG['acquisition_interval_s']))
        if not np.isfinite(self.acquisition_interval_s) or self.acquisition_interval_s <= 0:
            raise ValueError('Acquisition interval must be a positive finite number of seconds')
        if any(limit <= 0 for limit in self.max_events.values()) or not np.isfinite(self.interval_s) or self.interval_s <= 0:
            raise ValueError('Memory event limit and polling interval must be positive')
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = None
        self.timetagger_reader = None
        self.ttx_start_time_us = None
        self.ttx_start_monotonic_us = None
        self.remote = None
        self.buffers = {key: deque() for key in ('ttx', 'hh')}
        self.counts = dict(ttx=0, hh=0)
        self.total = dict(ttx=0, hh=0)
        self.ttx_channel_totals = {}
        self.session_started_monotonic = None
        self.evicted = dict(ttx=0, hh=0)
        self.interval_evicted = dict(ttx=0, hh=0)
        self.integrity_failure = None
        self.quality = None
        self.last_timestamp = dict(ttx=None, hh=None)
        self.interval_order_errors = []
        self.completed_intervals = 0
        self.discarded = dict(ttx=0, hh=0)
        self.next_interval_end = None
        self.overflow = 0
        self.resolution_ps = None
        self.status = 'Idle'
        self.error = None
        self.running = False
        self.updates = queue.Queue(maxsize=1)

    def start(self):
        if self.running:
            raise RuntimeError('QCS acquisition is already running')
        if self.worker is not None:
            raise RuntimeError('Create a new QCS controller for each session')
        if getattr(self.app, 'time_tagger', None) is None:
            raise RuntimeError('Time Tagger X is unavailable')
        if getattr(self.app, 'qutag_timestamp_writing', False):
            raise RuntimeError('Stop file recording before starting QCS')
        self.remote = getattr(self.app, 'timestamp_internet', None)
        if self.remote is None or not self.remote.snapshot()['ready']:
            raise RuntimeError('Enable Internet and wait for the remote HydraHarp service to be ready')
        self.status = 'Starting TTX stream before remote HH acquisition...'
        self.running = True
        try:
            from qcs_analyzer._implementation import _normalize_timetagger_physical_channels
            self.normalize_ttx_channels = _normalize_timetagger_physical_channels
            with self.lock:
                if self.stop_event.is_set():
                    raise RuntimeError('QCS startup cancelled')
                # Match Start Writing's ordering: begin local TTX capture before
                # asking the remote HH service to acquire.
                self.timetagger_reader = self.app.create_time_tagger_stream(self.max_events['ttx'])
                # Record integer microseconds immediately after stream creation
                # returns, before requesting remote acquisition.
                self.ttx_start_time_us = time.time_ns() // 1_000
                self.ttx_start_monotonic_us = time.monotonic_ns() // 1_000
            epoch_s, subsecond_us = divmod(self.ttx_start_time_us, 1_000_000)
            utc = datetime.fromtimestamp(epoch_s, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
            print(
                f'TTX TimeTagStream created: {utc}.{subsecond_us:06d}Z '
                f'(Unix us={self.ttx_start_time_us}, monotonic us={self.ttx_start_monotonic_us})',
                flush=True,
            )
            self.status = 'Requesting remote acquisition into local memory...'
            self.remote.record_memory(self._receive_hh, self._remote_ready,
                                      duration=float(self.config.get('duration_s', 3600)))
            self.worker = threading.Thread(target=self._run_worker, name='qcs-acquisition', daemon=True)
            self.worker.start()
        except Exception:
            self.running = False
            self.remote.stop_recording()
            with self.lock:
                reader = self.timetagger_reader
                self.timetagger_reader = None
            if reader is not None:
                try:
                    reader.stop()
                except Exception:
                    pass
            raise

    def _remote_ready(self, metadata):
        resolution = self.config.get('hydraharp_base_resolution_ps')
        if resolution is None:
            resolution = metadata.get('source', {}).get('base_resolution_ps')
        if resolution is None or not np.isfinite(float(resolution)) or float(resolution) <= 0 or not float(resolution).is_integer():
            raise ValueError('Remote HydraHarp must report a positive integer base_resolution_ps')
        with self.lock:
            if self.stop_event.is_set():
                raise RuntimeError('QCS startup cancelled')
            self.resolution_ps = int(resolution)
            self.next_interval_end = time.monotonic() + self.init_interval_s
            self.session_started_monotonic = self.next_interval_end - self.init_interval_s
            self.status = f'Initializing: acquiring {self.init_interval_s:g} s into memory'

    def _append(self, source, times, channels):
        times, channels = np.asarray(times), np.asarray(channels)
        if times.ndim != 1 or channels.ndim != 1 or times.size != channels.size:
            raise ValueError('Timestamp/channel arrays must be parallel and one-dimensional')
        if not times.size:
            return
        if not np.issubdtype(times.dtype, np.integer) or not np.issubdtype(channels.dtype, np.integer):
            raise ValueError('Timestamp and channel arrays must contain integers')
        times = times.astype(np.int64, copy=True)
        channels = channels.astype(np.int64, copy=True)
        with self.lock:
            if self.stop_event.is_set():
                return
            completed = self._rotate_interval_locked(time.monotonic())
            previous = self.last_timestamp[source]
            if previous is not None and times[0] < previous:
                self.interval_order_errors.append(f'{source}: timestamps moved backwards between batches')
            self.last_timestamp[source] = int(times[-1])
            self.buffers[source].append((times, channels))
            self.counts[source] += len(times)
            self.total[source] += len(times)
            if source == 'ttx':
                channel_ids, channel_counts = np.unique(channels, return_counts=True)
                for channel, count in zip(channel_ids, channel_counts):
                    channel = int(channel)
                    self.ttx_channel_totals[channel] = (
                        self.ttx_channel_totals.get(channel, 0) + int(count)
                    )
            excess = self.counts[source] - self.max_events[source]
            while excess > 0:
                t, ch = self.buffers[source].popleft()
                removed = min(excess, len(t))
                if removed < len(t):
                    self.buffers[source].appendleft((t[removed:].copy(), ch[removed:].copy()))
                self.counts[source] -= removed
                self.evicted[source] += removed
                self.interval_evicted[source] += removed
                excess -= removed
        if completed is not None:
            self._handle_completed_interval(completed)

    def _rotate_interval_locked(self, now):
        """Swap buffers atomically; callers must hold self.lock.

        Intervals currently use local arrival time, not device time. Remote
        network latency means they are not yet matched analysis windows.
        """
        if self.next_interval_end is None or now < self.next_interval_end:
            return None
        elapsed = int((now - self.next_interval_end) // self.acquisition_interval_s) + 1
        completed = self.buffers
        if self.completed_intervals == 0:
            self.status = f'Acquiring in {self.acquisition_interval_s:g} s memory intervals'
        self.buffers = {key: deque() for key in ('ttx', 'hh')}
        for source in self.counts:
            self.discarded[source] += self.counts[source]
            self.counts[source] = 0
        self.completed_intervals += elapsed
        self.next_interval_end += elapsed * self.acquisition_interval_s
        interval = dict(buffers=completed, sequence=self.completed_intervals,
                        initialization=self.completed_intervals == elapsed,
                        drops=self.interval_evicted, order_errors=self.interval_order_errors)
        self.interval_evicted = dict(ttx=0, hh=0)
        self.interval_order_errors = []
        return interval

    def _handle_completed_interval(self, completed):
        """Check acquisition integrity and timestamps, then release interval data."""
        reasons = list(completed['order_errors'])
        drops = completed['drops']
        if any(drops.values()):
            reasons.append(f'Memory capacity dropped events: {drops}')
        source_checks = {}
        offset = int(self.config.get('hydraharp_channel_offset', 5))
        required = {'ttx': self.config.get('required_ttx_channels', (1, 2)),
                    'hh': self.config.get('required_hh_channels', (offset + 1,))}
        configured = getattr(self.app, 'time_tagger_setting_channels', (1, 2, 3, 4))
        allowed = {'ttx': [abs(int(ch)) if abs(int(ch)) <= 4 else int(ch) for ch in configured],
                   'hh': range(offset, offset + 65)}
        minimum = int(self.config.get('min_events_per_channel', 2))
        insufficient = []
        arrays = {}
        try:
            for source, chunks in completed['buffers'].items():
                times = np.concatenate([item[0] for item in chunks]) if chunks else np.empty(0, dtype=np.int64)
                channels = np.concatenate([item[1] for item in chunks]) if chunks else np.empty(0, dtype=np.int64)
                arrays[source] = (times, channels)
                check = check_timestamp_quality(times, channels, allowed_channels=allowed[source],
                                                required_channels=required[source], min_events_per_channel=minimum)
                source_checks[source] = check
                reasons.extend(f'{source}: {reason}' for reason in check['errors'])
                insufficient.extend(f'{source}: {reason}' for reason in check['insufficient'])
            status = 'data loss' if any(drops.values()) else ('invalid timestamps' if reasons else
                     ('insufficient data' if insufficient else 'checks passed'))
            report = dict(sequence=completed['sequence'], status=status,
                          reasons=reasons + insufficient, sources=source_checks,
                          network='Frame CRC/sequence checked on receipt; final SHA-256 pending session end')
            with self.lock:
                if self.quality is None or report['sequence'] >= self.quality['sequence']:
                    self.quality = report
            if completed['initialization']:
                if status == 'checks passed' and not self.integrity_failure and not self.stop_event.is_set():
                    self.initialization_status = 'Sweeping global frequency difference...'
                    self.initialization_worker = threading.Thread(
                        target=self._initialize_alignment, args=(arrays,),
                        name='qcs-initial-alignment', daemon=True)
                    self.initialization_worker.start()
                else:
                    self.initialization_status = 'Initialization failed: ' + status + '; ' + '; '.join(report['reasons'])
        finally:
            completed['buffers'].clear()

    def _initialize_alignment(self, arrays):
        """Apply the start-time offset, then sweep the rebased photon streams."""
        try:
            from qcs_analyzer.global_freq_analysis import global_freq_alignment_g2
            t1_ps, ch1 = arrays['ttx']
            t2_ps, ch2 = arrays['hh']
            # Hardware timestamps are integer picoseconds. Keep them integer
            # through the Unix-start/latency correction; converting large
            # timestamps to float before rebasing can discard low bits.
            if not np.issubdtype(t1_ps.dtype, np.integer) or not np.issubdtype(t2_ps.dtype, np.integer):
                raise TypeError('TTX and HydraHarp timestamps must be integer picoseconds')
            t1_ps = np.asarray(t1_ps, dtype=np.int64)
            t2_ps = np.asarray(t2_ps, dtype=np.int64)
            ch1 = np.asarray(ch1, dtype=np.int64)
            ch2 = np.asarray(ch2, dtype=np.int64)
            offset = int(self.config.get('hydraharp_channel_offset', 5))
            photon1 = int(self.config.get('init_ttx_photon_channel', 1))
            photon2 = int(self.config.get('init_hh_photon_channel', offset + 1))
            first = t1_ps[ch1 == photon1]
            second = t2_ps[ch2 == photon2]
            if first.size < 2 or second.size < 2:
                raise ValueError(f'Need at least two photon events on channels {photon1}/{photon2}')
            origins = (int(first[0]), int(second[0]))
            t01_us = self.ttx_start_time_us
            remote_start = self.remote.snapshot().get('remote_start_time_us')
            if t01_us is None or remote_start is None:
                raise RuntimeError('TTX and HH Unix start timestamps are required for global alignment')
            t02_us = int(remote_start)
            latency_ps = int(round(self.latency_ms * 1_000_000_000))
            offset_ps = (t02_us - int(t01_us)) * 1_000_000 - latency_ps
            first = (first - np.int64(origins[0])).astype(np.int64, copy=False)
            second = (second - np.int64(origins[1]) + np.int64(offset_ps)).astype(
                np.int64, copy=False
            )
            options = dict(self.config.get('global_alignment_kwargs', {}))
            # Matplotlib/Tk figures must be created by the GUI thread. The UI
            # plots the completed result after this worker returns.
            options['show_plot'] = False
            # Initial global-frequency sweep is called here, once per QCS session.
            result = global_freq_alignment_g2(first, second, **options)
            # Keep estimates/diagnostic curves, release the full aligned tag array.
            result.pop('t2_aligned_ps', None)
            result['start_time_offset_ps'] = offset_ps
            result['latency_ms'] = self.latency_ms
            result['ttx_start_time_us'] = int(t01_us)
            result['hh_start_time_us'] = t02_us
            with self.lock:
                self.timestamp_origins = origins
                self.alignment_result = result
                unique = result.get('unique_best_candidate', False)
                self.initialization_status = (
                    f"Global sweep complete: correction {result['skew_ps_per_s']:.2f} ps/s, "
                    f"offset {result['offset_ps'] * 1e-6:.6g} us" +
                    ('' if unique else ' (tied candidates; skew unresolved)'))
        except Exception as exc:
            with self.lock:
                self.initialization_status = 'Initialization failed: ' + str(exc)
        finally:
            arrays.clear()
            self._publish()

    def _record_integrity_failure(self, reason):
        with self.lock:
            self.integrity_failure = str(reason)

    def _receive_hh(self, data):
        from qcs_analyzer._implementation import decode_hydraharp_t2_records
        times, channels, self.overflow = decode_hydraharp_t2_records(
            np.frombuffer(data, dtype='<u4'), self.overflow,
            channel_offset=int(self.config.get('hydraharp_channel_offset', 5)),
            base_resolution_ps=self.resolution_ps)
        self._append('hh', times, channels)
        self._publish()

    def _read_ttx(self, reader):
        data = reader.getData()
        times = np.asarray(data.getTimestamps(), dtype=np.int64)
        if len(times) >= self.max_events['ttx'] or data.hasOverflows:
            self._record_integrity_failure('Time Tagger X overflow or full SDK buffer')
            raise RuntimeError('Time Tagger X buffer overflow; QCS data are incomplete')
        types = np.asarray(data.getEventTypes())
        if np.any(types != 0):
            self._record_integrity_failure('Time Tagger X error or lost-event marker')
            raise RuntimeError('Time Tagger X reported an error or lost-event marker')
        channels = self.normalize_ttx_channels(data.getChannels())
        self._append('ttx', times, channels)

    def _run_worker(self):
        try:
            while not self.stop_event.wait(self.interval_s):
                with self.lock:
                    reader = self.timetagger_reader
                if reader is not None:
                    self._read_ttx(reader)
                with self.lock:
                    completed = self._rotate_interval_locked(time.monotonic())
                if completed is not None:
                    self._handle_completed_interval(completed)
                state = self.remote.snapshot()
                if not state['active']:
                    if state['result'] is None or state['result']['status'] != 'complete':
                        raise RuntimeError(state['status'])
                    break
                self._publish()
        except Exception as exc:
            self.error = str(exc)
            self._record_integrity_failure(exc)
        finally:
            self.remote.stop_recording()
            with self.lock:
                self.stop_event.set()
                reader = self.timetagger_reader
                self.timetagger_reader = None
            if reader is not None:
                try:
                    reader.stop()
                    self._read_ttx(reader)
                except Exception as exc:
                    self.error = self.error or str(exc)
            with self.lock:
                for source in self.counts:
                    self.discarded[source] += self.counts[source]
                    self.counts[source] = 0
                    self.buffers[source].clear()
            self.running = False
            self.status = 'QCS error: ' + self.error if self.error else 'QCS stopped; memory buffers discarded'
            self._publish()

    def stop(self):
        """Request a nonblocking stop; GUI waits until both streams drain."""
        self.status = 'Stopping QCS acquisition...'
        self.stop_event.set()
        if self.remote is not None:
            self.remote.stop_recording()

    @property
    def busy(self):
        return (self.running or (self.remote is not None and self.remote.snapshot()['active'])
                or (self.initialization_worker is not None and self.initialization_worker.is_alive()))

    def snapshot_arrays(self):
        """Return copies of the current interval's t1_ps, ch1, t2_ps, ch2."""
        with self.lock:
            arrays = []
            for source in ('ttx', 'hh'):
                chunks = self.buffers[source]
                for index in (0, 1):
                    arrays.append(np.concatenate([chunk[index] for chunk in chunks])
                                  if chunks else np.empty(0, dtype=np.int64))
            return tuple(arrays)

    def _publish(self):
        with self.lock:
            elapsed = (max(0.0, time.monotonic() - self.session_started_monotonic)
                       if self.session_started_monotonic is not None else 0.0)
            update = dict(status=self.status, error=self.error, running=self.running,
                          retained=dict(self.counts), total=dict(self.total), evicted=dict(self.evicted),
                          completed_intervals=self.completed_intervals, discarded=dict(self.discarded),
                          quality=self.quality, integrity_failure=self.integrity_failure,
                          initialization_status=self.initialization_status,
                          elapsed_s=elapsed, ttx_channel_totals=dict(self.ttx_channel_totals))
        try:
            self.updates.get_nowait()
        except queue.Empty:
            pass
        try:
            self.updates.put_nowait(update)
        except queue.Full:
            pass

    def poll_update(self):
        try:
            return self.updates.get_nowait()
        except queue.Empty:
            return None
