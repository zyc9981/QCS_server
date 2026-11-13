"""Live in-memory QCS acquisition, alignment, and interval clock tracking."""
from __future__ import annotations

import csv
from collections import deque
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from numba import njit
from qcs_analyzer._implementation import *

import numpy as np


DEFAULT_QCS_CONFIG = {
    'init_interval_s': 2.0,
    'acquisition_interval_s': 1.0,

    'latency_ms': 500.0,
    'init_time_offset_range_ms': 40.0,
    'init_bin_number': 1_000_000,

    'sweep_range_ps_per_s': 2000.0,
    'coarse_steps': 40,

    'fine_steps': 20,
    'fine_sweep_range_ps_per_s': 100.0,

    # Temporary diagnostic gain; restore 1.0 for full live correction.
    'live_correction_gain': 0.1,
}

MAX_TAU_HISTORY = 86_400


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


def g2_histogram_numba(
    acquisition_a,
    acquisition_b,
    edges,
    *,
    Ta_s,
    chA,
    chB,
    time_bin_ps_init,
):
    """
    Build a coincidence histogram and return a fit_g2-compatible dictionary.

    Requirements:
        - Acquisitions are sorted, finite, 1-D NumPy timestamp arrays.
        - Timestamps and edges use picoseconds.
        - Edges are uniformly spaced.
        - Bins include their lower edge and exclude their upper edge.
    """
    acquisition_a = np.asarray(acquisition_a)
    acquisition_b = np.asarray(acquisition_b)
    edges = np.asarray(edges, dtype=np.float64)

    if acquisition_a.ndim != 1 or acquisition_b.ndim != 1:
        raise ValueError("Acquisitions must be one-dimensional.")

    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("edges must contain at least two boundaries.")

    widths = np.diff(edges)
    time_bin_ps = widths[0]

    if (
        not np.all(np.isfinite(edges))
        or np.any(widths <= 0)
        or not np.allclose(widths, time_bin_ps, rtol=1e-8, atol=0)
    ):
        raise ValueError("edges must be finite, increasing, and uniform.")

    if not np.isfinite(time_bin_ps_init) or time_bin_ps_init <= 0:
        raise ValueError("time_bin_ps_init must be finite and positive.")
    if not np.isfinite(Ta_s) or Ta_s <= 0:
        raise ValueError("Ta_s must be finite and positive.")

    # Create the compiled helper once, then reuse it on subsequent calls.
    if not hasattr(g2_histogram_numba, "_count_pairs"):

        @njit(fastmath=False)
        def count_pairs(t_a, t_b, bin_edges):
            n_bins = len(bin_edges) - 1
            bin_width = bin_edges[1] - bin_edges[0]
            lower = bin_edges[0]
            upper = bin_edges[-1]

            counts = np.zeros(n_bins, dtype=np.int64)
            n_b = len(t_b)
            j_start = 0

            for i in range(len(t_a)):
                timestamp_a = t_a[i]

                while j_start < n_b:
                    if t_b[j_start] - timestamp_a >= lower:
                        break
                    j_start += 1

                if j_start == n_b:
                    break

                j = j_start
                while j < n_b:
                    delay = t_b[j] - timestamp_a

                    if delay >= upper:
                        break

                    k = int(np.floor((delay - lower) / bin_width))

                    # Resolve floating-point rounding at bin boundaries.
                    k = min(max(k, 0), n_bins - 1)

                    while k > 0 and delay < bin_edges[k]:
                        k -= 1
                    while k < n_bins - 1 and delay >= bin_edges[k + 1]:
                        k += 1

                    counts[k] += 1
                    j += 1

            return counts

        g2_histogram_numba._count_pairs = count_pairs

    counts = g2_histogram_numba._count_pairs(
        acquisition_a, acquisition_b, edges
    )
    centers = edges[:-1] + widths / 2.0

    return {
        "centers": centers.copy(),
        "histogram": counts / time_bin_ps * time_bin_ps_init,
        "histogram_raw": counts.copy(),
        "T_acq_s": float(Ta_s),
        "chA": chA,
        "chB": chB,
    }


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
        self.live_clock_adjustment_enabled = (
            self.config.get('live_clock_adjustment_enabled', True) is True
        )
        self.live_correction_gain = float(self.config.get(
            'live_correction_gain', DEFAULT_QCS_CONFIG['live_correction_gain']))
        if not np.isfinite(self.live_correction_gain) or self.live_correction_gain <= 0:
            raise ValueError('live_correction_gain must be positive and finite')
        alignment_options = dict(self.config.get('global_alignment_kwargs', {}))
        alignment_options.setdefault(
            'sweep_range_ps_per_s', DEFAULT_QCS_CONFIG['sweep_range_ps_per_s'])
        alignment_options.setdefault(
            'init_time_offset_range_ms', DEFAULT_QCS_CONFIG['init_time_offset_range_ms'])
        alignment_options.setdefault('init_bin_number', DEFAULT_QCS_CONFIG['init_bin_number'])
        alignment_options.setdefault('coarse_steps', DEFAULT_QCS_CONFIG['coarse_steps'])
        alignment_options.setdefault('fine_steps', DEFAULT_QCS_CONFIG['fine_steps'])
        alignment_options.setdefault(
            'fine_sweep_range_ps_per_s', DEFAULT_QCS_CONFIG['fine_sweep_range_ps_per_s'])
        steps = float(alignment_options['coarse_steps'])
        if not np.isfinite(steps) or not steps.is_integer() or steps < 3:
            raise ValueError('sweep_num must be an integer of at least 3')
        alignment_options['coarse_steps'] = int(steps)
        fine_steps = float(alignment_options['fine_steps'])
        if not np.isfinite(fine_steps) or not fine_steps.is_integer() or fine_steps < 3:
            raise ValueError('fine_steps must be an integer of at least 3')
        alignment_options['fine_steps'] = int(fine_steps)
        init_bins = float(alignment_options['init_bin_number'])
        if not np.isfinite(init_bins) or not init_bins.is_integer() or init_bins < 2:
            raise ValueError('init_bin_number must be an integer of at least 2')
        alignment_options['init_bin_number'] = int(init_bins)
        for name in ('sweep_range_ps_per_s', 'init_time_offset_range_ms',
                     'fine_sweep_range_ps_per_s'):
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
        self.clock_command_id = None
        self.initial_clock_adjustment_applied = False
        self.live_discard_through_sequence = None
        self.live_interval_queue = queue.Queue(maxsize=2)
        self.live_analysis_worker = None
        self.live_state = None
        self.live_t10_ps = None
        self.live_t20_ps = None
        self.calibrated_offset_ps = None
        # Cumulative QCS-applied deltas relative to the node's clock setting
        # at the start of this session; node commands themselves are deltas.
        self.qcs_clock_adjustment_total_ps_per_s = 0.0
        self.live_step_result = None
        self.live_analysis_steps = 0
        self.tau_time_s_history = np.full(MAX_TAU_HISTORY, np.nan, dtype=np.float64)
        self.tau_ps_history = np.full(MAX_TAU_HISTORY, np.nan, dtype=np.float64)
        self.tau_history_count = 0
        self.tau_history_version = 0
        self.live_log_origin_ns = None
        self.live_log_path = None
        self.live_log_lock = threading.Lock()
        self.live_log_error_reported = False
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
            self.live_analysis_worker = threading.Thread(
                target=self._live_analysis_worker, name='qcs-live-analysis', daemon=True)
            self.live_analysis_worker.start()
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
            now_ns = time.monotonic_ns()
            completed = self._rotate_interval_locked(now_ns * 1e-9, now_ns)
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

    def _rotate_interval_locked(self, now, now_ns=None):
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
                        finished_monotonic_ns=(int(now_ns) if now_ns is not None
                                               else time.monotonic_ns()),
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
            elif not completed['initialization']:
                # Skip intervals until the initial clock write is acknowledged,
                # including the interval that overlaps the write itself.
                with self.lock:
                    live_ready = self.initial_clock_adjustment_applied
                    discard_through = self.live_discard_through_sequence
                if live_ready and completed['sequence'] > discard_through:
                    acquired_ns = completed['finished_monotonic_ns']
                    self._log_live_tracking_event(
                        'acquisition_finished', completed['sequence'], acquired_ns,
                        acquisition_status=status,
                        details='; '.join(report['reasons']),
                    )
                    if status == 'checks passed' and not self.integrity_failure and not self.stop_event.is_set():
                        try:
                            self.live_interval_queue.put_nowait((completed['sequence'], arrays))
                        except queue.Full:
                            self._record_integrity_failure(
                                'Live analysis fell behind; a completed interval could not be queued')
        finally:
            completed['buffers'].clear()

    def _send_clock_adjustment(self, correction_ps_per_s):
        """Send a correction through the shared remote command path."""
        command_id = self.remote.set_clock_correction(correction_ps_per_s)
        with self.lock:
            self.clock_command_id = command_id
        return command_id

    def _wait_for_clock_adjustment(self, command_id):
        """Wait for the matching remote CLOCK_RESULT terminal status."""
        terminal = {'applied', 'failed', 'timed_out', 'connection_lost'}
        while True:
            result = self.remote.get_clock_command(command_id)
            if result is None:
                return {'status': 'missing', 'message': 'Remote command result was not found'}
            if result.get('status') in terminal:
                return result
            if self.stop_event.wait(0.05):
                return {'status': 'stopped', 'message': 'QCS stopped while waiting for clock adjustment'}

    def _analyze_live_interval(
            self,
            sequence,
            t1_ps,
            ch1,
            t2_ps,
            ch2,
            state,
            ch_0=1,
            ch_1=6,
            ch_2=2,
            ch_ref_0=3,
            ch_ref_1=7,

            t10_ps=0,
            t20_ps=0,
            fit_curve="g2calib_2",
            fit_parameters={
                "height": 40,
                "sigma": 100,
                "offset": 1,
            },
            # sour_freq=100e6,
            fwhm_calib_ps=1000,
            second_order_fit=False,
            show_fit=False,
            Freq=50e6,

            T_calib = 0.0,
            T_calib_rt = 0.0,
            ):
        """Analyze one interval, update and return persistent tracking state."""
        if not isinstance(state, dict):
            raise TypeError("state must be a mutable dictionary")

        # ------------------------------------------------------------
        # STEP 1:
        # Prepare for fitting and measurement.
        # ------------------------------------------------------------
        fit_curve = FIT_CURVE_ALIASES.get(str(fit_curve).strip().lower(), str(fit_curve).strip().lower())
        if fit_curve not in FIT_CURVES:
            available = ", ".join(sorted(FIT_CURVES))
            raise ValueError(f"Unknown fit_curve '{fit_curve}'. Choose one of: {available}.")
        try:
            fit_parameter_guess = {} if fit_parameters is None else dict(fit_parameters)
        except (TypeError, ValueError) as exc:
            raise TypeError("fit_parameters must be a dictionary of initial fit values.") from exc

        # Keep each channel chronological.
        t_0 = t1_ps[ch1 == ch_0] - t10_ps
        # t_ref_0 = t1_ps[ch1 == ch_ref_0]
        t_1 = t2_ps[ch2 == ch_1] - t20_ps
        # t_ref_1 = t2_ps[ch2 == ch_ref_1]
        t_2 = t1_ps[ch1 == ch_2] - t10_ps
        # print("TTX: ", t_0[0], "HH: ", t_1[0])

        # ------------------------------------------------------------
        # STEP 2:
        # g2 processing.
        # ------------------------------------------------------------
        t_lo = state["tau_ps"] - state["tau_span_ps"] * 0.5
        t_hi = state["tau_ps"] + state["tau_span_ps"] * 0.5
        time_bin_ps = state["time_bin_ps"]
        edges = np.arange(t_lo, t_hi + time_bin_ps, time_bin_ps)
        centers = edges[:-1] + time_bin_ps / 2
        start_stop_results = np.zeros(len(centers), dtype=np.int64)

        results = g2_histogram_numba(
            t_0,
            t_1,
            edges,
            Ta_s=self.acquisition_interval_s,
            chA=ch_0,
            chB=ch_1,
            time_bin_ps_init=state["time_bin_ps"],
        )

        t_lo_rt = state["tau_rt_ps"] - state["tau_span_ps"] * 0.5
        t_hi_rt = state["tau_rt_ps"] + state["tau_span_ps"] * 0.5
        time_bin_rt_ps = state["time_bin_rt_ps"]
        edges_rt = np.arange(t_lo_rt, t_hi_rt + time_bin_rt_ps, time_bin_rt_ps)
        centers_rt = edges_rt[:-1] + time_bin_rt_ps / 2
        start_stop_results_rt = np.zeros(len(centers_rt), dtype=np.int64)

        results_rt = g2_histogram_numba(
            t_0,
            t_2,
            edges_rt,
            Ta_s=self.acquisition_interval_s,
            chA=ch_0,
            chB=ch_2,
            time_bin_ps_init=state["time_bin_rt_ps"],
        )

        # save_g2_results(results_rt, output_filename="./test.npz")

        # ------------------------------------------------------------
        # STEP 3:
        # Estimate time offset and frequency skew from g2 fit.
        # ------------------------------------------------------------

        try:
            fit_parameters = fit_parameter_guess.copy()
            fit_parameters["center"] = float(
                results["centers"][np.argmax(results["histogram"])]
            )
            fit = fit_g2(
                results,
                curve=fit_curve,
                parameters=fit_parameters,
                fit_window=np.rint(float(1e12)/Freq) * 1.2,
                show=show_fit,
            )
        except Exception as exc:
            raise

            print(f"Time-dependent g2 fit failed.")
            fit = None
            exc_value = exc
        else:
            exc_value = None

        try:
            fit_parameters = fit_parameter_guess.copy()
            fit_parameters["center"] = float(
                results_rt["centers"][np.argmax(results_rt["histogram"])]
            )
            # print(fit_parameters["center"])
            fit_rt = fit_g2(
                results_rt,
                curve=fit_curve,
                parameters=fit_parameters,
                fit_window=np.rint(float(1e12)/Freq) * 1.2,
                show=show_fit,
                # show=True,
            )
        except Exception as exc:
            raise

            print(f"Time-dependent g2 fit failed.")
            fit = None
            exc_value = exc
        else:
            exc_value = None

        if not fit.get("success", True):
            raise RuntimeError(
                f"g2 fit failed for channels {ch_0}/{ch_1}: {fit.get('message') or 'optimizer did not converge'}"
            )
        if not fit_rt.get("success", True):
            raise RuntimeError(
                f"g2 fit failed for channels {ch_0}/{ch_2}: {fit_rt.get('message') or 'optimizer did not converge'}"
            )

        T_mid_ps = (t_0[0] + t_0[-1]) / 2
        elapsed_mid_ps = T_mid_ps - state["T_lastmid_ps"]
        if elapsed_mid_ps <= 0:
            raise ValueError("Live interval midpoint did not advance")
        state["u0"] = (fit["center"] - state["tau_ps"]) / elapsed_mid_ps
        state["tau_ps"] = fit["center"]
        # print(state["tau_ps"])
        state["tau_rt_ps"] = fit_rt["center"]
        state["epsilon_ps"] = state["tau_ps"] - state["tau_rt_ps"] / 2
        state["T_lastmid_ps"] = T_mid_ps

        return state

    def _append_tau_history(self, elapsed_s, tau_ps):
        """Store up to one day of live offset estimates without rolling."""
        with self.lock:
            if self.tau_history_count >= MAX_TAU_HISTORY:
                return
            index = self.tau_history_count
            self.tau_time_s_history[index] = float(elapsed_s)
            self.tau_ps_history[index] = float(tau_ps)
            self.tau_history_count += 1
            self.tau_history_version += 1

    def _log_live_tracking_event(self, event, sequence, event_monotonic_ns, **values):
        """Append a structured event with time relative to the first live interval."""
        fields = (
            'event', 'interval', 'relative_s', 'tau_ps',
            'u0_ps_per_s', 'T_lastmid_ms',
            'correction_ps_per_s', 'applied_correction_ps_per_s',
            'clock_status', 'command_id', 'acquisition_status', 'details',
        )
        try:
            with self.live_log_lock:
                if self.live_log_origin_ns is None:
                    origin_ns = int(event_monotonic_ns)
                    log_dir = Path(__file__).resolve().parent / 'qcs_live_tracking_logs'
                    log_dir.mkdir(parents=True, exist_ok=True)
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
                    log_path = log_dir / f'qcs_live_tracking_{stamp}.csv'
                    with log_path.open('w', newline='', encoding='utf-8') as file:
                        csv.DictWriter(file, fieldnames=fields).writeheader()
                    self.live_log_path = log_path
                    self.live_log_origin_ns = origin_ns
                    print(f'QCS live tracking log: {self.live_log_path}', flush=True)

                relative_ns = int(event_monotonic_ns) - self.live_log_origin_ns
                row = {
                    'event': event,
                    'interval': int(sequence),
                    'relative_s': f'{relative_ns * 1e-9:.9f}',
                }
                row.update(values)
                with self.live_log_path.open('a', newline='', encoding='utf-8') as file:
                    csv.DictWriter(file, fieldnames=fields).writerow(row)
                    file.flush()
        except Exception as exc:
            if not self.live_log_error_reported:
                self.live_log_error_reported = True
                print(f'QCS live tracking logger failed: {exc}', flush=True)

    def tau_history_snapshot(self, known_version=-1):
        """Return a stable copy of new history data for the GUI thread."""
        with self.lock:
            version = self.tau_history_version
            if version == known_version:
                return version, None, None
            count = self.tau_history_count
            if count == 0:
                return version, None, None
            return (version, self.tau_time_s_history[:count].copy(),
                    self.tau_ps_history[:count].copy())

    def _live_analysis_worker(self):
        """Process completed intervals sequentially outside acquisition callbacks."""
        while not self.stop_event.is_set():
            try:
                item = self.live_interval_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if item is None:
                    return

                # t0 = time.time()
                sequence, arrays = item
                (t1_ps, ch1), (t2_ps, ch2) = arrays['ttx'], arrays['hh']
                if self.live_state is None or self.live_t10_ps is None or self.live_t20_ps is None:
                    raise RuntimeError('Live tracking state was not initialized after alignment')
                state = self._analyze_live_interval(
                    sequence, t1_ps, ch1, t2_ps, ch2,
                    state=self.live_state,
                    t10_ps=self.live_t10_ps,
                    t20_ps=self.live_t20_ps,
                )
                analysis_finished_ns = time.monotonic_ns()
                # print("Processing: ", time.time() - t0)
                residual_drift_ps_per_s = float(state['u0']) * 1e12
                self._log_live_tracking_event(
                    'analysis_finished', sequence, analysis_finished_ns,
                    tau_ps=state['tau_ps'],
                    u0_ps_per_s=residual_drift_ps_per_s,
                    T_lastmid_ms=state['T_lastmid_ps'] * 1e-9,
                )

                self.live_state = state
                self._append_tau_history(state['T_lastmid_ps'] * 1e-12, state['tau_ps'])
                self.live_analysis_steps += 1
                first_live_step = self.live_analysis_steps == 1
                with self.lock:
                    self.live_step_result = {
                        'sequence': int(sequence),
                        'tau_ps': float(state['tau_ps']),
                        'u0_ps_per_s': residual_drift_ps_per_s,
                        'clock_status': (
                            'unchanged (first live fit)' if first_live_step else 'pending'
                        ),
                    }
                self._publish()
                if first_live_step:
                    # Use the first live fit as a measurement only. Start
                    # applying interval corrections from the next fit onward.
                    self._log_live_tracking_event(
                        'clock_adjustment_skipped', sequence, time.monotonic_ns(),
                        tau_ps=state['tau_ps'],
                        u0_ps_per_s=residual_drift_ps_per_s,
                        T_lastmid_ms=state['T_lastmid_ps'] * 1e-9,
                        clock_status='first live fit is measurement-only',
                    )
                    continue
                if not self.live_clock_adjustment_enabled:
                    with self.lock:
                        self.live_step_result['clock_status'] = 'disabled'
                    self._publish()
                    self._log_live_tracking_event(
                        'clock_adjustment_skipped', sequence, time.monotonic_ns(),
                        tau_ps=state['tau_ps'],
                        u0_ps_per_s=residual_drift_ps_per_s,
                        T_lastmid_ms=state['T_lastmid_ps'] * 1e-9,
                        clock_status='live clock adjustment disabled',
                    )
                    continue

                # The node treats each command as a signed increment to its
                # current clock correction. Scale the opposite residual by
                # the diagnostic feedback gain; fitted u0 remains unscaled.
                correction_ps_per_s = -self.live_correction_gain * residual_drift_ps_per_s
                command_id = self._send_clock_adjustment(correction_ps_per_s)
                command_result = self._wait_for_clock_adjustment(command_id)
                clock_finished_ns = time.monotonic_ns()
                command_result_details = command_result.get('result') or {}
                self._log_live_tracking_event(
                    'clock_adjustment_finished', sequence, clock_finished_ns,
                    tau_ps=state['tau_ps'],
                    u0_ps_per_s=residual_drift_ps_per_s,
                    T_lastmid_ms=state['T_lastmid_ps'] * 1e-9,
                    correction_ps_per_s=correction_ps_per_s,
                    applied_correction_ps_per_s=command_result_details.get(
                        'applied_correction_ps_per_s'),
                    clock_status=('unchanged (below resolution)' if command_result.get('no_change')
                                  else command_result.get('status')),
                    command_id=command_id,
                    details=command_result_details.get('message', ''),
                )
                if command_result.get('status') != 'applied':
                    raise RuntimeError(
                        f"Clock command {command_id}: {command_result.get('status')}: "
                        f"{command_result.get('message', '')}")
                applied = command_result.get('applied_correction_ps_per_s')
                applied_delta_ps_per_s = (
                    float(applied) if applied is not None else correction_ps_per_s
                )
                self.qcs_clock_adjustment_total_ps_per_s += applied_delta_ps_per_s
                with self.lock:
                    self.live_step_result['clock_status'] = (
                        'unchanged (below resolution)' if command_result.get('no_change')
                        else 'adjustment applied'
                    )
                    self.live_step_result['applied_clock_delta_ps_per_s'] = applied_delta_ps_per_s
                    self.live_step_result['qcs_clock_adjustment_total_ps_per_s'] = (
                        self.qcs_clock_adjustment_total_ps_per_s
                    )
                self._publish()
                # print("Send: ", time.time() - t0)
            except Exception as exc:
                message = f'Live analysis/adjustment failed: {exc}'
                print(f'QCS live tracking error (interval {sequence}): {message}', flush=True)
                self._record_integrity_failure(message)
                self._publish()
            finally:
                self.live_interval_queue.task_done()

    def _initialize_alignment(self, arrays):
        """Find global alignment, command the correction, and await its result."""
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
            unique = result.get('unique_best_candidate', False)
            command_summary = 'not sent'
            adjustment_applied = False
            if not unique:
                clock_status = 'skipped: alignment did not identify a unique skew'
            elif not self.remote.snapshot().get('clock_adjustment_supported', False):
                clock_status = 'skipped: remote node has not advertised clock-adjustment support'
            else:
                try:
                    command_id = self._send_clock_adjustment(result['skew_ps_per_s'])
                    result['clock_command_id'] = command_id
                    command_summary = f'sent #{command_id}'
                    with self.lock:
                        self.initialization_status = (
                            f"Offset: {(latency_ps + result['residual_offset_ps']) * 1e-9:.6g} ms | "
                            f"correction: {result['skew_ps_per_s']:+.2f} ps/s | "
                            f"command: {command_summary}")
                    self._publish()
                    command_result = self._wait_for_clock_adjustment(command_id)
                    command_state = command_result.get('status', 'unknown')
                    if command_state == 'applied':
                        adjustment_applied = True
                        if command_result.get('no_change'):
                            command_summary = f'unchanged #{command_id}'
                            clock_status = f'unchanged (below resolution), command {command_id}'
                        else:
                            command_summary = f'applied #{command_id}'
                            clock_status = f'applied command {command_id}'
                        applied = command_result.get('applied_correction_ps_per_s')
                        applied_delta_ps_per_s = (
                            float(applied) if applied is not None
                            else float(result['skew_ps_per_s'])
                        )
                        self.qcs_clock_adjustment_total_ps_per_s += applied_delta_ps_per_s
                    else:
                        command_summary = f'{command_state} #{command_id}'
                        clock_status = (
                            f"{command_state}: "
                            f"{command_result.get('message', 'remote adjustment did not complete')}")
                except Exception as exc:
                    clock_status = f'not sent: {exc}'
            result['clock_command_status'] = clock_status
            # Keep estimates/diagnostic curves, release the full aligned tag array.
            result.pop('t2_aligned_ps', None)
            result['start_time_offset_ps'] = offset_ps
            result['latency_ms'] = self.latency_ms
            # Report the requested sum; offset_ps itself remains the residual
            # correction applied to the already pre-aligned HH timestamps.
            result['final_offset_ps'] = latency_ps + result['residual_offset_ps']
            result['final_offset_ms'] = result['final_offset_ps'] * 1e-9
            result['ttx_start_time_us'] = int(t01_us)
            result['hh_start_time_us'] = t02_us
            print(
                f"QCS final alignment: skew={result['skew_ps_per_s']:+.3f} ps/s, "
                f"final offset={result['final_offset_ms']:.9g} ms "
                f"(latency {self.latency_ms:g} ms + residual "
                f"{result['residual_offset_ps'] * 1e-6:.6g} us); "
                f"clock command: {clock_status}",
                flush=True,
            )
            with self.lock:
                self.timestamp_origins = origins
                self.alignment_result = result
                if adjustment_applied:
                    # The global routine reports the additive residual needed
                    # on the HH-relative timestamps. Express that calibrated
                    # offset as the initial Unix/latency offset minus the
                    # measured residual delay (which is -residual_offset_ps).
                    calibrated_offset_ps = int(offset_ps) + int(round(
                        result['residual_offset_ps']))
                    self.calibrated_offset_ps = calibrated_offset_ps
                    self.live_t10_ps = int(origins[0])
                    self.live_t20_ps = int(origins[1]) - calibrated_offset_ps
                    self.live_state = {
                        'time_bin_ps': 20.0,
                        'time_bin_rt_ps': 20.0,
                        'tau_span_ps': 500_000.0,
                        'tau_ps': 0.0,
                        'tau_rt_ps': 58_968_630.0,
                        'epsilon_ps': 0.0,
                        'u0': 0.0,
                        'T_lastmid_ps': 0.0,
                    }
                    self.initial_clock_adjustment_applied = True
                    # The active interval can contain events captured before
                    # the remote write completed. Skip it and start on the
                    # following full interval.
                    self.live_discard_through_sequence = self.completed_intervals + 1
                self.initialization_status = (
                    f"Offset: {result['final_offset_ms']:.6g} ms | "
                    f"correction: {result['skew_ps_per_s']:+.2f} ps/s | "
                    f"command: {command_summary}")
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
                    now_ns = time.monotonic_ns()
                    completed = self._rotate_interval_locked(now_ns * 1e-9, now_ns)
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
            clock_command_id = self.clock_command_id
            update = dict(status=self.status, error=self.error, running=self.running,
                          retained=dict(self.counts), total=dict(self.total), evicted=dict(self.evicted),
                          completed_intervals=self.completed_intervals, discarded=dict(self.discarded),
                          quality=self.quality, integrity_failure=self.integrity_failure,
                          initialization_status=self.initialization_status,
                          live_step_result=(dict(self.live_step_result)
                                            if self.live_step_result is not None else None),
                          elapsed_s=elapsed, ttx_channel_totals=dict(self.ttx_channel_totals))
        update['clock_command'] = (
            self.remote.get_clock_command(clock_command_id)
            if self.remote is not None and clock_command_id is not None else None
        )
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
