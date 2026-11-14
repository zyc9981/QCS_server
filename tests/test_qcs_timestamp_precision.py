"""Precision regressions using synthetic tags, without acquisition hardware."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from QCS_algorithm import RealtimeG2Controller, g2_histogram_numba
from qcs_analyzer._implementation import decode_hydraharp_t2_records
from qcs_analyzer.global_freq_analysis import global_freq_alignment_g2
from qcs_analyzer.timestamp_math import (
    INT64_MAX, INT64_MIN, int64_timestamps, integer_scalar,
    milliseconds_to_picoseconds, midpoint_elapsed_picoseconds, shift_picoseconds,
)


class TimestampPrecisionTests(unittest.TestCase):
    def test_decimal_latency_and_explicit_ps_rounding(self):
        self.assertEqual(milliseconds_to_picoseconds('20.000000001'), 20_000_000_001)
        self.assertEqual(milliseconds_to_picoseconds('9007199.254740993'), 2**53 + 1)
        self.assertEqual(milliseconds_to_picoseconds('0.0000000025'), 2)
        self.assertEqual(milliseconds_to_picoseconds('0.0000000035'), 4)
        self.assertEqual(milliseconds_to_picoseconds('0.000000002500000000000000000001'), 3)
        for value in ('nan', 'inf', '-1'):
            with self.assertRaises(ValueError):
                milliseconds_to_picoseconds(value)

    def test_integer_inputs_do_not_silently_truncate_or_wrap(self):
        with self.assertRaises(TypeError):
            int64_timestamps(np.array([1.1, 2.1]))
        with self.assertRaises(OverflowError):
            int64_timestamps(np.array([2**63], dtype=np.uint64))
        with self.assertRaises(TypeError):
            integer_scalar(1794434110689783.0, 'Unix us')

    def test_rebase_preserves_ps_above_float_exact_integer_limit(self):
        tags = np.array([INT64_MAX - 20, INT64_MAX - 17, INT64_MAX - 1])
        output = np.empty(3, dtype=np.int64)
        shift_picoseconds(tags, -int(tags[0]), out=output)
        np.testing.assert_array_equal(output, [0, 3, 19])
        for tags, shift in ((np.array([INT64_MAX]), 1), (np.array([INT64_MIN]), -1)):
            with self.assertRaises(OverflowError):
                shift_picoseconds(tags, shift)

    def test_midpoint_sum_and_delta_are_exact_on_long_runs(self):
        base = 9_000_000_000_000_000_000
        previous_twice = 2 * base + 1
        state = {'T_lastmid_ps': float(base), 'T_lastmid_twice_ps': previous_twice}
        twice, delta = midpoint_elapsed_picoseconds(
            np.int64(base + 1), np.int64(base + 4), state)
        self.assertEqual(twice, 2 * base + 5)
        self.assertEqual(delta, 2.0)
        # A legacy/manual half-ps midpoint is also accepted exactly.
        self.assertEqual(midpoint_elapsed_picoseconds(3, 4, {'T_lastmid_ps': 1.5})[1], 2.0)

    def test_live_histogram_is_invariant_to_a_large_common_origin(self):
        first = np.array([0, 1000, 2000], dtype=np.int64)
        second = first + 3
        edges = np.array([-4, -2, 0, 2, 4, 6], dtype=float)
        base = (1 << 60) + 12345
        kwargs = dict(Ta_s=1, time_bin_ps_init=1)
        expected = g2_histogram_numba(first, second, edges, **kwargs)['histogram_raw']
        actual = g2_histogram_numba(first + base, second + base, edges, **kwargs)['histogram_raw']
        np.testing.assert_array_equal(expected, [0, 0, 0, 3, 0])
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaises(OverflowError):
            g2_histogram_numba(np.array([INT64_MIN]), np.array([INT64_MAX]), edges, **kwargs)

    def test_decoder_retains_low_ps_bits_and_rejects_overflow(self):
        ticks = 2**53 + 1
        times, channels, correction = decode_hydraharp_t2_records(
            np.array([3, 5], dtype=np.uint32), ticks, channel_offset=5)
        np.testing.assert_array_equal(times, [ticks + 3, ticks + 5])
        np.testing.assert_array_equal(channels, [6, 6])
        self.assertEqual(correction, ticks)
        resolution = 2**53 + 1
        scaled, _, _ = decode_hydraharp_t2_records(
            np.array([1], dtype=np.uint32), base_resolution_ps=resolution)
        self.assertEqual(int(scaled[0]), resolution)
        with self.assertRaises(ValueError):
            decode_hydraharp_t2_records(np.array([1], dtype=np.uint32), base_resolution_ps=1.5)
        for records, correction in (([10], INT64_MAX - 3),
                                     ([0xFE000001], INT64_MAX - 10)):
            with self.assertRaises(OverflowError):
                decode_hydraharp_t2_records(np.array(records, dtype=np.uint32), correction)

    def test_preallocated_buffers_preserve_integer_rebasing(self):
        controller = RealtimeG2Controller(SimpleNamespace(), {
            'max_events_ttx': 8, 'max_events_hh': 8, 'latency_ms': '20.000000001'})
        self.assertEqual(controller.latency_ps, 20_000_000_001)
        origin = (1 << 60) + 7
        controller.live_t10_ps = origin
        controller.live_t20_ps = origin - 10
        controller._append('ttx', np.array([origin + 1, origin + 2]), np.array([1, 2]))
        controller._append('hh', np.array([origin + 3]), np.array([6]))
        self.assertEqual(controller.channel_buffer['data'][('ttx', 1)][0], 1)
        self.assertEqual(controller.channel_buffer['data'][('ttx', 2)][0], 2)
        self.assertEqual(controller.channel_buffer['data'][('hh', 6)][0], 13)

    def test_live_skew_uses_exact_previous_midpoint_not_display_float(self):
        controller = RealtimeG2Controller(SimpleNamespace(), {
            'max_events_ttx': 8, 'max_events_hh': 8})
        base = 9_000_000_000_000_000_000
        state = dict(time_bin_ps=20., tau_span_ps=5000., tau_span_min_ps=5000.,
                     tau_ps=0., T_lastmid_ps=float(base),
                     T_lastmid_twice_ps=2 * (base - 2_000_000_000_000 + 50))
        tags = np.array([base, base + 100], dtype=np.int64)
        histogram = {'centers': np.array([0.]), 'histogram': np.array([1.])}
        with patch('QCS_algorithm.g2_histogram_numba', return_value=histogram), \
                patch('QCS_algorithm.fit_g2', side_effect=[
                    {'success': True, 'center': 2.}, {'success': True, 'center': 4.}]):
            controller._analyze_live_interval(1, tags, tags, state)
            self.assertEqual(state['u0'], 1e-12)
            self.assertEqual(state['T_lastmid_twice_ps'], 2 * base + 100)
            tags = tags + 2_000_000_000_000
            controller._analyze_live_interval(2, tags, tags, state)
            self.assertEqual(state['u0'], 1e-12)

    def test_global_alignment_is_invariant_to_large_absolute_tags(self):
        first = np.arange(100, dtype=np.int64) * 10000
        second = first + 137
        options = dict(init_time_offset_range_ms=0.000001, init_bin_number=64,
                       init_fft_method='full', sweep_range_ps_per_s=100,
                       fine_sweep_range_ps_per_s=10, coarse_steps=3, fine_steps=3,
                       time_offset_range_ns=1, max_bin_number=100, sweep_workers=1)
        expected = global_freq_alignment_g2(first, second, **options)
        origin = INT64_MAX - 2_000_000
        actual = global_freq_alignment_g2(first + origin, second + origin, **options)
        np.testing.assert_array_equal(actual['g2_histogram'], expected['g2_histogram'])
        np.testing.assert_array_equal(actual['init_g2_histogram'], expected['init_g2_histogram'])
        self.assertEqual(actual['offset_ps'], expected['offset_ps'])
        with self.assertRaises(ValueError):
            global_freq_alignment_g2(np.array([0, 2**53 + 1]), np.array([1, 2**53]), **options)

    def test_initialization_unit_conversion_stays_integer(self):
        controller = RealtimeG2Controller(SimpleNamespace(), {
            'latency_ms': '20.000000001', 'max_events_ttx': 8, 'max_events_hh': 8})
        controller.ttx_start_time_us = 1794434110142599
        controller.remote = SimpleNamespace(snapshot=lambda: {
            'remote_start_time_us': controller.ttx_start_time_us + 128097,
            'clock_adjustment_supported': False})
        origin = (1 << 60) + 7
        arrays = {'ttx_init': np.array([origin, origin + 3]),
                  'hh_init': np.array([origin + 7000, origin + 7005])}
        with patch('qcs_analyzer.global_freq_analysis.global_freq_alignment_g2',
                   return_value={'unique_best_candidate': False,
                                 'skew_ps_per_s': 0., 'residual_offset_ps': 0.}) as align:
            controller._initialize_alignment(arrays)
        first, second = align.call_args.args
        self.assertEqual(first.dtype, np.int64)
        self.assertEqual(second.dtype, np.int64)
        np.testing.assert_array_equal(first, [0, 3])
        expected_offset = 128097 * 1_000_000 - 20_000_000_001
        np.testing.assert_array_equal(second, [expected_offset, expected_offset + 5])


if __name__ == '__main__':
    unittest.main()
