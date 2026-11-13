"""Clock response handling without hardware, sockets, or a Tk window."""
from collections import deque
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from internet_timetag import InternetCoordinator
from QCS_algorithm import RealtimeG2Controller


class ClockAdjustmentResultTests(unittest.TestCase):
    def coordinator(self):
        remote = InternetCoordinator.__new__(InternetCoordinator)
        remote.lock = threading.Lock()
        remote.clock_commands = {}
        return remote

    def result(self, remote, key, *, success=True, no_change=False, applied=0.0, message=''):
        remote.clock_commands[key] = {
            'command_id': key, 'status': 'applied' if success else 'failed',
            'result': {'success': success, 'no_change': no_change,
                       'applied_correction_ps_per_s': applied, 'message': message},
        }

    def controller(self, remote):
        controller = RealtimeG2Controller(SimpleNamespace(time_tagger_setting_channels=(1, 2)),
                                          {'live_correction_gain': 0.3})
        controller.remote = remote
        controller.live_state = {}
        controller.live_t10_ps = controller.live_t20_ps = 0
        controller.initial_clock_adjustment_applied = True
        controller.live_discard_through_sequence = 0
        controller._log_live_tracking_event = Mock()
        controller._analyze_live_interval = Mock(side_effect=lambda seq, *args, **kwargs: {
            'u0': (0.2549461133477867 if seq == 2 else 30.0) * 1e-12,
            'tau_ps': -1846.0, 'T_lastmid_ps': seq * 1e12})
        return controller

    def arrays(self):
        return {'ttx': (np.array([10, 20, 30, 40]), np.array([1, 2, 1, 2])),
                'hh': (np.array([10, 30]), np.array([6, 6]))}

    def test_coordinator_extracts_zero_delta_and_remote_message(self):
        remote = self.coordinator()
        self.result(remote, 'zero', no_change=True, applied=0.0, message='Below resolution')
        result = remote.get_clock_command('zero')
        self.assertEqual(result['applied_correction_ps_per_s'], 0.0)
        self.assertTrue(result['no_change'])
        self.assertEqual(result['message'], 'Below resolution')
        result['result']['message'] = 'changed snapshot'
        self.assertEqual(remote.get_clock_command('zero')['message'], 'Below resolution')

    def test_failed_reply_and_timeout_are_not_treated_as_success(self):
        remote = self.coordinator()
        self.result(remote, 'failed', success=False, no_change=True, applied=0.0,
                    message='I2C write rejected')
        failed = remote.get_clock_command('failed')
        self.assertEqual(failed['status'], 'failed')
        self.assertFalse(failed['no_change'])
        self.assertEqual(failed['message'], 'I2C write rejected')
        remote.clock_commands['timeout'] = {'status': 'timed_out', 'result': None,
                                            'message': 'No CLOCK_RESULT received'}
        controller = self.controller(remote)
        timeout = controller._wait_for_clock_adjustment('timeout')
        self.assertEqual(timeout['status'], 'timed_out')
        self.assertEqual(timeout['message'], 'No CLOCK_RESULT received')

    def test_zero_adjustment_allows_next_acquisition_and_uses_actual_delta(self):
        remote = self.coordinator()
        controller = self.controller(remote)
        controller.qcs_clock_adjustment_total_ps_per_s = 100.0
        sent = []

        def send(correction):
            key = str(len(sent))
            sent.append(correction)
            if len(sent) == 1:
                self.result(remote, key, no_change=True, applied=0.0, message='Below resolution')
            else:
                self.result(remote, key, applied=-8.8, message='Write complete')
                controller.stop_event.set()
            return key

        remote.set_clock_correction = send
        original_publish = controller._publish
        published_status = []

        def publish():
            original_publish()
            step = controller.live_step_result
            if step is not None:
                published_status.append(step['clock_status'])
            if step is not None and step['clock_status'] == 'unchanged (below resolution)':
                self.assertIsNone(controller.integrity_failure)
                self.assertEqual(controller.qcs_clock_adjustment_total_ps_per_s, 100.0)
                # Exercise the actual acquisition gate after the no-change reply.
                controller._handle_completed_interval({
                    'sequence': 3, 'initialization': False,
                    'finished_monotonic_ns': 3_000_000_000,
                    'order_errors': [], 'drops': {'ttx': 0, 'hh': 0},
                    'buffers': {src: deque([data]) for src, data in self.arrays().items()},
                })

        controller._publish = publish
        for sequence in (1, 2):
            controller.live_interval_queue.put((sequence, self.arrays()))
        worker = threading.Thread(target=controller._live_analysis_worker, daemon=True)
        worker.start()
        worker.join(timeout=2)
        if worker.is_alive():
            controller.stop_event.set()
            worker.join(timeout=1)
            self.fail('Tracking did not continue to the next interval')
        self.assertEqual(controller._analyze_live_interval.call_count, 3)
        self.assertEqual(len(sent), 2)  # The first fit remains measurement-only.
        self.assertAlmostEqual(sent[0], -0.07648383400433602)
        self.assertAlmostEqual(sent[1], -9.0)
        self.assertIsNone(controller.integrity_failure)
        self.assertIn('unchanged (below resolution)', published_status)
        self.assertAlmostEqual(controller.qcs_clock_adjustment_total_ps_per_s, 91.2)
        self.assertEqual(controller.live_step_result['applied_clock_delta_ps_per_s'], -8.8)

    def test_initial_alignment_can_complete_with_successful_zero_adjustment(self):
        remote = self.coordinator()
        controller = RealtimeG2Controller(SimpleNamespace())
        controller.remote = remote
        controller.ttx_start_time_us = 1_000_000
        remote.snapshot = Mock(return_value={
            'remote_start_time_us': 1_000_000, 'clock_adjustment_supported': True})

        def send(correction):
            self.result(remote, 'initial', no_change=True, applied=0.0, message='Below resolution')
            return 'initial'

        remote.set_clock_correction = send
        arrays = {'ttx': (np.array([10, 20]), np.array([1, 1])),
                  'hh': (np.array([10, 20]), np.array([6, 6]))}
        with patch('qcs_analyzer.global_freq_analysis.global_freq_alignment_g2', return_value={
                'unique_best_candidate': True, 'skew_ps_per_s': 0.05,
                'residual_offset_ps': 0.0}):
            controller._initialize_alignment(arrays)
        self.assertTrue(controller.initial_clock_adjustment_applied)
        self.assertEqual(controller.qcs_clock_adjustment_total_ps_per_s, 0.0)
        self.assertIsNotNone(controller.live_state)
        self.assertIn('unchanged', controller.initialization_status)

    def test_live_write_failure_preserves_remote_error_and_cumulative_total(self):
        remote = self.coordinator()
        controller = self.controller(remote)
        controller.live_analysis_steps = 1
        controller.qcs_clock_adjustment_total_ps_per_s = 100.0

        def send(correction):
            self.result(remote, 'write-failed', success=False, applied=None,
                        message='I2C write rejected')
            controller.stop_event.set()
            return 'write-failed'

        remote.set_clock_correction = send
        controller.live_interval_queue.put((2, self.arrays()))
        controller._live_analysis_worker()
        self.assertIn('I2C write rejected', controller.integrity_failure)
        self.assertEqual(controller.qcs_clock_adjustment_total_ps_per_s, 100.0)


if __name__ == '__main__':
    unittest.main()
