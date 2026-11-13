"""Verify the GUI start path forwards and validates the live gain without hardware."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from QCS_algorithm import RealtimeG2Controller
from qutag_timestamp_helper import _start_qcs


class Value:
    def __init__(self, text):
        self.text = text

    def get(self):
        return self.text

    def set(self, text):
        self.text = text


class LiveGainGuiTests(unittest.TestCase):
    def app(self, gain):
        app = SimpleNamespace(qcs_controller=None, qutag_timestamp_win=None)
        for name, text in {
            'interval': '1', 'init_interval': '2', 'sweep_num': '40',
            'fine_steps': '20', 'init_bin_number': '1000000', 'latency_ms': '400',
            'sweep_range': '2000', 'offset_range': '40', 'correction_gain': gain,
        }.items():
            setattr(app, f'qcs_{name}_var', Value(text))
        app.qcs_status_var = Value('Idle')
        return app

    def test_entered_live_gain_is_used_and_retained(self):
        app = self.app('0.45')
        with patch.object(RealtimeG2Controller, 'start') as start, \
                patch('qutag_timestamp_helper.messagebox.showerror') as error:
            _start_qcs(app)
        error.assert_not_called()
        start.assert_called_once()
        self.assertEqual(app.qcs_controller.live_correction_gain, 0.45)
        self.assertEqual(app.qcs_correction_gain_text, '0.45')

    def test_invalid_gain_cannot_start_acquisition(self):
        for gain in ('0', '-0.3', 'nan', 'inf', 'abc'):
            with self.subTest(gain=gain):
                app = self.app(gain)
                with patch.object(RealtimeG2Controller, 'start') as start, \
                        patch('qutag_timestamp_helper.messagebox.showerror') as error:
                    _start_qcs(app)
                start.assert_not_called()
                error.assert_called_once()
                self.assertIsNone(app.qcs_controller)


if __name__ == '__main__':
    unittest.main()
