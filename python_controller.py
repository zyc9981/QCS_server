# -*- coding: utf-8 -*-
import ast
import collections
from datetime import datetime
import json
import logging
import os
import queue
import threading
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENABLE_TTX_DIAGNOSTICS = False
from optical_stream import OpticalLink, hydraharp_blocks, capture_hh_qsfp
from qsfp_mode_ui import get_qsfp_modes

from tkinter import (
    Tk,
    Toplevel,
    Frame,
    Label,
    Entry,
    Button,
    Checkbutton,
    Spinbox,
    LEFT,
    RIGHT,
    END,
    BooleanVar,
    StringVar,
    messagebox,
)
from tkinter.font import Font
from tkinter.scrolledtext import ScrolledText

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from automated_calibration.remote_laser_control import DLCProSimple
from controller_common import open_arduino, readline_str
from rx_commands import RxCommandsMixin
from tx_commands import TxCommandsMixin
from ring_calibration_helper import tx_scan_frequency
from highQ_piezoscan_helper import tx_scan_piezo
from qutag_piezoscan_helper import tx_scan_piezo_qutag
from qutag_timestamp_helper import open_qutag_timestamp_window, _stop_timestamp_writing
from scan_2v_analyzer import run_scan_2v_analyzer

try:
    from Swabian import TimeTagger
except Exception:
    TimeTagger = None
    print("Time Tagger X import failed.")

# Keep the HydraHarp settings and QSFP code for later use, but do not load
# its SDK or open the device while it is disabled.
ENABLE_HYDRAHARP = False
LogLevel = MeasMode = RefSource = create_hydraharp_api = None
if ENABLE_HYDRAHARP:
    try:
        from picoquant_snapi import LogLevel, MeasMode, RefSource, create_hydraharp_api
    except Exception as exc:
        print(f"HydraHarp snAPI import failed: {exc}")

LASER_HOST = "192.168.1.200"
TX_PORT = "/dev/ttyACM0"
RX_PORT = "/dev/ttyACM1"
DEFAULT_VOLTAGES_PATH = Path(__file__).with_name("20260924_QCSTX1_RB4.json")
RX_BACKGROUND_OFFSETS_PATH = Path(__file__).with_name("rx_background_offsets.json")

UI = {
    "bg": "#0b1118",
    "surface": "#141c26",
    "surface_alt": "#1f2a37",
    "border": "#2d3a4a",
    "text": "#e5edf7",
    "muted": "#9aa8ba",
    "accent": "#2f9e8f",
    "accent_hover": "#248577",
    "danger": "#b84a42",
    "danger_hover": "#963a34",
    "log_bg": "#070b12",
    "log_fg": "#d7e3f4",
    "plot_bg": "#101822",
    "plot_grid": "#253244",
    "plot_blue": "#58a6ff",
}



class App(TxCommandsMixin, RxCommandsMixin):
    def __init__(self):
        self.ui = UI
        self.tx_port = TX_PORT
        self.rx_port = RX_PORT
        self.default_voltages_path = DEFAULT_VOLTAGES_PATH
        self.rx_background_offsets_path = RX_BACKGROUND_OFFSETS_PATH
        self.rx_pd_to_channel = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
        }
        self.rx_channel_to_pd = {
            channel: pd_index for pd_index, channel in self.rx_pd_to_channel.items()
        }
        self.rx_pd_background_offsets = self.load_rx_background_offsets()
        self.tx_mode_pin_high = False
        self.rx_decode_detector_index = 1
        self.rx_decode_channel_number = self.rx_pd_to_channel.get(self.rx_decode_detector_index, 2)
        self.rx_threshold_value = None
        self.rx_channel_groups = [
            (1, 2),
            (3, 4),
            (5, 6),
            (7, 8),
            (0,),
        ]
        self.rx_detector_count = len(self.rx_pd_to_channel)
        self.qutag_hist_start_channel = 1
        self.qutag_hist_stop_channel = 2
        self.qutag_hist_pair_options = [(1, 2)]
        self.qutag_hist_pairs = [(1, 2)]
        self.qutag_hist_bin_width_ps = 500
        self.qutag_hist_bin_count = 200
        self.qutag_hist_integration_time_ms = 1000.0
        self.qutag_hist_configured = False
        self.qutag_hist_error_logged = False
        self.qutag_hist_next_poll = 0.0
        self.time_tagger_histograms = {}
        # The remaining qutag_* acquisition state is converted in later
        # steps.  Device configuration now belongs to Time Tagger X.
        self.qutag = None
        self.time_tagger = None
        # self.time_tagger_setting_channels = [1, 2, 3, 4]
        self.time_tagger_setting_channels = [1, 2, -3, -4]
        self.time_tagger_trigger_levels = {
            1: 0.5,
            2: 0.5,
            # 3: 0.3,
            -3: -0.15,
            # 4: 0.3,
            -4: -0.15
        }
        self.time_tagger_channel_delays_ps = {
            1: 0,
            2: 0,
            # -3: 0,
            3: 0,
            # -4: 0,
            4: 0,
        }
        # The Counter bin width replaces QuTAG's exposure time.  Counts from
        # the Time Tagger and the HydraHarp plot are both displayed per bin.
        self.time_tagger_count_bin_width_ms = 50.0
        self.time_tagger_counter_channels = list(self.time_tagger_setting_channels)
        self.time_tagger_counter = None
        self.time_tagger_counter_lock = threading.Lock()
        # Temporary display selection: keep Time Tagger X channels 3 and 4 available
        # to the rest of the controller, but hide them from Rx/TimeTag plots.
        self.spd_counter_indices = [1, 2]
        # snAPI time-trace channel 0 is the sync input; the photon-detector
        # channels start at 1.  Add further HydraHarp input channels here when
        # they are connected.
        self.hydraharp_channel_indices = [1]
        self.hydraharp_time_trace_num_bins = 200
        self.hydraharp_time_trace_history_size_s = 10.0
        self.hydraharp_time_trace_integration_ms = 50.0
        # Choose the HydraHarp clock before device initialization.  Set this
        # to RefSource.Internal when no stable external 10 MHz reference is
        # connected to the HydraHarp reference-clock input.
        self.hydraharp_ref_source = (
            RefSource.External_10MHZ if RefSource is not None else None
            # RefSource.Internal if RefSource is not None else None
        )
        self.hydraharp_config_path = Path(__file__).with_name("HH.ini")
        self.hydraharp_time_trace_settings_path = Path(__file__).with_name(
            "hydraharp_time_trace_settings.json"
        )
        self.load_hydraharp_time_trace_settings()
        self.hydraharp = None
        self.hydraharp_raw_recording = False
        self.optical_capture_active = False
        self.optical_capture_thread = None
        self.optical_capture_stop = threading.Event()
        self.optical_capture_events = queue.Queue()
        self.optical_closing = False
        self.hydraharp_setting_channels = []
        self.hydraharp_enabled_channels = set()
        self.hydraharp_cfd_levels_mV = {}
        self.hydraharp_zero_cross_levels_mV = {}
        self.hydraharp_channel_delays_ps = {}
        self.hydraharp_sync_cfd_level_mV = 300
        self.hydraharp_sync_zero_cross_mV = 10
        self.hydraharp_sync_delay_ps = 0
        # Keys use snAPI's zero-based input-channel indices.  Thus key 0 is
        # the physical HydraHarp input labelled "Ch 1" in the UI.
        self.hydraharp_startup_channel_defaults = {
            0: {"cfd_mV": 300, "zero_cross_mV": 10, "delay_ps": 0},
            1: {"cfd_mV": 150, "zero_cross_mV": 10, "delay_ps": 0},
        }
        self.spd_series = (
            [("Time Tagger X", channel) for channel in self.spd_counter_indices]
            + [("HydraHarp", channel) for channel in self.hydraharp_channel_indices]
        )
        self.spd_detector_count = len(self.spd_series)
        self.rx_plot_window_s = 30.0
        self.rx_grid_positions = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
        ]

        self.tx = open_arduino(self.tx_port, 115200, 0.2)
        while True:
            line = readline_str(self.tx)
            if line.strip() == "TX Ready.":
                break
        print("Connected to TX")

        self.rx = open_arduino(self.rx_port, 115200, 0.05)
        while True:
            line = readline_str(self.rx)
            if line.strip() == "RX Ready.":
                break
        print("Connected to RX")

        self.init_time_tagger()
        self.init_hydraharp()

        try:
            self.laser = DLCProSimple(LASER_HOST).open()
            print(f"Connected to laser at {LASER_HOST}")
        except Exception as exc:
            self.laser = None
            print("ERROR: Could not connect to laser:", exc)

        self.root = Tk()
        if ENABLE_TTX_DIAGNOSTICS:
            self.root.bind_all("<Control-Shift-D>", lambda event: self.open_time_tagger_diagnostics())
        self.root.title("Hybrid TX/RX Control")
        self.root.withdraw()

        self.build_tx_panel()
        if ENABLE_TTX_DIAGNOSTICS:
            Button(self.tx_win, text="Time Tagger diagnostics / event marker",
                   command=self.open_time_tagger_diagnostics).pack()
        self.build_rx_panel()
        self.tx_log_print(f"[TX] Sending default voltages from {self.default_voltages_path.name}...")
        self.tx_init_voltages()
        self.tx_log_print("[TX] Defaults sent.")

        self.root.protocol("WM_DELETE_WINDOW", self._quit_all)
        self.tx_win.protocol("WM_DELETE_WINDOW", self._quit_all)
        self.rx_win.protocol("WM_DELETE_WINDOW", self._quit_all)

    def _start_time_tagger_log(self):
        """Create one diagnostic log per device initialization attempt."""
        if not ENABLE_TTX_DIAGNOSTICS:
            self.time_tagger_logger = logging.Logger(f"TTX.disabled.{id(self)}")
            self.time_tagger_logger.addHandler(logging.NullHandler())
            self.time_tagger_log_path = None
            return
        if getattr(self, "time_tagger_sdk_callback", None) is not None:
            # setLogger is process-wide: never replace our callback mid-session.
            self.time_tagger_marker("Reusing registered SDK logger")
            return
        self._close_time_tagger_log()
        logger = logging.Logger(f"TTX.{id(self)}", level=logging.INFO)
        self.time_tagger_logger = logger
        self.time_tagger_log_path = None
        try:
            log_dir = PROJECT_ROOT / "TTX_log"
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = log_dir / f"TTX_{stamp}_{os.getpid()}.log"
            handler = logging.FileHandler(path, mode="x", encoding="utf-8")
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s [%(threadName)s] %(message)s"
            ))
            logger.addHandler(handler)
            self.time_tagger_log_path = path
            logger.info("Time Tagger X startup requested")
            print(f"Time Tagger X log: {path}")
        except OSError as exc:
            raise RuntimeError(f"Cannot open Time Tagger diagnostic log: {exc}") from exc

    def time_tagger_marker(self, message):
        logger = getattr(self, "time_tagger_logger", None)
        if logger is not None:
            logger.info("APP MARKER: %s", message)

    def open_time_tagger_diagnostics(self):
        dialog = Toplevel(self.root)
        dialog.title("Time Tagger X diagnostics")
        Label(dialog, text=f"Log: {self.time_tagger_log_path}", wraplength=700).pack(padx=10, pady=10)
        Label(dialog, text="Markers record observations; LED changes may produce no SDK message.").pack()
        for marker in ("CLK IN connected (Si5341)", "CLK IN disconnected",
                       "CH3 FPGA clock connected", "CH3 FPGA clock disconnected",
                       "CH4 FPGA clock connected", "CH4 FPGA clock disconnected",
                       "CH3 LED observed red", "CH4 LED observed red"):
            Button(dialog, text=marker,
                   command=lambda text=marker: self.time_tagger_marker(text)).pack(fill="x")
        entry = Entry(dialog, width=80)
        entry.pack(padx=10, pady=10)
        Button(dialog, text="Insert custom marker",
               command=lambda: self.time_tagger_marker(entry.get())).pack(pady=5)

    def _close_time_tagger_log(self):
        # Keep the callback and its file alive even for late SDK shutdown logs.
        # logging.shutdown flushes/closes handlers when this process exits.
        if getattr(self, "time_tagger_sdk_callback", None) is not None:
            return
        logger = getattr(self, "time_tagger_logger", None)
        if logger is not None:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
        self.time_tagger_logger = None

    def init_time_tagger(self):
        self._start_time_tagger_log()
        if TimeTagger is None:
            self.time_tagger_logger.error("Python package import failed; device unavailable")
            print("Time Tagger X unavailable: Python package import failed.")
            return False

        try:
            if ENABLE_TTX_DIAGNOSTICS and getattr(self, "time_tagger_sdk_callback", None) is None:
                logger = self.time_tagger_logger

                def sdk_callback(level, message):
                    # Do not map SDK levels to Python levels: even 0/unknown
                    # levels must pass. FileHandler serializes and flushes each
                    # complete record, including multiline messages, under a lock.
                    logger.info("SDK level=%s message=%s", level, message)

                self.time_tagger_marker(f"Registering SDK logger in hardware process pid={os.getpid()}")
                TimeTagger.setLogger(sdk_callback)
                self.time_tagger_sdk_callback = sdk_callback
                self.time_tagger_marker("SDK logger registered; all callback levels captured")
            self.time_tagger_marker("Device opening: createTimeTagger")
            self.time_tagger = TimeTagger.createTimeTagger()
            serial = self.time_tagger.getSerial()
            model = self.time_tagger.getModel()
        except Exception as exc:
            self.time_tagger = None
            self.time_tagger_logger.exception("Could not initialize Time Tagger X")
            print("ERROR: Could not initialize Time Tagger X:", exc)
            return False

        print(f"Connected to Time Tagger X: model {model}, serial {serial}")
        self.time_tagger_logger.info("Connected: model=%s serial=%s", model, serial)
        try:
            self.apply_time_tagger_settings()
            self.configure_time_tagger_counter()
        except Exception as exc:
            self.time_tagger_logger.exception("Could not apply default settings")
            print("ERROR: Could not apply default Time Tagger X settings:", exc)
        return True

    def init_hydraharp(self):
        """Start the optional HydraHarp T2 time trace used by the Rx SPD plot."""
        # The rest of the controller is designed to run without a HydraHarp.
        # Keep that state explicit before attempting discovery, so any failed
        # initialization leaves all later code treating it as unavailable.
        self.hydraharp = None
        if not ENABLE_HYDRAHARP:
            print("HydraHarp disabled; continuing without it.")
            return False
        if create_hydraharp_api is None:
            print("HydraHarp unavailable: snAPI wrapper import failed.")
            return False

        hydraharp = None

        def release_hydraharp():
            if hydraharp is None:
                return
            try:
                hydraharp.closeDevice()
            except Exception:
                pass
            # snAPI.__del__ releases the native API. An explicit exitAPI()
            # here can make some Linux builds release it a second time.

        try:
            hydraharp = create_hydraharp_api()
            if not hydraharp.getDevice():
                print("HydraHarp not detected; continuing without it.")
                release_hydraharp()
                return False
            if not hydraharp.initDevice(
                MeasMode.T2,
                refSrc=self.hydraharp_ref_source,
            ):
                print(
                    "HydraHarp could not be initialized in T2 mode "
                    f"({self.hydraharp_ref_source.name}); continuing without it."
                )
                release_hydraharp()
                return False

            hydraharp.setLogLevel(LogLevel.Config, True)
            if self.hydraharp_config_path.is_file():
                if not hydraharp.loadIniConfig(str(self.hydraharp_config_path)):
                    raise RuntimeError(f"Could not load {self.hydraharp_config_path.name}.")
            else:
                print(
                    f"HydraHarp configuration {self.hydraharp_config_path.name} was not found; "
                    "using device defaults."
                )

            self.apply_hydraharp_startup_channel_defaults(hydraharp)
            hydraharp.timeTrace.setNumBins(self.hydraharp_time_trace_num_bins)
            hydraharp.timeTrace.setHistorySize(self.hydraharp_time_trace_history_size_s)
            if not hydraharp.timeTrace.measure(0, waitFinished=False, savePTU=False):
                raise RuntimeError("Could not start HydraHarp time trace.")
        except Exception as exc:
            release_hydraharp()
            print(f"HydraHarp unavailable ({exc}); continuing without it.")
            return False

        self.hydraharp = hydraharp
        self.refresh_hydraharp_settings_from_device()
        print(
            "Connected to HydraHarp: "
            f"T2 time trace running ({self.hydraharp_time_trace_num_bins} bins over "
            f"{self.hydraharp_time_trace_history_size_s:g} s), "
            f"reference={self.hydraharp_ref_source.name}."
        )
        return True

    def apply_hydraharp_startup_channel_defaults(self, hydraharp):
        """Apply the lab's default sync/input CFD and delay values before measuring."""
        if not hydraharp.device.setSyncCFD(300, 10):
            raise RuntimeError("Could not set default HydraHarp sync CFD.")

        num_channels = int(getattr(hydraharp, "deviceConfig", {}).get("NumChans", 0))
        for channel, settings in self.hydraharp_startup_channel_defaults.items():
            if channel >= num_channels:
                continue
            if not hydraharp.device.setInputCFD(
                channel,
                int(settings["cfd_mV"]),
                int(settings["zero_cross_mV"]),
            ):
                raise RuntimeError(f"Could not set default CFD for HydraHarp Ch {channel + 1}.")
            if not hydraharp.device.setInputChannelOffset(channel, int(settings["delay_ps"])):
                raise RuntimeError(f"Could not set default delay for HydraHarp Ch {channel + 1}.")

    def refresh_hydraharp_settings_from_device(self):
        """Copy the active HydraHarp configuration into editable UI state."""
        hydraharp = getattr(self, "hydraharp", None)
        if hydraharp is None:
            return

        config = getattr(hydraharp, "deviceConfig", {})
        channel_configs = config.get("ChansCfg", [])
        num_channels = int(config.get("NumChans", len(channel_configs)))
        self.hydraharp_setting_channels = list(range(num_channels))
        channel_config = lambda channel: (
            channel_configs[channel] if channel < len(channel_configs) else {}
        )
        self.hydraharp_enabled_channels = {
            channel
            for channel in self.hydraharp_setting_channels
            if bool(channel_config(channel).get("ChanEna", 1))
        }
        self.hydraharp_cfd_levels_mV = {
            channel: int(channel_config(channel).get("DiscrLvl", 100))
            for channel in self.hydraharp_setting_channels
        }
        self.hydraharp_zero_cross_levels_mV = {
            channel: int(channel_config(channel).get("ZeroXLvl", 0))
            for channel in self.hydraharp_setting_channels
        }
        self.hydraharp_channel_delays_ps = {
            channel: int(channel_config(channel).get("ChanOffs", 0))
            for channel in self.hydraharp_setting_channels
        }
        self.hydraharp_sync_cfd_level_mV = int(config.get("SyncDiscrLvl", 100))
        self.hydraharp_sync_zero_cross_mV = int(config.get("SyncZeroXLvL", 0))
        self.hydraharp_sync_delay_ps = int(config.get("SyncChannelOffset", 0))

    def apply_hydraharp_settings(self):
        """Apply HydraHarp CFD thresholds, input delays, and enabled channels."""
        hydraharp = getattr(self, "hydraharp", None)
        if hydraharp is None:
            raise RuntimeError("HydraHarp is not initialized.")
        if self.hydraharp_raw_recording or getattr(self, "qutag_timestamp_preparing_hydraharp", False):
            raise RuntimeError("Stop the HydraHarp T2 recording before changing its settings.")

        for channel in self.hydraharp_setting_channels:
            enabled = int(channel in self.hydraharp_enabled_channels)
            threshold_mV = int(self.hydraharp_cfd_levels_mV[channel])
            zero_cross_mV = int(self.hydraharp_zero_cross_levels_mV[channel])
            delay_ps = int(self.hydraharp_channel_delays_ps[channel])

            if not 0 <= threshold_mV <= 1000:
                raise ValueError(f"HydraHarp Ch {channel + 1} threshold must be 0 to 1000 mV.")
            if not 0 <= zero_cross_mV <= 40:
                raise ValueError(f"HydraHarp Ch {channel + 1} zero-cross level must be 0 to 40 mV.")
            if not -99999 <= delay_ps <= 99999:
                raise ValueError(f"HydraHarp Ch {channel + 1} delay must be -99999 to 99999 ps.")

            if not hydraharp.device.setInputChannelEnable(channel, enabled):
                raise RuntimeError(f"Could not set HydraHarp Ch {channel + 1} enabled state.")
            if not hydraharp.device.setInputCFD(channel, threshold_mV, zero_cross_mV):
                raise RuntimeError(f"Could not set HydraHarp Ch {channel + 1} CFD threshold.")
            if not hydraharp.device.setInputChannelOffset(channel, delay_ps):
                raise RuntimeError(f"Could not set HydraHarp Ch {channel + 1} delay.")

        if not 0 <= self.hydraharp_sync_cfd_level_mV <= 1000:
            raise ValueError("HydraHarp sync threshold must be 0 to 1000 mV.")
        if not 0 <= self.hydraharp_sync_zero_cross_mV <= 40:
            raise ValueError("HydraHarp sync zero-cross level must be 0 to 40 mV.")
        if not -99999 <= self.hydraharp_sync_delay_ps <= 99999:
            raise ValueError("HydraHarp sync delay must be -99999 to 99999 ps.")
        if not hydraharp.device.setSyncCFD(
            self.hydraharp_sync_cfd_level_mV,
            self.hydraharp_sync_zero_cross_mV,
        ):
            raise RuntimeError("Could not set HydraHarp sync CFD threshold.")
        if not hydraharp.device.setSyncChannelOffset(self.hydraharp_sync_delay_ps):
            raise RuntimeError("Could not set HydraHarp sync delay.")

    def load_hydraharp_time_trace_settings(self):
        """Load the settings to apply before starting a HydraHarp time trace."""
        settings_path = self.hydraharp_time_trace_settings_path
        if not settings_path.is_file():
            return

        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            self.set_hydraharp_time_trace_settings(
                settings["history_size_s"],
                settings["integration_ms"],
                save=False,
            )
        except Exception as exc:
            print(f"WARNING: Could not load {settings_path.name}: {exc}")

    def set_hydraharp_time_trace_settings(self, history_size_s, integration_ms, save=True):
        """Save live-rate settings for the next safe HydraHarp measurement start."""
        history_size_s = float(history_size_s)
        integration_ms = float(integration_ms)
        if history_size_s <= 0:
            raise ValueError("HydraHarp time-trace history must be > 0 s.")
        if integration_ms <= 0 or integration_ms > history_size_s * 1000:
            raise ValueError("HydraHarp count integration must be > 0 and no longer than the history.")

        num_bins = max(1, int(round(history_size_s * 1000.0 / integration_ms)))
        actual_integration_ms = history_size_s * 1000.0 / num_bins
        self.hydraharp_time_trace_history_size_s = history_size_s
        self.hydraharp_time_trace_num_bins = num_bins
        self.hydraharp_time_trace_integration_ms = actual_integration_ms

        if save:
            settings = {
                "history_size_s": history_size_s,
                "integration_ms": actual_integration_ms,
            }
            self.hydraharp_time_trace_settings_path.write_text(
                json.dumps(settings, indent=2) + "\n",
                encoding="utf-8",
            )

    def shutdown_hydraharp(self):
        hydraharp = getattr(self, "hydraharp", None)
        self.hydraharp = None
        if hydraharp is None:
            return

        try:
            hydraharp.timeTrace.stopMeasure()
        except Exception:
            pass
        try:
            hydraharp.closeDevice()
        except Exception:
            pass
        # The installed snAPI object calls exitAPI() in its destructor.

    def restart_hydraharp_time_trace(self):
        """Resume live counts after a raw T2 measurement has fully stopped."""
        if self.hydraharp is None:
            raise RuntimeError("HydraHarp is not connected.")
        self.hydraharp.timeTrace.setNumBins(self.hydraharp_time_trace_num_bins)
        self.hydraharp.timeTrace.setHistorySize(self.hydraharp_time_trace_history_size_s)
        if not self.hydraharp.timeTrace.measure(0, waitFinished=False, savePTU=False):
            raise RuntimeError("timeTrace.measure returned False")

    def open_optical_capture_window(self):
        window = getattr(self, "optical_capture_win", None)
        if window is not None and window.winfo_exists():
            window.lift()
            return

        window = Toplevel(self.tx_win)
        self.optical_capture_win = window
        window.title("HydraHarp to QSFP")
        self._style_window(window, minsize=(620, 220))
        body = self._frame(window)
        body.pack(fill="both", expand=True, padx=16, pady=14)
        self._label(body, "HydraHarp T2 optical loopback", size=14, weight="bold").pack(anchor="w")
        self._label(body, "Returned records are saved as raw 32-bit T2 data (.bin).", fg=UI["muted"]).pack(anchor="w", pady=(4, 12))

        duration_row = self._frame(body)
        duration_row.pack(fill="x", pady=4)
        self._label(duration_row, "Duration (s)").pack(side=LEFT)
        self.optical_duration_var = StringVar(value="10")
        Entry(duration_row, textvariable=self.optical_duration_var, width=9).pack(side=LEFT, padx=10)

        output_row = self._frame(body)
        output_row.pack(fill="x", pady=4)
        self._label(output_row, "Output file").pack(side=LEFT)
        default_path = PROJECT_ROOT / f"hh_t2_returned_{datetime.now():%Y%m%d_%H%M%S_%f}.bin"
        self.optical_output_var = StringVar(value=str(default_path))
        Entry(output_row, textvariable=self.optical_output_var).pack(side=LEFT, fill="x", expand=True, padx=10)

        mode_row = self._frame(body)
        mode_row.pack(fill="x", pady=4)
        modes = get_qsfp_modes(self)
        modes.add_panel(mode_row)
        self._label(body, "", textvariable=modes.status, wraplength=580,
                    justify="left").pack(anchor="w")

        buttons = self._frame(body)
        buttons.pack(fill="x", pady=(12, 6))
        self.optical_start_button = self._button(buttons, text="Start", command=self.start_optical_capture, variant="primary")
        self.optical_start_button.pack(side=LEFT, padx=(0, 8))
        self.optical_stop_button = self._button(buttons, text="Stop", command=self.stop_optical_capture)
        self.optical_stop_button.pack(side=LEFT)
        self.optical_stop_button.configure(state="disabled")
        self.optical_status_var = StringVar(value="Ready. HydraHarp and XDMA must be connected.")
        self._label(body, "", textvariable=self.optical_status_var, wraplength=580, justify="left").pack(anchor="w")
        window.protocol("WM_DELETE_WINDOW", self._close_optical_capture_window)

    def _close_optical_capture_window(self):
        if self.optical_capture_active:
            self.stop_optical_capture()
        else:
            self.optical_capture_win.destroy()

    def start_optical_capture(self):
        if self.optical_capture_active:
            return
        if self.hydraharp is None:
            messagebox.showerror("HH to QSFP", "HydraHarp is not connected.", parent=self.optical_capture_win)
            return
        if self.hydraharp_raw_recording or getattr(self, "qutag_timestamp_preparing_hydraharp", False) or getattr(self, "qutag_timestamp_writing", False):
            messagebox.showerror("HH to QSFP", "Stop the TimeTag recording before starting optical capture.", parent=self.optical_capture_win)
            return
        try:
            self.optical_capture_options = get_qsfp_modes(self).capture_options()
            duration_ms = int(round(float(self.optical_duration_var.get()) * 1000))
            if not 1 <= duration_ms <= 3600000:
                raise ValueError("Duration must be between 0.001 and 3600 seconds.")
            output = Path(self.optical_output_var.get().strip()).expanduser()
            if not output.name or output.suffix.lower() != ".bin":
                raise ValueError("Choose a .bin output file.")
            if output.exists():
                raise FileExistsError(f"Output already exists: {output}")
            if not output.parent.is_dir():
                raise FileNotFoundError(f"Output directory does not exist: {output.parent}")
            for device in ("/dev/xdma0_h2c_0", "/dev/xdma0_c2h_0"):
                if not os.access(device, os.R_OK | os.W_OK):
                    raise PermissionError(f"Cannot access {device}")
        except (ValueError, OverflowError, OSError, RuntimeError) as exc:
            messagebox.showerror("HH to QSFP", str(exc), parent=self.optical_capture_win)
            return

        self.optical_capture_stop.clear()
        self.optical_capture_active = True
        self.hydraharp_raw_recording = True
        self.optical_start_button.configure(state="disabled")
        self.optical_stop_button.configure(state="normal")
        self.optical_status_var.set("Stopping the live time trace before T2 capture...")
        try:
            self.hydraharp.timeTrace.stopMeasure()
        except Exception as exc:
            self.optical_capture_active = False
            self.hydraharp_raw_recording = False
            self.optical_start_button.configure(state="normal")
            self.optical_stop_button.configure(state="disabled")
            try:
                self.hydraharp.timeTrace.measure(0, waitFinished=False, savePTU=False)
            except Exception:
                pass
            messagebox.showerror("HH to QSFP", f"Could not stop the live trace: {exc}", parent=self.optical_capture_win)
            return
        # Let snAPI's time-trace worker finish before starting its raw worker.
        self.root.after(1000, lambda: self._launch_optical_capture(duration_ms, output))

    def _launch_optical_capture(self, duration_ms, output):
        if not self.optical_capture_active:
            return
        if self.optical_capture_stop.is_set():
            self._finish_optical_capture({"blocks": 0, "bytes": 0, "error": None})
            return
        self.optical_status_var.set("Capturing HH T2 records and checking QSFP loopback...")
        self.optical_capture_thread = threading.Thread(
            target=self._optical_capture_worker,
            args=(duration_ms, output),
            daemon=False,
        )
        try:
            self.optical_capture_thread.start()
        except Exception as exc:
            self.optical_capture_thread = None
            self._finish_optical_capture({"blocks": 0, "bytes": 0, "error": str(exc)})
            return
        self.root.after(100, self._poll_optical_capture)

    def _optical_capture_worker(self, duration_ms, output):
        try:
            result = capture_hh_qsfp(
                self.hydraharp, duration_ms, output, self.optical_capture_stop,
                progress=lambda count, byte_count: self.optical_capture_events.put(
                    ("progress", count, byte_count)
                ),
                **self.optical_capture_options,
            )
        except Exception as exc:
            result = {"blocks": 0, "bytes": 0, "error": str(exc), "path": output}
        self.optical_capture_events.put(("done", result))

    def _poll_optical_capture(self):
        result = None
        while True:
            try:
                event = self.optical_capture_events.get_nowait()
            except queue.Empty:
                break
            if event[0] == "progress":
                _, count, byte_count = event
                self.optical_status_var.set(f"Verified {count} blocks, {byte_count:,} raw T2 bytes returned.")
            else:
                result = event[1]
        worker = self.optical_capture_thread
        if result is not None:
            self.optical_capture_result = result
        if worker is not None and worker.is_alive():
            self.root.after(100, self._poll_optical_capture)
        elif self.optical_capture_active:
            result = getattr(self, "optical_capture_result", {"blocks": 0, "bytes": 0, "error": "Capture ended unexpectedly."})
            # snAPI can finish its native measurement thread just after
            # stopMeasure() returns; give it time before starting timeTrace.
            self.root.after(300, lambda: self._finish_optical_capture(result))

    def _finish_optical_capture(self, result):
        trace_error = None
        if not self.optical_closing and self.hydraharp is not None:
            try:
                self.restart_hydraharp_time_trace()
            except Exception as exc:
                trace_error = str(exc)
        self.hydraharp_raw_recording = False
        self.optical_capture_active = False
        self.optical_capture_thread = None
        self.optical_start_button.configure(state="normal")
        self.optical_stop_button.configure(state="disabled")
        if "path" in result:
            message = f"{result['blocks']} blocks, {result['bytes']:,} T2 bytes saved to {result['path']}."
            self.optical_output_var.set(str(PROJECT_ROOT / f"hh_t2_returned_{datetime.now():%Y%m%d_%H%M%S_%f}.bin"))
        else:
            message = "Capture stopped before acquiring data."
        if result.get("error"):
            message += f" Capture error: {result['error']}"
        if result.get("source_path"):
            message += f" Original HH data: {result['source_path']}. Log: {result['error_path']}."
        if trace_error:
            message += f" Live time trace could not restart: {trace_error}"
        elif not self.optical_closing:
            message += " Live time trace restarted."
        self.optical_status_var.set(message)
        self.tx_log_print("[HH → QSFP] " + message)

    def stop_optical_capture(self):
        if self.optical_capture_active:
            self.optical_capture_stop.set()
            self.optical_status_var.set("Stopping after the current block...")

    def apply_time_tagger_settings(self):
        if self.time_tagger is None:
            raise RuntimeError("Time Tagger X is not initialized.")

        for channel in self.time_tagger_setting_channels:
            trigger_level = float(self.time_tagger_trigger_levels.get(channel, 0.5))
            delay_ps = int(round(float(self.time_tagger_channel_delays_ps.get(channel, 0))))
            self.time_tagger.setTriggerLevel(int(channel), trigger_level)
            self.time_tagger.setInputDelay(int(channel), delay_ps)

        # Wait until the settings have reached the hardware before a future
        # measurement object begins consuming time tags.
        self.time_tagger.sync()
        logger = getattr(self, "time_tagger_logger", None)
        if logger is not None:
            logger.info("Settings applied: channels=%s trigger_levels=%s delays_ps=%s",
                        self.time_tagger_setting_channels,
                        self.time_tagger_trigger_levels,
                        self.time_tagger_channel_delays_ps)

    def configure_time_tagger_counter(self):
        """Create the shared fixed-bin counter used by the live count views."""
        if self.time_tagger is None:
            raise RuntimeError("Time Tagger X is not initialized.")

        bin_width_ps = int(round(float(self.time_tagger_count_bin_width_ms) * 1e9))
        if bin_width_ps <= 0:
            raise ValueError("Time Tagger X count bin width must be positive.")

        with self.time_tagger_counter_lock:
            previous_counter = self.time_tagger_counter
            self.time_tagger_counter = None
            if previous_counter is not None:
                try:
                    previous_counter.stop()
                    self.time_tagger_marker("Acquisition stopped: previous Counter")
                except Exception:
                    pass

            self.time_tagger_counter_channels = list(self.time_tagger_setting_channels)
            self.time_tagger_counter = TimeTagger.Counter(
                tagger=self.time_tagger,
                channels=self.time_tagger_counter_channels,
                binwidth=bin_width_ps,
                n_values=20,
            )
            self.time_tagger_marker("Acquisition started: live Counter")

    def read_time_tagger_counts(self, timeout_s=0.0, clear=False):
        """Return the newest completed Counter bin as {channel: counts}.

        ``clear=True`` starts a fresh integration window, which is used after
        every piezo movement so a scan point cannot include pre-move events.
        """
        counter = self.time_tagger_counter
        if counter is None:
            raise RuntimeError("Time Tagger X live counter is not initialized.")

        if clear:
            with self.time_tagger_counter_lock:
                counter.clear()

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            with self.time_tagger_counter_lock:
                data = np.asarray(counter.getDataObject(remove=True).getData())

            if data.ndim == 2 and data.shape[0] and data.shape[1]:
                values = data[:, -1]
                return {
                    int(channel): float(value)
                    for channel, value in zip(self.time_tagger_counter_channels, values)
                }

            if time.monotonic() >= deadline:
                return None
            time.sleep(0.002)

    def create_time_tagger_histogram(self, start_channel, stop_channel, bin_width_ps, bin_count):
        if self.time_tagger is None or TimeTagger is None:
            raise RuntimeError("Time Tagger X is not initialized.")

        measurement = TimeTagger.Histogram(
            tagger=self.time_tagger,
            click_channel=int(stop_channel),
            start_channel=int(start_channel),
            binwidth=int(bin_width_ps),
            n_bins=int(bin_count),
        )
        self.time_tagger_marker(f"Acquisition started: Histogram start={start_channel} stop={stop_channel}")
        return measurement

    def create_time_tagger_file_writer(self, filename, channels):
        """Start a Time Tagger X FileWriter for the selected input channels."""
        if self.time_tagger is None or TimeTagger is None:
            raise RuntimeError("Time Tagger X is not initialized.")

        self.time_tagger_marker(f"Timestamp acquisition starting: filename={filename} channels={channels}")
        measurement = TimeTagger.FileWriter(
            tagger=self.time_tagger,
            filename=str(filename),
            channels=[int(channel) for channel in channels],
        )
        self.time_tagger_marker("Timestamp acquisition started: FileWriter")
        return measurement

    def shutdown_time_tagger(self):
        logger = getattr(self, "time_tagger_logger", None)
        if logger is not None:
            logger.info("Time Tagger X shutdown requested")
        file_writer = getattr(self, "time_tagger_file_writer", None)
        self.time_tagger_file_writer = None
        if file_writer is not None:
            try:
                file_writer.stop()
                self.time_tagger_marker("Timestamp acquisition stopped during shutdown")
            except Exception:
                if logger is not None:
                    logger.exception("Timestamp acquisition stop failed during shutdown")

        histograms = getattr(self, "time_tagger_histograms", {})
        self.time_tagger_histograms = {}
        for histogram in histograms.values():
            try:
                histogram.stop()
                self.time_tagger_marker("Acquisition stopped: Histogram during shutdown")
            except Exception:
                if logger is not None:
                    logger.exception("Histogram stop failed during shutdown")

        with self.time_tagger_counter_lock:
            counter = self.time_tagger_counter
            self.time_tagger_counter = None
            if counter is not None:
                try:
                    counter.stop()
                    self.time_tagger_marker("Acquisition stopped: Counter during shutdown")
                except Exception:
                    if logger is not None:
                        logger.exception("Counter stop failed during shutdown")

        time_tagger = getattr(self, "time_tagger", None)
        self.time_tagger = None
        if time_tagger is None or TimeTagger is None:
            self._close_time_tagger_log()
            return

        try:
            self.time_tagger_marker("Acquisition stop attempts complete; device closing: freeTimeTagger")
            TimeTagger.freeTimeTagger(time_tagger)
            if logger is not None:
                logger.info("Time Tagger X released")
        except Exception as exc:
            if logger is not None:
                logger.exception("Could not release Time Tagger X")
            print("WARNING: Could not release Time Tagger X:", exc)
        finally:
            self._close_time_tagger_log()

    def open_tagger_settings_dialog(self):
        if self.time_tagger is None:
            messagebox.showerror("Time Tagger X Settings", "Time Tagger X is not initialized.", parent=self.tx_win)
            return

        dialog = Toplevel(self.tx_win)
        dialog.title("Time Tagger X Settings")
        dialog.transient(self.tx_win)
        dialog.grab_set()
        dialog.resizable(False, False)
        self._style_window(dialog)

        frame = self._surface(dialog)
        frame.pack(fill="both", expand=True, padx=14, pady=14, ipadx=8, ipady=8)

        self._label(frame, text="Channel", size=10, weight="bold", fg=UI["muted"]).grid(row=0, column=0, padx=6, pady=6)
        self._label(frame, text="Trigger level (V)", size=10, weight="bold", fg=UI["muted"]).grid(row=0, column=1, padx=6, pady=6)
        self._label(frame, text="Delay (ps)", size=10, weight="bold", fg=UI["muted"]).grid(row=0, column=2, padx=6, pady=6)

        trigger_vars = {}
        delay_vars = {}

        hydraharp_enabled_vars = {}
        hydraharp_threshold_vars = {}
        hydraharp_zero_cross_vars = {}
        hydraharp_delay_vars = {}
        hydraharp_sync_threshold_var = None
        hydraharp_sync_zero_cross_var = None
        hydraharp_sync_delay_var = None
        hydraharp_history_var = None
        hydraharp_integration_var = None
        dialog_button_row = 7

        for row, channel in enumerate(self.time_tagger_setting_channels, start=1):
            trigger_vars[channel] = StringVar(value=str(self.time_tagger_trigger_levels.get(channel, 0.5)))
            delay_vars[channel] = StringVar(value=str(self.time_tagger_channel_delays_ps.get(channel, 0)))

            self._label(frame, text=str(channel)).grid(row=row, column=0, padx=6, pady=4)
            self._entry(frame, textvariable=trigger_vars[channel], width=10).grid(row=row, column=1, padx=6, pady=4)
            self._entry(frame, textvariable=delay_vars[channel], width=10).grid(row=row, column=2, padx=6, pady=4)

        hydraharp = getattr(self, "hydraharp", None)
        if hydraharp is not None:
            self._label(frame, text="HydraHarp", size=10, weight="bold", fg=UI["muted"]).grid(
                row=0, column=5, columnspan=5, padx=6, pady=6
            )
            for column, text in enumerate(("Input", "Enabled", "CFD (mV)", "Zero (mV)", "Delay (ps)"), start=5):
                self._label(frame, text=text, size=10, weight="bold", fg=UI["muted"]).grid(
                    row=1, column=column, padx=6, pady=4
                )
            for row, channel in enumerate(self.hydraharp_setting_channels, start=2):
                hydraharp_enabled_vars[channel] = BooleanVar(value=channel in self.hydraharp_enabled_channels)
                hydraharp_threshold_vars[channel] = StringVar(value=str(self.hydraharp_cfd_levels_mV[channel]))
                hydraharp_zero_cross_vars[channel] = StringVar(value=str(self.hydraharp_zero_cross_levels_mV[channel]))
                hydraharp_delay_vars[channel] = StringVar(value=str(self.hydraharp_channel_delays_ps[channel]))
                self._label(frame, text=f"Ch {channel + 1}").grid(row=row, column=5, padx=6, pady=4)
                self._checkbutton(frame, variable=hydraharp_enabled_vars[channel]).grid(row=row, column=6, padx=6, pady=4)
                self._entry(frame, textvariable=hydraharp_threshold_vars[channel], width=9).grid(row=row, column=7, padx=6, pady=4)
                self._entry(frame, textvariable=hydraharp_zero_cross_vars[channel], width=9).grid(row=row, column=8, padx=6, pady=4)
                self._entry(frame, textvariable=hydraharp_delay_vars[channel], width=10).grid(row=row, column=9, padx=6, pady=4)

            hydraharp_settings_row = max(7, len(self.hydraharp_setting_channels) + 2)
            hydraharp_sync_threshold_var = StringVar(value=str(self.hydraharp_sync_cfd_level_mV))
            hydraharp_sync_zero_cross_var = StringVar(value=str(self.hydraharp_sync_zero_cross_mV))
            hydraharp_sync_delay_var = StringVar(value=str(self.hydraharp_sync_delay_ps))
            hydraharp_history_var = StringVar(value=str(self.hydraharp_time_trace_history_size_s))
            hydraharp_integration_var = StringVar(value=str(self.hydraharp_time_trace_integration_ms))
            self._label(frame, text="Sync").grid(row=hydraharp_settings_row, column=5, padx=6, pady=(10, 4))
            self._entry(frame, textvariable=hydraharp_sync_threshold_var, width=9).grid(row=hydraharp_settings_row, column=7, padx=6, pady=(10, 4))
            self._entry(frame, textvariable=hydraharp_sync_zero_cross_var, width=9).grid(row=hydraharp_settings_row, column=8, padx=6, pady=(10, 4))
            self._entry(frame, textvariable=hydraharp_sync_delay_var, width=10).grid(row=hydraharp_settings_row, column=9, padx=6, pady=(10, 4))
            self._label(frame, text="History (s)").grid(row=hydraharp_settings_row + 1, column=5, columnspan=2, sticky="e", padx=6, pady=4)
            self._entry(frame, textvariable=hydraharp_history_var, width=9).grid(row=hydraharp_settings_row + 1, column=7, padx=6, pady=4)
            self._label(frame, text="Count integration (ms, next start)").grid(row=hydraharp_settings_row + 2, column=5, columnspan=3, sticky="e", padx=6, pady=4)
            self._entry(frame, textvariable=hydraharp_integration_var, width=9).grid(row=hydraharp_settings_row + 2, column=8, padx=6, pady=4)
            dialog_button_row = hydraharp_settings_row + 3

        def on_ok():
            try:
                new_trigger_levels = {
                    channel: float(trigger_vars[channel].get())
                    for channel in self.time_tagger_setting_channels
                }
                new_delays = {
                    channel: int(round(float(delay_vars[channel].get())))
                    for channel in self.time_tagger_setting_channels
                }
                if hydraharp is not None:
                    new_hydraharp_thresholds = {
                        channel: int(round(float(hydraharp_threshold_vars[channel].get())))
                        for channel in self.hydraharp_setting_channels
                    }
                    new_hydraharp_zero_cross_levels = {
                        channel: int(round(float(hydraharp_zero_cross_vars[channel].get())))
                        for channel in self.hydraharp_setting_channels
                    }
                    new_hydraharp_delays = {
                        channel: int(round(float(hydraharp_delay_vars[channel].get())))
                        for channel in self.hydraharp_setting_channels
                    }
                    new_hydraharp_sync_threshold = int(round(float(hydraharp_sync_threshold_var.get())))
                    new_hydraharp_sync_zero_cross = int(round(float(hydraharp_sync_zero_cross_var.get())))
                    new_hydraharp_sync_delay = int(round(float(hydraharp_sync_delay_var.get())))
                    new_hydraharp_history_s = float(hydraharp_history_var.get())
                    new_hydraharp_integration_ms = float(hydraharp_integration_var.get())
            except ValueError:
                messagebox.showerror(
                    "Tagger Settings",
                    "Tagger thresholds, delays, and time-trace values must be numeric.",
                    parent=dialog,
                )
                return
            old_trigger_levels = dict(self.time_tagger_trigger_levels)
            old_delays = dict(self.time_tagger_channel_delays_ps)
            old_hydraharp_enabled_channels = set(self.hydraharp_enabled_channels)
            old_hydraharp_thresholds = dict(self.hydraharp_cfd_levels_mV)
            old_hydraharp_zero_cross_levels = dict(self.hydraharp_zero_cross_levels_mV)
            old_hydraharp_delays = dict(self.hydraharp_channel_delays_ps)
            old_hydraharp_sync_threshold = self.hydraharp_sync_cfd_level_mV
            old_hydraharp_sync_zero_cross = self.hydraharp_sync_zero_cross_mV
            old_hydraharp_sync_delay = self.hydraharp_sync_delay_ps

            self.time_tagger_trigger_levels = new_trigger_levels
            self.time_tagger_channel_delays_ps = new_delays
            if hydraharp is not None:
                self.hydraharp_enabled_channels = {
                    channel
                    for channel in self.hydraharp_setting_channels
                    if hydraharp_enabled_vars[channel].get()
                }
                self.hydraharp_cfd_levels_mV = new_hydraharp_thresholds
                self.hydraharp_zero_cross_levels_mV = new_hydraharp_zero_cross_levels
                self.hydraharp_channel_delays_ps = new_hydraharp_delays
                self.hydraharp_sync_cfd_level_mV = new_hydraharp_sync_threshold
                self.hydraharp_sync_zero_cross_mV = new_hydraharp_sync_zero_cross
                self.hydraharp_sync_delay_ps = new_hydraharp_sync_delay

            try:
                self.apply_time_tagger_settings()
                self.configure_time_tagger_counter()
                if hydraharp is not None:
                    self.apply_hydraharp_settings()
                    self.set_hydraharp_time_trace_settings(
                        new_hydraharp_history_s,
                        new_hydraharp_integration_ms,
                    )
            except Exception as exc:
                self.time_tagger_trigger_levels = old_trigger_levels
                self.time_tagger_channel_delays_ps = old_delays
                try:
                    self.apply_time_tagger_settings()
                except Exception:
                    pass
                self.hydraharp_enabled_channels = old_hydraharp_enabled_channels
                self.hydraharp_cfd_levels_mV = old_hydraharp_thresholds
                self.hydraharp_zero_cross_levels_mV = old_hydraharp_zero_cross_levels
                self.hydraharp_channel_delays_ps = old_hydraharp_delays
                self.hydraharp_sync_cfd_level_mV = old_hydraharp_sync_threshold
                self.hydraharp_sync_zero_cross_mV = old_hydraharp_sync_zero_cross
                self.hydraharp_sync_delay_ps = old_hydraharp_sync_delay
                messagebox.showerror("Tagger Settings", f"Could not apply settings:\n{exc}", parent=dialog)
                return

            if hasattr(self, "tx_log_print"):
                self.tx_log_print(
                    "[Time Tagger X] Settings applied: "
                    f"triggers={self.time_tagger_trigger_levels}, "
                    f"delays_ps={self.time_tagger_channel_delays_ps}"
                )
                if hydraharp is not None:
                    self.tx_log_print(
                        "[HydraHarp] Settings applied: "
                        f"enabled={sorted(self.hydraharp_enabled_channels)}, "
                        f"thresholds_mV={self.hydraharp_cfd_levels_mV}, "
                        f"zero_cross_mV={self.hydraharp_zero_cross_levels_mV}, "
                        f"delays_ps={self.hydraharp_channel_delays_ps}, "
                        f"history_s={self.hydraharp_time_trace_history_size_s:g}, "
                        f"integration_ms={self.hydraharp_time_trace_integration_ms:g} "
                        "(applies on next program start)"
                    )
            dialog.destroy()

        self._button(frame, text="OK", command=on_ok, variant="primary").grid(row=dialog_button_row, column=2, padx=6, pady=10)
        self._button(frame, text="Cancel", command=dialog.destroy).grid(row=dialog_button_row, column=3, padx=6, pady=10)

    def _normalize_voltage_entry(self, name, entry):
        if isinstance(entry, dict):
            chip_id = entry["chip_id"]
            pin = entry["pin"]
            voltage = entry["voltage"]
        elif isinstance(entry, (list, tuple)) and len(entry) == 3:
            chip_id, pin, voltage = entry
        else:
            raise ValueError(f"Invalid voltage entry for {name!r}: {entry!r}")

        return int(chip_id), int(pin), float(voltage)

    def load_default_voltages(self):
        raw_text = self.default_voltages_path.read_text(encoding="utf-8")

        try:
            raw_data = json.loads(raw_text)
        except json.JSONDecodeError:
            raw_data = ast.literal_eval(raw_text)

        if not isinstance(raw_data, dict):
            raise ValueError(
                f"{self.default_voltages_path.name} must contain an object mapping names to voltage entries."
            )

        return {
            str(name): self._normalize_voltage_entry(str(name), entry)
            for name, entry in raw_data.items()
        }

    def save_default_voltages(self):
        payload = {
            name: {
                "chip_id": int(chip_id),
                "pin": int(pin),
                "voltage": float(voltage),
            }
            for name, (chip_id, pin, voltage) in self.tx_dac_map.items()
        }
        self.default_voltages_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def default_rx_background_offsets(self):
        return {
            int(pd_index): 0.0
            for pd_index in self.rx_pd_to_channel
        }

    def load_rx_background_offsets(self):
        offsets = self.default_rx_background_offsets()
        path = self.rx_background_offsets_path

        if not path.exists():
            return offsets

        try:
            raw_data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw_data, dict):
                raise ValueError("Background offset file must contain a JSON object.")

            for pd_index in offsets:
                pd_key = f"PD{pd_index}"
                if pd_key in raw_data:
                    offsets[pd_index] = float(raw_data[pd_key])
                elif str(pd_index) in raw_data:
                    offsets[pd_index] = float(raw_data[str(pd_index)])
        except Exception as exc:
            print(f"ERROR: Could not load RX background offsets from {path.name}: {exc}")

        return offsets

    def save_rx_background_offsets(self):
        payload = {
            f"PD{pd_index}": float(value)
            for pd_index, value in sorted(self.rx_pd_background_offsets.items())
        }
        self.rx_background_offsets_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def _font(self, size=10, weight="normal", family="Segoe UI"):
        return Font(family=family, size=size, weight=weight)

    def _style_window(self, window, minsize=None):
        window.configure(bg=UI["bg"])
        if minsize is not None:
            window.minsize(*minsize)

    def _frame(self, parent, bg=None, **kwargs):
        return Frame(parent, bg=bg or UI["bg"], bd=0, highlightthickness=0, **kwargs)

    def _surface(self, parent, bg=None, **kwargs):
        return Frame(
            parent,
            bg=bg or UI["surface"],
            bd=0,
            highlightthickness=1,
            highlightbackground=UI["border"],
            highlightcolor=UI["border"],
            **kwargs,
        )

    def _label(self, parent, text, size=10, weight="normal", fg=None, bg=None, **kwargs):
        return Label(
            parent,
            text=text,
            font=self._font(size=size, weight=weight),
            fg=fg or UI["text"],
            bg=bg or parent.cget("bg"),
            **kwargs,
        )

    def _button(
        self,
        parent,
        text=None,
        command=None,
        textvariable=None,
        variant="secondary",
        size=10,
        padx=12,
        pady=7,
    ):
        if variant == "primary":
            bg = UI["accent"]
            active_bg = UI["accent_hover"]
            fg = "#ffffff"
        elif variant == "danger":
            bg = UI["danger"]
            active_bg = UI["danger_hover"]
            fg = "#ffffff"
        else:
            bg = UI["surface_alt"]
            active_bg = UI["border"]
            fg = UI["text"]

        options = {
            "command": command,
            "font": self._font(size=size, weight="bold"),
            "bg": bg,
            "fg": fg,
            "activebackground": active_bg,
            "activeforeground": fg,
            "relief": "flat",
            "bd": 0,
            "padx": padx,
            "pady": pady,
            "cursor": "hand2",
            "highlightthickness": 0,
        }
        if text is not None:
            options["text"] = text
        if textvariable is not None:
            options["textvariable"] = textvariable

        return Button(
            parent,
            **options,
        )

    def _entry(self, parent, textvariable=None, width=10):
        return Entry(
            parent,
            textvariable=textvariable,
            width=width,
            font=self._font(size=10),
            bg=UI["surface"],
            fg=UI["text"],
            insertbackground=UI["text"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=UI["border"],
            highlightcolor=UI["accent"],
        )

    def _spinbox(self, parent, **kwargs):
        return Spinbox(
            parent,
            font=self._font(size=11),
            bg=UI["surface"],
            fg=UI["text"],
            insertbackground=UI["text"],
            relief="flat",
            bd=0,
            buttonbackground=UI["surface_alt"],
            highlightthickness=1,
            highlightbackground=UI["border"],
            highlightcolor=UI["accent"],
            **kwargs,
        )

    def _checkbutton(self, parent, text=None, variable=None, command=None):
        options = {
            "variable": variable,
            "command": command,
            "font": self._font(size=10),
            "fg": UI["text"],
            "bg": parent.cget("bg"),
            "activebackground": parent.cget("bg"),
            "activeforeground": UI["text"],
            "selectcolor": UI["surface"],
            "highlightthickness": 0,
        }
        if text is not None:
            options["text"] = text

        return Checkbutton(parent, **options)

    def _log(self, parent, height, width):
        log = ScrolledText(
            parent,
            height=height,
            width=width,
            font=("Consolas", 10),
            bg=UI["log_bg"],
            fg=UI["log_fg"],
            insertbackground=UI["log_fg"],
            relief="flat",
            bd=0,
            padx=8,
            pady=8,
        )
        return log

    def _style_plot(self, figure, axis):
        figure.patch.set_facecolor(UI["surface"])
        axis.set_facecolor(UI["plot_bg"])
        axis.set_prop_cycle(color=[UI["plot_blue"], "#f2cc60", "#56d364", "#ff7b72", "#d2a8ff"])
        axis.grid(True, color=UI["plot_grid"], linewidth=0.6, alpha=0.9)
        axis.tick_params(colors=UI["muted"])
        for spine in axis.spines.values():
            spine.set_color(UI["border"])
        axis.title.set_color(UI["text"])
        axis.xaxis.label.set_color(UI["muted"])
        axis.yaxis.label.set_color(UI["muted"])

    def _style_legend(self, legend):
        if legend is None:
            return
        legend.get_frame().set_facecolor(UI["plot_bg"])
        legend.get_frame().set_edgecolor(UI["border"])
        for text in legend.get_texts():
            text.set_color(UI["text"])

    def build_tx_panel(self):
        voltage_max = 28.0
        voltage_increment = 0.05
        ranges = {"PD_DWDM": (1.0, 4.0, 1.0), "RT_DWDM": (0.0, 1.0, 0.01)}

        self.tx_dac_map = self.load_default_voltages()

        self.tx_write_order = list(self.tx_dac_map.keys())

        self.tx_win = Toplevel(self.root)
        self.tx_win.title("TX Control Panel")
        self._style_window(self.tx_win, minsize=(1100, 400))

        shell = self._frame(self.tx_win)
        shell.pack(fill="both", expand=True, padx=18, pady=16)

        header = self._frame(shell)
        header.pack(fill="x", pady=(0, 12))
        self._label(header, "TX Control Panel", size=18, weight="bold").pack(side=LEFT)
        self._label(
            header,
            self.default_voltages_path.name,
            size=10,
            fg=UI["muted"],
        ).pack(side=RIGHT, pady=(5, 0))

        frame = self._surface(shell)
        frame.pack(fill="x", pady=(0, 12), ipadx=16, ipady=14)

        self.tx_log = self._log(shell, height=8, width=90)
        self.tx_log.pack(fill="both", expand=False, pady=(0, 10))
        self.tx_log_print = lambda s: (self.tx_log.insert(END, s + "\n"), self.tx_log.see(END))

        self.voltage_vars = {}
        for name, (_chip_id, _pin, volt) in self.tx_dac_map.items():
            self.voltage_vars[name] = StringVar(value=f"{float(volt):.3f}")

        def tx_names_between(start_name, end_name):
            try:
                start_index = self.tx_write_order.index(start_name)
                end_index = self.tx_write_order.index(end_name)
            except ValueError:
                return []
            if start_index > end_index:
                start_index, end_index = end_index, start_index
            return self.tx_write_order[start_index : end_index + 1]

        preferred_tx_rows = [
            tx_names_between("CQS", "RTRbot"),
            tx_names_between("QF11", "QF42"),
            [
                "MUXa",
                "PCa1",
                "PCa2",
                "PCa3",
                "PCa4",
                "MUXb",
                "PCb1",
                "PCb2",
                "PCb3",
                "PCb4",
            ],
        ]
        placed_tx_names = set()
        tx_layout_rows = []
        for row_names in preferred_tx_rows:
            present_names = [
                name
                for name in row_names
                if name in self.tx_dac_map and name not in placed_tx_names
            ]
            tx_layout_rows.append(present_names)
            placed_tx_names.update(present_names)
        tx_layout_rows.append([
            name for name in self.tx_write_order if name not in placed_tx_names
        ])

        for row, row_names in enumerate(tx_layout_rows):
            for index, name in enumerate(row_names):
                column = index * 2
                self._label(frame, text=name, size=10, weight="bold", fg=UI["muted"]).grid(
                    row=row, column=column, sticky="e", padx=(8, 6), pady=6
                )
                low, high, increment = ranges.get(name, (0.0, voltage_max, voltage_increment))
                self._spinbox(
                    frame,
                    from_=low,
                    to=high,
                    increment=increment,
                    format="%.3f",
                    width=8,
                    textvariable=self.voltage_vars[name],
                ).grid(row=row, column=column + 1, padx=(0, 14), pady=6, sticky="w")

        button_row = self._frame(shell)
        button_row.pack(fill="x", pady=(2, 0))

        def tx_toolbar_button(side=LEFT, **kwargs):
            self._button(button_row, size=9, padx=7, pady=5, **kwargs).pack(
                side=side,
                padx=3,
                pady=4,
            )

        tx_toolbar_button(
            text="Set_Tx_OOK",
            command=self.tx_set_levels,
        )
        tx_toolbar_button(
            text="Tx_OOK",
            command=self.tx_pulse_test,
        )
        # self.tx_mode_pin_button_text = StringVar(value="Pulse")
        # tx_toolbar_button(
        #     textvariable=self.tx_mode_pin_button_text,
        #     command=self.tx_toggle_mode_pin,
        # )
        tx_toolbar_button(
            text="Set_Tagger",
            command=self.open_tagger_settings_dialog,
        )
        tx_toolbar_button(
            text="TimeTag",
            command=lambda: open_qutag_timestamp_window(self),
        )
        tx_toolbar_button(
            text="HH → QSFP",
            command=self.open_optical_capture_window,
        )
        tx_toolbar_button(
            text="Scan_WL",
            command=lambda: tx_scan_frequency(self),
        )

        tx_toolbar_button(
            text="Scan_PZ(Rx)",
            command=lambda: tx_scan_piezo(self),
        )
        tx_toolbar_button(
            text="Scan_PZ(SPD)",
            command=lambda: tx_scan_piezo_qutag(self),
        )

        tx_toolbar_button(
            text="Scan_2V",
            command=self.tx_scan_2v,
        )

        tx_toolbar_button(
            text="Plot_2v",
            command=lambda: run_scan_2v_analyzer(
                parent=self.tx_win,
                log_print=lambda message: self.tx_log_print(f"[2V Analyzer] {message}"),
            ),
        )

        tx_toolbar_button(
            text="QDCP_Status",
            command=self.tx_qdcp_slip_status,
        )

        tx_toolbar_button(
            text="Set_Offset",
            command=self.tx_set_offset,
        )

        tx_toolbar_button(
            text="Update",
            command=self.tx_update_changed_from_gui,
            variant="primary",
            side=RIGHT,
        )

        tx_toolbar_button(
            text="Quit",
            command=self._quit_all,
            variant="danger",
            side=RIGHT,
        )

    def build_rx_panel(self):
        self.rx_win = Toplevel(self.root)
        self.rx_win.title("RX Control Panel")
        self.rx_win.geometry("1500x900")
        self._style_window(self.rx_win, minsize=(1180, 760))

        shell = self._frame(self.rx_win)
        shell.pack(fill="both", expand=True, padx=14, pady=14)

        header = self._frame(shell)
        header.pack(fill="x", pady=(0, 10))
        self._label(header, "RX Control Panel", size=18, weight="bold").pack(side=LEFT)

        controls = self._frame(shell)
        controls.pack(fill="x", pady=(0, 10))

        self._button(
            controls,
            text="Start Stream",
            command=self.rx_stream_start,
        ).pack(side=LEFT, padx=6)
        self._button(
            controls,
            text="Stop Stream",
            command=self.rx_stream_stop,
        ).pack(side=LEFT, padx=6)
        self._button(
            controls,
            text="Decode Mode (blocking)",
            command=self.rx_decode_mode,
        ).pack(side=LEFT, padx=6)
        self._button(
            controls,
            text="QDCP Decode (blocking)",
            command=self.rx_qdcp_decode_mode,
        ).pack(side=LEFT, padx=6)
        self._button(
            controls,
            text="Exit Decode (Reconnect)",
            command=self.rx_reconnect,
        ).pack(side=LEFT, padx=6)
        self._button(
            controls,
            text="Settings",
            command=self.rx_open_plot_selector,
        ).pack(side=LEFT, padx=6)

        content = self._frame(shell)
        content.pack(fill="both", expand=True)

        for row in range(2):
            content.grid_rowconfigure(row, weight=1)
        for column in range(4):
            content.grid_columnconfigure(column, weight=1, uniform="rx_grid")

        self.rx_axes = []
        self.rx_canvases = []
        self.rx_axis_to_canvas = {}
        self.rx_axis_plot_key = {}
        self.rx_plot_labels = {}
        self.rx_plot_redraw_enabled = {}
        self.rx_lines = [None] * self.rx_detector_count
        self.spd_lines = [None] * self.spd_detector_count
        self.qutag_hist_line = None
        self.qutag_hist_lines = []

        for group_index, channel_group in enumerate(self.rx_channel_groups):
            plot_key = f"rx_{group_index}"
            plot_label = "Channels " + ",".join(str(channel) for channel in channel_group)
            self.rx_plot_labels[plot_key] = plot_label
            self.rx_plot_redraw_enabled[plot_key] = False

            row, column = self.rx_grid_positions[group_index]
            panel = self._surface(content)
            panel.grid(row=row, column=column, padx=4, pady=4, sticky="nsew")

            figure = Figure(figsize=(4.8, 2.7), dpi=100)
            axis = figure.add_subplot(111)
            self._style_plot(figure, axis)
            self.rx_axes.append(axis)
            self.rx_axis_plot_key[axis] = plot_key
            axis.set_xlim(0, self.rx_plot_window_s)
            axis.set_ylim(0, 4096)
            axis.margins(x=0.0, y=0.0)
            axis.tick_params(axis="both", which="major", labelsize=8, pad=1)

            for channel_number in channel_group:
                detector_index = self.rx_channel_to_pd[channel_number]
                line, = axis.plot([], [], label=f"Channel {channel_number}")
                self.rx_lines[detector_index] = line

            legend = axis.legend(
                loc="upper left",
                frameon=False,
                fontsize=8,
                borderaxespad=0.2,
                handlelength=1.2,
                handletextpad=0.4,
                labelspacing=0.2,
            )
            self._style_legend(legend)
            figure.subplots_adjust(left=0.09, right=0.985, bottom=0.08, top=0.985)

            canvas = FigureCanvasTkAgg(figure, master=panel)
            self.rx_axis_to_canvas[axis] = canvas
            canvas.get_tk_widget().configure(bg=UI["surface"], highlightthickness=0)
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
            self.rx_canvases.append(canvas)
        
        # Add Time Tagger X and HydraHarp channels to the single-photon count panel.
        plot_key = "spd"
        spd_series = list(getattr(self, "spd_series", []))
        self.rx_plot_labels[plot_key] = "Single Photon Counts (per Time Tagger X bin)"
        self.rx_plot_redraw_enabled[plot_key] = False
        row, column = 1, 1
        panel = self._surface(content)
        panel.grid(row=row, column=column, padx=4, pady=4, sticky="nsew")
        figure = Figure(figsize=(4.8, 2.7), dpi=100)
        axis = figure.add_subplot(111)
        self._style_plot(figure, axis)
        self.rx_axes.append(axis)
        self.rx_axis_plot_key[axis] = plot_key
        axis.set_xlim(0, self.rx_plot_window_s)
        axis.set_ylim(0, 4096)
        axis.margins(x=0.0, y=0.0)
        axis.tick_params(axis="both", which="major", labelsize=8, pad=1)
        for line_index, (tagger_name, channel_index) in enumerate(spd_series):
            line, = axis.plot([], [], label=f"{tagger_name} {channel_index}")
            self.spd_lines[line_index] = line
        legend = axis.legend(
            loc="upper left",
            frameon=False,
            fontsize=8,
            borderaxespad=0.2,
            handlelength=1.2,
            handletextpad=0.4,
            labelspacing=0.2,
        )
        self._style_legend(legend)
        figure.subplots_adjust(left=0.09, right=0.985, bottom=0.08, top=0.985)
        canvas = FigureCanvasTkAgg(figure, master=panel)
        self.rx_axis_to_canvas[axis] = canvas
        canvas.get_tk_widget().configure(bg=UI["surface"], highlightthickness=0)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self.rx_canvases.append(canvas)

        plot_key = "hist"
        self.rx_plot_labels[plot_key] = self.qutag_hist_plot_label()
        self.rx_plot_redraw_enabled[plot_key] = False
        row, column = 1, 2
        panel = self._surface(content)
        panel.grid(row=row, column=column, padx=4, pady=4, sticky="nsew")
        figure = Figure(figsize=(4.8, 2.7), dpi=100)
        axis = figure.add_subplot(111)
        self._style_plot(figure, axis)
        self.rx_axes.append(axis)
        self.rx_axis_plot_key[axis] = plot_key
        axis.set_xlim(0, self.qutag_hist_bin_width_ps * self.qutag_hist_bin_count / 1000.0)
        axis.set_ylim(0, 1)
        axis.margins(x=0.0, y=0.05)
        axis.tick_params(axis="both", which="major", labelsize=8, pad=1)
        self.qutag_hist_axis = axis
        self.qutag_hist_data = {}
        self.qutag_hist_stats = {}
        self.qutag_hist_index = {}
        self.qutag_hist_plot_dirty = False
        self.qutag_hist_lines = []
        for hist_pair in getattr(self, "qutag_hist_pairs", [(1, 2)]):
            line, = axis.plot([], [], label=self.qutag_hist_line_label(hist_pair))
            self.qutag_hist_lines.append(line)
        self.qutag_hist_line = self.qutag_hist_lines[0] if self.qutag_hist_lines else None
        legend = axis.legend(
            loc="upper left",
            frameon=False,
            fontsize=8,
            borderaxespad=0.2,
            handlelength=1.2,
            handletextpad=0.4,
            labelspacing=0.2,
        )
        self._style_legend(legend)
        figure.subplots_adjust(left=0.09, right=0.985, bottom=0.08, top=0.985)
        canvas = FigureCanvasTkAgg(figure, master=panel)
        self.rx_axis_to_canvas[axis] = canvas
        canvas.get_tk_widget().configure(bg=UI["surface"], highlightthickness=0)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self.rx_canvases.append(canvas)

        self.rx_history = collections.deque()
        self.spd_history = collections.deque()
        self.rx_serial_lock = threading.Lock()
        self.rx_line_queue = queue.SimpleQueue()
        self.rx_reader_stop = threading.Event()
        self.rx_reader_thread = None
        self.rx_max_lines_per_poll = 200
        self.rx_plot_interval_s = 0.05
        self.rx_next_plot_update = 0.0
        self.rx_plot_dirty = False
        self.spd_plot_dirty = False
        self.spd_poll_interval_s = 0.1
        self.spd_next_poll = 0.0
        self.spd_time_tagger_error_logged = False
        self.spd_hydraharp_error_logged = False
        self.time_tagger_latest_counts = None
        self.time_tagger_latest_counts_monotonic = 0.0
        self.rx_log_flush_interval_s = 0.1
        self.rx_next_log_flush = 0.0
        self.rx_log_max_lines = 2000
        self.rx_pending_log_lines = []
        self.rx_reader_idle_sleep_s = 0.01

        log_frame = self._surface(content)
        log_frame.grid(row=1, column=3, padx=4, pady=4, sticky="nsew")

        self.rx_log = self._log(log_frame, height=16, width=48)
        self.rx_log.pack(fill="both", expand=True, padx=8, pady=8)
        self.rx_log_print = lambda s: (self.rx_log.insert(END, s + "\n"), self.rx_log.see(END))

        self.rx_polling = True
        self.rx_streaming = False
        self.qutag_hist_configure()
        self.rx_reset_plot_data()
        self.rx_start_reader_thread()
        self.rx_poll()

    def rx_open_plot_selector(self):
        existing_window = getattr(self, "rx_plot_selector_win", None)
        if existing_window is not None and existing_window.winfo_exists():
            existing_window.lift()
            existing_window.focus_force()
            return

        selector = Toplevel(self.rx_win)
        selector.title("Settings")
        selector.resizable(False, False)
        self._style_window(selector)
        self.rx_plot_selector_win = selector
        self.rx_plot_selector_vars = {}
        self.qutag_hist_integration_time_var = StringVar(
            value=str(getattr(self, "qutag_hist_integration_time_ms", 500.0))
        )
        self.qutag_hist_bin_width_var = StringVar(
            value=str(getattr(self, "qutag_hist_bin_width_ps", 1))
        )
        self.qutag_hist_bin_count_var = StringVar(
            value=str(getattr(self, "qutag_hist_bin_count", 3000))
        )
        hist_pair_options = self.qutag_hist_get_pair_options()
        selected_hist_pairs = set(
            hist_pair for hist_pair in self.qutag_hist_get_pairs()
            if hist_pair in hist_pair_options
        )
        self.qutag_hist_pair_vars = {
            hist_pair: BooleanVar(value=hist_pair in selected_hist_pairs)
            for hist_pair in hist_pair_options
        }
        self.rx_background_offset_vars = {}

        frame = self._surface(selector)
        frame.pack(padx=14, pady=12, fill="both", expand=True, ipadx=8, ipady=8)

        self._label(frame, text="Histogram", size=12, weight="bold").pack(anchor="w")
        hist_frame = self._frame(frame, bg=UI["surface"])
        hist_frame.pack(anchor="w", fill="x", pady=(4, 12))

        self._label(hist_frame, text="Histogram interval").grid(row=0, column=0, sticky="e", padx=(0, 6), pady=3)
        self._entry(hist_frame, textvariable=self.qutag_hist_integration_time_var, width=10).grid(row=0, column=1, sticky="w", pady=3)
        self._label(hist_frame, text="ms", fg=UI["muted"]).grid(row=0, column=2, sticky="w", padx=(4, 10), pady=3)

        self._label(hist_frame, text="Hist Bin Width").grid(row=1, column=0, sticky="e", padx=(0, 6), pady=3)
        self._entry(hist_frame, textvariable=self.qutag_hist_bin_width_var, width=10).grid(row=1, column=1, sticky="w", pady=3)
        self._label(hist_frame, text="ps", fg=UI["muted"]).grid(row=1, column=2, sticky="w", padx=(4, 10), pady=3)

        self._label(hist_frame, text="Hist Bin Count").grid(row=2, column=0, sticky="e", padx=(0, 6), pady=3)
        self._entry(hist_frame, textvariable=self.qutag_hist_bin_count_var, width=10).grid(row=2, column=1, sticky="w", pady=3)
        self._label(hist_frame, text="bins", fg=UI["muted"]).grid(row=2, column=2, sticky="w", padx=(4, 10), pady=3)

        self._label(hist_frame, text="Hist Pairs").grid(row=3, column=0, sticky="ne", padx=(0, 6), pady=3)
        pair_frame = self._frame(hist_frame, bg=UI["surface"])
        pair_frame.grid(row=3, column=1, columnspan=2, sticky="w", pady=3)
        for hist_pair, pair_var in self.qutag_hist_pair_vars.items():
            self._checkbutton(
                pair_frame,
                text=self.qutag_hist_pair_option_label(hist_pair),
                variable=pair_var,
            ).pack(side=LEFT, padx=(0, 8))

        self._button(
            hist_frame,
            text="Apply",
            command=self.rx_apply_hist_integration_time_setting,
            variant="primary",
        ).grid(row=4, column=1, sticky="w", pady=(6, 0))

        self._label(frame, text="Background Offset", size=12, weight="bold").pack(anchor="w")
        background_frame = self._frame(frame, bg=UI["surface"])
        background_frame.pack(anchor="w", fill="x", pady=(4, 12))

        for row_index, pd_index in enumerate(sorted(self.rx_pd_to_channel)):
            saved_value = getattr(self, "rx_pd_background_offsets", {}).get(pd_index)
            entry_value = "0" if saved_value is None else f"{float(saved_value):g}"
            var = StringVar(value=entry_value)
            self.rx_background_offset_vars[pd_index] = var

            label_text = f"PD{pd_index}"

            self._label(background_frame, text=label_text).grid(
                row=row_index,
                column=0,
                sticky="e",
                padx=(0, 6),
                pady=3,
            )
            self._entry(background_frame, textvariable=var, width=10).grid(
                row=row_index,
                column=1,
                sticky="w",
                pady=3,
            )

        self._button(
            background_frame,
            text="Save",
            command=self.rx_save_background_offset_settings,
            variant="primary",
        ).grid(row=len(self.rx_pd_to_channel), column=1, sticky="w", pady=(6, 0))

        self._label(frame, text="Redraw during stream", size=12, weight="bold").pack(anchor="w", pady=(0, 8))

        for plot_key, plot_label in self.rx_plot_labels.items():
            var = BooleanVar(value=self.rx_plot_redraw_enabled.get(plot_key, False))
            self.rx_plot_selector_vars[plot_key] = var
            self._checkbutton(
                frame,
                text=plot_label,
                variable=var,
                command=lambda key=plot_key, value=var: self.rx_set_plot_redraw(key, value),
            ).pack(anchor="w")

        buttons = self._frame(frame, bg=UI["surface"])
        buttons.pack(fill="x", pady=(10, 0))
        self._button(
            buttons,
            text="All",
            command=lambda: self.rx_set_all_plot_redraw(True),
        ).pack(side=LEFT, padx=(0, 6))
        self._button(
            buttons,
            text="None",
            command=lambda: self.rx_set_all_plot_redraw(False),
        ).pack(side=LEFT, padx=(0, 6))
        self._button(
            buttons,
            text="Close",
            command=selector.destroy,
        ).pack(side=RIGHT)

    def rx_set_plot_redraw(self, plot_key, variable):
        enabled = bool(variable.get())
        self.rx_plot_redraw_enabled[plot_key] = enabled
        if enabled:
            if plot_key == "spd":
                self.spd_plot_dirty = True
            elif plot_key == "hist":
                self.qutag_hist_plot_dirty = True
            else:
                self.rx_plot_dirty = True

    def rx_set_all_plot_redraw(self, enabled):
        for plot_key, variable in getattr(self, "rx_plot_selector_vars", {}).items():
            variable.set(enabled)
            self.rx_set_plot_redraw(plot_key, variable)

    def rx_apply_hist_integration_time_setting(self):
        try:
            integration_time_ms = float(self.qutag_hist_integration_time_var.get())
            bin_width_ps = int(float(self.qutag_hist_bin_width_var.get()))
            bin_count = int(float(self.qutag_hist_bin_count_var.get()))
            hist_pairs = [
                hist_pair
                for hist_pair in self.qutag_hist_get_pair_options()
                if self.qutag_hist_pair_vars[hist_pair].get()
            ]
            self.qutag_hist_set_pairs(hist_pairs, configure=False)
            self.qutag_hist_set_params(bin_width_ps, bin_count)
            self.qutag_hist_set_integration_time_ms(integration_time_ms)
        except Exception as exc:
            messagebox.showerror("Settings", str(exc), parent=self.rx_plot_selector_win)
            return

        self.rx_log_print(
            f"[Time Tagger X Histogram] Settings applied: pairs={self.qutag_hist_plot_label()}, "
            f"integration={integration_time_ms:.1f} ms, "
            f"bin_width={bin_width_ps} ps, bin_count={bin_count}"
        )

    def rx_save_background_offset_settings(self):
        background_vars = getattr(self, "rx_background_offset_vars", {})
        offsets = {}

        try:
            for pd_index, variable in background_vars.items():
                raw_value = variable.get().strip()
                if not raw_value:
                    offsets[pd_index] = 0.0
                    continue
                offsets[pd_index] = float(raw_value)
        except ValueError:
            messagebox.showerror(
                "Settings",
                "Background offsets must be numeric, or left blank.",
                parent=self.rx_plot_selector_win,
            )
            return

        try:
            saved_offsets = self.rx_set_background_offsets(offsets)
        except Exception as exc:
            messagebox.showerror(
                "Settings",
                f"Could not save background offsets:\n{exc}",
                parent=self.rx_plot_selector_win,
            )
            return

        self.rx_log_print(
            "[RX Background] Saved offsets (applied): "
            + self.rx_format_background_offsets(saved_offsets)
        )

    def rx_set_background_offsets(self, offsets):
        normalized_offsets = self.default_rx_background_offsets()
        for pd_index, value in offsets.items():
            pd_index = int(pd_index)
            if pd_index in normalized_offsets:
                normalized_offsets[pd_index] = float(value)

        previous_offsets = dict(getattr(self, "rx_pd_background_offsets", {}))
        self.rx_pd_background_offsets = normalized_offsets
        try:
            self.save_rx_background_offsets()
        except Exception:
            self.rx_pd_background_offsets = previous_offsets
            raise

        self.rx_rebase_background_offset_history(previous_offsets, normalized_offsets)
        self.rx_update_background_offset_fields()
        return normalized_offsets

    def rx_format_background_offsets(self, offsets):
        return ", ".join(
            f"PD{pd_index}={value:g}"
            for pd_index, value in sorted(offsets.items())
        )

    def rx_update_background_offset_fields(self):
        background_vars = getattr(self, "rx_background_offset_vars", {})
        if not background_vars:
            return

        offsets = getattr(self, "rx_pd_background_offsets", {})
        for pd_index, variable in background_vars.items():
            variable.set(f"{float(offsets.get(pd_index, 0.0)):g}")

    def rx_rebase_background_offset_history(self, previous_offsets, current_offsets):
        history = getattr(self, "rx_history", None)
        if not history:
            return

        rebased_history = collections.deque()
        for timestamp, values in history:
            rebased_values = []
            for index, value in enumerate(values):
                pd_index = int(index)
                previous_offset = float(previous_offsets.get(pd_index, 0.0))
                current_offset = float(current_offsets.get(pd_index, 0.0))
                rebased_values.append(
                    int(max(0.0, float(value) + previous_offset - current_offset))
                )
            rebased_history.append((timestamp, rebased_values))

        self.rx_history = rebased_history
        self.rx_plot_dirty = True

    def _quit_all(self):
        self.optical_closing = True
        if (
            getattr(self, "qutag_timestamp_writing", False)
            or getattr(self, "qutag_timestamp_preparing_hydraharp", False)
            or getattr(self, "qutag_timestamp_hydraharp_recording", False)
        ):
            _stop_timestamp_writing(self)
        if getattr(self, "qutag_timestamp_qsfp_active", False):
            self.root.after(100, self._quit_all)
            return
        internet = getattr(self, "timestamp_internet", None)
        if internet is not None:
            if internet.snapshot()["active"]:
                internet.stop_recording()
                self.root.after(100, self._quit_all)
                return
            internet.close()
            self.timestamp_internet = None
        if self.optical_capture_active:
            self.optical_capture_stop.set()
            worker = self.optical_capture_thread
            if worker is not None and worker.is_alive():
                self.root.after(100, self._quit_all)
                return
            self.optical_capture_active = False
        modes = getattr(self, "qsfp_modes", None)
        if modes is not None:
            if modes.applying:
                self.root.after(100, self._quit_all)
                return
            try:
                modes.close()
            except Exception as exc:
                self.tx_log_print(f"[QSFP] Shutdown: {exc}")
        try:
            self.rx_polling = False
        except Exception:
            pass
        try:
            self.rx_stop_reader_thread()
        except Exception:
            pass
        try:
            self.tx.close()
        except Exception:
            pass
        try:
            if hasattr(self, "rx_serial_lock"):
                with self.rx_serial_lock:
                    self.rx.close()
            else:
                self.rx.close()
        except Exception:
            pass
        self.shutdown_time_tagger()
        self.shutdown_hydraharp()
        self.root.destroy()


if __name__ == "__main__":
    app = App()
    app.root.mainloop()
