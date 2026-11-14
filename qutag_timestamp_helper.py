import collections
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import (
    Toplevel,
    Frame,
    Label,
    Entry,
    Button,
    Checkbutton,
    Radiobutton,
    BooleanVar,
    StringVar,
    messagebox,
    filedialog,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from optical_stream import OpticalLink, hydraharp_blocks, capture_hh_qsfp
from qsfp_mode_ui import get_qsfp_modes
from internet_timetag import InternetCoordinator


TIMETAG_UI = {
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
    "plot_bg": "#101822",
    "plot_grid": "#253244",
}


def _ui(app):
    return getattr(app, "ui", TIMETAG_UI)


def _style_window(app, window):
    if hasattr(app, "_style_window"):
        app._style_window(window)
    else:
        window.configure(bg=_ui(app)["bg"])


def _frame(app, parent, surface=False):
    if surface and hasattr(app, "_surface"):
        return app._surface(parent)
    if hasattr(app, "_frame"):
        return app._frame(parent, bg=_ui(app)["surface"] if surface else _ui(app)["bg"])
    colors = _ui(app)
    return Frame(
        parent,
        bg=colors["surface"] if surface else colors["bg"],
        bd=0,
        highlightthickness=1 if surface else 0,
        highlightbackground=colors["border"],
        highlightcolor=colors["border"],
    )


def _label(app, parent, text=None, textvariable=None, fg=None):
    colors = _ui(app)
    return Label(
        parent,
        text=text,
        textvariable=textvariable,
        fg=fg or colors["text"],
        bg=parent.cget("bg"),
        font=("Segoe UI", 10),
    )


def _entry(app, parent, textvariable=None, width=10):
    if hasattr(app, "_entry"):
        return app._entry(parent, textvariable=textvariable, width=width)
    colors = _ui(app)
    return Entry(
        parent,
        textvariable=textvariable,
        width=width,
        bg=colors["surface"],
        fg=colors["text"],
        insertbackground=colors["text"],
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground=colors["border"],
        highlightcolor=colors["accent"],
    )


def _button(app, parent, text=None, command=None, variant="secondary"):
    if hasattr(app, "_button"):
        return app._button(parent, text=text, command=command, variant=variant, size=10, padx=10, pady=6)

    colors = _ui(app)
    if variant == "primary":
        bg = colors["accent"]
        active_bg = colors["accent_hover"]
        fg = "#ffffff"
    elif variant == "danger":
        bg = colors["danger"]
        active_bg = colors["danger_hover"]
        fg = "#ffffff"
    else:
        bg = colors["surface_alt"]
        active_bg = colors["border"]
        fg = colors["text"]
    return Button(
        parent,
        text=text,
        command=command,
        font=("Segoe UI", 10, "bold"),
        bg=bg,
        fg=fg,
        activebackground=active_bg,
        activeforeground=fg,
        relief="flat",
        bd=0,
        padx=10,
        pady=6,
        highlightthickness=0,
    )


def _checkbutton(app, parent, text=None, variable=None, command=None):
    if hasattr(app, "_checkbutton"):
        return app._checkbutton(parent, text=text, variable=variable, command=command)

    colors = _ui(app)
    return Checkbutton(
        parent,
        text=text,
        variable=variable,
        command=command,
        font=("Segoe UI", 10),
        fg=colors["text"],
        bg=parent.cget("bg"),
        activebackground=parent.cget("bg"),
        activeforeground=colors["text"],
        selectcolor=colors["surface"],
        highlightthickness=0,
    )


def _radiobutton(app, parent, text=None, variable=None, value=None, command=None):
    colors = _ui(app)
    return Radiobutton(
        parent,
        text=text,
        variable=variable,
        value=value,
        command=command,
        font=("Segoe UI", 10),
        fg=colors["text"],
        bg=parent.cget("bg"),
        activebackground=parent.cget("bg"),
        activeforeground=colors["text"],
        selectcolor=colors["surface"],
        highlightthickness=0,
    )


def _style_plot(app, figure, axis):
    if hasattr(app, "_style_plot"):
        app._style_plot(figure, axis)
        return
    colors = _ui(app)
    figure.patch.set_facecolor(colors["surface"])
    axis.set_facecolor(colors["plot_bg"])
    if "plot_blue" in colors:
        axis.set_prop_cycle(color=[colors["plot_blue"], "#f2cc60", "#56d364", "#ff7b72", "#d2a8ff"])
    axis.grid(True, color=colors["plot_grid"], linewidth=0.6, alpha=0.9)
    axis.tick_params(colors=colors["muted"])
    for spine in axis.spines.values():
        spine.set_color(colors["border"])
    axis.title.set_color(colors["text"])
    axis.xaxis.label.set_color(colors["muted"])
    axis.yaxis.label.set_color(colors["muted"])


def _style_legend(app, legend):
    if hasattr(app, "_style_legend"):
        app._style_legend(legend)
        return
    if legend is None:
        return
    colors = _ui(app)
    legend.get_frame().set_facecolor(colors["plot_bg"])
    legend.get_frame().set_edgecolor(colors["border"])
    for text in legend.get_texts():
        text.set_color(colors["text"])


def _qcs_integer_value(text, label):
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a whole number") from exc
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"{label} must be a whole number")
    return int(value)


def open_qutag_timestamp_window(app):
    from QCS_algorithm import DEFAULT_QCS_CONFIG

    existing_window = getattr(app, "qutag_timestamp_win", None)
    if existing_window is not None and existing_window.winfo_exists():
        existing_window.lift()
        existing_window.focus_force()
        return

    app.qutag_timestamp_win = Toplevel(app.tx_win)
    app.qutag_timestamp_win.title("Time Stamps")
    # Keep the two plot windows aligned across the top, with acquisition controls below the left plot.
    screen_width = app.qutag_timestamp_win.winfo_screenwidth()
    screen_height = app.qutag_timestamp_win.winfo_screenheight()
    app.qutag_timestamp_win.geometry(f"{screen_width}x{screen_height}+0+0")
    app.qutag_timestamp_win.minsize(1100, 700)
    app.qutag_timestamp_win.grid_columnconfigure(0, weight=1)
    app.qutag_timestamp_win.grid_rowconfigure(0, weight=1)
    _style_window(app, app.qutag_timestamp_win)
    app.qutag_timestamp_writing = False
    app.qutag_timestamp_preparing_hydraharp = False
    app.qutag_timestamp_hydraharp_recording = False
    app.qutag_timestamp_hydraharp_ptu_path = None
    app.qutag_timestamp_hydraharp_h5_path = None
    conversion_running = getattr(app, "qutag_timestamp_hydraharp_conversion_running", False)
    app.qutag_timestamp_hydraharp_conversion_running = conversion_running
    app.qutag_timestamp_hydraharp_warning = None
    app.qutag_timestamp_qsfp_active = False
    app.qutag_timestamp_qsfp_stopping = False
    app.qutag_timestamp_qsfp_worker = None
    if not conversion_running:
        app.qutag_timestamp_qsfp_result = None
    app.qutag_timestamp_qsfp_stop = threading.Event()
    app.qutag_timestamp_qsfp_output = None
    app.qutag_timestamp_close_pending = False
    # This flag belongs to the shared HydraHarp, so opening the TimeTag
    # window must not clear an optical capture already in progress.
    app.qutag_timestamp_streaming = True
    app.qutag_timestamp_history = collections.deque()
    app.qutag_timestamp_poll_interval_s = 0.1
    app.qutag_timestamp_plot_window_s = getattr(app, "rx_plot_window_s", 30.0)
    app.qutag_timestamp_time_tagger_available = (
        getattr(app, "time_tagger", None) is not None
    )
    # The Time Stamps view normally follows the Rx live-count selection, but
    # also shows Time Tagger X channels 3 and 4 for the trigger/reference
    # signals.
    # Keep only physical inputs that the shared Time Tagger X counter is
    # configured to acquire. A negative Time Tagger X channel is a falling
    # edge, but it still belongs to the same physical input for this plot.
    live_counter_channels = list(
        getattr(app, "time_tagger_setting_channels", [1, 2, 3, 4])
    )
    app.qutag_timestamp_counter_hardware_channels = {}
    for hardware_channel in live_counter_channels:
        physical_channel = abs(int(hardware_channel))
        if physical_channel in (1, 2, 3, 4):
            app.qutag_timestamp_counter_hardware_channels.setdefault(
                physical_channel, int(hardware_channel)
            )
    requested_counter_indices = list(getattr(app, "spd_counter_indices", [1, 2])) + [3, 4]
    app.qutag_timestamp_counter_indices = (
        list(
            dict.fromkeys(
                abs(int(channel))
                for channel in requested_counter_indices
                if abs(int(channel)) in app.qutag_timestamp_counter_hardware_channels
            )
        )
        if app.qutag_timestamp_time_tagger_available
        else []
    )
    # HydraHarp is optional for this window.  Do not create unavailable plot
    # series when the application is running with Time Tagger X alone.
    app.qutag_timestamp_hydraharp_channel_indices = (
        list(getattr(app, "hydraharp_channel_indices", [1]))
        if getattr(app, "hydraharp", None) is not None
        else []
    )
    app.qutag_timestamp_count_series = (
        [("Time Tagger X", counter_index) for counter_index in app.qutag_timestamp_counter_indices]
        + [("HydraHarp", channel_index) for channel_index in app.qutag_timestamp_hydraharp_channel_indices]
    )
    app.qutag_lock_running = False
    app.qutag_lock_target_ratio = getattr(app, "qutag_lock_target_ratio", 0.9)
    app.qutag_lock_channel = getattr(app, "qutag_lock_channel", 1)
    app.qutag_lock_rate = getattr(app, "qutag_lock_rate", 0.3)
    app.qutag_lock_initial_voltage = None
    app.qutag_lock_start_voltage = None
    app.qutag_lock_peak_voltage = None
    from qutag_piezoscan_helper import load_latest_qutag_piezoscan_csv
    load_latest_qutag_piezoscan_csv(app)
    app.qutag_timestamp_written_path = None
    app.qutag_timestamp_size_poll_active = False
    app.qutag_timestamp_duration_limit_s = None
    app.qutag_timestamp_start_monotonic = None
    app.qutag_timestamp_stop_monotonic = None
    app.qutag_timestamp_finish_after_id = None
    app.qutag_timestamp_elapsed_s = 0.0
    app.qutag_timestamp_trigger_repeating = False
    app.qutag_timestamp_trigger_after_id = None
    app.qutag_timestamp_trigger_stop_after_id = None
    app.qutag_timestamp_trigger_count = 0
    app.time_tagger_file_writer = None

    content = _frame(app, app.qutag_timestamp_win)
    content.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 6))
    content.grid_columnconfigure(0, weight=1, uniform="timestamp_layout")
    content.grid_columnconfigure(1, weight=1, uniform="timestamp_layout")
    content.grid_rowconfigure(0, weight=1, uniform="timestamp_rows")
    content.grid_rowconfigure(1, weight=1, uniform="timestamp_rows")
    count_plot_frame = _frame(app, content, surface=True)
    count_plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
    placeholder_plot_frame = _frame(app, content, surface=True)
    placeholder_plot_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
    controls_panel = _frame(app, content, surface=True)
    controls_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
    controls_panel.grid_columnconfigure(0, weight=1)
    qcs_panel = _frame(app, content)
    qcs_panel.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
    qcs_panel.grid_columnconfigure(0, weight=1)
    qcs_panel.grid_rowconfigure(0, weight=1, uniform="qcs_sections")
    qcs_panel.grid_rowconfigure(1, weight=1, uniform="qcs_sections")
    init_panel = _frame(app, qcs_panel, surface=True)
    init_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
    tracking_panel = _frame(app, qcs_panel, surface=True)
    tracking_panel.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
    _label(app, init_panel, text="QCS initialization").grid(
        row=0, column=0, columnspan=4, sticky="w", padx=12, pady=(8, 4))

    def qcs_parameter(parent, label, variable_name, entry_name, saved_name, default,
                      row_index, column_index):
        # Stack each label above its entry in a shared four-column grid.
        group = _frame(app, parent)
        group.grid(row=row_index, column=column_index, sticky="ew", padx=5, pady=(6, 6))
        group.grid_columnconfigure(0, weight=1)
        _label(app, group, text=label).grid(row=0, column=0, sticky="w", pady=(0, 3))
        initial_value = getattr(app, saved_name, default)
        if variable_name in ("qcs_sweep_num_var", "qcs_init_bin_number_var", "qcs_fine_steps_var"):
            try:
                initial_value = str(_qcs_integer_value(initial_value, label))
            except ValueError:
                pass
        var = StringVar(value=initial_value)
        setattr(app, variable_name, var)
        entry = _entry(app, group, textvariable=var, width=9)
        entry.configure(bg="white", fg="black", insertbackground="black",
                        disabledbackground="white", disabledforeground="#666666")
        entry.grid(row=1, column=0, sticky="ew")
        setattr(app, entry_name, entry)

    for column in range(4):
        init_panel.grid_columnconfigure(column, weight=1, uniform="qcs_init_parameters")

    qcs_parameter(init_panel, "init_Ta_s", "qcs_init_interval_var",
                  "qcs_init_interval_entry", "qcs_init_interval_text",
                  f"{DEFAULT_QCS_CONFIG['init_interval_s']:g}", 1, 0)
    qcs_parameter(init_panel, "Ta_s", "qcs_interval_var",
                  "qcs_interval_entry", "qcs_interval_text",
                  f"{DEFAULT_QCS_CONFIG['acquisition_interval_s']:g}", 1, 1)
    qcs_parameter(init_panel, "latency_ms", "qcs_latency_ms_var",
                  "qcs_latency_ms_entry", "qcs_latency_ms_text",
                  f"{DEFAULT_QCS_CONFIG['latency_ms']:g}", 1, 2)
    qcs_parameter(init_panel, "init_tau_range_ms", "qcs_offset_range_var",
                  "qcs_offset_range_entry", "qcs_offset_range_text",
                  f"{DEFAULT_QCS_CONFIG['init_time_offset_range_ms']:g}", 1, 3)

    qcs_parameter(init_panel, "init_bin_num", "qcs_init_bin_number_var",
                  "qcs_init_bin_number_entry", "qcs_init_bin_number_text",
                  str(DEFAULT_QCS_CONFIG['init_bin_number']), 2, 0)
    qcs_parameter(init_panel, "coarse_range", "qcs_sweep_range_var",
                  "qcs_sweep_range_entry", "qcs_sweep_range_text",
                  f"{DEFAULT_QCS_CONFIG['sweep_range_ps_per_s']:g}", 2, 1)
    qcs_parameter(init_panel, "coarse_steps", "qcs_sweep_num_var",
                  "qcs_sweep_num_entry", "qcs_sweep_num_text",
                  str(DEFAULT_QCS_CONFIG['coarse_steps']), 2, 2)
    qcs_parameter(init_panel, "fine_steps", "qcs_fine_steps_var",
                  "qcs_fine_steps_entry", "qcs_fine_steps_text",
                  str(DEFAULT_QCS_CONFIG['fine_steps']), 2, 3)

    app.qcs_initialization_var = StringVar(value="Waiting for initialization")
    init_label = _label(app, init_panel, textvariable=app.qcs_initialization_var)
    init_label.configure(wraplength=500, justify="left")
    init_label.grid(row=3, column=0, columnspan=4, sticky="w", padx=12, pady=4)

    _label(app, tracking_panel, text="QCS live tracking").grid(
        row=0, column=0, sticky="w", padx=12, pady=(8, 4))
    tracking_panel.grid_columnconfigure(0, weight=1)
    live_parameters = _frame(app, tracking_panel)
    live_parameters.grid(row=1, column=0, sticky="w", padx=7)
    qcs_parameter(live_parameters, "correction_gain", "qcs_correction_gain_var",
                  "qcs_correction_gain_entry", "qcs_correction_gain_text",
                  f"{DEFAULT_QCS_CONFIG['live_correction_gain']:g}", 0, 0)
    app.qcs_status_var = StringVar(value="Idle — enable Internet and wait for the remote node")
    app.qcs_quality_var = StringVar(value="Live step: waiting for a completed interval")
    for row, variable in enumerate(
        (app.qcs_status_var, app.qcs_quality_var), start=2
    ):
        label = _label(app, tracking_panel, textvariable=variable)
        label.configure(wraplength=500, justify="left")
        label.grid(row=row, column=0, sticky="w", padx=12, pady=3)

    figure = Figure(figsize=(4.8, 2.7), dpi=100)
    axis = figure.add_subplot(111)
    _style_plot(app, figure, axis)
    axis.set_title(
        "Single-photon counts per Time Tagger X bin"
        if app.qutag_timestamp_time_tagger_available
        else "Single-photon counts (Time Tagger X unavailable)"
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Counts")
    axis.set_xlim(0, app.qutag_timestamp_plot_window_s)
    axis.set_ylim(0, 4096)

    app.qutag_timestamp_axis = axis
    app.qutag_timestamp_lines = []
    for tagger_name, channel_index in app.qutag_timestamp_count_series:
        (line,) = axis.plot([], [], label=f"{tagger_name} {channel_index}")
        app.qutag_timestamp_lines.append(line)
    legend = axis.legend(loc="upper left", frameon=False)
    _style_legend(app, legend)
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.12, top=0.92)

    app.qutag_timestamp_canvas = FigureCanvasTkAgg(figure, master=count_plot_frame)
    app.qutag_timestamp_canvas.get_tk_widget().configure(
        bg=_ui(app)["surface"],
        highlightthickness=0,
    )
    app.qutag_timestamp_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    placeholder_figure = Figure(figsize=(4.8, 2.7), dpi=100)
    placeholder_axis = placeholder_figure.add_subplot(111)
    _style_plot(app, placeholder_figure, placeholder_axis)
    placeholder_axis.set_title("Live QCS time offset")
    placeholder_axis.set_xlabel("Elapsed time (s)")
    placeholder_axis.set_ylabel("tau_ps (ps)")
    (app.qutag_timestamp_tau_line,) = placeholder_axis.plot([], [], color="tab:blue", lw=1.2)
    app.qutag_timestamp_tau_controller = None
    app.qutag_timestamp_tau_version = -1
    placeholder_figure.subplots_adjust(left=0.16, right=0.985, bottom=0.12, top=0.92)
    app.qutag_timestamp_placeholder_axis = placeholder_axis
    app.qutag_timestamp_placeholder_canvas = FigureCanvasTkAgg(
        placeholder_figure, master=placeholder_plot_frame
    )
    app.qutag_timestamp_placeholder_canvas.get_tk_widget().configure(
        bg=_ui(app)["surface"], highlightthickness=0
    )
    app.qutag_timestamp_placeholder_canvas.get_tk_widget().pack(
        fill="both", expand=True, padx=8, pady=8
    )

    file_frame = _frame(app, controls_panel)
    file_frame.pack(fill="x", padx=10, pady=(10, 6))

    _label(app, file_frame, text="File:").pack(side="left")
    # FileWriter stores losslessly compressed Time Tagger data in .ttbin files.
    app.qutag_timestamp_format = "ttbin"
    default_path = str(Path.cwd() / "time_tagger_timestamps.ttbin")
    app.qutag_timestamp_file_var = StringVar(value=default_path)
    _entry(app, file_frame, textvariable=app.qutag_timestamp_file_var, width=80).pack(
        side="left",
        fill="x",
        expand=True,
        padx=6,
    )
    _button(app, file_frame, text="Browse", command=lambda: _browse_timestamp_file(app)).pack(side="left")

    format_frame = _frame(app, controls_panel)
    format_frame.pack(fill="x", padx=10, pady=(0, 6))
    _label(
        app,
        format_frame,
        text="Format: Time Tagger X .ttbin (lossless compressed)",
        fg=_ui(app)["muted"],
    ).pack(side="left")

    qsfp_frame = _frame(app, controls_panel)
    qsfp_frame.pack(fill="x", padx=10, pady=(0, 6))
    modes = get_qsfp_modes(app)
    app.qutag_timestamp_qsfp_var = modes.enabled
    app.qutag_timestamp_qsfp_checkbox = modes.add_panel(qsfp_frame)
    qsfp_status_frame = _frame(app, controls_panel)
    qsfp_status_frame.pack(fill="x", padx=10, pady=(0, 6))
    _label(
        app, qsfp_status_frame, textvariable=modes.status,
        fg=_ui(app)["muted"],
    ).pack(anchor="w")

    internet_frame = _frame(app, controls_panel)
    internet_frame.pack(fill="x", padx=10, pady=(0, 6))
    app.qutag_timestamp_internet_var = BooleanVar(value=False)
    app.qutag_timestamp_internet_checkbox = _checkbutton(
        app, internet_frame, text="Internet", variable=app.qutag_timestamp_internet_var,
        command=lambda: _toggle_timestamp_internet(app))
    app.qutag_timestamp_internet_checkbox.pack(side="left")
    app.qutag_timestamp_internet_entries = []
    for label, name, default, width in (
        ("Bind:", "bind", "0.0.0.0", 15),
        ("Control port:", "control_port", "50051", 7),
        ("Data port:", "data_port", "50050", 7),
    ):
        _label(app, internet_frame, text=label).pack(side="left", padx=(10, 4))
        var = StringVar(value=default)
        setattr(app, f"qutag_timestamp_internet_{name}_var", var)
        entry = _entry(app, internet_frame, textvariable=var, width=width)
        entry.pack(side="left")
        app.qutag_timestamp_internet_entries.append(entry)
    internet_status_frame = _frame(app, controls_panel)
    internet_status_frame.pack(fill="x", padx=10, pady=(0, 6))
    app.qutag_timestamp_internet_status_var = StringVar(value="Internet disabled")
    app.qutag_timestamp_internet_size_var = StringVar(value="Internet file: 0 B")
    app.qutag_timestamp_internet_speed_var = StringVar(value="Internet speed: 0.00 MB/s")
    _label(app, internet_status_frame, textvariable=app.qutag_timestamp_internet_status_var,
           fg=_ui(app)["muted"]).pack(anchor="w")
    internet_metrics = _frame(app, internet_status_frame)
    internet_metrics.pack(fill="x")
    for var in (app.qutag_timestamp_internet_size_var, app.qutag_timestamp_internet_speed_var):
        _label(app, internet_metrics, textvariable=var, fg=_ui(app)["muted"]).pack(
            side="left", padx=(0, 16))
    app.qutag_timestamp_internet_stopping = False
    app.qutag_timestamp_internet_requested = False
    app.qutag_timestamp_internet_stop_after_id = None
    app.qutag_timestamp_internet_sample = (time.monotonic(), 0)
    _poll_timestamp_internet(app)

    status_frame = _frame(app, controls_panel)
    status_frame.pack(fill="x", padx=10, pady=(0, 6))
    app.qutag_timestamp_recording_var = StringVar(value="[REC] OFF")
    recording_status_frame = _frame(app, status_frame)
    recording_status_frame.pack(anchor="w", fill="x")
    app.qutag_timestamp_recording_label = _label(
        app,
        recording_status_frame,
        textvariable=app.qutag_timestamp_recording_var,
        fg=_ui(app)["muted"],
    )
    app.qutag_timestamp_recording_label.pack(side="left")
    app.qutag_timestamp_trigger_var = StringVar(value="[TRIG] OFF")
    app.qutag_timestamp_trigger_label = _label(
        app,
        recording_status_frame,
        textvariable=app.qutag_timestamp_trigger_var,
        fg=_ui(app)["muted"],
    )
    app.qutag_timestamp_trigger_label.pack(side="left", padx=(14, 0))
    info_status_frame = _frame(app, status_frame)
    info_status_frame.pack(anchor="w", fill="x")
    _label(app, info_status_frame, text="Info:", fg=_ui(app)["muted"]).pack(side="left")
    if not app.qutag_timestamp_time_tagger_available:
        initial_status = "Time Tagger X unavailable. Trigger control only; recording disabled."
    elif conversion_running:
        initial_status = "HydraHarp H5 conversion is still running."
    else:
        initial_status = "Ready."
    app.qutag_timestamp_status_var = StringVar(value=initial_status)
    _label(app, info_status_frame, textvariable=app.qutag_timestamp_status_var).pack(
        side="left", padx=(6, 0)
    )
    lock_status_frame = _frame(app, status_frame)
    lock_status_frame.pack(anchor="w", fill="x")
    _label(app, lock_status_frame, text="Lock:", fg=_ui(app)["muted"]).pack(side="left")
    app.qutag_timestamp_lock_status_var = StringVar(value="Stopped.")
    _label(app, lock_status_frame, textvariable=app.qutag_timestamp_lock_status_var).pack(
        side="left", padx=(6, 0)
    )
    app.qutag_timestamp_file_size_var = StringVar(value="Stored data: 0 B")
    app.qutag_timestamp_elapsed_var = StringVar(value="Recording time: 0.0 s")
    data_status_frame = _frame(app, status_frame)
    data_status_frame.pack(anchor="w", fill="x")
    _label(
        app,
        data_status_frame,
        textvariable=app.qutag_timestamp_file_size_var,
        fg=_ui(app)["muted"],
    ).pack(side="left")
    _label(
        app,
        data_status_frame,
        textvariable=app.qutag_timestamp_elapsed_var,
        fg=_ui(app)["muted"],
    ).pack(side="left", padx=(16, 0))

    limit_frame = _frame(app, controls_panel)
    limit_frame.pack(fill="x", padx=10, pady=(0, 8))
    app.qutag_timestamp_auto_stop_var = BooleanVar(value=False)
    app.qutag_timestamp_size_limit_var = StringVar(
        value=getattr(app, "qutag_timestamp_size_limit_text", "100 MB")
    )
    _checkbutton(
        app,
        limit_frame,
        text="Auto stop at size",
        variable=app.qutag_timestamp_auto_stop_var,
    ).pack(side="left")
    _entry(app, limit_frame, textvariable=app.qutag_timestamp_size_limit_var, width=12).pack(
        side="left",
        padx=(6, 4),
    )
    _label(app, limit_frame, text="B / KB / MB / GB", fg=_ui(app)["muted"]).pack(side="left")
    app.qutag_timestamp_time_auto_stop_var = BooleanVar(value=False)
    app.qutag_timestamp_duration_limit_var = StringVar(
        value=getattr(app, "qutag_timestamp_duration_limit_text", "60")
    )
    _checkbutton(
        app,
        limit_frame,
        text="Auto stop after time",
        variable=app.qutag_timestamp_time_auto_stop_var,
    ).pack(side="left", padx=(18, 0))
    _entry(app, limit_frame, textvariable=app.qutag_timestamp_duration_limit_var, width=12).pack(
        side="left",
        padx=(6, 4),
    )
    _label(app, limit_frame, text="s", fg=_ui(app)["muted"]).pack(side="left")

    button_frame = _frame(app, app.qutag_timestamp_win, surface=True)
    button_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

    app.qutag_timestamp_start_writing_button = _button(
        app,
        button_frame,
        text="Start Writing",
        command=lambda: _start_timestamp_writing(app),
    )
    app.qutag_timestamp_start_writing_button.pack(side="left", padx=(0, 8))
    if not app.qutag_timestamp_time_tagger_available:
        app.qutag_timestamp_start_writing_button.configure(state="disabled")
    _button(
        app,
        button_frame,
        text="Stop Writing",
        command=lambda: _stop_timestamp_writing(app),
    ).pack(side="left", padx=(0, 8))
    _button(
        app,
        button_frame,
        text="Trigger",
        command=lambda: _trigger_arduino(app),
        variant="primary",
    ).pack(side="left", padx=(0, 8))
    _button(
        app,
        button_frame,
        text="Lock Setting",
        command=lambda: _open_lock_settings(app),
    ).pack(side="left", padx=(0, 8))
    _button(
        app,
        button_frame,
        text="Lock",
        command=lambda: _start_lock(app),
        variant="primary",
    ).pack(side="left", padx=(0, 8))
    _button(
        app,
        button_frame,
        text="Unlock",
        command=lambda: _stop_lock(app),
    ).pack(side="left", padx=(0, 8))
    _button(
        app,
        button_frame,
        text="Quit",
        command=lambda: _close_timestamp_window(app),
        variant="danger",
    ).pack(side="right")

    app.qcs_start_button = _button(app, button_frame, text="QCS_start",
                                   command=lambda: _start_qcs(app), variant="primary")
    app.qcs_start_button.pack(side="left", padx=(0, 8))
    app.qcs_stop_button = _button(app, button_frame, text="QCS_stop",
                                  command=lambda: _stop_qcs(app))
    app.qcs_stop_button.pack(side="left", padx=(0, 8))
    app.qcs_stop_button.configure(state="disabled")
    _poll_qcs(app)

    app.qutag_timestamp_win.protocol("WM_DELETE_WINDOW", lambda: _close_timestamp_window(app))
    _poll_qutag_counts(app)


def _start_qcs(app):
    from QCS_algorithm import RealtimeG2Controller
    previous = getattr(app, "qcs_controller", None)
    if previous is not None and previous.busy:
        return
    try:
        interval = float(app.qcs_interval_var.get())
        sweep_num = _qcs_integer_value(app.qcs_sweep_num_var.get(), "sweep_num")
        fine_steps = _qcs_integer_value(app.qcs_fine_steps_var.get(), "fine_steps")
        init_bin_number = _qcs_integer_value(
            app.qcs_init_bin_number_var.get(), "init_bin_number")
        app.qcs_sweep_num_var.set(str(sweep_num))
        app.qcs_fine_steps_var.set(str(fine_steps))
        app.qcs_init_bin_number_var.set(str(init_bin_number))
        controller = RealtimeG2Controller(app, {'acquisition_interval_s': interval,
                                                   'live_clock_adjustment_enabled': True,
                                                   'live_correction_gain': float(app.qcs_correction_gain_var.get()),
                                                   'init_interval_s': float(app.qcs_init_interval_var.get()),
                                                   'latency_ms': float(app.qcs_latency_ms_var.get()),
                                                   'global_alignment_kwargs': {
                                                       'coarse_steps': sweep_num,
                                                       'fine_steps': fine_steps,
                                                       'sweep_range_ps_per_s': float(app.qcs_sweep_range_var.get()),
                                                       'init_time_offset_range_ms': float(app.qcs_offset_range_var.get()),
                                                       'init_bin_number': init_bin_number}})
        app.qcs_sweep_range_text = app.qcs_sweep_range_var.get()
        app.qcs_sweep_num_text = app.qcs_sweep_num_var.get()
        app.qcs_fine_steps_text = app.qcs_fine_steps_var.get()
        app.qcs_offset_range_text = app.qcs_offset_range_var.get()
        app.qcs_init_bin_number_text = app.qcs_init_bin_number_var.get()
        app.qcs_latency_ms_text = app.qcs_latency_ms_var.get()
        app.qcs_init_interval_text = app.qcs_init_interval_var.get()
        app.qcs_interval_text = app.qcs_interval_var.get()
        app.qcs_correction_gain_text = app.qcs_correction_gain_var.get()
        controller.start()
        app.qcs_controller = controller
        app.qcs_status_var.set(controller.status)
    except Exception as exc:
        messagebox.showerror("QCS acquisition", str(exc), parent=app.qutag_timestamp_win)


def _stop_qcs(app):
    controller = getattr(app, "qcs_controller", None)
    if controller is not None:
        controller.stop()
        app.qcs_status_var.set(controller.status)


def _poll_qcs(app):
    """Run one GUI poll and always schedule the next poll while the window lives."""
    window = getattr(app, "qutag_timestamp_win", None)
    if window is None or not window.winfo_exists():
        return
    try:
        _poll_qcs_once(app)
    except Exception as exc:
        # Keep the Tk event loop and subsequent QCS status updates alive if
        # one display update encounters malformed or temporarily empty data.
        print(f"QCS GUI update failed: {exc}", flush=True)
    window = getattr(app, "qutag_timestamp_win", None)
    if window is not None and window.winfo_exists():
        window.after(100, lambda: _poll_qcs(app))


def _poll_qcs_once(app):
    window = getattr(app, "qutag_timestamp_win", None)
    if window is None or not window.winfo_exists():
        return
    controller = getattr(app, "qcs_controller", None)
    busy = controller is not None and controller.busy
    if controller is not None:
        if app.qutag_timestamp_tau_controller is not controller:
            app.qutag_timestamp_tau_controller = controller
            app.qutag_timestamp_tau_version = -1
            app.qutag_timestamp_tau_line.set_data([], [])
            app.qutag_timestamp_placeholder_axis.set_xlim(0, 1)
            app.qutag_timestamp_placeholder_axis.set_ylim(-1, 1)
            app.qutag_timestamp_placeholder_axis.set_autoscaley_on(True)
            app.qutag_timestamp_placeholder_canvas.draw_idle()
        version, tau_time_s, tau_ps = controller.tau_history_snapshot(
            app.qutag_timestamp_tau_version)
        if tau_time_s is not None and tau_time_s.size:
            app.qutag_timestamp_tau_version = version
            app.qutag_timestamp_tau_line.set_data(tau_time_s, tau_ps)
            axis = app.qutag_timestamp_placeholder_axis
            axis.relim()
            axis.set_autoscaley_on(True)
            axis.autoscale_view(scalex=False, scaley=True)
            axis.set_xlim(0, max(float(tau_time_s[-1]), 1.0))
            app.qutag_timestamp_placeholder_canvas.draw_idle()
        update = controller.poll_update()
        if update is not None:
            app.qcs_status_var.set(update['status'])
            initialization_status = update['initialization_status']
            clock_command = update.get('clock_command')
            if clock_command and clock_command['status'] in (
                'applied', 'failed', 'timed_out', 'connection_lost'
            ):
                status_prefix = initialization_status.partition(' | command: ')[0]
                command_id = clock_command.get('command_id')
                command_label = (
                    'unchanged (below resolution)' if clock_command.get('no_change')
                    else clock_command['status']
                )
                if command_id is not None:
                    command_label += f" #{command_id}"
                initialization_status = f"{status_prefix} | command: {command_label}"
            app.qcs_initialization_var.set(initialization_status)
            live_step = update.get('live_step_result')
            if live_step is not None:
                rt_status = (
                    f" | RT fit failed: {live_step['rt_fit_error']}"
                    if live_step.get('rt_fit_error') else ''
                )
                app.qcs_quality_var.set(
                    f"Interval {live_step['sequence']} | "
                    f"u0 {live_step['u0_ps_per_s']:+.3f} ps/s | "
                    f"time offset {live_step['tau_ps']:+.3f} ps | "
                    f"clock {live_step['clock_status']}"
                    f"{rt_status}"
                )
            if update.get('integrity_failure'):
                app.qcs_quality_var.set('Acquisition integrity failure: ' + update['integrity_failure'])
        if controller.alignment_result is not None and not controller.alignment_plot_shown:
            controller.alignment_plot_shown = True
            try:
                from qcs_analyzer.global_freq_analysis import plot_global_freq_alignment_result
                plot_global_freq_alignment_result(controller.alignment_result)
            except Exception as exc:
                app.qcs_initialization_var.set(f"Alignment complete, but plotting failed: {exc}")
                print(f"QCS alignment plot failed: {exc}", flush=True)
    if controller is not None and not controller.running:
        remote = controller.remote.snapshot() if controller.remote is not None else None
        if remote is not None and remote['active']:
            app.qcs_status_var.set("Stopping QCS — draining the remote stream...")
        elif remote is not None and remote['result'] is not None and remote['result']['status'] != 'complete':
            reason = remote['result'].get('error', remote['status'])
            app.qcs_status_var.set("QCS remote error: " + reason)
            app.qcs_quality_var.set("Session integrity failed: " + reason)
        elif remote is not None and remote['result'] is not None and controller.quality is not None and not controller.integrity_failure:
            app.qcs_status_var.set("Remote acquisition session complete")
    app.qcs_sweep_range_entry.configure(state="disabled" if busy else "normal")
    app.qcs_sweep_num_entry.configure(state="disabled" if busy else "normal")
    app.qcs_fine_steps_entry.configure(state="disabled" if busy else "normal")
    app.qcs_offset_range_entry.configure(state="disabled" if busy else "normal")
    app.qcs_init_bin_number_entry.configure(state="disabled" if busy else "normal")
    app.qcs_latency_ms_entry.configure(state="disabled" if busy else "normal")
    app.qcs_init_interval_entry.configure(state="disabled" if busy else "normal")
    app.qcs_interval_entry.configure(state="disabled" if busy else "normal")
    app.qcs_correction_gain_entry.configure(state="disabled" if busy else "normal")
    app.qcs_start_button.configure(state="disabled" if busy else "normal")
    app.qcs_stop_button.configure(state="normal" if busy else "disabled")
    app.qutag_timestamp_start_writing_button.configure(
        state="disabled" if busy or not app.qutag_timestamp_time_tagger_available else "normal")
def _timestamp_internet_enabled(app):
    var = getattr(app, "qutag_timestamp_internet_var", None)
    return bool(var is not None and var.get())


def _timestamp_internet_active(app):
    coordinator = getattr(app, "timestamp_internet", None)
    return bool(coordinator is not None and coordinator.snapshot()["active"])


def _toggle_timestamp_internet(app):
    if (getattr(app, "qutag_timestamp_writing", False)
            or getattr(app, "qutag_timestamp_preparing_hydraharp", False)
            or _timestamp_internet_active(app)):
        app.qutag_timestamp_internet_var.set(getattr(app, "timestamp_internet", None) is not None)
        return
    previous = getattr(app, "timestamp_internet", None)
    if previous is not None:
        previous.close()
        app.timestamp_internet = None
    if _timestamp_internet_enabled(app):
        try:
            control = int(app.qutag_timestamp_internet_control_port_var.get())
            data = int(app.qutag_timestamp_internet_data_port_var.get())
            if not (1 <= control <= 65535 and 1 <= data <= 65535) or control == data:
                raise ValueError("Choose two distinct ports between 1 and 65535")
            app.timestamp_internet = InternetCoordinator(
                app.qutag_timestamp_internet_bind_var.get().strip(), control, data)
        except (ValueError, OSError) as exc:
            app.qutag_timestamp_internet_var.set(False)
            app.qutag_timestamp_internet_status_var.set(f"Internet settings error: {exc}")
            return
    else:
        app.qutag_timestamp_internet_status_var.set("Internet disabled")
    for entry in app.qutag_timestamp_internet_entries:
        entry.configure(state="disabled" if _timestamp_internet_enabled(app) else "normal")


def _poll_timestamp_internet(app):
    window = getattr(app, "qutag_timestamp_win", None)
    if window is None or not window.winfo_exists():
        return
    coordinator = getattr(app, "timestamp_internet", None)
    busy = (getattr(app, "qutag_timestamp_writing", False)
            or getattr(app, "qutag_timestamp_preparing_hydraharp", False)
            or _timestamp_internet_active(app))
    app.qutag_timestamp_internet_checkbox.configure(state="disabled" if busy else "normal")
    if coordinator is not None:
        state = coordinator.snapshot()
        app.qutag_timestamp_internet_status_var.set(state["status"])
        app.qutag_timestamp_internet_size_var.set(
            f"Internet file: {_format_byte_size(state['bytes'])}")
        now = time.monotonic()
        last_time, last_size = app.qutag_timestamp_internet_sample
        rate = max(0, state["bytes"] - last_size) / max(now - last_time, 1e-6) / 1e6
        if state["finished"] is not None and state["started"] is not None:
            rate = state["bytes"] / max(state["finished"] - state["started"], 1e-6) / 1e6
        app.qutag_timestamp_internet_speed_var.set(
            f"Internet speed: {rate:.2f} MB/s" + (" (average)" if state["finished"] else ""))
        app.qutag_timestamp_internet_sample = (now, state["bytes"])
        if (state["result"] is not None and not state["active"]
                and getattr(app, "qutag_timestamp_internet_requested", False)
                and not getattr(app, "qutag_timestamp_internet_stopping", False)
                and getattr(app, "qutag_timestamp_writing", False)):
            _stop_timestamp_writing(app)
    if window.winfo_exists():
        window.after(250, lambda: _poll_timestamp_internet(app))


def _trigger_arduino(app, force_on=False):
    """Toggle the Arduino's latched AWG burst gate, or ensure it is on."""
    gate_is_high = getattr(app, "qutag_timestamp_trigger_repeating", False)
    # The automatic post-start action must never turn a manually enabled
    # gate back off while the recording is running.
    if force_on and gate_is_high:
        return

    try:
        app.tx_trigger()
    except Exception as exc:
        message = f"Arduino trigger failed: {exc}"
        messagebox.showerror("TimeTag Trigger", message, parent=app.qutag_timestamp_win)
        return

    next_is_high = not gate_is_high
    app.qutag_timestamp_trigger_repeating = next_is_high
    _set_timestamp_trigger_indicator(app, next_is_high)


def _trigger_TTX_AUX1(app):
    """Toggle a 100 kHz square wave on Time Tagger X AUX OUT 1.

    The AUX output state is queried from the device rather than inferred from
    the UI.  The function returns ``True`` when AUX OUT 1 is enabled after the
    call and ``False`` when it is disabled.
    """
    aux_channel = 1  # Physical AUX OUT 1.
    frequency_hz = 100e3
    try:
        tagger = getattr(app, "time_tagger", None)
        if tagger is None:
            raise RuntimeError("Time Tagger X is not initialized.")

        is_enabled = bool(tagger.xtra_getAuxOut(aux_channel))
        if is_enabled:
            tagger.xtra_setAuxOut(aux_channel, False)
            expected_enabled = False
        else:
            tagger.xtra_setAuxOutSignal(aux_channel, frequency_hz)
            tagger.xtra_setAuxOut(aux_channel, True)
            expected_enabled = True
        tagger.sync()
        is_enabled = bool(tagger.xtra_getAuxOut(aux_channel))
        if is_enabled != expected_enabled:
            raise RuntimeError(
                "the device did not report the requested AUX OUT 1 state"
            )
    except Exception as exc:
        message = f"Could not toggle Time Tagger X AUX OUT 1: {exc}"
        status_var = getattr(app, "qutag_timestamp_status_var", None)
        if status_var is not None:
            status_var.set(message)
        window = getattr(app, "qutag_timestamp_win", None)
        if window is not None and window.winfo_exists():
            messagebox.showerror("Time Tagger X AUX OUT 1", message, parent=window)
        return None

    app.qutag_timestamp_ttx_aux1_enabled = is_enabled
    status_var = getattr(app, "qutag_timestamp_status_var", None)
    if status_var is not None:
        status_var.set(
            "Time Tagger X AUX OUT 1: "
            + ("ON (100 kHz square wave)." if is_enabled else "OFF.")
        )
    return is_enabled


def _stop_recording_triggers(app):
    """Turn off the latched Arduino AWG burst gate when recording stops."""
    gate_was_high = getattr(app, "qutag_timestamp_trigger_repeating", False)
    if gate_was_high:
        try:
            app.tx_trigger()
        except Exception as exc:
            message = f"Could not turn off Arduino trigger gate: {exc}"
            status_var = getattr(app, "qutag_timestamp_status_var", None)
            if status_var is not None:
                status_var.set(message)
            window = getattr(app, "qutag_timestamp_win", None)
            if window is not None and window.winfo_exists():
                messagebox.showerror("TimeTag Trigger", message, parent=window)
            return

    app.qutag_timestamp_trigger_repeating = False
    _set_timestamp_trigger_indicator(app, False)


def _set_timestamp_trigger_indicator(app, is_running):
    indicator_var = getattr(app, "qutag_timestamp_trigger_var", None)
    if indicator_var is not None:
        indicator_var.set("[TRIG] ON" if is_running else "[TRIG] OFF")

    indicator_label = getattr(app, "qutag_timestamp_trigger_label", None)
    if indicator_label is not None:
        try:
            indicator_label.configure(fg=_ui(app)["primary" if is_running else "muted"])
        except Exception:
            pass


def _browse_timestamp_file(app):
    file_var = getattr(app, "qutag_timestamp_file_var", None)
    current_path = file_var.get() if file_var is not None else ""
    initial_dir = str(Path(current_path).parent) if current_path else str(Path.cwd())
    initial_file = Path(current_path).name if current_path else "time_tagger_timestamps.ttbin"

    filename = filedialog.asksaveasfilename(
        parent=app.qutag_timestamp_win,
        title="Save Time Tagger X time tags",
        initialdir=initial_dir,
        initialfile=initial_file,
        defaultextension=".ttbin",
        filetypes=[("Time Tagger files", "*.ttbin"), ("All files", "*.*")],
    )
    if filename:
        filename = _normalize_timestamp_filename(filename)
        app.qutag_timestamp_file_var.set(filename)
        if not getattr(app, "qutag_timestamp_writing", False):
            app.qutag_timestamp_written_path = filename
        _update_timestamp_file_size(app)


def _normalize_timestamp_filename(filename):
    path = Path(filename)
    return str(path if path.suffix.lower() == ".ttbin" else path.with_suffix(".ttbin"))


def _start_timestamp_writing(app):
    qcs = getattr(app, "qcs_controller", None)
    if qcs is not None and qcs.busy:
        messagebox.showerror("Time Stamps", "Stop QCS before writing files.", parent=app.qutag_timestamp_win)
        return
    if getattr(app, "optical_capture_active", False):
        messagebox.showerror(
            "Time Stamps",
            "Stop the HH to QSFP capture before starting TimeTag recording.",
            parent=app.qutag_timestamp_win,
        )
        return
    if getattr(app, "hydraharp_raw_recording", False) and not getattr(app, "qutag_timestamp_writing", False):
        messagebox.showerror(
            "Time Stamps", "HydraHarp is finishing another raw capture. Wait for its live trace to resume.",
            parent=app.qutag_timestamp_win,
        )
        return
    if (
        getattr(app, "qutag_timestamp_writing", False)
        or getattr(app, "qutag_timestamp_preparing_hydraharp", False)
    ):
        return
    if getattr(app, "qutag_timestamp_hydraharp_conversion_running", False):
        messagebox.showerror(
            "Time Stamps", "Wait for the previous HydraHarp H5 conversion to finish.",
            parent=app.qutag_timestamp_win,
        )
        return

    if getattr(app, "time_tagger", None) is None:
        message = "Time Tagger X is not initialized. Recording is unavailable."
        status_var = getattr(app, "qutag_timestamp_status_var", None)
        if status_var is not None:
            status_var.set(message)
        messagebox.showerror(
            "Time Stamps",
            message,
            parent=getattr(app, "qutag_timestamp_win", None),
        )
        return

    raw_filename = app.qutag_timestamp_file_var.get().strip()
    if not raw_filename:
        messagebox.showerror("Time Stamps", "Please choose a timestamp file.", parent=app.qutag_timestamp_win)
        return
    filename = _normalize_timestamp_filename(raw_filename)
    app.qutag_timestamp_file_var.set(filename)
    app.qutag_timestamp_qsfp_result = None
    app.qutag_timestamp_stop_errors = []
    qsfp_requested = bool(app.qutag_timestamp_qsfp_var.get())
    internet_requested = _timestamp_internet_enabled(app)
    if internet_requested:
        coordinator = getattr(app, "timestamp_internet", None)
        if coordinator is None or not coordinator.snapshot()["ready"] or _timestamp_internet_active(app):
            messagebox.showerror("Time Stamps", "Internet Windows service is not ready.",
                                 parent=app.qutag_timestamp_win)
            return
        output = Path(filename).with_suffix(".internet.bin")
        if any(Path(str(output) + suffix).exists() for suffix in ("", ".part", ".json")):
            messagebox.showerror("Time Stamps", f"Internet output already exists: {output}",
                                 parent=app.qutag_timestamp_win)
            return
    hydraharp = getattr(app, "hydraharp", None)
    qsfp_output = Path(filename).with_suffix(".qsfp.bin") if qsfp_requested else None
    if qsfp_requested:
        try:
            app.qutag_timestamp_qsfp_options = get_qsfp_modes(app).capture_options()
            if hydraharp is None:
                raise RuntimeError("HydraHarp is not connected; QSFP needs HydraHarp T2 records.")
            if qsfp_output.exists():
                raise FileExistsError(f"QSFP output already exists: {qsfp_output}")
            for device in ("/dev/xdma0_h2c_0", "/dev/xdma0_c2h_0"):
                if not os.access(device, os.R_OK | os.W_OK):
                    raise PermissionError(f"Cannot access {device}")
        except (RuntimeError, OSError, ValueError, OverflowError) as exc:
            messagebox.showerror("Time Stamps", str(exc), parent=app.qutag_timestamp_win)
            return

    if _timestamp_auto_stop_enabled(app):
        try:
            app.qutag_timestamp_size_limit_bytes = _parse_size_limit_text(
                app.qutag_timestamp_size_limit_var.get()
            )
            app.qutag_timestamp_size_limit_text = app.qutag_timestamp_size_limit_var.get().strip()
        except ValueError as exc:
            messagebox.showerror("Time Stamps", str(exc), parent=app.qutag_timestamp_win)
            return
    else:
        app.qutag_timestamp_size_limit_bytes = None

    if _timestamp_time_auto_stop_enabled(app):
        try:
            app.qutag_timestamp_duration_limit_s = _parse_duration_limit_text(
                app.qutag_timestamp_duration_limit_var.get()
            )
            app.qutag_timestamp_duration_limit_text = app.qutag_timestamp_duration_limit_var.get().strip()
        except ValueError as exc:
            messagebox.showerror("Time Stamps", str(exc), parent=app.qutag_timestamp_win)
            return
    else:
        app.qutag_timestamp_duration_limit_s = None

    if internet_requested and app.qutag_timestamp_duration_limit_s is not None:
        duration = app.qutag_timestamp_duration_limit_s
        if not 0.001 <= duration <= 3600:
            messagebox.showerror("Time Stamps", "Internet duration must be 0.001..3600 seconds.",
                                 parent=app.qutag_timestamp_win)
            return
    app.qutag_timestamp_internet_requested = internet_requested
    app.qutag_timestamp_internet_output = Path(filename).with_suffix(".internet.bin")
    app.qutag_timestamp_internet_stopping = False

    app.qutag_timestamp_format = "ttbin"

    try:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        messagebox.showerror(
            "Time Stamps",
            f"Could not prepare Time Tagger X timestamp file:\n{exc}",
            parent=app.qutag_timestamp_win,
        )
        return

    # When available, preserve the synchronized Time Tagger X + HydraHarp workflow.
    # Without it, proceed with a Time Tagger X-only recording session.
    ptu_path = None
    hydraharp_warning = None
    if hydraharp is not None:
        ptu_path = str(Path(filename).with_suffix(".ptu"))
        try:
            Path(ptu_path).parent.mkdir(parents=True, exist_ok=True)
            # A time-trace and a raw TTTR measurement cannot run concurrently on
            # the HydraHarp.  Stop the live trace first, then allow its worker to
            # finish before starting the raw T2 acquisition below.
            hydraharp.timeTrace.stopMeasure()
        except Exception as exc:
            if qsfp_requested:
                messagebox.showerror(
                    "Time Stamps", f"Cannot start QSFP recording: {exc}",
                    parent=app.qutag_timestamp_win,
                )
                return
            # A connected-but-unusable HydraHarp must not prevent a valid
            # Time Tagger X-only recording.
            hydraharp_warning = f"HydraHarp skipped during preparation: {exc}"
            hydraharp = None
            ptu_path = None

    app.qutag_timestamp_preparing_hydraharp = True
    app.qutag_timestamp_pending_filename = filename
    app.qutag_timestamp_hydraharp_ptu_path = ptu_path
    app.qutag_timestamp_hydraharp_warning = hydraharp_warning
    app.qutag_timestamp_qsfp_requested = qsfp_requested
    app.qutag_timestamp_qsfp_output = qsfp_output
    app.qutag_timestamp_qsfp_checkbox.configure(state="disabled")
    if hydraharp is not None:
        app.qutag_timestamp_status_var.set("Preparing HydraHarp T2 TTTR recording...")
        app.qutag_timestamp_win.after(1000, lambda: _start_timestamp_recording(app))
    else:
        app.qutag_timestamp_status_var.set("Starting Time Tagger X timestamp recording...")
        _start_timestamp_recording(app)


def _start_timestamp_recording(app):
    """Start Time Tagger X writing, with HydraHarp TTTR when it was prepared."""
    if not getattr(app, "qutag_timestamp_preparing_hydraharp", False):
        return

    window = getattr(app, "qutag_timestamp_win", None)
    if window is None or not window.winfo_exists():
        app.qutag_timestamp_preparing_hydraharp = False
        if getattr(app, "hydraharp", None) is not None:
            app.hydraharp_raw_recording = True
            app.root.after(300, lambda: _resume_hydraharp_trace(app))
        return

    filename = getattr(app, "qutag_timestamp_pending_filename", None)
    ptu_path = getattr(app, "qutag_timestamp_hydraharp_ptu_path", None)
    hydraharp = getattr(app, "hydraharp", None)
    hydraharp_warning = getattr(app, "qutag_timestamp_hydraharp_warning", None)
    use_hydraharp = hydraharp is not None and ptu_path is not None
    qsfp_requested = bool(getattr(app, "qutag_timestamp_qsfp_requested", False))
    if not filename:
        app.qutag_timestamp_preparing_hydraharp = False
        app.qutag_timestamp_qsfp_checkbox.configure(state="normal")
        if hydraharp is not None:
            app.hydraharp_raw_recording = True
            app.root.after(300, lambda: _resume_hydraharp_trace(app))
        messagebox.showerror(
            "Time Stamps",
            "Time Tagger X recording preparation was incomplete.",
            parent=window,
        )
        return

    if use_hydraharp:
        try:
            hydraharp.setPTUFilePath(ptu_path)
            if qsfp_requested:
                # Same practical maximum used by the live trace; Stop Writing
                # ends the measurement at the requested time.
                started = hydraharp.raw.startBlock(360000000, 1048576, savePTU=True)
            else:
                started = hydraharp.raw.measure(0, 0, waitFinished=False, savePTU=True)
            if not started:
                raise RuntimeError("HydraHarp raw measurement could not start.")
            app.hydraharp_raw_recording = True
            app.qutag_timestamp_hydraharp_recording = True
        except Exception as exc:
            if qsfp_requested:
                app.qutag_timestamp_preparing_hydraharp = False
                app.qutag_timestamp_qsfp_checkbox.configure(state="normal")
                _stop_raw_and_resume_trace(app, hydraharp)
                messagebox.showerror("Time Stamps", f"Could not start QSFP recording: {exc}", parent=window)
                return
            # Preserve the usable Time Tagger X recording path if the optional
            # HydraHarp acquisition cannot start.
            hydraharp_warning = f"HydraHarp skipped at start: {exc}"
            app.qutag_timestamp_hydraharp_warning = hydraharp_warning
            _stop_raw_and_resume_trace(app, hydraharp)
            app.qutag_timestamp_hydraharp_recording = False
            app.qutag_timestamp_hydraharp_ptu_path = None
            use_hydraharp = False
            ptu_path = None

    try:
        channels = list(getattr(app, "time_tagger_setting_channels", [1, 2, 3, 4]))
        app.time_tagger_file_writer = app.create_time_tagger_file_writer(filename, channels)
    except Exception as exc:
        if use_hydraharp:
            _stop_raw_and_resume_trace(app, hydraharp)
        app.qutag_timestamp_preparing_hydraharp = False
        app.qutag_timestamp_hydraharp_recording = False
        app.qutag_timestamp_qsfp_checkbox.configure(state="normal")
        messagebox.showerror(
            "Time Stamps",
            f"Could not start Time Tagger X timestamp writing:\n{exc}",
            parent=window,
        )
        return

    if getattr(app, "qutag_timestamp_internet_requested", False):
        try:
            app.timestamp_internet.record(
                app.qutag_timestamp_internet_output,
                duration=getattr(app, "qutag_timestamp_duration_limit_s", None) or 3600)
            app.qutag_timestamp_internet_sample = (time.monotonic(), 0)
        except Exception as exc:
            try:
                app.time_tagger_file_writer.stop()
                getattr(app, "time_tagger_marker", lambda message: None)("Timestamp acquisition stopped: Internet startup failed")
            except Exception as cleanup_exc:
                app.tx_log_print(f"[TimeTag] Writer cleanup failed: {cleanup_exc}")
            finally:
                app.time_tagger_file_writer = None
            app.qutag_timestamp_preparing_hydraharp = False
            app.qutag_timestamp_qsfp_checkbox.configure(state="normal")
            if use_hydraharp:
                _stop_raw_and_resume_trace(app, hydraharp)
            messagebox.showerror("Time Stamps", f"Cannot start Internet recording: {exc}", parent=window)
            return

    if qsfp_requested:
        app.qutag_timestamp_qsfp_stop.clear()
        app.qutag_timestamp_qsfp_result = None
        app.qutag_timestamp_qsfp_count = 0
        app.qutag_timestamp_qsfp_bytes = 0
        app.qutag_timestamp_qsfp_displayed_count = 0
        app.qutag_timestamp_qsfp_stopping = False
        app.qutag_timestamp_qsfp_active = True
        app.qutag_timestamp_qsfp_worker = threading.Thread(
            target=_run_qsfp_stream,
            args=(app, hydraharp, app.qutag_timestamp_qsfp_output),
            daemon=False,
        )
        try:
            app.qutag_timestamp_qsfp_worker.start()
        except Exception as exc:
            app.qutag_timestamp_qsfp_active = False
            app.qutag_timestamp_qsfp_worker = None
            if getattr(app, "qutag_timestamp_internet_requested", False):
                app.timestamp_internet.stop_recording()
            cleanup_errors = []
            try:
                app.time_tagger_file_writer.stop()
                getattr(app, "time_tagger_marker", lambda message: None)("Timestamp acquisition stopped: QSFP startup failed")
            except Exception as cleanup_exc:
                cleanup_errors.append(f"Time Tagger X: {cleanup_exc}")
            app.time_tagger_file_writer = None
            try:
                hydraharp.raw.stopMeasure()
            except Exception as cleanup_exc:
                cleanup_errors.append(f"HydraHarp: {cleanup_exc}")
            app.hydraharp_raw_recording = True
            app.qutag_timestamp_hydraharp_recording = False
            app.qutag_timestamp_preparing_hydraharp = False
            app.qutag_timestamp_qsfp_checkbox.configure(state="normal")
            if not cleanup_errors:
                app.root.after(300, lambda: _resume_hydraharp_trace(app))
            detail = f"Could not start QSFP worker: {exc}"
            if cleanup_errors:
                detail += "\nCleanup errors: " + "; ".join(cleanup_errors)
            messagebox.showerror("Time Stamps", detail, parent=window)
            return
        app.root.after(200, lambda: _poll_qsfp_stream(app))

    app.qutag_timestamp_preparing_hydraharp = False
    app.qutag_timestamp_writing = True
    app.qutag_timestamp_written_path = filename
    app.qutag_timestamp_start_monotonic = time.monotonic()
    app.qutag_timestamp_stop_monotonic = None
    app.qutag_timestamp_elapsed_s = 0.0
    _set_timestamp_recording_indicator(app, True)
    _update_timestamp_elapsed(app)
    # Let the recorder(s) settle before enabling the AWG gate. ``force_on``
    # prevents this delayed action from toggling a manually enabled gate off.
    def enable_recording_trigger():
        app.qutag_timestamp_trigger_after_id = None
        if not getattr(app, "qutag_timestamp_writing", False):
            return

        # A very short timed recording can reach its one-second trigger
        # lead-out before the normal delayed gate start. In that case, keep
        # the trigger off for the whole recording.
        duration_limit_s = getattr(app, "qutag_timestamp_duration_limit_s", None)
        start_monotonic = getattr(app, "qutag_timestamp_start_monotonic", None)
        if duration_limit_s is not None and start_monotonic is not None:
            elapsed_s = time.monotonic() - float(start_monotonic)
            if elapsed_s >= max(0.0, float(duration_limit_s) - 1.0):
                return

        _trigger_arduino(app, force_on=True)

    app.qutag_timestamp_trigger_after_id = app.qutag_timestamp_win.after(
        2000, enable_recording_trigger
    )

    # For a time-limited acquisition, turn off the gate one second before the
    # recorder auto-stop deadline while both taggers continue recording.
    duration_limit_s = getattr(app, "qutag_timestamp_duration_limit_s", None)
    if duration_limit_s is not None:
        def stop_recording_trigger_early():
            app.qutag_timestamp_trigger_stop_after_id = None
            if not getattr(app, "qutag_timestamp_writing", False):
                return
            _stop_recording_triggers(app)
            app.qutag_timestamp_status_var.set(
                "Trigger stopped 1 s before the timed recording end."
            )

        trigger_stop_delay_ms = max(
            0, int(round((float(duration_limit_s) - 1.0) * 1000))
        )
        app.qutag_timestamp_trigger_stop_after_id = app.qutag_timestamp_win.after(
            trigger_stop_delay_ms, stop_recording_trigger_early
        )
        # Use a dedicated finish timer rather than depending on the 500 ms
        # file-size poll to notice that the recording deadline has passed.
        app.qutag_timestamp_finish_after_id = app.qutag_timestamp_win.after(
            max(1, int(round(float(duration_limit_s) * 1000))),
            lambda: _finish_timed_timestamp_recording(app),
        )
    if use_hydraharp:
        app.qutag_timestamp_status_var.set(
            "Software-aligned recording started. "
            f"Time Tagger X: {filename}; HydraHarp PTU: {ptu_path}"
            + (f"; QSFP: {app.qutag_timestamp_qsfp_output}" if qsfp_requested else "")
        )
    else:
        warning_suffix = f" ({hydraharp_warning})" if hydraharp_warning else ""
        app.qutag_timestamp_status_var.set(
            f"Time Tagger X recording started: {filename}{warning_suffix}"
        )
    _poll_timestamp_file_size(app)


def _run_qsfp_stream(app, hydraharp, output_path):
    """Keep the HH capture running while recording QSFP verification errors."""
    def progress(count, byte_count):
        app.qutag_timestamp_qsfp_count = count
        app.qutag_timestamp_qsfp_bytes = byte_count

    try:
        app.qutag_timestamp_qsfp_result = capture_hh_qsfp(
            hydraharp, 0, output_path, app.qutag_timestamp_qsfp_stop,
            started=True, progress=progress,
            **app.qutag_timestamp_qsfp_options,
        )
    except Exception as exc:
        # Includes file-creation failures before acquisition can be drained.
        stop_error = None
        try:
            hydraharp.raw.stopMeasure()
        except Exception as stop_exc:
            stop_error = str(stop_exc)
        app.qutag_timestamp_qsfp_result = {
            "blocks": app.qutag_timestamp_qsfp_count,
            "bytes": app.qutag_timestamp_qsfp_bytes,
            "error": str(exc), "stop_error": stop_error, "path": output_path,
        }


def _poll_qsfp_stream(app):
    worker = getattr(app, "qutag_timestamp_qsfp_worker", None)
    if worker is None:
        return
    if worker.is_alive():
        count = getattr(app, "qutag_timestamp_qsfp_count", 0)
        if not getattr(app, "qutag_timestamp_qsfp_stopping", False) and count != getattr(app, "qutag_timestamp_qsfp_displayed_count", 0):
            app.qutag_timestamp_qsfp_displayed_count = count
            app.qutag_timestamp_status_var.set(
                f"QSFP: {count} blocks, {app.qutag_timestamp_qsfp_bytes:,} T2 bytes verified."
            )
        app.root.after(200, lambda: _poll_qsfp_stream(app))
    elif getattr(app, "qutag_timestamp_qsfp_active", False):
        _stop_timestamp_writing(app)


def _resume_hydraharp_trace(app):
    if getattr(app, "optical_closing", False):
        return
    try:
        app.restart_hydraharp_time_trace()
    except Exception as exc:
        message = f"HydraHarp live trace could not restart: {exc}"
        app.qutag_timestamp_status_var.set(message)
        app.tx_log_print("[TimeTag] " + message)
    else:
        app.tx_log_print("[TimeTag] HydraHarp live trace restarted.")
    finally:
        app.hydraharp_raw_recording = False


def _stop_raw_and_resume_trace(app, hydraharp):
    app.hydraharp_raw_recording = True
    try:
        hydraharp.raw.stopMeasure()
    except Exception as exc:
        message = f"HydraHarp raw measurement could not stop: {exc}"
        app.qutag_timestamp_status_var.set(message)
        app.tx_log_print("[TimeTag] " + message)
        return False
    app.root.after(300, lambda: _resume_hydraharp_trace(app))
    return True


def _finish_timed_timestamp_recording(app):
    app.qutag_timestamp_finish_after_id = None
    if (getattr(app, "qutag_timestamp_writing", False)
            and getattr(app, "qutag_timestamp_stop_monotonic", None) is None):
        _stop_timestamp_writing(app)


def _stop_timestamp_writing(app):
    window = getattr(app, "qutag_timestamp_win", None)
    for timer_attribute in (
        "qutag_timestamp_trigger_after_id",
        "qutag_timestamp_trigger_stop_after_id",
        "qutag_timestamp_internet_stop_after_id",
        "qutag_timestamp_finish_after_id",
    ):
        timer_id = getattr(app, timer_attribute, None)
        if timer_id is not None and window is not None and window.winfo_exists():
            try:
                window.after_cancel(timer_id)
            except Exception:
                pass
        setattr(app, timer_attribute, None)

    # Gate off first, so the AWG stops before either tagger is stopped.
    _stop_recording_triggers(app)
    time_tagger_writing = getattr(app, "qutag_timestamp_writing", False)
    hydraharp_recording = getattr(app, "qutag_timestamp_hydraharp_recording", False)
    preparing_hydraharp = getattr(app, "qutag_timestamp_preparing_hydraharp", False)
    if not (time_tagger_writing or hydraharp_recording or preparing_hydraharp):
        return

    # Draining Internet data can outlast acquisition. Freeze the displayed
    # recording time at the first stop request, while transfer checks continue.
    if getattr(app, "qutag_timestamp_stop_monotonic", None) is None:
        app.qutag_timestamp_stop_monotonic = time.monotonic()
    _update_timestamp_elapsed(app)
    _set_timestamp_recording_indicator(app, False)

    errors = getattr(app, "qutag_timestamp_stop_errors", [])
    file_writer = getattr(app, "time_tagger_file_writer", None)
    if time_tagger_writing and file_writer is not None:
        try:
            file_writer.stop()
            getattr(app, "time_tagger_marker", lambda message: None)("Timestamp acquisition stopped: FileWriter")
        except Exception as exc:
            getattr(app, "time_tagger_marker", lambda message: None)(f"Timestamp acquisition stop failed: {exc}")
            errors.append(f"Time Tagger X: {exc}")
    app.time_tagger_file_writer = None

    if _timestamp_internet_active(app):
        if not getattr(app, "qutag_timestamp_internet_stopping", False):
            app.timestamp_internet.stop_recording()
            app.qutag_timestamp_internet_stopping = True
        app.qutag_timestamp_stop_errors = errors
        app.qutag_timestamp_status_var.set("Waiting for Internet recording to stop and verify...")
        app.qutag_timestamp_internet_stop_after_id = app.root.after(
            100, lambda: _stop_timestamp_writing(app))
        return
    if getattr(app, "qutag_timestamp_internet_requested", False):
        coordinator = getattr(app, "timestamp_internet", None)
        result = coordinator.snapshot()["result"] if coordinator else None
        if result and result["status"] != "complete":
            errors.append("Internet: " + result.get("error", "incomplete transfer"))
        app.qutag_timestamp_internet_requested = False
        app.qutag_timestamp_internet_stopping = False

    qsfp_active = getattr(app, "qutag_timestamp_qsfp_active", False)
    if qsfp_active:
        worker = getattr(app, "qutag_timestamp_qsfp_worker", None)
        if worker is not None and worker.is_alive():
            app.qutag_timestamp_stop_errors = errors
            app.qutag_timestamp_qsfp_stop.set()
            app.qutag_timestamp_qsfp_stopping = True
            app.qutag_timestamp_writing = False
            app.qutag_timestamp_preparing_hydraharp = False
            app.qutag_timestamp_size_poll_active = False
            _set_timestamp_recording_indicator(app, False)
            burst = getattr(app, "qutag_timestamp_qsfp_options", {}).get("mode") == "burst"
            app.qutag_timestamp_status_var.set(
                "Finishing HH capture and transmitting the saved QSFP batch..." if burst else
                "Stopping QSFP and draining the final HydraHarp blocks...")
            app.root.after(100, lambda: _stop_timestamp_writing(app))
            return

    hydraharp_stop_succeeded = False
    if qsfp_active:
        result = getattr(app, "qutag_timestamp_qsfp_result", None)
        if result is None:
            errors.append("QSFP worker ended without a result")
        else:
            hydraharp_stop_succeeded = not bool(result["stop_error"])
            if result["stop_error"]:
                errors.append(f"HydraHarp: {result['stop_error']}")
            if result["error"]:
                errors.append(f"QSFP: {result['error']}")
            app.tx_log_print(
                f"[TimeTag QSFP] {result['blocks']} blocks, {result['bytes']:,} "
                f"returned T2 bytes: {result['path']}"
            )
            if result.get("source_path"):
                app.tx_log_print(
                    f"[TimeTag QSFP] Original HH data: {result['source_path']}; "
                    f"error log: {result['error_path']}"
                )
        app.qutag_timestamp_qsfp_active = False
        app.qutag_timestamp_qsfp_worker = None
        app.qutag_timestamp_qsfp_stopping = False
    elif hydraharp_recording:
        try:
            app.hydraharp.raw.stopMeasure()
            hydraharp_stop_succeeded = True
        except Exception as exc:
            errors.append(f"HydraHarp: {exc}")

    start_monotonic = getattr(app, "qutag_timestamp_start_monotonic", None)
    if start_monotonic is not None:
        app.qutag_timestamp_elapsed_s = max(
            0.0, app.qutag_timestamp_stop_monotonic - float(start_monotonic))

    app.qutag_timestamp_writing = False
    app.qutag_timestamp_preparing_hydraharp = False
    app.qutag_timestamp_hydraharp_recording = False
    resume_trace = (hydraharp_stop_succeeded or preparing_hydraharp) and getattr(app, "hydraharp", None) is not None
    app.hydraharp_raw_recording = resume_trace or (hydraharp_recording and not hydraharp_stop_succeeded)
    checkbox = getattr(app, "qutag_timestamp_qsfp_checkbox", None)
    if checkbox is not None and checkbox.winfo_exists():
        checkbox.configure(state="normal")
    app.qutag_timestamp_size_poll_active = False
    app.qutag_timestamp_start_monotonic = None
    app.qutag_timestamp_duration_limit_s = None
    app.qutag_timestamp_stop_errors = []
    _set_timestamp_recording_indicator(app, False)
    if errors:
        app.qutag_timestamp_status_var.set("Recording stop completed with errors: " + "; ".join(errors))
        app.tx_log_print("[TimeTag] " + "; ".join(errors))
        if window is not None and window.winfo_exists() and not getattr(app, "optical_closing", False):
            messagebox.showerror(
                "Time Stamps",
                "Some recording stops reported errors:\n" + "\n".join(errors),
                parent=window,
            )
    else:
        app.qutag_timestamp_status_var.set(
            "Time Tagger X and HydraHarp recording stopped. Converting the HydraHarp PTU to H5..."
            if hydraharp_stop_succeeded
            else "Time Tagger X recording stopped."
        )
    _update_timestamp_file_size(app)

    if resume_trace:
        app.root.after(300, lambda: _resume_hydraharp_trace(app))
    if hydraharp_stop_succeeded:
        _start_hydraharp_h5_conversion(app)
    if getattr(app, "qutag_timestamp_close_pending", False):
        app.qutag_timestamp_close_pending = False
        coordinator = getattr(app, "timestamp_internet", None)
        if coordinator is not None:
            coordinator.close()
            app.timestamp_internet = None
        if window is not None and window.winfo_exists():
            window.destroy()


def _start_hydraharp_h5_conversion(app):
    """Convert the just-recorded HH PTU in a separate process after stopping."""
    ptu_path = getattr(app, "qutag_timestamp_hydraharp_ptu_path", None)
    qutag_path = getattr(app, "qutag_timestamp_written_path", None)
    if not ptu_path or not qutag_path:
        return

    qutag_file = Path(qutag_path)
    h5_path = qutag_file.with_name(f"{qutag_file.stem}_HH.h5")
    app.qutag_timestamp_hydraharp_h5_path = str(h5_path)
    app.qutag_timestamp_hydraharp_conversion_running = True
    app.qutag_timestamp_status_var.set(
        f"Converting HydraHarp PTU to {h5_path.name} in the background..."
    )

    converter_path = Path(__file__).with_name("hydraharp_ptu_to_h5.py")
    command = [sys.executable, str(converter_path), str(ptu_path), str(h5_path)]
    worker = threading.Thread(
        target=_run_hydraharp_h5_conversion,
        args=(app, command, h5_path),
        # Keep conversion alive if the TimeTag window is closed immediately
        # after Stop Writing.
        daemon=False,
    )
    worker.start()


def _run_hydraharp_h5_conversion(app, command, h5_path):
    try:
        # Give the raw-acquisition worker a moment to finish flushing the PTU
        # after stopMeasure before the file-device reader opens it.
        time.sleep(0.5)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "Unknown converter error.").strip()
            raise RuntimeError(detail)
    except Exception as exc:
        message = f"HydraHarp PTU-to-H5 conversion failed: {exc}"
    else:
        message = f"Recording saved. HydraHarp H5: {h5_path}."
    qsfp_result = getattr(app, "qutag_timestamp_qsfp_result", None)
    if qsfp_result is not None:
        message += (
            f" QSFP: {qsfp_result['blocks']} blocks, "
            f"{qsfp_result['bytes']:,} T2 bytes verified in {qsfp_result['path']}."
        )
        if qsfp_result["error"] or qsfp_result["stop_error"]:
            message += " QSFP capture reported an error; see the TimeTag log."

    try:
        app.root.after(0, lambda: _finish_hydraharp_h5_conversion(app, message))
    except Exception:
        app.qutag_timestamp_hydraharp_conversion_running = False
        print(message)


def _finish_hydraharp_h5_conversion(app, message):
    app.qutag_timestamp_hydraharp_conversion_running = False
    status_var = getattr(app, "qutag_timestamp_status_var", None)
    if status_var is not None:
        status_var.set(message)
    _update_timestamp_elapsed(app)


def _format_byte_size(byte_count):
    byte_count = max(0, int(byte_count))
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(byte_count)
    unit = units[0]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            break
        size /= 1024.0
    if unit == "B":
        return f"{byte_count} B"
    return f"{size:.2f} {unit}"


def _parse_size_limit_text(text):
    raw_text = str(text).strip()
    if not raw_text:
        raise ValueError("Auto-stop file size must not be empty.")

    parts = raw_text.split()
    if len(parts) == 1:
        token = parts[0]
        split_index = 0
        while split_index < len(token) and (
            token[split_index].isdigit() or token[split_index] == "."
        ):
            split_index += 1
        value_text = token[:split_index]
        unit_text = token[split_index:] or "MB"
    elif len(parts) == 2:
        value_text, unit_text = parts
    else:
        raise ValueError("Use a size like 100 MB, 2 GB, 500 KB, or 100.")

    try:
        value = float(value_text)
    except ValueError as exc:
        raise ValueError("Auto-stop file size must start with a number.") from exc

    if value <= 0:
        raise ValueError("Auto-stop file size must be greater than 0.")

    normalized_unit = unit_text.strip().lower()
    multipliers = {
        "b": 1,
        "byte": 1,
        "bytes": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024 ** 2,
        "mb": 1024 ** 2,
        "mib": 1024 ** 2,
        "g": 1024 ** 3,
        "gb": 1024 ** 3,
        "gib": 1024 ** 3,
        "t": 1024 ** 4,
        "tb": 1024 ** 4,
        "tib": 1024 ** 4,
    }
    if normalized_unit not in multipliers:
        raise ValueError("Supported size units are B, KB, MB, GB, and TB.")

    return max(1, int(value * multipliers[normalized_unit]))


def _timestamp_auto_stop_enabled(app):
    enabled_var = getattr(app, "qutag_timestamp_auto_stop_var", None)
    return bool(enabled_var is not None and enabled_var.get())


def _timestamp_time_auto_stop_enabled(app):
    enabled_var = getattr(app, "qutag_timestamp_time_auto_stop_var", None)
    return bool(enabled_var is not None and enabled_var.get())


def _parse_duration_limit_text(text):
    raw_text = str(text).strip()
    if not raw_text:
        raise ValueError("Auto-stop time must not be empty.")

    try:
        duration_s = float(raw_text)
    except ValueError as exc:
        raise ValueError("Auto-stop time must be a number of seconds.") from exc

    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Auto-stop time must be a finite number greater than 0 seconds.")

    return duration_s


def _format_duration_s(duration_s):
    duration_s = max(0.0, float(duration_s))
    if duration_s < 10.0:
        return f"{duration_s:.2f} s"
    if duration_s < 100.0:
        return f"{duration_s:.1f} s"
    return f"{duration_s:.0f} s"


def _set_timestamp_recording_indicator(app, is_recording):
    indicator_var = getattr(app, "qutag_timestamp_recording_var", None)
    if indicator_var is not None:
        indicator_var.set("[REC] ON" if is_recording else "[REC] OFF")

    indicator_label = getattr(app, "qutag_timestamp_recording_label", None)
    if indicator_label is not None:
        try:
            indicator_label.configure(fg=_ui(app)["danger" if is_recording else "muted"])
        except Exception:
            pass


def _update_timestamp_elapsed(app):
    elapsed_var = getattr(app, "qutag_timestamp_elapsed_var", None)
    if elapsed_var is None:
        return 0.0

    if getattr(app, "qutag_timestamp_writing", False):
        start_monotonic = getattr(app, "qutag_timestamp_start_monotonic", None)
        if start_monotonic is None:
            elapsed_s = 0.0
        else:
            stopped = getattr(app, "qutag_timestamp_stop_monotonic", None)
            elapsed_s = max(0.0, (time.monotonic() if stopped is None else stopped)
                            - float(start_monotonic))
        app.qutag_timestamp_elapsed_s = elapsed_s
    else:
        elapsed_s = max(0.0, float(getattr(app, "qutag_timestamp_elapsed_s", 0.0)))

    elapsed_var.set(f"Recording time: {_format_duration_s(elapsed_s)}")
    return elapsed_s


def _update_timestamp_file_size(app):
    size_var = getattr(app, "qutag_timestamp_file_size_var", None)
    if size_var is None:
        return None

    filename = getattr(app, "qutag_timestamp_written_path", None)
    if not filename:
        file_var = getattr(app, "qutag_timestamp_file_var", None)
        filename = file_var.get().strip() if file_var is not None else ""

    if not filename:
        size_var.set("Stored data: 0 B")
        return None

    try:
        byte_count = _time_tagger_filewriter_size(Path(filename))
    except FileNotFoundError:
        size_var.set("Stored data: 0 B")
        return 0
    except OSError as exc:
        size_var.set(f"Stored data: unavailable ({exc})")
        return None
    else:
        size_var.set(f"Stored data: {_format_byte_size(byte_count)}")
        return byte_count


def _time_tagger_filewriter_size(filename):
    """Return the total size of a FileWriter header and numbered data blocks."""
    files = [filename]
    prefix = f"{filename.stem}."
    for candidate in filename.parent.glob(f"{filename.stem}.*{filename.suffix}"):
        suffix_part = candidate.name[len(prefix) : -len(filename.suffix)]
        if suffix_part.isdigit():
            files.append(candidate)

    found_file = False
    byte_count = 0
    for path in files:
        try:
            byte_count += path.stat().st_size
            found_file = True
        except FileNotFoundError:
            continue

    if not found_file:
        raise FileNotFoundError(filename)
    return byte_count


def _poll_timestamp_file_size(app):
    window = getattr(app, "qutag_timestamp_win", None)
    if window is None or not window.winfo_exists():
        app.qutag_timestamp_size_poll_active = False
        return

    if getattr(app, "qutag_timestamp_size_poll_active", False):
        return

    app.qutag_timestamp_size_poll_active = True

    def poll_once():
        current_window = getattr(app, "qutag_timestamp_win", None)
        if current_window is None or not current_window.winfo_exists():
            app.qutag_timestamp_size_poll_active = False
            return

        byte_count = _update_timestamp_file_size(app)
        elapsed_s = _update_timestamp_elapsed(app)
        if getattr(app, "qutag_timestamp_stop_monotonic", None) is not None:
            app.qutag_timestamp_size_poll_active = False
            return

        if (
            getattr(app, "qutag_timestamp_writing", False)
            and _timestamp_auto_stop_enabled(app)
            and byte_count is not None
        ):
            size_limit = getattr(app, "qutag_timestamp_size_limit_bytes", None)
            if size_limit is not None and byte_count >= size_limit:
                _stop_timestamp_writing(app)
                app.qutag_timestamp_status_var.set(
                    "Auto-stopped timestamp writing at "
                    f"{_format_byte_size(byte_count)} "
                    f"(limit {_format_byte_size(size_limit)})."
                )
                return

        if (
            getattr(app, "qutag_timestamp_writing", False)
            and _timestamp_time_auto_stop_enabled(app)
        ):
            duration_limit_s = getattr(app, "qutag_timestamp_duration_limit_s", None)
            start_monotonic = getattr(app, "qutag_timestamp_start_monotonic", None)
            if duration_limit_s is not None and start_monotonic is not None:
                if elapsed_s >= float(duration_limit_s):
                    _finish_timed_timestamp_recording(app)
                    return

        if getattr(app, "qutag_timestamp_writing", False):
            current_window.after(500, poll_once)
        else:
            app.qutag_timestamp_size_poll_active = False

    poll_once()


def _poll_qutag_counts(app):
    window = getattr(app, "qutag_timestamp_win", None)
    if window is None or not window.winfo_exists() or not getattr(app, "qutag_timestamp_streaming", False):
        return

    if getattr(app, "qutag_piezoscan_running", False):
        app.qutag_timestamp_status_var.set("Time Tagger X piezo scan running; live counts paused.")
        window.after(250, lambda: _poll_qutag_counts(app))
        return

    time_tagger_data = None
    cache_max_age_s = max(
        0.25,
        3.0 * float(getattr(app, "qutag_timestamp_poll_interval_s", 0.1)),
    )
    if getattr(app, "time_tagger", None) is not None:
        cached_data = getattr(app, "time_tagger_latest_counts", None)
        cached_at = float(getattr(app, "time_tagger_latest_counts_monotonic", 0.0))
        if cached_data is not None and time.monotonic() - cached_at <= cache_max_age_s:
            # Rx is the primary Time Tagger X reader. Reusing its fresh sample prevents
            # the TimeTag window from competing for the same counter update.
            time_tagger_data = cached_data
        else:
            try:
                data = app.read_time_tagger_counts()
            except Exception as exc:
                app.qutag_timestamp_status_var.set(f"Time Tagger X count stream error: {exc}")
            else:
                if data is not None:
                    time_tagger_data = data
                    app.time_tagger_latest_counts = dict(data)
                    app.time_tagger_latest_counts_monotonic = time.monotonic()

    hydraharp_values = _read_hydraharp_counts(app)
    if time_tagger_data is not None or hydraharp_values is not None:
        _append_count_sample(app, time_tagger_data, hydraharp_values)
        _update_count_plot(app)
        if time_tagger_data is not None and getattr(app, "qutag_lock_running", False):
            _apply_lock_from_counts(app, time_tagger_data)

    interval_ms = int(max(0.02, getattr(app, "qutag_timestamp_poll_interval_s", 0.1)) * 1000)
    window.after(interval_ms, lambda: _poll_qutag_counts(app))


def _read_hydraharp_counts(app):
    hydraharp = getattr(app, "hydraharp", None)
    channel_indices = getattr(app, "qutag_timestamp_hydraharp_channel_indices", [])
    if (
        hydraharp is None
        or not channel_indices
        or getattr(app, "hydraharp_raw_recording", False)
    ):
        return None

    try:
        counts_all, _times = hydraharp.timeTrace.getData()
        count_bin_s = max(
            0.0,
            float(getattr(app, "time_tagger_count_bin_width_ms", 50.0)) / 1000.0,
        )
        return [
            float(counts_all[channel_index, -1]) * count_bin_s
            for channel_index in channel_indices
        ]
    except Exception as exc:
        app.qutag_timestamp_status_var.set(f"HydraHarp count stream error: {exc}")
        return None


def _append_count_sample(app, qutag_data, hydraharp_values):
    now = time.time()
    unavailable = float("nan")
    if qutag_data is None:
        qutag_values = [unavailable] * len(app.qutag_timestamp_counter_indices)
    else:
        qutag_values = [
            float(
                qutag_data[
                    app.qutag_timestamp_counter_hardware_channels.get(
                        counter_index, counter_index
                    )
                ]
            )
            for counter_index in app.qutag_timestamp_counter_indices
        ]
    if hydraharp_values is None:
        hydraharp_values = [unavailable] * len(app.qutag_timestamp_hydraharp_channel_indices)
    values = qutag_values + list(hydraharp_values)
    app.qutag_timestamp_history.append((now, values))

    cutoff = now - app.qutag_timestamp_plot_window_s
    while app.qutag_timestamp_history and app.qutag_timestamp_history[0][0] < cutoff:
        app.qutag_timestamp_history.popleft()


def _update_count_plot(app):
    history = app.qutag_timestamp_history
    if not history:
        return

    start_time = history[0][0]
    xs = [timestamp - start_time for timestamp, _values in history]
    max_y = 0.0

    for index, line in enumerate(app.qutag_timestamp_lines):
        ys = [values[index] for _timestamp, values in history]
        line.set_data(xs, ys)
        finite_ys = [value for value in ys if value == value]
        if finite_ys:
            max_y = max(max_y, max(finite_ys))

    app.qutag_timestamp_axis.set_xlim(0, app.qutag_timestamp_plot_window_s)
    app.qutag_timestamp_axis.set_ylim(0, max(4096.0, max_y * 1.1))
    app.qutag_timestamp_canvas.draw_idle()


def _open_lock_settings(app):
    dialog = Toplevel(app.qutag_timestamp_win)
    dialog.title("Lock Setting")
    dialog.transient(app.qutag_timestamp_win)
    dialog.grab_set()
    dialog.resizable(False, False)
    _style_window(app, dialog)

    frame = _frame(app, dialog, surface=True)
    frame.pack(fill="both", expand=True, padx=12, pady=12, ipadx=6, ipady=6)

    target_var = StringVar(value=str(getattr(app, "qutag_lock_target_ratio", 0.9)))
    channel_var = StringVar(value=str(getattr(app, "qutag_lock_channel", 1)))
    rate_var = StringVar(value=str(getattr(app, "qutag_lock_rate", 0.3)))

    _label(app, frame, text="Target ratio (0-1):").grid(row=0, column=0, sticky="e", padx=6, pady=6)
    _entry(app, frame, textvariable=target_var, width=10).grid(row=0, column=1, padx=6, pady=6)
    _label(app, frame, text="Channel (1-4):").grid(row=1, column=0, sticky="e", padx=6, pady=6)
    _entry(app, frame, textvariable=channel_var, width=10).grid(row=1, column=1, padx=6, pady=6)
    _label(app, frame, text="Correction rate:").grid(row=2, column=0, sticky="e", padx=6, pady=6)
    _entry(app, frame, textvariable=rate_var, width=10).grid(row=2, column=1, padx=6, pady=6)

    def on_ok():
        try:
            target_ratio = float(target_var.get())
            channel = int(float(channel_var.get()))
            rate = float(rate_var.get())
        except ValueError:
            messagebox.showerror("Lock Setting", "Target, channel, and rate must be numeric.", parent=dialog)
            return

        if target_ratio <= 0 or target_ratio > 1:
            messagebox.showerror("Lock Setting", "Target ratio must be in (0, 1].", parent=dialog)
            return
        if channel not in (1, 2, 3, 4):
            messagebox.showerror("Lock Setting", "Channel must be 1, 2, 3, or 4.", parent=dialog)
            return
        if rate <= 0:
            messagebox.showerror("Lock Setting", "Correction rate must be > 0.", parent=dialog)
            return

        app.qutag_lock_target_ratio = target_ratio
        app.qutag_lock_channel = channel
        app.qutag_lock_rate = rate
        app.qutag_timestamp_lock_status_var.set(
            f"Lock setting: channel {channel}, target {target_ratio:.3f} of range, "
            f"rate {rate:.3f}."
        )
        dialog.destroy()

    _button(app, frame, text="OK", command=on_ok, variant="primary").grid(row=3, column=0, padx=6, pady=10)
    _button(app, frame, text="Cancel", command=dialog.destroy).grid(row=3, column=1, padx=6, pady=10)


def _start_lock(app):
    if getattr(app, "qutag_lock_running", False):
        return
    if getattr(app, "qutag_piezoscan_running", False):
        messagebox.showerror("Lock", "Wait for the QuTAG piezo scan to finish before locking.", parent=app.qutag_timestamp_win)
        return
    if getattr(app, "laser", None) is None:
        messagebox.showerror("Lock", "Laser is not connected.", parent=app.qutag_timestamp_win)
        return

    try:
        channel = int(getattr(app, "qutag_lock_channel", 1))
        calibration = _get_lock_calibration(app, channel)
        peak_voltage = _get_lock_peak_voltage(app, channel, calibration)
        initial_voltage = peak_voltage + 5.0
        app.laser.set_piezo_V(initial_voltage)
    except Exception as exc:
        messagebox.showerror("Lock", str(exc), parent=app.qutag_timestamp_win)
        return

    app.qutag_lock_initial_voltage = initial_voltage
    app.qutag_lock_start_voltage = initial_voltage
    app.qutag_lock_peak_voltage = peak_voltage
    app.qutag_lock_running = True
    app.qutag_timestamp_lock_status_var.set(
        f"Lock running from peak+5 start {initial_voltage:.4f} V; waiting for next plot sample."
    )


def _stop_lock(app):
    was_running = bool(getattr(app, "qutag_lock_running", False))
    app.qutag_lock_running = False
    status_var = getattr(app, "qutag_timestamp_lock_status_var", None)
    initial_voltage = getattr(app, "qutag_lock_initial_voltage", None)

    if was_running and initial_voltage is not None and getattr(app, "laser", None) is not None:
        try:
            app.laser.set_piezo_V(float(initial_voltage))
            if status_var is not None:
                status_var.set(f"Lock stopped; restored piezo to {float(initial_voltage):.4f} V.")
        except Exception as exc:
            if status_var is not None:
                status_var.set(f"Lock stopped; could not restore piezo voltage: {exc}")
        finally:
            app.qutag_lock_initial_voltage = None
            app.qutag_lock_start_voltage = None
            app.qutag_lock_peak_voltage = None
        return

    if status_var is not None:
        status_var.set("Lock stopped.")


def _apply_lock_from_counts(app, data):
    window = getattr(app, "qutag_timestamp_win", None)
    if window is None or not window.winfo_exists() or not getattr(app, "qutag_lock_running", False):
        return

    try:
        channel = int(getattr(app, "qutag_lock_channel", 1))
        calibration = _get_lock_calibration(app, channel)
        target_ratio = float(getattr(app, "qutag_lock_target_ratio", 0.9))
        target_count = (
            calibration["min_count"]
            + (calibration["max_count"] - calibration["min_count"]) * target_ratio
        )
        current_count = float(data[channel])

        target_voltage = _interpolate_voltage_for_count(calibration["branch"], target_count)
        current_equiv_voltage = _interpolate_voltage_for_count(calibration["branch"], current_count)
        full_correction_voltage = target_voltage - current_equiv_voltage
        rate = float(getattr(app, "qutag_lock_rate", 0.3))
        correction_voltage = full_correction_voltage * rate
        current_setpoint = _get_piezo_setpoint(app)
        lock_start_voltage = getattr(app, "qutag_lock_start_voltage", None)
        max_allowed_voltage = calibration["max_voltage"]
        if lock_start_voltage is not None:
            max_allowed_voltage = max(max_allowed_voltage, float(lock_start_voltage))
        next_voltage = _clamp(
            current_setpoint + correction_voltage,
            calibration["min_voltage"],
            max_allowed_voltage,
        )
        peak_voltage = getattr(app, "qutag_lock_peak_voltage", None)
        if peak_voltage is None:
            peak_voltage = _get_lock_peak_voltage(app, channel, calibration)
            app.qutag_lock_peak_voltage = peak_voltage

        if next_voltage <= float(peak_voltage) - 2.0:
            next_voltage = float(peak_voltage) + 5.0
            app.qutag_lock_start_voltage = next_voltage
            if hasattr(app, "tx_log_print"):
                app.tx_log_print("Lock failed. Reset done.")

        app.laser.set_piezo_V(next_voltage)
        app.qutag_timestamp_lock_status_var.set(
            f"Lock ch {channel}: count={current_count:.1f}, target={target_count:.1f}, "
            f"dV={correction_voltage:+.4f} (rate {rate:.3f}), set={next_voltage:.4f} V"
        )

    except Exception as exc:
        app.qutag_timestamp_lock_status_var.set(f"Lock error: {exc}")


def _get_lock_calibration(app, channel):
    voltages = list(getattr(app, "qutag_piezoscan_voltages", []))
    values = list(getattr(app, "qutag_piezoscan_values", []))
    counter_indices = list(getattr(app, "qutag_piezoscan_counter_indices", []))

    if not voltages or not values:
        from qutag_piezoscan_helper import load_latest_qutag_piezoscan_csv
        load_latest_qutag_piezoscan_csv(app, force=True)
        voltages = list(getattr(app, "qutag_piezoscan_voltages", []))
        values = list(getattr(app, "qutag_piezoscan_values", []))
        counter_indices = list(getattr(app, "qutag_piezoscan_counter_indices", []))
    if not voltages or not values:
        raise RuntimeError(
            "No saved QuTAG piezo calibration was found. Run Scan Piezo (SPD) first."
        )
    if channel not in counter_indices:
        raise RuntimeError(
            f"Channel {channel} is not in the last calibration. "
            f"Calibrated channels: {counter_indices}"
        )

    column = counter_indices.index(channel)
    pairs = [
        (float(voltage), float(row[column]))
        for voltage, row in zip(voltages, values)
        if len(row) > column
    ]
    if len(pairs) < 2:
        raise RuntimeError("Calibration needs at least two points.")

    max_index = max(range(len(pairs)), key=lambda index: pairs[index][1])
    branch = pairs[: max_index + 1]
    if len(branch) < 2:
        branch = pairs

    return {
        "branch": branch,
        "peak_voltage": pairs[max_index][0],
        "max_count": max(count for _voltage, count in pairs),
        "min_count": min(count for _voltage, count in pairs),
        "min_voltage": min(voltage for voltage, _count in pairs),
        "max_voltage": max(voltage for voltage, _count in pairs),
    }


def _get_lock_peak_voltage(app, channel, calibration):
    peak_voltages = getattr(app, "qutag_piezoscan_peak_voltages", {})
    try:
        return float(peak_voltages[int(channel)])
    except Exception:
        return float(calibration["peak_voltage"])


def _interpolate_voltage_for_count(branch, count):
    points = sorted((float(count_value), float(voltage)) for voltage, count_value in branch)
    target = float(count)

    if target <= points[0][0]:
        return points[0][1]
    if target >= points[-1][0]:
        return points[-1][1]

    for (count_a, voltage_a), (count_b, voltage_b) in zip(points, points[1:]):
        if count_a <= target <= count_b:
            if count_b == count_a:
                return voltage_b
            fraction = (target - count_a) / (count_b - count_a)
            return voltage_a + fraction * (voltage_b - voltage_a)

    return min(points, key=lambda point: abs(point[0] - target))[1]


def _get_piezo_setpoint(app):
    try:
        return float(app.laser.get_piezo_V_setpoint())
    except Exception:
        return float(app.laser.get_piezo_V())


def _clamp(value, low, high):
    return max(float(low), min(float(high), float(value)))


def _close_timestamp_window(app):
    controller = getattr(app, "qcs_controller", None)
    if controller is not None and controller.busy:
        controller.stop()
        app.qutag_timestamp_win.after(100, lambda: _close_timestamp_window(app))
        return
    _stop_lock(app)
    app.qutag_timestamp_streaming = False

    if (
        getattr(app, "qutag_timestamp_writing", False)
        or getattr(app, "qutag_timestamp_hydraharp_recording", False)
        or getattr(app, "qutag_timestamp_preparing_hydraharp", False)
    ):
        _stop_timestamp_writing(app)
        if getattr(app, "qutag_timestamp_qsfp_active", False) or _timestamp_internet_active(app):
            app.qutag_timestamp_close_pending = True
            return
    else:
        # A manually enabled gate must not be left running merely because the
        # TimeTag window is closed without a recording session.
        _stop_recording_triggers(app)

    coordinator = getattr(app, "timestamp_internet", None)
    if coordinator is not None:
        coordinator.close()
        app.timestamp_internet = None
    window = getattr(app, "qutag_timestamp_win", None)
    if window is not None and window.winfo_exists():
        window.destroy()
