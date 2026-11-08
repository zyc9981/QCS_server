import csv
import os
import queue
import threading
import time

import matplotlib.pyplot as plt
from tkinter import (
    Toplevel,
    Label,
    Entry,
    Frame,
    Checkbutton,
    Button,
    StringVar,
    BooleanVar,
    messagebox,
)

# Temporary scan selection; other HydraHarp acquisition paths are unchanged.
ENABLE_HYDRAHARP_PIEZOSCAN = False


def _ensure_qutag_piezoscan_state(app):
    if getattr(app, "_qutag_piezoscan_state_init", False):
        return

    app.qutag_piezoscan_queue = queue.Queue()
    app.qutag_piezoscan_thread = None
    app.qutag_piezoscan_running = False
    app.qutag_piezoscan_live_plot = False
    app.qutag_piezoscan_prev_rx_streaming = False

    app.qutag_piezoscan_voltages = []
    app.qutag_piezoscan_values = []
    app.qutag_piezoscan_peak_voltages = {}
    app.qutag_piezoscan_peak_counts = {}
    app.qutag_piezoscan_auto_correct_rtrbot = False
    app.qutag_piezoscan_rtrbot_shift_per_0p1V = 3.62
    app.qutag_piezoscan_counter_indices = list(
        getattr(app, "qutag_piezoscan_counter_indices", [1, 2])
    )
    app.qutag_piezoscan_hydraharp_channel_indices = list(
        getattr(app, "qutag_piezoscan_hydraharp_channel_indices", [1])
    )
    app.qutag_piezoscan_series = []

    app.qutag_piezoscan_fig = None
    app.qutag_piezoscan_ax = None
    app.qutag_piezoscan_lines = []

    app.last_qutag_piezoscan_range_V = getattr(app, "last_qutag_piezoscan_range_V", 20.0)
    app.last_qutag_piezoscan_step_V = getattr(app, "last_qutag_piezoscan_step_V", 0.2)
    app.last_qutag_piezoscan_settle_ms = getattr(app, "last_qutag_piezoscan_settle_ms", 10)
    app.last_qutag_piezoscan_live_plot = getattr(app, "last_qutag_piezoscan_live_plot", False)
    app.last_qutag_piezoscan_auto_correct_rtrbot = getattr(
        app,
        "last_qutag_piezoscan_auto_correct_rtrbot",
        False,
    )
    app.last_qutag_piezoscan_rtrbot_shift_per_0p1V = getattr(
        app,
        "last_qutag_piezoscan_rtrbot_shift_per_0p1V",
        3.62,
    )

    app._qutag_piezoscan_state_init = True


def _ask_qutag_piezoscan_params(app):
    _ensure_qutag_piezoscan_state(app)

    dialog = Toplevel(app.tx_win)
    dialog.title("Time Tagger X Piezo Scan")
    dialog.transient(app.tx_win)
    dialog.grab_set()

    Label(dialog, text="Sweep range (V):").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    Label(dialog, text="Sweep step (V):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    Label(dialog, text="Piezo settle (ms):").grid(row=2, column=0, sticky="e", padx=5, pady=5)
    Label(dialog, text="Ch1 shift per +0.1 V RTRbot:").grid(row=4, column=0, sticky="e", padx=5, pady=5)

    range_var = StringVar(value=str(app.last_qutag_piezoscan_range_V))
    step_var = StringVar(value=str(app.last_qutag_piezoscan_step_V))
    settle_var = StringVar(value=str(app.last_qutag_piezoscan_settle_ms))
    live_var = BooleanVar(value=bool(app.last_qutag_piezoscan_live_plot))
    auto_correct_var = BooleanVar(value=bool(app.last_qutag_piezoscan_auto_correct_rtrbot))
    rtrbot_shift_var = StringVar(value=str(app.last_qutag_piezoscan_rtrbot_shift_per_0p1V))

    Entry(dialog, textvariable=range_var, width=10).grid(row=0, column=1, padx=5, pady=5)
    Entry(dialog, textvariable=step_var, width=10).grid(row=1, column=1, padx=5, pady=5)
    Entry(dialog, textvariable=settle_var, width=10).grid(row=2, column=1, padx=5, pady=5)
    Checkbutton(
        dialog,
        text="Auto-correct RTRbot after scan",
        variable=auto_correct_var,
    ).grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="w")
    Entry(dialog, textvariable=rtrbot_shift_var, width=10).grid(row=4, column=1, padx=5, pady=5)
    Checkbutton(
        dialog,
        text="Live plotting during sweep",
        variable=live_var,
    ).grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="w")

    result = {}

    def on_ok():
        try:
            range_V = float(range_var.get())
            step_V = float(step_var.get())
            settle_ms = float(settle_var.get())
            rtrbot_shift_per_0p1V = float(rtrbot_shift_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Range, step, settle time, and RTRbot correction must be numbers.",
                parent=dialog,
            )
            return

        if range_V <= 0:
            messagebox.showerror("Invalid input", "Range must be > 0.", parent=dialog)
            return
        if step_V <= 0:
            messagebox.showerror("Invalid input", "Step must be > 0.", parent=dialog)
            return
        if settle_ms < 0:
            messagebox.showerror("Invalid input", "Settle time must be >= 0 ms.", parent=dialog)
            return
        if rtrbot_shift_per_0p1V <= 0:
            messagebox.showerror(
                "Invalid input",
                "RTRbot correction must be > 0.",
                parent=dialog,
            )
            return

        result["values"] = (
            range_V,
            step_V,
            settle_ms,
            bool(live_var.get()),
            bool(auto_correct_var.get()),
            rtrbot_shift_per_0p1V,
        )
        dialog.destroy()

    Button(dialog, text="OK", command=on_ok).grid(row=6, column=0, pady=10)
    Button(dialog, text="Cancel", command=dialog.destroy).grid(row=6, column=1, pady=10)

    app.tx_win.wait_window(dialog)
    return result.get("values")


def _get_current_piezo_voltage(app):
    if getattr(app, "laser", None) is None:
        return 70.0

    try:
        return float(app.laser.get_piezo_V_setpoint())
    except Exception:
        return 70.0


def _build_sweep_values(center_voltage, range_V, step_V):
    start = float(center_voltage) + float(range_V) / 2.0
    stop = float(center_voltage) - float(range_V) / 2.0
    step = abs(float(step_V))

    values = []
    value = start
    while value >= stop - 1e-12:
        values.append(value)
        value -= step

    if values and abs(values[-1] - stop) > 1e-9:
        values.append(stop)

    return values


def _measure_tagger_counters(app, timeout_s=1.0):
    if getattr(app, "time_tagger", None) is None:
        raise RuntimeError("Time Tagger X is not initialized.")

    counter_indices = list(getattr(app, "qutag_piezoscan_counter_indices", [1, 2]))
    data = app.read_time_tagger_counts(timeout_s=timeout_s, clear=True)
    if data is not None:
        time_tagger_values = [float(data[index]) for index in counter_indices]
        if ENABLE_HYDRAHARP_PIEZOSCAN:
            return time_tagger_values + _measure_hydraharp_counts(app)
        return time_tagger_values

    raise TimeoutError("Timed out waiting for a Time Tagger X Counter bin.")


def _measure_hydraharp_counts(app):
    hydraharp = getattr(app, "hydraharp", None)
    channel_indices = list(getattr(app, "qutag_piezoscan_hydraharp_channel_indices", [1]))
    if hydraharp is None:
        raise RuntimeError("HydraHarp is not initialized.")

    counts_all, _times = hydraharp.timeTrace.getData()
    count_bin_s = max(
        0.0,
        float(getattr(app, "time_tagger_count_bin_width_ms", 50.0)) / 1000.0,
    )
    return [
        float(counts_all[channel_index, -1]) * count_bin_s
        for channel_index in channel_indices
    ]


def _qutag_piezoscan_worker(app, range_V, step_V, center_voltage, settle_ms):
    restore_voltage = float(center_voltage)
    sweep_values = _build_sweep_values(center_voltage, range_V, step_V)
    settle_s = max(0.0, float(settle_ms)) / 1000.0
    first_settle_s = max(0.1, settle_s)

    try:
        for index, voltage in enumerate(sweep_values):
            app.laser.set_piezo_V(voltage)
            time.sleep(first_settle_s if index == 0 else settle_s)

            values = _measure_tagger_counters(app, timeout_s=1.0)
            app.qutag_piezoscan_queue.put(("point", voltage, values))

        app.qutag_piezoscan_queue.put(("restore_voltage", restore_voltage))
        app.qutag_piezoscan_queue.put(("done", "completed"))

    except Exception as exc:
        app.qutag_piezoscan_queue.put(("restore_voltage", restore_voltage))
        app.qutag_piezoscan_queue.put(("error", str(exc)))


def _qutag_piezoscan_process_queue(app):
    try:
        while True:
            item = app.qutag_piezoscan_queue.get_nowait()
            kind = item[0]

            if kind == "log":
                _, message = item
                app.tx_log_print(message)

            elif kind == "point":
                _, voltage, values = item
                app.qutag_piezoscan_voltages.append(float(voltage))
                app.qutag_piezoscan_values.append([float(value) for value in values])

                if app.qutag_piezoscan_live_plot and app.qutag_piezoscan_fig is not None:
                    _update_live_plot(app)

            elif kind == "restore_voltage":
                _, voltage = item
                _set_piezo(app, voltage)

            elif kind == "done":
                _, reason = item
                _qutag_piezoscan_finish(app, success=True, info=f"Finished ({reason})")

            elif kind == "error":
                _, message = item
                _qutag_piezoscan_finish(app, success=False, info=message)

    except queue.Empty:
        pass

    if app.qutag_piezoscan_running:
        app.tx_win.after(50, lambda: _qutag_piezoscan_process_queue(app))


def _create_live_plot(app):
    series = list(getattr(app, "qutag_piezoscan_series", []))

    app.qutag_piezoscan_fig, app.qutag_piezoscan_ax = plt.subplots()
    app.qutag_piezoscan_lines = []

    for tagger_name, channel_index in series:
        (line,) = app.qutag_piezoscan_ax.plot([], [], marker="o", label=f"{tagger_name} {channel_index}")
        app.qutag_piezoscan_lines.append(line)

    app.qutag_piezoscan_ax.set_xlabel("Piezo voltage (V)")
    app.qutag_piezoscan_ax.set_ylabel("Counts")
    app.qutag_piezoscan_ax.set_title("Single-photon piezo scan")
    app.qutag_piezoscan_ax.grid(True)
    app.qutag_piezoscan_ax.legend()
    app.qutag_piezoscan_fig.tight_layout()
    app.qutag_piezoscan_fig.show()


def _update_live_plot(app):
    voltages = app.qutag_piezoscan_voltages
    values_by_point = app.qutag_piezoscan_values

    for index, line in enumerate(app.qutag_piezoscan_lines):
        ys = [values[index] for values in values_by_point]
        line.set_data(voltages, ys)

    app.qutag_piezoscan_ax.relim()
    app.qutag_piezoscan_ax.autoscale_view()
    app.qutag_piezoscan_fig.canvas.draw_idle()
    app.qutag_piezoscan_fig.canvas.flush_events()


def _plot_qutag_piezoscan_results(app):
    series = list(getattr(app, "qutag_piezoscan_series", []))
    fig, ax = plt.subplots()

    for index, (tagger_name, channel_index) in enumerate(series):
        ys = [values[index] for values in app.qutag_piezoscan_values]
        ax.plot(app.qutag_piezoscan_voltages, ys, marker="o", label=f"{tagger_name} {channel_index}")

    ax.set_xlabel("Piezo voltage (V)")
    ax.set_ylabel("Counts")
    ax.set_title("Single-photon piezo scan")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


def _save_qutag_piezoscan_csv(app):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "automatic_piezoscan_logs")
    os.makedirs(log_dir, exist_ok=True)

    series = list(getattr(app, "qutag_piezoscan_series", []))
    filename = time.strftime("qutag_piezo_scan_%Y%m%d_%H%M%S.csv")
    filepath = os.path.join(log_dir, filename)

    try:
        with open(filepath, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["piezo_voltage_V"]
                + [f"{tagger_name.lower()}_{channel_index}" for tagger_name, channel_index in series]
            )
            for voltage, values in zip(app.qutag_piezoscan_voltages, app.qutag_piezoscan_values):
                writer.writerow([voltage] + list(values))
        app.tx_log_print(f"[QUTAG SCAN] Saved {len(app.qutag_piezoscan_voltages)} points to {filepath}")
    except Exception as exc:
        app.tx_log_print(f"[QUTAG SCAN] Could not save CSV: {exc}")


def _record_qutag_piezoscan_peaks(app):
    voltages = list(getattr(app, "qutag_piezoscan_voltages", []))
    values_by_point = list(getattr(app, "qutag_piezoscan_values", []))
    counter_indices = list(getattr(app, "qutag_piezoscan_counter_indices", [1, 2, 3, 4]))

    peak_voltages = {}
    peak_counts = {}

    for column, counter_index in enumerate(counter_indices):
        candidates = [
            (float(voltage), float(values[column]))
            for voltage, values in zip(voltages, values_by_point)
            if len(values) > column
        ]
        if not candidates:
            continue

        peak_voltage, peak_count = max(candidates, key=lambda item: item[1])
        peak_voltages[int(counter_index)] = peak_voltage
        peak_counts[int(counter_index)] = peak_count

    app.qutag_piezoscan_peak_voltages = peak_voltages
    app.qutag_piezoscan_peak_counts = peak_counts

    for counter_index in counter_indices:
        if int(counter_index) in peak_voltages:
            app.tx_log_print(
                f"[QUTAG SCAN] Peak Qutag {int(counter_index)}: "
                f"{peak_counts[int(counter_index)]:.1f} counts at "
                f"{peak_voltages[int(counter_index)]:.4f} V"
            )


def _get_rtrbot_dac_name(app):
    dac_map = getattr(app, "tx_dac_map", {})
    for candidate in ("RTRbot", "RTR_bot"):
        if candidate in dac_map:
            return candidate

    normalized_names = {
        str(name).replace("_", "").lower(): name
        for name in dac_map
    }
    return normalized_names.get("rtrbot")


def _auto_correct_rtrbot_from_qutag_peaks(app):
    peak_voltages = dict(getattr(app, "qutag_piezoscan_peak_voltages", {}))
    missing_channels = [channel for channel in (1, 4) if channel not in peak_voltages]
    if missing_channels:
        app.tx_log_print(
            "[QUTAG SCAN] RTRbot correction skipped: missing peak for "
            + ", ".join(f"Qutag {channel}" for channel in missing_channels)
        )
        return

    try:
        shift_per_0p1V = float(getattr(app, "qutag_piezoscan_rtrbot_shift_per_0p1V", 3.62))
    except Exception:
        app.tx_log_print("[QUTAG SCAN] RTRbot correction skipped: invalid correction parameter.")
        return

    if shift_per_0p1V <= 0:
        app.tx_log_print("[QUTAG SCAN] RTRbot correction skipped: correction parameter must be > 0.")
        return

    rtrbot_name = _get_rtrbot_dac_name(app)
    if not rtrbot_name:
        app.tx_log_print("[QUTAG SCAN] RTRbot correction skipped: RTRbot is not in tx_dac_map.")
        return

    _chip_id, _pin, current_voltage = app.tx_dac_map[rtrbot_name]
    peak_difference = float(peak_voltages[1]) - float(peak_voltages[4])
    correction_voltage = peak_difference * 0.1 / shift_per_0p1V
    requested_voltage = float(current_voltage) + correction_voltage
    new_voltage = min(28.0, max(0.0, requested_voltage))

    if abs(new_voltage - requested_voltage) > 1e-12:
        app.tx_log_print(
            f"[QUTAG SCAN] RTRbot correction clipped to {new_voltage:.6f} V "
            f"from requested {requested_voltage:.6f} V."
        )

    try:
        app.tx_send_voltages({rtrbot_name: new_voltage})
    except Exception as exc:
        app.tx_log_print(f"[QUTAG SCAN] RTRbot correction failed: {exc}")
        return

    if hasattr(app, "tx_update_cached_voltage_values"):
        app.tx_update_cached_voltage_values({rtrbot_name: new_voltage}, persist=False)
    if rtrbot_name in getattr(app, "voltage_vars", {}):
        app.voltage_vars[rtrbot_name].set(f"{new_voltage:.6f}")

    app.tx_log_print(
        f"[QUTAG SCAN] RTRbot correction: peak1 - peak4 = {peak_difference:.6f} V, "
        f"delta RTRbot = {correction_voltage:.6f} V, "
        f"{rtrbot_name}: {float(current_voltage):.6f} -> {new_voltage:.6f} V"
    )


def _set_piezo(app, voltage):
    try:
        app.laser.set_piezo_V(float(voltage))
        app.tx_log_print(f"[QUTAG SCAN] Restored piezo voltage to {float(voltage):.4f} V")
    except Exception as exc:
        app.tx_log_print(f"[QUTAG SCAN] Could not restore piezo voltage: {exc}")


def _qutag_piezoscan_finish(app, success, info):
    app.qutag_piezoscan_running = False
    app.rx_streaming = bool(getattr(app, "qutag_piezoscan_prev_rx_streaming", False))

    if not app.qutag_piezoscan_voltages:
        app.tx_log_print("[QUTAG SCAN] No data collected.")
        if not success:
            app.tx_log_print(f"[QUTAG SCAN] Aborted: {info}")
        return

    if success:
        app.tx_log_print(f"[QUTAG SCAN] Completed: {info}")
    else:
        app.tx_log_print(f"[QUTAG SCAN] Aborted: {info}")

    _record_qutag_piezoscan_peaks(app)
    if bool(getattr(app, "qutag_piezoscan_auto_correct_rtrbot", False)):
        if success:
            _auto_correct_rtrbot_from_qutag_peaks(app)
        else:
            app.tx_log_print("[QUTAG SCAN] RTRbot correction skipped because the scan did not complete.")

    if not app.qutag_piezoscan_live_plot:
        _plot_qutag_piezoscan_results(app)

    _save_qutag_piezoscan_csv(app)


def tx_scan_piezo_qutag(app):
    _ensure_qutag_piezoscan_state(app)

    if app.qutag_piezoscan_running:
        messagebox.showwarning(
            "Time Tagger X piezo scan running",
            "A Time Tagger X piezo scan is already in progress.",
            parent=app.tx_win,
        )
        return

    if getattr(app, "laser", None) is None:
        messagebox.showerror(
            "Laser error",
            "Laser is not connected. Check LASER_HOST and network.",
            parent=app.tx_win,
        )
        return

    if getattr(app, "time_tagger", None) is None:
        messagebox.showerror(
            "Time Tagger X error",
            "Time Tagger X is not initialized.",
            parent=app.tx_win,
        )
        return

    params = _ask_qutag_piezoscan_params(app)
    if params is None:
        return

    range_V, step_V, settle_ms, live_plot, auto_correct_rtrbot, rtrbot_shift_per_0p1V = params
    center_voltage = _get_current_piezo_voltage(app)

    app.last_qutag_piezoscan_range_V = range_V
    app.last_qutag_piezoscan_step_V = step_V
    app.last_qutag_piezoscan_settle_ms = settle_ms
    app.last_qutag_piezoscan_live_plot = bool(live_plot)
    app.last_qutag_piezoscan_auto_correct_rtrbot = bool(auto_correct_rtrbot)
    app.last_qutag_piezoscan_rtrbot_shift_per_0p1V = float(rtrbot_shift_per_0p1V)
    app.qutag_piezoscan_live_plot = bool(live_plot)
    # RTRbot correction needs Time Tagger X Ch 4, which is temporarily hidden here.
    app.qutag_piezoscan_auto_correct_rtrbot = False
    app.qutag_piezoscan_rtrbot_shift_per_0p1V = float(rtrbot_shift_per_0p1V)
    if auto_correct_rtrbot:
        app.tx_log_print(
            "[TIME TAGGER X SCAN] RTRbot auto-correction is temporarily disabled because channel 4 is hidden."
        )

    app.qutag_piezoscan_voltages = []
    app.qutag_piezoscan_values = []
    app.qutag_piezoscan_peak_voltages = {}
    app.qutag_piezoscan_peak_counts = {}
    # Temporary display selection: Time Tagger X channels 3 and 4 remain usable elsewhere,
    # but are omitted from this scan for a clearer plot.
    app.qutag_piezoscan_counter_indices = [1, 2]
    app.qutag_piezoscan_hydraharp_channel_indices = (
        [1] if ENABLE_HYDRAHARP_PIEZOSCAN else []
    )
    app.qutag_piezoscan_series = (
        [("Time Tagger X", counter_index) for counter_index in app.qutag_piezoscan_counter_indices]
        + [
            ("HydraHarp", channel_index)
            for channel_index in app.qutag_piezoscan_hydraharp_channel_indices
        ]
    )
    app.qutag_piezoscan_fig = None
    app.qutag_piezoscan_ax = None
    app.qutag_piezoscan_lines = []
    try:
        while True:
            app.qutag_piezoscan_queue.get_nowait()
    except queue.Empty:
        pass

    app.qutag_piezoscan_prev_rx_streaming = bool(getattr(app, "rx_streaming", False))
    app.rx_streaming = False

    app.tx_log_print(
        f"[TIME TAGGER X SCAN] Starting piezo scan centered at {center_voltage:.4f} V, "
        f"range={range_V:.4f} V, step={step_V:.4f} V, "
        f"settle={settle_ms:.1f} ms, "
        f"series={app.qutag_piezoscan_series}, "
        f"live_plot={app.qutag_piezoscan_live_plot}, "
        f"auto_correct_RTRbot={app.qutag_piezoscan_auto_correct_rtrbot}, "
        f"RTRbot_shift_per_0p1V={app.qutag_piezoscan_rtrbot_shift_per_0p1V:.6f}"
    )

    if app.qutag_piezoscan_live_plot:
        _create_live_plot(app)

    app.qutag_piezoscan_running = True
    app.qutag_piezoscan_thread = threading.Thread(
        target=_qutag_piezoscan_worker,
        args=(app, range_V, step_V, center_voltage, settle_ms),
        daemon=True,
    )
    app.qutag_piezoscan_thread.start()

    _qutag_piezoscan_process_queue(app)
