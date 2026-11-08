import os
import csv
import time
import threading
import queue
from contextlib import nullcontext
import numpy as np

import matplotlib.pyplot as plt
from tkinter import (
    Toplevel, Label, Entry, Frame, Radiobutton, Checkbutton, Button,
    StringVar, IntVar, BooleanVar, messagebox, OptionMenu,
)


# ---------- internal helpers / state ----------

def _rx_serial_guard(app):
    return getattr(app, "rx_serial_lock", nullcontext())

def _flush_rx_buffer(app):
    """
    Best-effort flush of any pending lines in the RX serial buffer.
    Called from worker threads to get rid of stale PD samples.
    """
    try:
        with _rx_serial_guard(app):
            while app.rx.in_waiting:
                _ = app.rx.readline()
    except Exception:
        # Don't kill the worker if flushing fails
        pass

def _ensure_calib_state(app):
    """Attach calibration-related attributes to app the first time we run."""
    if getattr(app, "_calib_state_init", False):
        return

    app.calib_queue = queue.Queue()
    app.calib_thread = None
    app.calib_running = False
    app.calib_live_plot = False

    app.calib_wavelengths = []
    app.calib_pd_values = []
    app.calib_pd_values_by_channel = {}
    app.calib_selected_pds = []
    app.calib_scan_mode = None
    app.calib_prev_rx_polling = False

    app.calib_fig = None
    app.calib_ax = None
    app.calib_line = None
    app.calib_lines = {}

    # Defaults for dialog, if not already present
    app.last_calib_start_nm = getattr(app, "last_calib_start_nm", 1545.0)
    app.last_calib_end_nm   = getattr(app, "last_calib_end_nm",   1550.0)
    app.last_calib_step_nm  = getattr(app, "last_calib_step_nm",  0.01)
    app.last_calib_pd       = getattr(app, "last_calib_pd",       1)
    app.last_calib_pds = list(getattr(app, "last_calib_pds", [app.last_calib_pd]))
    app.last_calib_live_plot = getattr(app, "last_calib_live_plot", False)

    app._calib_state_init = True


def _ask_calibration_params(app):
    """
    Modal dialog to get (start_nm, end_nm, step_nm, pd_channels, live_plot).
    Returns None if cancelled.
    """
    _ensure_calib_state(app)

    dlg = Toplevel(app.tx_win)
    dlg.title("Calibrate Rings")
    dlg.transient(app.tx_win)
    dlg.grab_set()

    Label(dlg, text="Sweep start (nm):").grid(row=0, column=0, sticky='e', padx=5, pady=5)
    Label(dlg, text="Sweep end (nm):").grid(row=1, column=0, sticky='e', padx=5, pady=5)
    Label(dlg, text="Step (nm):").grid(row=2, column=0, sticky='e', padx=5, pady=5)
    Label(dlg, text="Photon detectors:").grid(row=3, column=0, sticky='ne', padx=5, pady=10)

    start_var = StringVar(value=str(app.last_calib_start_nm))
    end_var   = StringVar(value=str(app.last_calib_end_nm))
    step_var  = StringVar(value=str(app.last_calib_step_nm))
    available_pds = sorted(getattr(app, "rx_pd_to_channel", {}).keys())
    if not available_pds:
        available_pds = list(range(int(getattr(app, "rx_detector_count", 0))))
    previous_pds = set(getattr(app, "last_calib_pds", [app.last_calib_pd]))
    pd_vars = {
        pd: BooleanVar(value=pd in previous_pds)
        for pd in available_pds
    }

    Entry(dlg, textvariable=start_var, width=10).grid(row=0, column=1, padx=5, pady=5)
    Entry(dlg, textvariable=end_var,   width=10).grid(row=1, column=1, padx=5, pady=5)
    Entry(dlg, textvariable=step_var,  width=10).grid(row=2, column=1, padx=5, pady=5)

    pd_frame = Frame(dlg)
    pd_frame.grid(row=3, column=1, padx=5, pady=5, sticky='w')
    for column, pd in enumerate(available_pds):
        Checkbutton(pd_frame, text=f"PD{pd}", variable=pd_vars[pd]).grid(
            row=column // 5,
            column=column % 5,
            sticky="w",
            padx=4,
        )

    live_var = BooleanVar(value=bool(app.last_calib_live_plot))
    Checkbutton(
        dlg,
        text="Live plotting during sweep (may be slower / less stable)",
        variable=live_var
    ).grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky='w')

    result = {}

    def on_ok():
        try:
            s = float(start_var.get())
            e = float(end_var.get())
            st = float(step_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Start, end, and step must be numbers.", parent=dlg)
            return
        if st <= 0:
            messagebox.showerror("Invalid input", "Step must be > 0.", parent=dlg)
            return
        selected_pds = [pd for pd in available_pds if pd_vars[pd].get()]
        if not selected_pds:
            messagebox.showerror(
                "No detector selected",
                "Select at least one photon detector.",
                parent=dlg,
            )
            return
        result["vals"] = (s, e, st, selected_pds, live_var.get())
        dlg.destroy()

    def on_cancel():
        dlg.destroy()

    Button(dlg, text="OK",     command=on_ok).grid(row=5, column=0, pady=10)
    Button(dlg, text="Cancel", command=on_cancel).grid(row=5, column=1, pady=10)

    app.tx_win.wait_window(dlg)
    return result.get("vals", None)

def _ask_ring_calibration_params(app):
    """
    Modal dialog to get:
      (start_nm, end_nm, step_nm, pd_channel,
       live_plot, voltage_name, tuning_slope_nm_per_v)

    Same sweep options as a single scan, plus:
      - voltage to tune (from TX control panel voltages)
      - tuning slope in nm / V
    """
    _ensure_calib_state(app)

    dlg = Toplevel(app.tx_win)
    dlg.title("Auto Ring Calibration")
    dlg.transient(app.tx_win)
    dlg.grab_set()

    # ---------- sweep params ----------
    Label(dlg, text="Sweep start (nm):").grid(row=0, column=0, sticky='e', padx=5, pady=5)
    Label(dlg, text="Sweep end (nm):").grid(row=1, column=0, sticky='e', padx=5, pady=5)
    Label(dlg, text="Step (nm):").grid(row=2, column=0, sticky='e', padx=5, pady=5)
    Label(dlg, text="Photon detector:").grid(row=3, column=0, sticky='e', padx=5, pady=10)

    start_var = StringVar(value=str(getattr(app, "last_calib_start_nm", 1550.0)))
    end_var   = StringVar(value=str(getattr(app, "last_calib_end_nm", 1550.5)))
    step_var  = StringVar(value=str(getattr(app, "last_calib_step_nm", 0.005)))
    pd_var    = IntVar(value=int(getattr(app, "last_calib_pd", 1)))
    live_var  = BooleanVar(value=bool(getattr(app, "last_calib_live_plot", False)))

    Entry(dlg, textvariable=start_var, width=10).grid(row=0, column=1, padx=5, pady=5)
    Entry(dlg, textvariable=end_var,   width=10).grid(row=1, column=1, padx=5, pady=5)
    Entry(dlg, textvariable=step_var,  width=10).grid(row=2, column=1, padx=5, pady=5)

    pd_frame = Frame(dlg)
    pd_frame.grid(row=3, column=1, padx=5, pady=5, sticky='w')
    for ch in range(1, 5):
        Radiobutton(pd_frame, text=f"PD{ch}", variable=pd_var, value=ch).pack(side="left", padx=4)

    Checkbutton(
        dlg,
        text="Live plotting during sweep (may be slower / less stable)",
        variable=live_var
    ).grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky='w')

    # ---------- voltage to tune ----------
    Label(dlg, text="Voltage to tune:").grid(row=5, column=0, sticky='e', padx=5, pady=5)

    # Prefer voltages starting with "V_" from tx_write_order
    try:
        all_names = list(app.tx_write_order)
    except Exception:
        all_names = list(getattr(app, "voltage_vars", {}).keys())

    volt_names = [n for n in all_names if n.startswith("V_")] or all_names

    default_volt = getattr(app, "last_ring_voltage_name", "V_QF2top")
    if default_volt not in volt_names:
        default_volt = volt_names[0]

    volt_var = StringVar(value=default_volt)

    volt_menu = OptionMenu(dlg, volt_var, *volt_names)
    volt_menu.grid(row=5, column=1, padx=5, pady=5, sticky='w')

    # ---------- tuning slope ----------
    Label(dlg, text="Tuning slope (nm / V):").grid(row=6, column=0, sticky='e', padx=5, pady=5)

    default_slope = getattr(app, "last_ring_tuning_slope_nm_per_v", 0.05)  # nm per volt
    slope_var = StringVar(value=str(default_slope))

    Entry(dlg, textvariable=slope_var, width=10).grid(row=6, column=1, padx=5, pady=5, sticky='w')

    # ---------- buttons / validation ----------
    result = {}

    def on_ok():
        try:
            s = float(start_var.get())
            e = float(end_var.get())
            st = float(step_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Start, end, and step must be numbers.", parent=dlg)
            return
        if st <= 0:
            messagebox.showerror("Invalid input", "Step must be > 0.", parent=dlg)
            return

        try:
            slope = float(slope_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Tuning slope must be a number.", parent=dlg)
            return
        if slope <= 0:
            messagebox.showerror("Invalid input", "Tuning slope must be > 0.", parent=dlg)
            return

        result["vals"] = (
            s, e, st, pd_var.get(),
            bool(live_var.get()),
            volt_var.get(),
            slope,
        )
        dlg.destroy()

    def on_cancel():
        dlg.destroy()

    Button(dlg, text="OK",     command=on_ok).grid(row=7, column=0, pady=10)
    Button(dlg, text="Cancel", command=on_cancel).grid(row=7, column=1, pady=10)

    app.tx_win.wait_window(dlg)
    return result.get("vals", None)



def _measure_pd_values_worker(app, pd_channels, timeout_s: float = 0.3):
    """Read all requested PD values from one RX streaming sample."""
    from python_controller import readline_str

    channels = tuple(dict.fromkeys(int(channel) for channel in pd_channels))
    detector_count = int(getattr(app, "rx_detector_count", 0))
    if not channels:
        raise ValueError("At least one photon detector must be selected.")
    if any(channel < 0 or channel >= detector_count for channel in channels):
        raise ValueError(
            f"PD channels must be in 0..{detector_count - 1}, got {list(channels)}"
        )

    with _rx_serial_guard(app):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            line = readline_str(app.rx)
            if not line:
                continue
            parts = line.strip().split()
            if not parts:
                continue
            try:
                values = [float(token) for token in parts]
            except ValueError:
                continue
            if len(values) < detector_count:
                continue

            return {
                channel: (
                    app.rx_apply_pd_background_offset_value(values[channel], channel)
                    if hasattr(app, "rx_apply_pd_background_offset_value")
                    else values[channel]
                )
                for channel in channels
            }

    selected = ", ".join(f"PD{channel}" for channel in channels)
    raise TimeoutError(f"Timeout waiting for streaming values from {selected}")


def _measure_pd_worker(app, pd_channel: int, timeout_s: float = 0.3) -> float:
    """Read one PD value; retained for automatic ring calibration."""
    return _measure_pd_values_worker(app, [pd_channel], timeout_s)[int(pd_channel)]



def _calibration_worker(app, start_nm, end_nm, step_nm, pd_channels):
    """
    Runs in a background thread. Does NOT touch Tk.
    Sends progress + data back via app.calib_queue.
    """
    from python_controller import readline_str

    selected_label = ", ".join(f"PD{channel}" for channel in pd_channels)

    direction = 1 if end_nm >= start_nm else -1
    step = abs(step_nm) * direction

    # Start PD streaming
    try:
        with _rx_serial_guard(app):
            app.rx.write(b"STREAM_START\n")
            t0 = time.time()
            while time.time() - t0 < 0.5:
                line = readline_str(app.rx)
                if not line:
                    continue
                if "PD1" in line and "PD2" in line:
                    break
    except Exception as e:
        app.calib_queue.put(("error", f"Failed to start PD stream: {e}"))
        return

    # Big jump to start wavelength + settle
    pre_settle_s = 1.0  # adjust as needed
    try:
        app.calib_queue.put((
            "log",
            f"[CAL] Jumping laser to start λ = {start_nm:.4f} nm and settling ({pre_settle_s:.1f}s)"
        ))
        app.laser.set_wavelength_nm(start_nm)
        time.sleep(pre_settle_s)
        _flush_rx_buffer(app)
        try:
            dummy = _measure_pd_values_worker(app, pd_channels, timeout_s=0.3)
            app.calib_queue.put((
                "log",
                f"[CAL] Discarded first transient sample for {selected_label}: {dummy} "
                f"at λ={start_nm:.4f} nm"
            ))
        except Exception as e:
            app.calib_queue.put((
                "log",
                f"[CAL] Could not discard first PD sample: {e}"
            ))
    except Exception as e:
        app.calib_queue.put(("error", f"Failed to jump to start λ={start_nm:.4f} nm: {e}"))
        # Stop stream before returning
        try:
            with _rx_serial_guard(app):
                app.rx.write(b"STREAM_STOP\n")
        except Exception:
            pass
        return

    lam = start_nm
    try:
        # Optional: discard first PD sample after the big jump to avoid transients
        try:
            _ = _measure_pd_values_worker(app, pd_channels, timeout_s=0.3)
        except Exception:
            pass

        while (direction == 1 and lam <= end_nm + 1e-12) or \
              (direction == -1 and lam >= end_nm - 1e-12):

            app.calib_queue.put(("log", f"[CAL] Setting λ = {lam:.4f} nm"))

            app.laser.set_wavelength_nm(lam)

            try:
                values = _measure_pd_values_worker(app, pd_channels, timeout_s=0.3)
            except Exception as e:
                app.calib_queue.put((
                    "error",
                    f"Error reading {selected_label} at λ={lam:.4f} nm: {e}"
                ))
                return

            readings = ", ".join(
                f"PD{channel}={values[channel]}" for channel in pd_channels
            )
            app.calib_queue.put(("log", f"[CAL]   -> {readings}"))
            app.calib_queue.put(("scan_point", lam, values))

            lam += step

        app.calib_queue.put(("done", "completed"))

    except Exception as e:
        app.calib_queue.put(("error", f"Calibration worker exception: {e}"))

    finally:
        # Always stop PD streaming
        try:
            with _rx_serial_guard(app):
                app.rx.write(b"STREAM_STOP\n")
        except Exception:
            pass



def _calib_process_queue(app):
    """
    Runs in the Tk thread via .after().
    Processes messages from calibration / scan worker(s).
    """
    import queue as _queue_mod

    try:
        while True:
            item = app.calib_queue.get_nowait()
            kind = item[0]

            if kind == "log":
                _, msg = item
                app.tx_log_print(msg)

            elif kind == "internal_rx_off":
                app.calib_prev_rx_polling = getattr(app, "rx_polling", False)
                app.rx_polling = False

            elif kind == "internal_rx_on":
                if getattr(app, "calib_prev_rx_polling", False):
                    app.rx_polling = True
                    app.rx_poll()

            elif kind == "scan_start":
                _, it, V_curr = item
                # Clear current scan data
                app.calib_wavelengths = []
                app.calib_pd_values = []
                app.tx_log_print(f"[SCAN] Starting scan {it} at V_QF2top = {V_curr:.3f} V")

                if app.calib_live_plot:
                    # NEW: create a brand new figure for this scan
                    app.calib_fig, app.calib_ax = plt.subplots()
                    (app.calib_line,) = app.calib_ax.plot([], [], marker='o')

                    app.calib_ax.set_xlabel("Wavelength (nm)")
                    app.calib_ax.set_ylabel(f"PD{app.last_calib_pd} (arb. units)")
                    app.calib_ax.set_title(
                        f"Ring calibration – PD{app.last_calib_pd} "
                        f"(scan {it}, V_QF2top={V_curr:.3f} V)"
                    )
                    app.calib_ax.grid(True)
                    app.calib_fig.tight_layout()
                    app.calib_fig.show()

            elif kind == "point":
                _, lam, val = item
                app.calib_wavelengths.append(lam)
                app.calib_pd_values.append(val)

                if app.calib_live_plot and app.calib_fig is not None:
                    app.calib_line.set_data(app.calib_wavelengths, app.calib_pd_values)
                    app.calib_ax.relim()
                    app.calib_ax.autoscale_view()
                    app.calib_fig.canvas.draw_idle()
                    app.calib_fig.canvas.flush_events()

            elif kind == "scan_point":
                _, lam, values = item
                app.calib_wavelengths.append(lam)
                for channel in app.calib_selected_pds:
                    app.calib_pd_values_by_channel[channel].append(values[channel])

                if app.calib_live_plot and app.calib_fig is not None:
                    for channel, line in app.calib_lines.items():
                        line.set_data(
                            app.calib_wavelengths,
                            app.calib_pd_values_by_channel[channel],
                        )
                    app.calib_ax.relim()
                    app.calib_ax.autoscale_view()
                    app.calib_fig.canvas.draw_idle()
                    app.calib_fig.canvas.flush_events()

            elif kind == "ring_scan_result":
                _, it, lambda_filter, lambda_dip, delta_nm = item
                app.tx_log_print(
                    f"[RING] Scan {it}: filter @ {lambda_filter:.4f} nm, "
                    f"dip @ {lambda_dip:.4f} nm (Δ = {delta_nm*1e3:.1f} pm)"
                )

            elif kind == "update_voltage":
                _, V_name, V_new, dV, delta_nm, it = item
                app.tx_log_print(
                    f"[RING] Iter {it}: Δλ = {delta_nm*1e3:.1f} pm, "
                    f"ΔV = {dV:+.3f} V -> {V_new:.3f} V ({V_name})"
                )
                try:
                    app.voltage_vars[V_name].set(f"{V_new:.3f}")
                    vd = {k: float(app.voltage_vars[k].get()) for k in app.tx_write_order}
                    app.tx_send_voltages(vd)
                    if hasattr(app, "tx_update_cached_voltage_values"):
                        app.tx_update_cached_voltage_values(vd, persist=False)
                except Exception as e:
                    app.tx_log_print(f"[RING] Error updating {V_name}: {e}")
                    # If voltage update fails, stop calibration
                    app.calib_running = False

            elif kind == "done":
                _, reason = item
                _calib_finish(app, success=True, info=f"Finished ({reason})")

            elif kind == "error":
                _, msg = item
                _calib_finish(app, success=False, info=msg)

    except _queue_mod.Empty:
        pass

    if app.calib_running:
        app.tx_win.after(50, lambda: _calib_process_queue(app))


def _calib_finish(app, success: bool, info: str):
    """Called in Tk thread when calibration is done or fails."""
    app.calib_running = False

    if app.calib_prev_rx_polling:
        app.rx_polling = True
        app.rx_poll()

    if getattr(app, "calib_scan_mode", None) == "wavelength_scan":
        if not app.calib_wavelengths:
            app.tx_log_print("[SCAN] No data collected.")
            if not success:
                app.tx_log_print(f"[SCAN] Aborted: {info}")
        elif success:
            app.tx_log_print(f"[SCAN] Completed: {info}")
        else:
            app.tx_log_print(f"[SCAN] Aborted: {info}")

        if app.calib_wavelengths:
            if not app.calib_live_plot:
                _plot_wavelength_scan_results(
                    app,
                    app.calib_wavelengths,
                    app.calib_pd_values_by_channel,
                )
            _save_wavelength_scan_csv(
                app,
                app.calib_wavelengths,
                app.calib_pd_values_by_channel,
            )
        app.calib_scan_mode = None
        return

    if not app.calib_wavelengths:
        app.tx_log_print("[CAL] No data collected.")
        if not success:
            app.tx_log_print(f"[CAL] Aborted: {info}")
        return

    if success:
        app.tx_log_print(f"[CAL] Completed: {info}")
    else:
        app.tx_log_print(f"[CAL] Aborted: {info}")

    # Static plot only if live plotting was OFF
    if not app.calib_live_plot:
        _plot_calibration_results(
            app,
            app.calib_wavelengths,
            app.calib_pd_values,
            app.last_calib_pd,
        )

    _save_calibration_csv(
        app,
        app.calib_wavelengths,
        app.calib_pd_values,
        app.last_calib_pd,
    )


def _plot_wavelength_scan_results(app, wavelengths, values_by_channel):
    fig, ax = plt.subplots()
    for channel in app.calib_selected_pds:
        ax.plot(
            wavelengths,
            values_by_channel[channel],
            marker="o",
            label=f"PD{channel}",
        )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Detector value (arb. units)")
    ax.set_title("Wavelength scan")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


def _save_wavelength_scan_csv(app, wavelengths, values_by_channel):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "automatic_calib_logs")
    os.makedirs(log_dir, exist_ok=True)

    channels = list(app.calib_selected_pds)
    channel_suffix = "_".join(f"PD{channel}" for channel in channels)
    fname = time.strftime(f"wavelength_scan_{channel_suffix}_%Y%m%d_%H%M%S.csv")
    fpath = os.path.join(log_dir, fname)

    try:
        with open(fpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wavelength_nm", *[f"PD{channel}" for channel in channels]])
            for row_index, wavelength in enumerate(wavelengths):
                writer.writerow(
                    [wavelength, *[values_by_channel[channel][row_index] for channel in channels]]
                )
        app.tx_log_print(f"[SCAN] Saved {len(wavelengths)} points to {fpath}")
    except Exception as e:
        app.tx_log_print(f"[SCAN] Could not save CSV: {e}")


def _plot_calibration_results(app, wavelengths, pd_values, pd_ch):
    fig, ax = plt.subplots()
    ax.plot(wavelengths, pd_values, marker='o')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f"PD{pd_ch} (arb. units)")
    ax.set_title(f"Ring calibration – PD{pd_ch}")
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def _save_calibration_csv(app, wavelengths, pd_values, pd_ch):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "automatic_calib_logs")
    os.makedirs(log_dir, exist_ok=True)

    fname = time.strftime(f"ring_calibration_PD{pd_ch}_%Y%m%d_%H%M%S.csv")
    fpath = os.path.join(log_dir, fname)

    try:
        with open(fpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wavelength_nm", f"PD{pd_ch}"])
            for lam, val in zip(wavelengths, pd_values):
                writer.writerow([lam, val])
        app.tx_log_print(f"[CAL] Saved {len(wavelengths)} points to {fpath}")
    except Exception as e:
        app.tx_log_print(f"[CAL] Could not save CSV: {e}")

def _blocking_scan_once(app, start_nm, end_nm, step_nm, pd_ch, pre_settle_s=1.0):
    """
    Do a single wavelength scan synchronously.
    - Stops RX polling during the scan.
    - Uses app.laser.set_wavelength_nm and app.rx_measure_pd(pd_ch).
    - Returns (wavelengths, pd_values) as numpy arrays.
    """
    direction = 1 if end_nm >= start_nm else -1
    step = abs(step_nm) * direction

    app.tx_log_print(
        f"[SCAN] Jump to start λ = {start_nm:.4f} nm and settle for {pre_settle_s:.1f}s"
    )

    # Jump laser to start and settle
    app.laser.set_wavelength_nm(start_nm)
    time.sleep(pre_settle_s)

    # Turn off RX polling
    prev_polling = getattr(app, "rx_polling", False)
    app.rx_polling = False

    wavelengths = []
    pd_values = []

    try:
        lam = start_nm
        while (direction == 1 and lam <= end_nm + 1e-12) or \
              (direction == -1 and lam >= end_nm - 1e-12):

            app.tx_log_print(f"[SCAN] λ = {lam:.4f} nm")

            app.laser.set_wavelength_nm(lam)

            # PD read – reuse your synchronous helper
            val = app.rx_measure_pd(pd_ch, timeout_s=0.5)
            wavelengths.append(lam)
            pd_values.append(val)

            # Keep Tk responsive
            try:
                app.tx_win.update_idletasks()
                app.tx_win.update()
            except Exception:
                pass

            lam += step

    finally:
        app.rx_polling = prev_polling
        if prev_polling:
            app.rx_poll()

    return np.array(wavelengths, dtype=float), np.array(pd_values, dtype=float)

def _find_filter_and_ring_centers(lam, y):
    """
    Given lam (nm) and PD y, return:
        (lambda_filter, lambda_dip, delta_nm)
    where:
        lambda_filter = center of broad tall peak (filter)
        lambda_dip    = center of narrow dip (ring)
        delta_nm      = lambda_dip - lambda_filter

    Designed to work even when the dip splits the peak into two lobes
    (bimodal-looking filter around the notch).
    """
    lam = np.asarray(lam, dtype=float)
    y = np.asarray(y, dtype=float)
    n = lam.size
    if n < 40:
        return None, None, None

    # Sort by wavelength just in case
    idx_sort = np.argsort(lam)
    lam = lam[idx_sort]
    y = y[idx_sort]

    # --- 1) Strong smoothing to get the broad envelope (filter shape) ---
    # Window ~ 10–15% of scan length, but at least 31 points
    win = max(31, n // 8)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / float(win)
    y_env = np.convolve(y, kernel, mode="same")

    # Broad filter center is max of smoothed envelope
    idx_filter = int(np.argmax(y_env))
    lambda_filter = lam[idx_filter]

    # --- 2) Look for a "real" narrow dip near the filter center ---
    # Difference between envelope and actual signal
    diff = y_env - y

    # Focus search window: +/- ~10% of scan around the filter center
    span = max(20, n // 10)
    lo = max(0, idx_filter - span)
    hi = min(n, idx_filter + span)

    diff_seg = diff[lo:hi]
    lam_seg = lam[lo:hi]
    y_seg = y[lo:hi]

    # A dip is where y is significantly below the envelope.
    # Threshold: at least 10% of the full-scale envelope variation.
    env_span = float(y_env.max() - y_env.min()) if y_env.max() > y_env.min() else 0.0
    depth_thr = 0.10 * env_span

    candidate_idx_local = np.where(diff_seg > depth_thr)[0]

    if candidate_idx_local.size > 0:
        # Among candidates, pick the deepest point
        candidate_idx = candidate_idx_local + lo
        idx_dip = int(candidate_idx[np.argmin(y[candidate_idx])])
    else:
        # Fallback: local minimum in the neighborhood around filter center
        idx_local_min = int(np.argmin(y_seg))
        idx_dip = lo + idx_local_min

    lambda_dip = lam[idx_dip]
    delta_nm = lambda_dip - lambda_filter

    return lambda_filter, lambda_dip, delta_nm

def _ring_calibration_worker(app, start_nm, end_nm, step_nm, pd_ch,
                             V_init, V_name, tuning_slope_nm_per_v):
    """
    Runs in a background thread. Performs several scans and nudges the chosen
    voltage (V_name) until the broad filter peak and narrow dip align.

    All UI interactions go through app.calib_queue.
    """
    from python_controller import readline_str

    direction = 1 if end_nm >= start_nm else -1
    step = abs(step_nm) * direction

    # ----------- tuning / limits -----------
    DAMPING = 0.7               # 0 < DAMPING <= 1
    max_step_V = 0.5            # hard cap on per-iteration ΔV
    V_min, V_max = 0.0, 32.0    # adjust as needed
    max_iters = 6
    tol_nm = 0.01               # stop when |Δλ| < 0.01 nm
    pre_settle_s = 1.0          # after big wavelength/voltage changes

    if tuning_slope_nm_per_v is not None and tuning_slope_nm_per_v > 0:
        k_gain = DAMPING / tuning_slope_nm_per_v   # V per nm
    else:
        # Fallback: simple proportional gain if slope unknown
        k_gain = 0.5

    V_curr = float(V_init)

    # Turn off RX polling while worker uses serial
    app.calib_queue.put(("internal_rx_off", None))

    # Start PD streaming for entire calibration
    try:
        with _rx_serial_guard(app):
            app.rx.write(b"STREAM_START\n")
            t0 = time.time()
            while time.time() - t0 < 0.5:
                line = readline_str(app.rx)
                if not line:
                    continue
                if "PD1" in line and "PD2" in line:
                    break
    except Exception as e:
        app.calib_queue.put(("error", f"Failed to start PD stream: {e}"))
        app.calib_queue.put(("internal_rx_on", None))
        return

    try:
        for it in range(1, max_iters + 1):
            # Tell GUI we're starting a new scan
            app.calib_queue.put(("scan_start", it, V_curr))
            app.calib_queue.put((
                "log",
                f"[RING] Iter {it}/{max_iters}, {V_name} = {V_curr:.3f} V"
            ))

            # Jump to start wavelength and settle
            try:
                app.laser.set_wavelength_nm(start_nm)
                time.sleep(pre_settle_s)

                # Optional: flush buffer + discard first PD sample of this scan
                try:
                    _flush_rx_buffer(app)
                except NameError:
                    pass  # if you didn't define _flush_rx_buffer, ignore

                try:
                    dummy = _measure_pd_worker(app, pd_ch, timeout_s=0.5)
                    app.calib_queue.put((
                        "log",
                        f"[RING] Iter {it}: discarded first transient "
                        f"PD{pd_ch}={dummy} at λ={start_nm:.4f} nm"
                    ))
                except Exception as e:
                    app.calib_queue.put((
                        "log",
                        f"[RING] Iter {it}: could not discard first PD sample: {e}"
                    ))

            except Exception as e:
                app.calib_queue.put(("error", f"Failed to set start λ: {e}"))
                return

            lam_list = []
            y_list = []

            # Do one scan, streaming points to GUI
            lam = start_nm
            try:
                while (direction == 1 and lam <= end_nm + 1e-12) or \
                      (direction == -1 and lam >= end_nm - 1e-12):

                    app.laser.set_wavelength_nm(lam)
                    val = _measure_pd_worker(app, pd_ch, timeout_s=0.5)

                    lam_list.append(lam)
                    y_list.append(val)

                    # live plotting in GUI thread
                    app.calib_queue.put(("point", lam, val))

                    lam += step
            except Exception as e:
                app.calib_queue.put(("error", f"Scan failed in iter {it}: {e}"))
                return

            lam_arr = np.asarray(lam_list, dtype=float)
            y_arr = np.asarray(y_list, dtype=float)

            lambda_filter, lambda_dip, delta_nm = _find_filter_and_ring_centers(lam_arr, y_arr)

            if lambda_filter is None or lambda_dip is None or delta_nm is None:
                app.calib_queue.put(("error", "[RING] Could not reliably detect filter/dip; stopping."))
                return

            app.calib_queue.put((
                "ring_scan_result",
                it, lambda_filter, lambda_dip, delta_nm
            ))

            # Check alignment
            if abs(delta_nm) < tol_nm:
                app.calib_queue.put(("done", "aligned"))
                return

            # Compute new voltage from delta_nm and tuning slope
            dV = k_gain * delta_nm
            # Clip step size
            if dV > max_step_V:
                dV = max_step_V
            elif dV < -max_step_V:
                dV = -max_step_V

            V_new = V_curr + dV
            V_new = max(V_min, min(V_max, V_new))

            if abs(V_new - V_curr) < 1e-3:
                app.calib_queue.put(("done", "voltage_change_too_small"))
                return

            app.calib_queue.put((
                "update_voltage",
                V_name, V_new, dV, delta_nm, it
            ))

            V_curr = V_new
            # give time for heater to settle after changing voltage
            time.sleep(pre_settle_s)

        # If we exit the loop normally, we hit max_iters
        app.calib_queue.put(("done", "max_iters"))

    finally:
        # Stop PD streaming and restore RX polling
        try:
            with _rx_serial_guard(app):
                app.rx.write(b"STREAM_STOP\n")
        except Exception:
            pass
        app.calib_queue.put(("internal_rx_on", None))


# ---------- public entry point ----------

def tx_scan_frequency(app):
    """
    Public entry point: call this from App as tx_calibrate_rings(self).
    """
    _ensure_calib_state(app)

    if app.calib_running:
        messagebox.showwarning(
            "Calibration running",
            "A calibration is already in progress.",
            parent=app.tx_win
        )
        return

    if app.laser is None:
        messagebox.showerror(
            "Laser error",
            "Laser is not connected. Check LASER_HOST and network.",
            parent=app.tx_win
        )
        return

    params = _ask_calibration_params(app)
    if params is None:
        return

    start_nm, end_nm, step_nm, pd_channels, live_plot = params

    # Remember for next time
    app.last_calib_start_nm = start_nm
    app.last_calib_end_nm   = end_nm
    app.last_calib_step_nm  = step_nm
    app.last_calib_pds = list(pd_channels)
    app.last_calib_pd = pd_channels[0]
    app.last_calib_live_plot = bool(live_plot)

    app.calib_live_plot = bool(live_plot)
    selected_label = ", ".join(f"PD{channel}" for channel in pd_channels)
    app.tx_log_print(
        f"[SCAN] Starting wavelength scan: {start_nm:.4f} to {end_nm:.4f} nm "
        f"in steps of {abs(step_nm):.4f} nm, {selected_label}, "
        f"live_plot={app.calib_live_plot}"
    )

    # Stop RX polling so worker can use serial safely
    app.calib_prev_rx_polling = getattr(app, "rx_polling", False)
    app.rx_polling = False

    # Reset data buffers
    app.calib_wavelengths = []
    app.calib_pd_values = []
    app.calib_selected_pds = list(pd_channels)
    app.calib_pd_values_by_channel = {channel: [] for channel in pd_channels}
    app.calib_scan_mode = "wavelength_scan"

    # Optional live plot setup in main thread
    app.calib_fig = app.calib_ax = app.calib_line = None
    app.calib_lines = {}
    if app.calib_live_plot:
        app.calib_fig, app.calib_ax = plt.subplots()
        for channel in pd_channels:
            (app.calib_lines[channel],) = app.calib_ax.plot(
                [], [], marker="o", label=f"PD{channel}"
            )
        app.calib_ax.legend()
        app.calib_ax.set_xlabel("Wavelength (nm)")
        app.calib_ax.set_ylabel("Detector value (arb. units)")
        app.calib_ax.set_title("Wavelength scan")
        app.calib_ax.grid(True)
        app.calib_fig.tight_layout()
        app.calib_fig.show()

    # Start worker thread
    app.calib_running = True
    app.calib_thread = threading.Thread(
        target=_calibration_worker,
        args=(app, start_nm, end_nm, step_nm, pd_channels),
        daemon=True,
    )
    app.calib_thread.start()

    # Start processing queue in Tk thread
    _calib_process_queue(app)


def tx_calibrate_rings(app):
    """
    Public entry point: auto-calibrate rings by tuning a chosen voltage.
    Uses a dialog (similar to Scan Frequency) to set sweep, PD, voltage,
    live plotting, and tuning slope, then runs in a background thread with
    queue-based updates.
    """
    _ensure_calib_state(app)

    if app.calib_running:
        messagebox.showwarning(
            "Calibration running",
            "A calibration / scan is already in progress.",
            parent=app.tx_win
        )
        return

    if app.laser is None:
        messagebox.showerror(
            "Laser error",
            "Laser is not connected. Please connect the laser first.",
            parent=app.tx_win
        )
        return

    params = _ask_ring_calibration_params(app)
    if params is None:
        # user cancelled
        return

    (start_nm, end_nm, step_nm, pd_ch,
     live_plot, V_name, tuning_slope_nm_per_v) = params

    # Remember for future runs
    app.last_calib_start_nm = start_nm
    app.last_calib_end_nm   = end_nm
    app.last_calib_step_nm  = step_nm
    app.last_calib_pd       = pd_ch
    app.last_calib_live_plot = bool(live_plot)
    app.last_ring_voltage_name = V_name
    app.last_ring_tuning_slope_nm_per_v = tuning_slope_nm_per_v

    app.calib_live_plot = bool(live_plot)

    # Initial voltage for this channel
    try:
        V_init = float(app.voltage_vars[V_name].get())
    except Exception:
        V_init = 0.0

    app.tx_log_print(
        f"[RING] Auto-calibration starting: λ from {start_nm:.4f} to {end_nm:.4f} nm, "
        f"step {abs(step_nm):.4f} nm, PD{pd_ch}, tuning {V_name} with slope "
        f"{tuning_slope_nm_per_v:.4f} nm/V, live_plot={app.calib_live_plot}"
    )

    # Prepare data buffers
    app.calib_wavelengths = []
    app.calib_pd_values = []
    app.last_calib_pd = pd_ch  # for labels/plot titles

    # Let the queue handler create per-scan figures when needed
    app.calib_fig = app.calib_ax = app.calib_line = None

    # Start worker thread
    app.calib_running = True
    app.calib_thread = threading.Thread(
        target=_ring_calibration_worker,
        args=(app, start_nm, end_nm, step_nm, pd_ch, V_init, V_name, tuning_slope_nm_per_v),
        daemon=True,
    )
    app.calib_thread.start()

    # Start queue processing in Tk thread
    _calib_process_queue(app)

