import os
import csv
import time
import threading
import queue
from contextlib import nullcontext
import numpy as np

import matplotlib.pyplot as plt
from tkinter import (
    Toplevel, Label, Entry, Frame, Checkbutton, Button,
    StringVar, BooleanVar, messagebox,
)

V_f_Scale = 35.0 / 140


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

def _ensure_piezoscan_state(app):
    """Attach piezoscan-related attributes to app the first time we run."""
    if getattr(app, "_piezoscan_state_init", False):
        return

    app.piezoscan_queue = queue.Queue()
    app.piezoscan_thread = None
    app.piezoscan_running = False
    app.piezoscan_live_plot = False

    app.piezoscan_frequencies = []
    app.piezoscan_pd_values = []
    app.piezoscan_prev_rx_polling = False
    app.piezoscan_prev_rx_streaming = False

    app.piezoscan_fig = None
    app.piezoscan_ax = None
    app.piezoscan_line = None
    app.piezoscan_lines = []

    # Defaults for dialog, if not already present
    app.last_piezoscan_range_V = getattr(app, "last_piezoscan_range_V", 60.0)
    app.last_piezoscan_step_V   = getattr(app, "last_piezoscan_step_V",   0.1)
    app.last_piezoscan_central_nm = getattr(app, "last_piezoscan_central_nm", 1550.0)
    app.last_piezoscan_pd       = getattr(app, "last_piezoscan_pd",       1)
    app.last_piezoscan_pds = getattr(app, "last_piezoscan_pds", [1, 2])
    app.last_piezoscan_live_plot = getattr(app, "last_piezoscan_live_plot", False)

    app._piezoscan_state_init = True

def _ask_piezoscan_params(app):
    """
    Modal dialog to get (range_V, step_V, central_nm, pd_channels, live_plot).
    Returns None if cancelled.
    """
    _ensure_piezoscan_state(app)

    dlg = Toplevel(app.tx_win)
    dlg.title("Piezo Scan")
    dlg.transient(app.tx_win)
    dlg.grab_set()

    Label(dlg, text="Sweep range (V):").grid(row=0, column=0, sticky='e', padx=5, pady=5)
    Label(dlg, text="Sweep step (V):").grid(row=1, column=0, sticky='e', padx=5, pady=5)
    Label(dlg, text="Central (nm):").grid(row=2, column=0, sticky='e', padx=5, pady=5)
    Label(dlg, text="Photon detectors:").grid(row=3, column=0, sticky='e', padx=5, pady=10)

    range_var = StringVar(value=str(app.last_piezoscan_range_V))
    step_var   = StringVar(value=str(app.last_piezoscan_step_V))
    central_var  = StringVar(value=str(app.last_piezoscan_central_nm))
    rx_channel_to_pd = getattr(app, "rx_channel_to_pd", {})
    if rx_channel_to_pd:
        pd_options = [
            (int(detector_index), f"Ch{int(channel_number)} (PD{int(detector_index)})")
            for channel_number, detector_index in sorted(rx_channel_to_pd.items())
        ]
    else:
        pd_options = [(ch, f"PD{ch}") for ch in range(1, 9)]

    last_pds = {
        int(pd_ch)
        for pd_ch in getattr(app, "last_piezoscan_pds", [getattr(app, "last_piezoscan_pd", 1)])
    }
    pd_vars = {
        ch: BooleanVar(value=(ch in last_pds))
        for ch, _label in pd_options
    }

    Entry(dlg, textvariable=range_var, width=10).grid(row=0, column=1, padx=5, pady=5)
    Entry(dlg, textvariable=step_var,   width=10).grid(row=1, column=1, padx=5, pady=5)
    Entry(dlg, textvariable=central_var,  width=10).grid(row=2, column=1, padx=5, pady=5)

    pd_frame = Frame(dlg)
    pd_frame.grid(row=3, column=1, padx=5, pady=5, sticky='w')
    for ch, label in pd_options:
        Checkbutton(pd_frame, text=label, variable=pd_vars[ch]).pack(side="left", padx=3)

    live_var = BooleanVar(value=bool(app.last_piezoscan_live_plot))
    Checkbutton(
        dlg,
        text="Live plotting during sweep (may be slower / less stable)",
        variable=live_var
    ).grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky='w')

    result = {}

    def on_ok():
        try:
            r = float(range_var.get())
            st = float(step_var.get())
            c = float(central_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Range, step, and central must be numbers.", parent=dlg)
            return
        if st <= 0:
            messagebox.showerror("Invalid input", "Step must be > 0.", parent=dlg)
            return
        pd_channels = [ch for ch, _label in pd_options if pd_vars[ch].get()]
        if not pd_channels:
            messagebox.showerror("Invalid input", "Choose at least one PD.", parent=dlg)
            return
        result["vals"] = (r, st, c, pd_channels, live_var.get())
        dlg.destroy()

    def on_cancel():
        dlg.destroy()

    Button(dlg, text="OK",     command=on_ok).grid(row=5, column=0, pady=10)
    Button(dlg, text="Cancel", command=on_cancel).grid(row=5, column=1, pady=10)

    app.tx_win.wait_window(dlg)
    return result.get("vals", None)

def format_pd_values(pd_channels, values):
    return ", ".join(
        f"PD{int(pd_channel)}={float(value):.3f}"
        for pd_channel, value in zip(pd_channels, values)
    )


def format_pd_label(pd_channels):
    return ", ".join(f"PD{int(pd_channel)}" for pd_channel in pd_channels)


def _measure_pd_worker(app, pd_channel: int, timeout_s: float = 0.3) -> float:
    """
    Blocking PD read for use in the calibration worker thread.

    Assumes the RX Arduino is already in STREAM mode and is continuously
    printing lines like:
        "1234  567  890  1150"
    corresponding to PD1..PD4.

    We read one such line and return the requested channel.
    """
    return _measure_pds_worker(app, [pd_channel], timeout_s=timeout_s)[0]


def _measure_pds_worker(app, pd_channels, timeout_s: float = 0.3):
    """
    Blocking PD read for use in the calibration worker thread.
    Reads one stream line and returns all requested PD channels from that line.
    """
    from python_controller import readline_str

    chan_indices = [int(pd_channel) for pd_channel in pd_channels]
    for chan_idx in chan_indices:
        if not (0 <= chan_idx <= 8):
            raise ValueError(f"pd_channel must be 0..8, got {chan_idx}")

    with _rx_serial_guard(app):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            line = readline_str(app.rx)
            if not line:
                continue
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # Skip header or non-numeric lines (e.g. "ACK", "PD1 PD2 PD3 PD4")
            try:
                vals = [float(tok) for tok in parts]
            except ValueError:
                continue

            if len(vals) <= max(chan_indices):
                continue

            if hasattr(app, "rx_apply_pd_background_offset_value"):
                return [
                    app.rx_apply_pd_background_offset_value(vals[chan_idx], chan_idx)
                    for chan_idx in chan_indices
                ]
            if hasattr(app, "rx_apply_pd_dark_noise_value"):
                return [
                    app.rx_apply_pd_dark_noise_value(vals[chan_idx], chan_idx)
                    for chan_idx in chan_indices
                ]
            return [vals[chan_idx] for chan_idx in chan_indices]

    raise TimeoutError(
        f"Timeout waiting for PD{','.join(str(channel) for channel in pd_channels)} streaming value"
    )


def _piezoscan_worker(app, range_V, step_V, central_nm, pd_channels):
    """
    Runs in a background thread. Does NOT touch Tk.
    Sends progress + data back via app.calib_queue.
    """
    from python_controller import readline_str

    # initial_V = app.laser.get_piezo_V()  # This line stores the initial wavelength
    initial_V = 70.0  # This line stores the initial wavelength

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
        app.piezoscan_queue.put(("error", f"Failed to start PD stream: {e}"))
        return

    # Big jump to start wavelength + settle
    pre_settle_s = 1.0  # adjust as needed
    try:
        app.piezoscan_queue.put((
            "log",
            f"[CAL] Jumping laser to start λ = {central_nm:.4f} nm and settling ({pre_settle_s:.1f}s)"
        ))
        app.laser.set_wavelength_nm(central_nm)
        time.sleep(pre_settle_s)
        _flush_rx_buffer(app)
        try:
            dummy = _measure_pds_worker(app, pd_channels, timeout_s=0.3)
            app.piezoscan_queue.put((
                "log",
                f"[CAL] Discarded first transient {format_pd_values(pd_channels, dummy)} at λ={central_nm:.4f} nm"
            ))
        except Exception as e:
            app.piezoscan_queue.put((
                "log",
                f"[CAL] Could not discard first PD sample: {e}"
            ))
    except Exception as e:
        app.piezoscan_queue.put(("error", f"Failed to jump to start λ={central_nm:.4f} nm: {e}"))
        # Stop stream before returning
        try:
            with _rx_serial_guard(app):
                app.rx.write(b"STREAM_STOP\n")
        except Exception:
            pass
        return

    V = range_V / 2
    try:
        # Optional: discard first PD sample after the big jump to avoid transients
        try:
            _ = _measure_pds_worker(app, pd_channels, timeout_s=0.3)
        except Exception:
            pass

        while V >= -range_V / 2 + 1e-12:

            app.piezoscan_queue.put(("log", f"[CAL] Setting piezo voltage = {V:.4f} V"))

            app.laser.set_piezo_V(initial_V + V)

            try:
                values = _measure_pds_worker(app, pd_channels, timeout_s=0.3)
            except Exception as e:
                app.piezoscan_queue.put((
                    "error",
                    f"Error reading PD{','.join(str(ch) for ch in pd_channels)} at V={V:.4f} nm: {e}"
                ))
                return

            app.piezoscan_queue.put(("log", f"[CAL]   -> {format_pd_values(pd_channels, values)}"))
            app.piezoscan_queue.put(("point", initial_V + V, values))

            V -= step_V

        app.piezoscan_queue.put(("restore_voltage", initial_V))
        app.piezoscan_queue.put(("done", "completed"))

    except Exception as e:
        app.piezoscan_queue.put(("error", f"Calibration worker exception: {e}"))

    finally:
        # Always stop PD streaming
        try:
            with _rx_serial_guard(app):
                app.rx.write(b"STREAM_STOP\n")
        except Exception:
            pass


def _piezoscan_process_queue(app):
    """
    Runs in the Tk thread via .after().
    Processes messages from calibration / scan worker(s).
    """
    import queue as _queue_mod

    try:
        while True:
            item = app.piezoscan_queue.get_nowait()
            kind = item[0]

            if kind == "log":
                _, msg = item
                app.tx_log_print(msg)

            elif kind == "internal_rx_off":
                app.piezoscan_prev_rx_polling = getattr(app, "rx_polling", False)
                app.piezoscan_prev_rx_streaming = getattr(app, "rx_streaming", False)
                app.rx_polling = False

            elif kind == "internal_rx_on":
                _resume_rx_panel_after_piezoscan(app)

            elif kind == "scan_start":
                _, it, V_curr = item
                # Clear current scan data
                app.piezoscan_frequencies = []
                app.piezoscan_pd_values = []
                app.tx_log_print(f"[SCAN] Starting scan {it} at V_QF2top = {V_curr:.3f} V")

                if app.piezoscan_live_plot:
                    # NEW: create a brand new figure for this scan
                    app.piezoscan_fig, app.piezoscan_ax = plt.subplots()
                    app.piezoscan_lines = []
                    for pd_ch in app.last_piezoscan_pds:
                        (line,) = app.piezoscan_ax.plot([], [], marker='o', label=f"PD{pd_ch}")
                        app.piezoscan_lines.append(line)

                    app.piezoscan_ax.set_xlabel("Frequency (GHz)")
                    app.piezoscan_ax.set_ylabel("PD value (arb. units)")
                    app.piezoscan_ax.set_title(
                        f"Ring piezo scan - {format_pd_label(app.last_piezoscan_pds)} "
                        f"(scan {it}, V_QF2top={V_curr:.3f} V)"
                    )
                    app.piezoscan_ax.legend()
                    app.piezoscan_ax.grid(True)
                    app.piezoscan_ax.set_ylim(bottom=0, auto=None)
                    app.piezoscan_fig.tight_layout()
                    app.piezoscan_fig.show()

            elif kind == "point":
                _, V, values = item
                values = list(values) if isinstance(values, (list, tuple)) else [values]
                app.piezoscan_frequencies.append(V * V_f_Scale)
                app.piezoscan_pd_values.append(values)

                if app.piezoscan_live_plot and app.piezoscan_fig is not None:
                    for index, line in enumerate(getattr(app, "piezoscan_lines", [])):
                        ys = [
                            row[index]
                            for row in app.piezoscan_pd_values
                            if len(row) > index
                        ]
                        xs = app.piezoscan_frequencies[:len(ys)]
                        line.set_data(xs, ys)
                    app.piezoscan_ax.relim()
                    app.piezoscan_ax.autoscale_view()
                    # Preserve y autoscaling so the upper limit follows new data.
                    app.piezoscan_ax.set_ylim(bottom=0, auto=None)
                    app.piezoscan_fig.canvas.draw_idle()
                    app.piezoscan_fig.canvas.flush_events()

            elif kind == "done":
                _, reason = item
                _piezoscan_finish(app, success=True, info=f"Finished ({reason})")

            elif kind == "restore_voltage":
                _, V = item
                _set_piezo(app, voltage=V)

            elif kind == "error":
                _, msg = item
                _piezoscan_finish(app, success=False, info=msg)

    except _queue_mod.Empty:
        pass

    if app.piezoscan_running:
        app.tx_win.after(50, lambda: _piezoscan_process_queue(app))


def _piezoscan_finish(app, success: bool, info: str):
    """Called in Tk thread when calibration is done or fails."""
    app.piezoscan_running = False

    _resume_rx_panel_after_piezoscan(app)

    if not app.piezoscan_frequencies:
        app.tx_log_print("[CAL] No data collected.")
        if not success:
            app.tx_log_print(f"[CAL] Aborted: {info}")
        return

    if success:
        app.tx_log_print(f"[CAL] Completed: {info}")
    else:
        app.tx_log_print(f"[CAL] Aborted: {info}")

    # Static plot only if live plotting was OFF
    if not app.piezoscan_live_plot:
        _plot_piezoscan_results(
            app,
            app.piezoscan_frequencies,
            app.piezoscan_pd_values,
            app.last_piezoscan_pds,
        )

    _save_piezoscan_csv(
        app,
        app.piezoscan_frequencies,
        app.piezoscan_pd_values,
        app.last_piezoscan_pds,
    )


def _resume_rx_panel_after_piezoscan(app):
    if getattr(app, "piezoscan_prev_rx_streaming", False):
        app.rx_streaming = True
        try:
            app.rx_clear_line_queue()
        except Exception:
            pass
        try:
            with _rx_serial_guard(app):
                app.rx.reset_input_buffer()
                app.rx.write(b"STREAM_START\n")
                app.rx.flush()
        except Exception as exc:
            app.tx_log_print(f"[RX] Could not resume panel stream after piezo scan: {exc}")

    if getattr(app, "piezoscan_prev_rx_polling", False):
        app.rx_polling = True
        try:
            app.rx_start_reader_thread()
        except Exception:
            pass
        app.rx_plot_dirty = True


def _plot_piezoscan_results(app, frequencies, pd_values, pd_channels):
    fig, ax = plt.subplots()
    for index, pd_ch in enumerate(pd_channels):
        ys = [
            row[index] if isinstance(row, (list, tuple)) else row
            for row in pd_values
            if not isinstance(row, (list, tuple)) or len(row) > index
        ]
        xs = frequencies[:len(ys)]
        ax.plot(xs, ys, marker='o', label=f"PD{pd_ch}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("PD value (arb. units)")
    ax.set_title(f"Ring piezo scan - {format_pd_label(pd_channels)}")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(bottom=0, auto=None)
    fig.tight_layout()
    plt.show()


def _save_piezoscan_csv(app, frequencies, pd_values, pd_channels):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "automatic_piezoscan_logs")
    os.makedirs(log_dir, exist_ok=True)

    pd_label = "_".join(f"PD{int(pd_ch)}" for pd_ch in pd_channels)
    fname = time.strftime(f"ring_calibration_{pd_label}_%Y%m%d_%H%M%S.csv")
    fpath = os.path.join(log_dir, fname)

    try:
        with open(fpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wavelength_nm", *[f"PD{int(pd_ch)}" for pd_ch in pd_channels]])
            for lam, values in zip(frequencies, pd_values):
                values = list(values) if isinstance(values, (list, tuple)) else [values]
                writer.writerow([lam, *values])
        app.tx_log_print(f"[CAL] Saved {len(frequencies)} points to {fpath}")
    except Exception as e:
        app.tx_log_print(f"[CAL] Could not save CSV: {e}")


def _set_wavelength(app, wavelength):
    """Restore the laser wavelength to its initial value."""
    if app.laser is not None:
        app.laser.set_wavelength_nm(wavelength)
        app.tx_log_print(f"[CAL] Set wavelength to {wavelength:.4f} nm")
    else:
        app.tx_log_print("[CAL] Laser object is None")


def _set_piezo(app, voltage):
    """Restore the laser wavelength to its initial value."""
    if app.laser is not None:
        app.laser.set_piezo_V(voltage)
        app.tx_log_print(f"[CAL] Set piezo voltage to {voltage:.4f} V")
    else:
        app.tx_log_print("[CAL] Laser object is None")


# ---------- public entry point ----------
def tx_scan_piezo(app):
    """
    Public entry point: call this from App as tx_calibrate_rings(self).
    """

    _ensure_piezoscan_state(app)

    if app.piezoscan_running:
        messagebox.showwarning(
            "Piezo scanning running",
            "A piezo scan is already in progress.",
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

    params = _ask_piezoscan_params(app)
    if params is None:
        return

    range_V, step_V, central_nm, pd_channels, live_plot = params
    pd_channels = [int(pd_ch) for pd_ch in pd_channels]

    if hasattr(app, "load_rx_background_offsets"):
        stored_offsets = app.load_rx_background_offsets()
        app.rx_pd_background_offsets = stored_offsets
        if hasattr(app, "rx_update_background_offset_fields"):
            app.rx_update_background_offset_fields()
        app.tx_log_print(
            "[CAL] Applying RX background offsets: "
            + app.rx_format_background_offsets(stored_offsets)
        )

    # Remember for next time
    app.last_piezoscan_range_V = range_V
    app.last_piezoscan_step_V   = step_V
    app.last_piezoscan_central_nm  = central_nm
    app.last_piezoscan_pd       = pd_channels[0]
    app.last_piezoscan_pds      = pd_channels
    app.last_piezoscan_live_plot = bool(live_plot)

    app.piezoscan_live_plot = bool(live_plot)

    app.tx_log_print(
        f"[CAL] Starting piezo scan: λ at {central_nm:.4f} "
        f"{format_pd_label(pd_channels)}, live_plot={app.piezoscan_live_plot}"
    )

    # Stop RX polling so worker can use serial safely
    app.piezoscan_prev_rx_polling = getattr(app, "rx_polling", False)
    app.piezoscan_prev_rx_streaming = getattr(app, "rx_streaming", False)
    app.rx_polling = False
    app.rx_streaming = False

    # Reset data buffers
    app.piezoscan_frequencies = []
    app.piezoscan_pd_values = []

    # Optional live plot setup in main thread
    app.piezoscan_fig = app.piezoscan_ax = app.piezoscan_line = None
    app.piezoscan_lines = []
    if app.piezoscan_live_plot:
        app.piezoscan_fig, app.piezoscan_ax = plt.subplots()
        for pd_ch in pd_channels:
            (line,) = app.piezoscan_ax.plot([], [], marker='o', label=f"PD{pd_ch}")
            app.piezoscan_lines.append(line)
        app.piezoscan_ax.set_xlabel("Frequency (GHz)")
        app.piezoscan_ax.set_ylabel("PD value (arb. units)")
        app.piezoscan_ax.set_title(f"Ring piezo scan - {format_pd_label(pd_channels)}")
        app.piezoscan_ax.legend()
        app.piezoscan_ax.grid(True)
        app.piezoscan_ax.set_ylim(bottom=0, auto=None)
        app.piezoscan_fig.tight_layout()
        app.piezoscan_fig.show()

    # Start worker thread
    app.piezoscan_running = True
    app.piezoscan_thread = threading.Thread(
        target=_piezoscan_worker,
        args=(app, range_V, step_V, central_nm, pd_channels),
        daemon=True,
    )
    app.piezoscan_thread.start()

    # Start processing queue in Tk thread
    _piezoscan_process_queue(app)
