import time

import matplotlib.pyplot as plt
import numpy as np

from tkinter import (
    Toplevel,
    Frame,
    Label,
    OptionMenu,
    Entry,
    Button,
    LEFT,
    RIGHT,
    StringVar,
    messagebox,
    simpledialog,
)
from tkinter import filedialog

from controller_common import readline_str, write_line, write_float_line, vsq_linspace


class Scan2VDialog:
    def __init__(self, parent, names, defaults=None):
        self.top = Toplevel(parent)
        self.top.title("Scan 2V")
        self.top.transient(parent)
        self.top.grab_set()

        self.result = None
        defaults = defaults or {}

        default_v1 = defaults.get("voltage1", names[0] if names else "")
        if default_v1 not in names and names:
            default_v1 = names[0]
        default_v2 = defaults.get(
            "voltage2",
            names[1] if len(names) > 1 else (names[0] if names else ""),
        )
        if default_v2 not in names and names:
            default_v2 = names[1] if len(names) > 1 else names[0]

        self.v1 = StringVar(value=default_v1)
        self.v2 = StringVar(value=default_v2)

        self.a1s = StringVar(value=str(defaults.get("axis1_start", 5.0)))
        self.a1e = StringVar(value=str(defaults.get("axis1_stop", 20.0)))
        self.a1n = StringVar(value=str(defaults.get("axis1_count", 21)))

        self.a2s = StringVar(value=str(defaults.get("axis2_start", 5.0)))
        self.a2e = StringVar(value=str(defaults.get("axis2_stop", 20.0)))
        self.a2n = StringVar(value=str(defaults.get("axis2_count", 21)))

        self.pd1 = StringVar(value=str(defaults.get("pd1", 3)))
        self.pd2 = StringVar(value=str(defaults.get("pd2", 4)))

        frame = Frame(self.top)
        frame.pack(padx=12, pady=12)

        def add_row(row, label_text, widget):
            Label(frame, text=label_text).grid(row=row, column=0, sticky="e", padx=6, pady=4)
            widget.grid(row=row, column=1, sticky="w", padx=6, pady=4)

        add_row(0, "Voltage #1", OptionMenu(frame, self.v1, *names))
        add_row(1, "Voltage #2", OptionMenu(frame, self.v2, *names))

        axis1_frame = Frame(frame)
        Entry(axis1_frame, width=10, textvariable=self.a1s).pack(side=LEFT, padx=(0, 6))
        Entry(axis1_frame, width=10, textvariable=self.a1e).pack(side=LEFT, padx=(0, 6))
        Entry(axis1_frame, width=6, textvariable=self.a1n).pack(side=LEFT)
        add_row(2, "Axis1 start / stop / N", axis1_frame)

        axis2_frame = Frame(frame)
        Entry(axis2_frame, width=10, textvariable=self.a2s).pack(side=LEFT, padx=(0, 6))
        Entry(axis2_frame, width=10, textvariable=self.a2e).pack(side=LEFT, padx=(0, 6))
        Entry(axis2_frame, width=6, textvariable=self.a2n).pack(side=LEFT)
        add_row(3, "Axis2 start / stop / N", axis2_frame)

        pd_frame = Frame(frame)
        Entry(pd_frame, width=6, textvariable=self.pd1).pack(side=LEFT, padx=(0, 8))
        Entry(pd_frame, width=6, textvariable=self.pd2).pack(side=LEFT)
        add_row(4, "PD indices (1-4)", pd_frame)

        buttons = Frame(frame)
        buttons.grid(row=5, column=0, columnspan=2, pady=(10, 0), sticky="e")
        Button(buttons, text="Cancel", command=self._cancel).pack(side=RIGHT, padx=6)
        Button(buttons, text="Start", command=self._ok).pack(side=RIGHT)

        frame.grid_columnconfigure(1, weight=1)

    def _cancel(self):
        self.top.destroy()

    def _ok(self):
        try:
            voltage1 = self.v1.get().strip()
            voltage2 = self.v2.get().strip()
            axis1_start = float(self.a1s.get())
            axis1_stop = float(self.a1e.get())
            axis1_count = int(self.a1n.get())
            axis2_start = float(self.a2s.get())
            axis2_stop = float(self.a2e.get())
            axis2_count = int(self.a2n.get())
            pd1 = int(self.pd1.get())
            pd2 = int(self.pd2.get())

            if axis1_count < 1 or axis2_count < 1:
                raise ValueError("N must be >= 1 for both axes.")
            if not (1 <= pd1 <= 8 and 1 <= pd2 <= 8):
                raise ValueError("PD indices must be in 1..8.")

            self.result = (
                voltage1,
                voltage2,
                axis1_start,
                axis1_stop,
                axis1_count,
                axis2_start,
                axis2_stop,
                axis2_count,
                pd1,
                pd2,
            )
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc), parent=self.top)
            return

        self.top.destroy()


class TxLevelSweepDialog:
    def __init__(self, parent, part_names, channel_choices, defaults=None):
        self.top = Toplevel(parent)
        self.top.title("Set TX Voltages")
        self.top.transient(parent)
        self.top.grab_set()

        self.result = None
        defaults = defaults or {}

        default_part = defaults.get("part_name", part_names[0] if part_names else "")
        if default_part not in part_names and part_names:
            default_part = part_names[0]

        self.channel_lookup = {
            label: (int(channel_number), int(detector_index))
            for label, channel_number, detector_index in channel_choices
        }
        channel_labels = list(self.channel_lookup.keys())
        default_channel_label = defaults.get(
            "channel_label",
            channel_labels[0] if channel_labels else "",
        )
        if default_channel_label not in self.channel_lookup and channel_labels:
            default_channel_label = channel_labels[0]

        self.part_var = StringVar(value=default_part)
        self.channel_var = StringVar(value=default_channel_label)
        self.start_var = StringVar(value=str(defaults.get("start_voltage", 0.0)))
        self.stop_var = StringVar(value=str(defaults.get("stop_voltage", 20.0)))
        self.step_var = StringVar(value=str(defaults.get("step_voltage", 0.5)))

        frame = Frame(self.top)
        frame.pack(padx=12, pady=12)

        def add_row(row, label_text, widget):
            Label(frame, text=label_text).grid(row=row, column=0, sticky="e", padx=6, pady=4)
            widget.grid(row=row, column=1, sticky="w", padx=6, pady=4)

        add_row(0, "Chip part", OptionMenu(frame, self.part_var, *part_names))
        add_row(1, "RX channel", OptionMenu(frame, self.channel_var, *channel_labels))

        start_stop_step_frame = Frame(frame)
        Entry(start_stop_step_frame, width=10, textvariable=self.start_var).pack(side=LEFT, padx=(0, 6))
        Entry(start_stop_step_frame, width=10, textvariable=self.stop_var).pack(side=LEFT, padx=(0, 6))
        Entry(start_stop_step_frame, width=10, textvariable=self.step_var).pack(side=LEFT)
        add_row(2, "Start / stop / step (V)", start_stop_step_frame)

        buttons = Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky="e")
        Button(buttons, text="Cancel", command=self._cancel).pack(side=RIGHT, padx=6)
        Button(buttons, text="Start", command=self._ok).pack(side=RIGHT)

        frame.grid_columnconfigure(1, weight=1)

    def _cancel(self):
        self.top.destroy()

    def _ok(self):
        try:
            part_name = self.part_var.get().strip()
            channel_label = self.channel_var.get().strip()
            start_voltage = float(self.start_var.get())
            stop_voltage = float(self.stop_var.get())
            step_voltage = float(self.step_var.get())

            if not part_name:
                raise ValueError("Pick a TX chip part to sweep.")
            if channel_label not in self.channel_lookup:
                raise ValueError("Pick a valid RX channel.")
            if step_voltage <= 0.0:
                raise ValueError("Step voltage must be > 0.")
            if abs(stop_voltage - start_voltage) < 1e-12:
                raise ValueError("Start and stop voltages must be different.")
            if not (0.0 <= start_voltage <= 30.0 and 0.0 <= stop_voltage <= 30.0):
                raise ValueError("Start and stop voltages must be within 0..30 V.")

            channel_number, detector_index = self.channel_lookup[channel_label]
            self.result = (
                part_name,
                channel_label,
                channel_number,
                detector_index,
                start_voltage,
                stop_voltage,
                step_voltage,
            )
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc), parent=self.top)
            return

        self.top.destroy()


class TxCommandsMixin:
    def _tx_wait_for_serial_line(self, expected_text: str, timeout_s: float, context: str):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            response = readline_str(self.tx)
            if not response:
                continue

            text = response.strip()
            if not text:
                continue

            print(text)
            if text.startswith("ERR"):
                raise RuntimeError(f"{context}: {text}")

            if text == expected_text:
                return

            if hasattr(self, "tx_log_print"):
                self.tx_log_print(f"[TX<-] {text}")

        raise TimeoutError(f"{context}: timed out waiting for '{expected_text}'")

    def _tx_normalize_voltage_updates(self, dac_map_updates: dict):
        normalized_updates = {}

        for name, entry in dac_map_updates.items():
            if isinstance(entry, (int, float)):
                if name not in self.tx_dac_map:
                    raise KeyError(f"{name}: missing from tx_dac_map")
                chip_id, pin, _old_voltage = self.tx_dac_map[name]
                voltage = float(entry)
            else:
                try:
                    chip_id, pin, voltage = entry
                except Exception as exc:
                    raise ValueError(
                        f"{name}: expected voltage update as float or (chip_id, pin, voltage)"
                    ) from exc

            normalized_updates[name] = (int(chip_id), int(pin), float(voltage))

        return normalized_updates

    def tx_update_cached_voltage_values(self, voltage_values: dict, persist=False):
        if not hasattr(self, "tx_dac_map"):
            return

        for name, voltage in voltage_values.items():
            if name not in self.tx_dac_map:
                continue
            chip_id, pin, _old_voltage = self.tx_dac_map[name]
            self.tx_dac_map[name] = (chip_id, pin, float(voltage))

        if persist and hasattr(self, "save_default_voltages"):
            self.save_default_voltages()

    def _tx_get_rx_channel_choices(self):
        channel_to_pd = getattr(self, "rx_channel_to_pd", {})
        if channel_to_pd:
            return [
                (f"Channel {int(channel_number)} (PD{int(detector_index)})", channel_number, detector_index)
                for channel_number, detector_index in sorted(channel_to_pd.items())
            ]

        detector_count = int(getattr(self, "rx_detector_count", 0))
        return [(f"PD{detector_index}", detector_index, detector_index) for detector_index in range(detector_count)]

    def _tx_build_sweep_values(self, start_voltage: float, stop_voltage: float, step_voltage: float):
        start_voltage = float(start_voltage)
        stop_voltage = float(stop_voltage)
        step_voltage = abs(float(step_voltage))
        if step_voltage <= 0.0:
            raise ValueError("Step voltage must be > 0.")

        direction = 1.0 if stop_voltage >= start_voltage else -1.0
        step_voltage *= direction

        sweep_values = []
        current_voltage = start_voltage
        if direction > 0:
            while current_voltage <= stop_voltage + 1e-12:
                sweep_values.append(round(current_voltage, 9))
                current_voltage += step_voltage
        else:
            while current_voltage >= stop_voltage - 1e-12:
                sweep_values.append(round(current_voltage, 9))
                current_voltage += step_voltage

        if not sweep_values or abs(sweep_values[-1] - stop_voltage) > 1e-9:
            sweep_values.append(float(stop_voltage))

        if len(sweep_values) < 2:
            raise ValueError("Sweep must contain at least two voltage points.")

        return sweep_values

    def _tx_apply_level_config(self, part_name: str, low_voltage: float, high_voltage: float):
        chip_id, pin, _old_voltage = self.tx_dac_map[part_name]

        write_line(self.tx, "SET_TX_LEVELS")
        write_line(self.tx, str(int(chip_id)))
        write_line(self.tx, str(int(pin)))
        write_float_line(self.tx, low_voltage)
        write_float_line(self.tx, high_voltage)
        self._tx_wait_for_serial_line("Done", timeout_s=5.0, context="SET_TX_LEVELS")

        if hasattr(self, "tx_log_print"):
            self.tx_log_print(
                f"[TX] Active TX output set to {part_name} (chip {int(chip_id)}, pin {int(pin)})"
            )

    def _tx_measure_rx_value_for_voltage(
        self,
        part_name: str,
        voltage: float,
        detector_index: int,
        settle_s: float = 0.12,
        timeout_s: float = 1.0,
    ):
        self.tx_send_voltages({part_name: float(voltage)})
        time.sleep(settle_s)

        with self.rx_serial_lock:
            try:
                self.rx.reset_input_buffer()
            except Exception:
                pass

        values = self.rx_read_stream_sample_blocking(timeout_s=timeout_s)
        detector_index = int(detector_index)
        if detector_index < 0 or detector_index >= len(values):
            raise IndexError(f"RX detector index out of range: {detector_index}")
        return float(values[detector_index])

    def tx_send_voltages(self, dac_map_updates: dict):
        normalized_updates = self._tx_normalize_voltage_updates(dac_map_updates)
        if not normalized_updates:
            return

        write_line(self.tx, "UPDATE_VOLTAGES")
        self._tx_wait_for_serial_line("ACK", timeout_s=5.0, context="UPDATE_VOLTAGES")

        # The startup batch is large enough to outrun the Arduino serial parser
        # if we dump every line at once, so pace each update slightly.
        time.sleep(0.02)

        for _name, (chip_id, pin, volt) in normalized_updates.items():
            write_line(self.tx, f"{int(chip_id)} {int(pin)} {float(volt):.6f}")
            try:
                self.tx.flush()
            except Exception:
                pass
            time.sleep(0.02)

        write_line(self.tx, "END")
        try:
            self.tx.flush()
        except Exception:
            pass
        self._tx_wait_for_serial_line("Done.", timeout_s=10.0, context="UPDATE_VOLTAGES")

    def tx_init_voltages(self):

        if not hasattr(self, "tx_dac_map"):
            messagebox.showerror("Error", "tx_dac_map not found.")
            return

        # Use the map directly as the payload for tx_send_voltages()
        updates = {name: (chip, pin, float(v)) for name, (chip, pin, v) in self.tx_dac_map.items()}

        try:
            self.tx_send_voltages(updates)
        except Exception as e:
            if hasattr(self, "tx_log_print"):
                self.tx_log_print(f"[TX] Init-from-map failed: {e}")
            messagebox.showerror("TX init failed", str(e))
            return

        if hasattr(self, "tx_log_print"):
            self.tx_log_print(
                f"[TX] Initialized from {getattr(self.default_voltages_path, 'name', 'default_voltages.json')} "
                f"({len(updates)} channels)."
            )

    def tx_update_changed_from_gui(self, eps: float = 1e-4):
        if not hasattr(self, "tx_dac_map"):
            messagebox.showerror("Error", "tx_dac_map not found (master DAC map missing).")
            return
        if not hasattr(self, "tx_write_order"):
            messagebox.showerror("Error", "tx_write_order not found.")
            return

        updates = {}
        changed_voltage_values = {}

        try:
            for name in self.tx_write_order:
                if name not in self.voltage_vars or name not in self.tx_dac_map:
                    continue

                new_voltage = float(self.voltage_vars[name].get())
                chip_id, pin, old_voltage = self.tx_dac_map[name]
                if abs(new_voltage - float(old_voltage)) > eps:
                    updates[name] = (chip_id, pin, float(new_voltage))
                    changed_voltage_values[name] = float(new_voltage)
        except ValueError as exc:
            messagebox.showerror("Invalid input", f"Bad numeric value: {exc}")
            return
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        if not updates:
            if hasattr(self, "tx_log_print"):
                self.tx_log_print("[TX] No voltage changes.")
            return

        try:
            print(updates)
            self.tx_send_voltages(updates)
        except Exception as exc:
            if hasattr(self, "tx_log_print"):
                self.tx_log_print(f"[TX] Update failed: {exc}")
            messagebox.showerror("TX update failed", str(exc))
            return

        try:
            self.tx_update_cached_voltage_values(changed_voltage_values, persist=True)
        except Exception as exc:
            if hasattr(self, "tx_log_print"):
                self.tx_log_print(
                    f"[TX] Voltages were updated, but saving {getattr(self.default_voltages_path, 'name', 'default_voltages.json')} failed: {exc}"
                )
            messagebox.showwarning("Default voltage save failed", str(exc))

        if hasattr(self, "tx_log_print"):
            self.tx_log_print(f"[TX] Updated {len(updates)} channel(s): {', '.join(updates.keys())}")

    def _tx_run_scan_2v(
        self,
        name1,
        name2,
        s1,
        e1,
        n1,
        s2,
        e2,
        n2,
        pd1,
        pd2,
        timeout_s=60.0,
    ):
        def _read_exact(nbytes, timeout=timeout_s):
            start_time = time.time()
            buffer = bytearray()
            while len(buffer) < nbytes:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Timeout while reading {nbytes} bytes (got {len(buffer)})")
                chunk = self.tx.read(nbytes - len(buffer))
                if chunk:
                    buffer.extend(chunk)
            return bytes(buffer)

        def _read_begin(timeout=10.0):
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError("Timeout waiting for BEGIN")
                line = readline_str(self.tx)
                if not line:
                    continue
                text = line.strip()
                if not text:
                    continue
                if text.startswith("ERR"):
                    raise RuntimeError(text)
                if text.startswith("BEGIN"):
                    parts = text.split()
                    if len(parts) != 3:
                        raise RuntimeError(f"Malformed BEGIN line: {text}")
                    return int(parts[1]), int(parts[2])

        def _read_done(timeout=10.0):
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError("Timeout waiting for DONE")
                line = readline_str(self.tx)
                if not line:
                    continue
                text = line.strip()
                if not text:
                    continue
                if text.startswith("ERR"):
                    raise RuntimeError(text)
                if text == "DONE":
                    return

        chip1, pin1, voltage1_init = self.tx_dac_map[name1]
        chip2, pin2, voltage2_init = self.tx_dac_map[name2]

        write_line(self.tx, "SCAN_2V")

        while True:
            if readline_str(self.tx).strip() == "ACK":
                break

        write_line(self.tx, f"{chip1} {pin1} {voltage1_init} {s1} {e1} {n1}")
        write_line(self.tx, f"{chip2} {pin2} {voltage2_init} {s2} {e2} {n2}")
        write_line(self.tx, f"{pd1} {pd2}")
        write_line(self.tx, "END")

        while True:
            if readline_str(self.tx).strip() == "ACK":
                break

        while True:
            if readline_str(self.tx).strip() == "Scan2V Finished":
                break

        returned_n1, returned_n2 = _read_begin(timeout=10.0)

        total_points = returned_n1 * returned_n2
        raw1 = _read_exact(total_points * 2, timeout=timeout_s)
        raw2 = _read_exact(total_points * 2, timeout=timeout_s)

        _read_done(timeout=10.0)

        pd1_values = np.frombuffer(raw1, dtype="<u2").astype(np.float32).reshape((returned_n1, returned_n2)).T
        pd2_values = np.frombuffer(raw2, dtype="<u2").astype(np.float32).reshape((returned_n1, returned_n2)).T
        stored_offsets = self.load_rx_background_offsets()
        self.rx_pd_background_offsets = stored_offsets
        if hasattr(self, "rx_update_background_offset_fields"):
            self.rx_update_background_offset_fields()

        pd1_offset = float(stored_offsets.get(int(pd1), 0.0))
        pd2_offset = float(stored_offsets.get(int(pd2), 0.0))
        pd1_values = self._tx_apply_scan_background_offset(pd1_values, pd1_offset)
        pd2_values = self._tx_apply_scan_background_offset(pd2_values, pd2_offset)

        if hasattr(self, "tx_log_print"):
            self.tx_log_print(
                f"[TX] Scan 2V applied stored background offsets: "
                f"PD{int(pd1)}={pd1_offset:g}, PD{int(pd2)}={pd2_offset:g}"
            )

        return {
            "name1": name1,
            "name2": name2,
            "n1": returned_n1,
            "n2": returned_n2,
            "V1_axis": vsq_linspace(s1, e1, returned_n1),
            "V2_axis": vsq_linspace(s2, e2, returned_n2),
            "PD1": pd1_values,
            "PD2": pd2_values,
            "pd1": pd1,
            "pd2": pd2,
            "background_offset_pd1": pd1_offset,
            "background_offset_pd2": pd2_offset,
            "background_offsets": stored_offsets,
        }

    def _tx_apply_scan_background_offset(self, values, offset):
        corrected_values = np.asarray(values, dtype=np.float64) - float(offset)
        return np.floor(np.maximum(0.0, corrected_values)).astype(np.int64)

    def tx_scan_2v(self):
        dialog = Scan2VDialog(
            self.tx_win,
            list(self.tx_dac_map.keys()),
            defaults=getattr(self, "tx_scan_2v_last_params", None),
        )
        self.tx_win.wait_window(dialog.top)
        if dialog.result is None:
            return

        (
            voltage1,
            voltage2,
            axis1_start,
            axis1_stop,
            axis1_count,
            axis2_start,
            axis2_stop,
            axis2_count,
            pd1,
            pd2,
        ) = dialog.result
        self.tx_scan_2v_last_params = {
            "voltage1": voltage1,
            "voltage2": voltage2,
            "axis1_start": axis1_start,
            "axis1_stop": axis1_stop,
            "axis1_count": axis1_count,
            "axis2_start": axis2_start,
            "axis2_stop": axis2_stop,
            "axis2_count": axis2_count,
            "pd1": pd1,
            "pd2": pd2,
        }

        data = self._tx_run_scan_2v(*dialog.result)
        self._tx_save_scan_2v(data)
        self._tx_plot_scan_2v(data)

    def tx_pulse_test(self):
        message = simpledialog.askstring("Pulse Test", "Enter message to send:", parent=self.tx_win)
        if message is None:
            return
        write_line(self.tx, "SEND_MESSAGE_d")
        write_line(self.tx, message)
        self.tx_log_print(f"[TX] SEND_MESSAGE '{message}'")
        while True:
            response = readline_str(self.tx)
            if not response:
                self.tx_win.update_idletasks()
                self.tx_win.update()
                continue
            self.tx_log_print("[TX<-] " + response.strip())
            if response.strip() == "Done":
                break

    def tx_toggle_mode_pin(self):
        next_high = not bool(getattr(self, "tx_mode_pin_high", False))
        try:
            write_line(self.tx, "SET_TX_MODE_PIN")
            write_line(self.tx, "1" if next_high else "0")
            self._tx_wait_for_serial_line("Done", timeout_s=2.0, context="SET_TX_MODE_PIN")
        except Exception as exc:
            if hasattr(self, "tx_log_print"):
                self.tx_log_print(f"[TX] Mode pin toggle failed: {exc}")
            messagebox.showerror("TX Mode Pin", str(exc), parent=self.tx_win)
            return

        self.tx_mode_pin_high = next_high
        label = "Continuous" if next_high else "  Pulse  "
        not_label = "Continuous" if not next_high else "Pulse"
        if hasattr(self, "tx_mode_pin_button_text"):
            self.tx_mode_pin_button_text.set(label)
        if hasattr(self, "tx_log_print"):
            level = "HIGH" if next_high else "LOW"
            self.tx_log_print(f"[TX] Digital pin 53 set {level} ({not_label}).")

    def tx_trigger(self, timeout_s: float = 2.0):
        """Toggle the TX Arduino's retained AWG burst-gate output level."""
        if getattr(self, "tx", None) is None:
            raise RuntimeError("TX Arduino is not connected.")

        write_line(self.tx, "TRIGGER")
        try:
            self.tx.flush()
        except Exception:
            pass
        self._tx_wait_for_serial_line("Trigger Done", timeout_s=timeout_s, context="TRIGGER")

        if hasattr(self, "tx_log_print"):
            self.tx_log_print("[TX] Trigger gate toggled.")
        return "Trigger Done"

    def tx_teensy_test(self):
        mode = simpledialog.askstring("Teensy Test", "Enter 0 or 1:", parent=self.tx_win)
        if mode not in ("0", "1"):
            return
        write_line(self.tx, "TEST_TEENSY")
        self.tx.write(mode.encode("utf-8"))
        self.tx.write(b"\n")
        while True:
            response = readline_str(self.tx)
            if not response:
                self.tx_win.update_idletasks()
                self.tx_win.update()
                continue
            self.tx_log_print("[TX<-] " + response.strip())
            if response.strip() == "Done":
                break

    def tx_qdcp_slip_status(self, timeout_s: float = 2.0):
        try:
            self.tx.reset_input_buffer()
        except Exception:
            pass

        write_line(self.tx, "QDCP_SLIP_STATUS")
        self.tx_log_print("[TX] QDCP_SLIP_STATUS")

        deadline = time.time() + timeout_s
        saw_begin = False
        while time.time() < deadline:
            response = readline_str(self.tx)
            if not response:
                self.tx_win.update_idletasks()
                self.tx_win.update()
                continue

            text = response.strip()
            if not text:
                continue

            self.tx_log_print("[TX<-] " + text)
            if text == "QDCP_STATUS_BEGIN":
                saw_begin = True
            if text == "QDCP_STATUS_END":
                return

        if saw_begin:
            self.tx_log_print("[TX] QDCP status timed out before END")
        else:
            self.tx_log_print("[TX] QDCP status timed out")

    def tx_set_offset(self):
        duration_s = simpledialog.askfloat(
            "Set Offset",
            "Measurement time (s):",
            parent=self.tx_win,
            initialvalue=1.0,
            minvalue=0.1,
        )
        if duration_s is None:
            return

        expected_count = int(getattr(self, "rx_detector_count", 0))
        if expected_count <= 0:
            messagebox.showerror("Set Offset", "No RX PD channels are available.", parent=self.tx_win)
            return

        previous_rx_polling = bool(getattr(self, "rx_polling", False))
        was_streaming = bool(getattr(self, "rx_streaming", False))
        sample_sums = np.zeros(expected_count, dtype=np.float64)
        sample_count = 0
        offsets_applied = False

        if hasattr(self, "tx_log_print"):
            self.tx_log_print(f"[TX] Measuring RX background offsets for {float(duration_s):.3f} s.")

        try:
            self.rx_polling = False
            self.rx_reset_runtime_state()

            with self.rx_serial_lock:
                try:
                    self.rx.reset_input_buffer()
                except Exception:
                    pass

            if not was_streaming:
                self.rx_start_stream_blocking(timeout_s=1.0)

            deadline = time.monotonic() + float(duration_s)
            while time.monotonic() < deadline:
                remaining_s = deadline - time.monotonic()
                sample_timeout_s = min(0.25, max(0.02, remaining_s))
                try:
                    values = self.rx_read_stream_sample_blocking(timeout_s=sample_timeout_s)
                except TimeoutError:
                    self.tx_win.update_idletasks()
                    self.tx_win.update()
                    continue

                if len(values) < expected_count:
                    continue

                sample_sums += np.asarray(values[:expected_count], dtype=np.float64)
                sample_count += 1
                self.tx_win.update_idletasks()
                self.tx_win.update()

            if sample_count <= 0:
                raise TimeoutError("No RX samples were captured.")

            averages = {
                int(pd_index): float(sample_sums[int(pd_index)] / sample_count)
                for pd_index in sorted(getattr(self, "rx_pd_to_channel", {}))
            }
            saved_offsets = self.rx_set_background_offsets(averages)
            offsets_applied = True

            if hasattr(self, "tx_log_print"):
                self.tx_log_print(
                    f"[TX] Set RX background offsets from {sample_count} sample(s): "
                    + self.rx_format_background_offsets(saved_offsets)
                )
            if hasattr(self, "rx_log_print"):
                self.rx_log_print(
                    "[RX Background] Auto-set offsets (applied): "
                    + self.rx_format_background_offsets(saved_offsets)
                )

        except Exception as exc:
            if hasattr(self, "tx_log_print"):
                self.tx_log_print(f"[TX] Set Offset failed: {exc}")
            messagebox.showerror("Set Offset failed", str(exc), parent=self.tx_win)
            return
        finally:
            try:
                if not was_streaming:
                    self.rx_stop_stream_blocking(timeout_s=1.0)
                else:
                    with self.rx_serial_lock:
                        try:
                            self.rx.reset_input_buffer()
                        except Exception:
                            pass
            except Exception as exc:
                if hasattr(self, "tx_log_print"):
                    self.tx_log_print(f"[RX] Stream cleanup warning: {exc}")

            self.rx_polling = previous_rx_polling
            self.rx_reset_runtime_state()
            if offsets_applied:
                self.rx_plot_dirty = True

    def tx_set_levels(self):
        part_names = list(getattr(self, "tx_write_order", self.tx_dac_map.keys()))
        channel_choices = self._tx_get_rx_channel_choices()
        if not part_names:
            messagebox.showerror("Set TX Voltages", "No TX voltages are available to sweep.")
            return
        if not channel_choices:
            messagebox.showerror("Set TX Voltages", "No RX channels are available to read.")
            return

        dialog = TxLevelSweepDialog(
            self.tx_win,
            part_names,
            channel_choices,
            defaults={
                "part_name": getattr(self, "last_tx_level_part_name", part_names[0]),
                "channel_label": getattr(self, "last_tx_level_channel_label", channel_choices[0][0]),
                "start_voltage": getattr(self, "last_tx_level_start_voltage", 0.0),
                "stop_voltage": getattr(self, "last_tx_level_stop_voltage", 20.0),
                "step_voltage": getattr(self, "last_tx_level_step_voltage", 0.5),
            },
        )
        self.tx_win.wait_window(dialog.top)
        if dialog.result is None:
            return

        (
            part_name,
            channel_label,
            channel_number,
            detector_index,
            start_voltage,
            stop_voltage,
            step_voltage,
        ) = dialog.result

        self.last_tx_level_part_name = part_name
        self.last_tx_level_channel_label = channel_label
        self.last_tx_level_start_voltage = float(start_voltage)
        self.last_tx_level_stop_voltage = float(stop_voltage)
        self.last_tx_level_step_voltage = float(step_voltage)

        try:
            sweep_values = self._tx_build_sweep_values(start_voltage, stop_voltage, step_voltage)
        except Exception as exc:
            messagebox.showerror("Set TX Voltages", str(exc), parent=self.tx_win)
            return

        previous_rx_polling = bool(getattr(self, "rx_polling", False))
        was_streaming = bool(getattr(self, "rx_streaming", False))
        measurements = []

        if hasattr(self, "tx_log_print"):
            self.tx_log_print(
                f"[TX] Sweeping {part_name} from {start_voltage:.3f} V to {stop_voltage:.3f} V "
                f"in {step_voltage:.3f} V steps while reading {channel_label}."
            )

        try:
            self.rx_polling = False
            self.rx_reset_runtime_state()

            with self.rx_serial_lock:
                try:
                    self.rx.reset_input_buffer()
                except Exception:
                    pass

            if not was_streaming:
                self.rx_start_stream_blocking(timeout_s=1.0)

            for voltage in sweep_values:
                measured_value = self._tx_measure_rx_value_for_voltage(
                    part_name,
                    voltage,
                    detector_index,
                )
                measurements.append((float(voltage), float(measured_value)))
                if hasattr(self, "tx_log_print"):
                    self.tx_log_print(
                        f"[TX]   {part_name}={float(voltage):.3f} V -> Channel {int(channel_number)} = {float(measured_value):.1f}"
                    )
                self.tx_win.update_idletasks()
                self.tx_win.update()

            low_voltage, low_value = min(measurements, key=lambda item: item[1])
            high_voltage, high_value = max(measurements, key=lambda item: item[1])
            threshold_value = low_value + 0.3 * (high_value - low_value)

            self._tx_apply_level_config(part_name, low_voltage, high_voltage)
            applied_detector_index = self.rx_set_decode_pd(detector_index)
            applied_threshold = self.rx_set_threshold(threshold_value)

            self.tx_update_cached_voltage_values({part_name: low_voltage}, persist=False)
            if part_name in getattr(self, "voltage_vars", {}):
                self.voltage_vars[part_name].set(f"{float(low_voltage):.6f}")

            self.tx_active_output_name = part_name
            self.tx_active_output_channel = int(channel_number)
            self.tx_low_level_voltage = float(low_voltage)
            self.tx_high_level_voltage = float(high_voltage)
            self.rx_decode_detector_index = int(applied_detector_index)
            self.rx_decode_channel_number = int(channel_number)
            self.rx_threshold_value = int(applied_threshold)

            if hasattr(self, "tx_log_print"):
                self.tx_log_print(
                    f"[TX] Sweep complete: VL={float(low_voltage):.3f} V ({float(low_value):.1f}), "
                    f"VH={float(high_voltage):.3f} V ({float(high_value):.1f}), "
                    f"RX threshold={int(applied_threshold)}."
                )

        except Exception as exc:
            if hasattr(self, "tx_log_print"):
                self.tx_log_print(f"[TX] TX level sweep failed: {exc}")
            messagebox.showerror("Set TX Voltages failed", str(exc), parent=self.tx_win)
            return
        finally:
            try:
                if not was_streaming:
                    self.rx_stop_stream_blocking(timeout_s=1.0)
                else:
                    with self.rx_serial_lock:
                        try:
                            self.rx.reset_input_buffer()
                        except Exception:
                            pass
            except Exception as exc:
                if hasattr(self, "tx_log_print"):
                    self.tx_log_print(f"[RX] Stream cleanup warning: {exc}")

            self.rx_polling = previous_rx_polling
            self.rx_reset_runtime_state()

    def _tx_scan_2v_plot_extent(self, v1, v2):
        def _axis_bounds(axis_values):
            squared = np.asarray(axis_values, dtype=np.float64) ** 2
            lower = float(squared[0])
            upper = float(squared[-1])
            if lower == upper:
                pad = max(abs(lower) * 0.01, 0.01)
                return lower - pad, upper + pad
            return lower, upper

        x0, x1 = _axis_bounds(v1)
        y0, y1 = _axis_bounds(v2)
        return [x0, x1, y0, y1]

    def _tx_scan_2v_1d_axis_and_values(self, data, values):
        v1 = np.asarray(data["V1_axis"], dtype=np.float64)
        v2 = np.asarray(data["V2_axis"], dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)

        if len(v1) == 1 and len(v2) == 1:
            return np.square(v1), values.reshape(-1), str(data["name1"])
        if len(v1) == 1:
            return np.square(v2), values[:, 0], str(data["name2"])
        if len(v2) == 1:
            return np.square(v1), values[0, :], str(data["name1"])
        raise ValueError("Scan 2V data is not one-dimensional.")

    def _tx_fit_scan_2v_sinusoid(self, x, y, initial_half_period=300.0):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if len(x) < 4 or np.ptp(x) <= 0:
            return None

        def sinusoid(x_values, offset, amplitude, phase, half_period):
            return offset + amplitude * np.sin(np.pi * x_values / half_period + phase)

        try:
            from scipy.optimize import curve_fit
        except Exception as exc:
            if hasattr(self, "tx_log_print") and not getattr(
                self, "_tx_scan_2v_scipy_fit_unavailable_logged", False
            ):
                self.tx_log_print(f"[TX] Scan 2V sinusoid fit skipped: scipy unavailable ({exc})")
                self._tx_scan_2v_scipy_fit_unavailable_logged = True
            return None

        offset0 = float(np.mean(y))
        amplitude0 = float((np.max(y) - np.min(y)) / 2.0)
        if amplitude0 == 0.0:
            amplitude0 = 1.0
        p0 = [offset0, amplitude0, 0.0, float(initial_half_period)]
        lower_bounds = [-np.inf, -np.inf, -2.0 * np.pi, 1e-9]
        upper_bounds = [np.inf, np.inf, 2.0 * np.pi, np.inf]

        try:
            params, _ = curve_fit(
                sinusoid,
                x,
                y,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=20000,
            )
        except Exception as exc:
            if hasattr(self, "tx_log_print"):
                self.tx_log_print(f"[TX] Scan 2V sinusoid fit failed: {exc}")
            return None

        return sinusoid, params

    def _tx_plot_scan_2v_1d_series(self, data, series, title, ylabel):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        xlabel = None
        for label, values in series:
            x, y, current_xlabel = self._tx_scan_2v_1d_axis_and_values(data, values)
            xlabel = current_xlabel if xlabel is None else xlabel
            (line,) = ax.plot(x, y, marker="o", linestyle="-", label=f"{label} data")

            fit = self._tx_fit_scan_2v_sinusoid(x, y)
            if fit is not None:
                sinusoid, params = fit
                x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 500)
                y_fit = sinusoid(x_fit, *params)
                half_period = float(params[3])
                ax.plot(
                    x_fit,
                    y_fit,
                    linestyle="--",
                    color=line.get_color(),
                    label=f"{label} fit, half-period={half_period:.3g}",
                )
                if hasattr(self, "tx_log_print"):
                    self.tx_log_print(
                        f"[TX] Scan 2V fit {label}: half-period={half_period:.6g}"
                    )

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        ax.legend()
        fig.tight_layout()

    def _tx_plot_scan_2v_1d_curve(self, data, values, title, ylabel):
        self._tx_plot_scan_2v_1d_series(data, [(title, values)], title, ylabel)

    def _tx_plot_scan_2v_1d(self, data, pd_sum, pd_diff):
        pd1_label = f"PD{data['pd1']}"
        pd2_label = f"PD{data['pd2']}"

        self._tx_plot_scan_2v_1d_series(
            data,
            [(pd1_label, data["PD1"]), (pd2_label, data["PD2"])],
            f"{pd1_label} and {pd2_label}",
            "PD counts",
        )
        self._tx_plot_scan_2v_1d_curve(
            data,
            pd_sum,
            f"{pd1_label} + {pd2_label}",
            f"{pd1_label} + {pd2_label}",
        )
        self._tx_plot_scan_2v_1d_curve(
            data,
            pd_diff,
            f"{pd1_label} - {pd2_label}",
            f"{pd1_label} - {pd2_label}",
        )
        plt.show()

    def _tx_plot_scan_2v(self, data):
        pd_sum = data["PD1"] + data["PD2"]
        pd_diff = np.divide(
            data["PD1"] - data["PD2"],
            pd_sum,
            out=np.zeros_like(pd_sum, dtype=np.float64),
            where=pd_sum != 0,
        )

        v1 = data["V1_axis"]
        v2 = data["V2_axis"]
        if len(v1) == 1 or len(v2) == 1:
            self._tx_plot_scan_2v_1d(data, pd_sum, pd_diff)
            return

        extent = self._tx_scan_2v_plot_extent(v1, v2)
        pd1_label = f"PD{data['pd1']}"
        pd2_label = f"PD{data['pd2']}"

        plt.figure()
        plt.title(pd1_label)
        image = plt.imshow(data["PD1"], origin="lower", aspect="auto", extent=extent)
        plt.colorbar(image, label=pd1_label)
        plt.xlabel(data["name1"])
        plt.ylabel(data["name2"])

        plt.figure()
        plt.title(pd2_label)
        image = plt.imshow(data["PD2"], origin="lower", aspect="auto", extent=extent)
        plt.colorbar(image, label=pd2_label)
        plt.xlabel(data["name1"])
        plt.ylabel(data["name2"])

        plt.figure()
        plt.title(f"{pd1_label} + {pd2_label}")
        image = plt.imshow(pd_sum, origin="lower", aspect="auto", extent=extent)
        plt.colorbar(image, label=f"{pd1_label} + {pd2_label}")
        plt.xlabel(data["name1"])
        plt.ylabel(data["name2"])

        plt.figure()
        plt.title(f"{pd1_label} - {pd2_label}")
        image = plt.imshow(pd_diff, origin="lower", aspect="auto", extent=extent)
        plt.colorbar(image, label=f"{pd1_label} - {pd2_label}")
        plt.xlabel(data["name1"])
        plt.ylabel(data["name2"])

        plt.show()

    def _tx_save_scan_2v(self, data, filepath=None):
        pd1 = np.asarray(data["PD1"], dtype=np.float64)
        pd2 = np.asarray(data["PD2"], dtype=np.float64)

        pd_sum = pd1 + pd2
        pd_diff = np.divide(
            pd1 - pd2,
            pd_sum,
            out=np.zeros_like(pd_sum, dtype=np.float64),
            where=pd_sum != 0,
        )

        if filepath is None:
            default_name = f"scan2v_{data['name1']}_{data['name2']}.npz"
            filepath = filedialog.asksaveasfilename(
                parent=self.tx_win,
                title="Save Scan 2V Result",
                defaultextension=".npz",
                initialfile=default_name,
                filetypes=[("NumPy compressed", "*.npz")],
            )
            if not filepath:
                if hasattr(self, "tx_log_print"):
                    self.tx_log_print("[TX] Save cancelled.")
                return

        np.savez_compressed(
            filepath,
            PD1=pd1,
            PD2=pd2,
            PDsum=pd_sum,
            PDdiff=pd_diff,
            V1_axis=np.asarray(data["V1_axis"], dtype=np.float64),
            V2_axis=np.asarray(data["V2_axis"], dtype=np.float64),
            pd1=int(data["pd1"]),
            pd2=int(data["pd2"]),
            name1=str(data["name1"]),
            name2=str(data["name2"]),
            background_offset_pd1=float(data.get("background_offset_pd1", 0.0)),
            background_offset_pd2=float(data.get("background_offset_pd2", 0.0)),
            background_offsets=np.asarray(
                [
                    float(data.get("background_offsets", {}).get(pd_index, 0.0))
                    for pd_index in sorted(getattr(self, "rx_pd_to_channel", {}))
                ],
                dtype=np.float64,
            ),
        )

        if hasattr(self, "tx_log_print"):
            self.tx_log_print(f"[TX] Saved: {filepath}")
