"""Small DLC pro control window with keyboard digit selection."""

import argparse
from decimal import Decimal, InvalidOperation
import tkinter as tk
from tkinter import ttk

DEFAULT_HOST = "192.168.1.200"

# Power uses the DLC pro power-stabilization setpoint. Its unit and available
# range depend on the laser's configuration, so the GUI does not assume either.
CONTROLS = (
    ("wavelength", "Wavelength", "nm", 3, "laser1:ctl:wavelength-set", "set_wavelength_nm"),
    ("piezo", "Piezo voltage", "V", 3, "laser1:dl:pc:voltage-set", "set_piezo_V"),
    ("power", "Power setpoint", "device units", 2, "laser1:power-stabilization:setpoint", "set_power"),
)


class CTLTopticaControl(tk.Tk):
    def __init__(self, host: str):
        super().__init__()
        self.title("TOPTICA DLC pro control")
        self.resizable(False, False)
        self.laser = None
        self.fields = {}

        frame = ttk.Frame(self, padding=16)
        frame.grid(sticky="nsew")

        ttk.Label(frame, text="DLC pro address").grid(row=0, column=0, sticky="w")
        self.host_var = tk.StringVar(value=host)
        ttk.Entry(frame, textvariable=self.host_var, width=23).grid(row=0, column=1, columnspan=2, sticky="ew", padx=(8, 8))
        ttk.Button(frame, text="Connect / refresh", command=self.connect).grid(row=0, column=3, columnspan=2, sticky="ew")

        ttk.Label(frame, text="Setpoint").grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(14, 0))
        ttk.Label(frame, text="Selected step").grid(row=1, column=2, sticky="w", padx=(8, 0), pady=(14, 0))

        for row, (key, label, unit, precision, parameter, setter) in enumerate(CONTROLS, start=2):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=5)
            value_var = tk.StringVar()
            place_var = tk.StringVar(value=f"{Decimal(1).scaleb(-precision):f}")
            entry = ttk.Entry(frame, textvariable=value_var, width=16, state="disabled")
            entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=5)
            entry.bind("<Left>", lambda event, name=key: self._move_digit(name, 1))
            entry.bind("<Right>", lambda event, name=key: self._move_digit(name, -1))
            entry.bind("<Up>", lambda event, name=key: self._arrow(event, name, 1))
            entry.bind("<Down>", lambda event, name=key: self._arrow(event, name, -1))
            entry.bind("<Return>", lambda event, name=key: self._enter(event, name))
            entry.bind("<ButtonRelease-1>", lambda event, name=key: self.after_idle(self._select_clicked_digit, name))
            ttk.Label(frame, textvariable=place_var, width=10).grid(row=row, column=2, sticky="w", padx=(8, 0), pady=5)

            down = ttk.Button(frame, text="▼", width=3, state="disabled", command=lambda name=key: self.adjust(name, -1))
            down.grid(row=row, column=3, padx=(8, 0), pady=5)
            up = ttk.Button(frame, text="▲", width=3, state="disabled", command=lambda name=key: self.adjust(name, 1))
            up.grid(row=row, column=4, padx=(4, 0), pady=5)
            ttk.Label(frame, text=unit).grid(row=row, column=5, sticky="w", padx=(8, 0))

            self.fields[key] = {
                "label": label,
                "parameter": parameter,
                "setter": setter,
                "precision": precision,
                "value": value_var,
                "display": "",
                "selected_exponent": -precision,
                "place": place_var,
                "entry": entry,
                "buttons": (down, up),
            }

        ttk.Label(frame, text="Click a digit or use ←/→ to select it; ↑/↓ changes it immediately. Type a value and press Enter.").grid(
            row=5, column=0, columnspan=6, sticky="w", pady=(12, 0)
        )
        self.status_var = tk.StringVar(value="Connecting…")
        ttk.Label(frame, textvariable=self.status_var, wraplength=510).grid(
            row=6, column=0, columnspan=6, sticky="w", pady=(8, 0)
        )

        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(100, self.connect)

    def connect(self):
        if self.laser is not None:
            try:
                self.laser.close()
            finally:
                self.laser = None
        self._enable_controls(False)
        host = self.host_var.get().strip()
        if not host:
            self.status_var.set("Enter the DLC pro address.")
            return

        self.status_var.set(f"Connecting to {host}…")
        self.update_idletasks()
        try:
            from automated_calibration.remote_laser_control import DLCProSimple

            self.laser = DLCProSimple(host).open()
        except Exception as exc:
            self.laser = None
            self.status_var.set(f"Connection failed: {exc}")
            return

        self._enable_controls(True)
        unreadable = []
        for field in self.fields.values():
            field["selected_exponent"] = -field["precision"]
            try:
                setpoint = Decimal(str(self.laser.client.get(field["parameter"])))
                field["display"] = self._format_value(field, setpoint)
                field["value"].set(field["display"])
            except Exception:
                field["display"] = ""
                field["value"].set("")
                unreadable.append(field["label"])
            self._show_selected_digit(field)

        if unreadable:
            self.status_var.set(
                f"Connected to {host}. Could not read: {', '.join(unreadable)}. "
                "You can type a setpoint and press Enter."
            )
        else:
            self.status_var.set(f"Connected to {host}. Setpoints loaded.")
        self.fields["wavelength"]["entry"].focus_set()
        self._show_selected_digit(self.fields["wavelength"])

    def _enable_controls(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for field in self.fields.values():
            field["entry"].configure(state=state)
            for button in field["buttons"]:
                button.configure(state=state)

    def _arrow(self, event, key: str, direction: int):
        self.adjust(key, direction)
        return "break"

    def _enter(self, event, key: str):
        self.send_value(key)
        return "break"

    @staticmethod
    def _format_value(field, value: Decimal) -> str:
        if not value.is_finite():
            raise ValueError("setpoint must be finite")
        precision = field["precision"]
        quantum = Decimal(1).scaleb(-precision)
        return format(value.quantize(quantum), f".{precision}f")

    def _show_selected_digit(self, field):
        exponent = field["selected_exponent"]
        text = field["display"]
        if not text or field["value"].get() != text:
            field["place"].set(f"{Decimal(1).scaleb(exponent):f}")
            return
        decimal_index = text.index(".")
        max_exponent = len(text[:decimal_index].lstrip("-")) - 1
        exponent = min(max(exponent, -field["precision"]), max_exponent)
        field["selected_exponent"] = exponent
        field["place"].set(f"{Decimal(1).scaleb(exponent):f}")
        position = decimal_index - 1 - exponent if exponent >= 0 else decimal_index - exponent
        field["entry"].icursor(position)
        field["entry"].selection_range(position, position + 1)

    def _move_digit(self, key: str, direction: int):
        field = self.fields[key]
        if not field["display"] or field["value"].get() != field["display"]:
            return None  # Allow normal cursor movement while editing a typed value.
        text = field["display"]
        max_exponent = len(text.split(".", 1)[0].lstrip("-")) - 1
        field["selected_exponent"] = min(
            max(field["selected_exponent"] + direction, -field["precision"]),
            max_exponent,
        )
        self._show_selected_digit(field)
        return "break"

    def _select_clicked_digit(self, key: str):
        field = self.fields[key]
        text = field["display"]
        if not text or field["value"].get() != text:
            return
        position = min(field["entry"].index(tk.INSERT), len(text) - 1)
        decimal_index = text.index(".")
        if position == decimal_index:
            position += 1
        elif text[position] == "-":
            position += 1
        field["selected_exponent"] = (
            decimal_index - position - 1 if position < decimal_index else decimal_index - position
        )
        self._show_selected_digit(field)

    def adjust(self, key: str, direction: int):
        field = self.fields[key]
        if not field["display"] or field["value"].get() != field["display"]:
            self.status_var.set("Press Enter to send the typed value before adjusting a digit.")
            return
        try:
            current = Decimal(field["display"])
            step = Decimal(1).scaleb(field["selected_exponent"])
        except (InvalidOperation, ValueError):
            self.status_var.set(f"Enter a valid {field['label'].lower()} setpoint.")
            return
        self.send_value(key, current + direction * step)

    def send_value(self, key: str, value=None):
        field = self.fields[key]
        if self.laser is None:
            self.status_var.set("Connect to the DLC pro first.")
            return
        try:
            requested = Decimal(field["value"].get().strip()) if value is None else value
            formatted = self._format_value(field, requested)
            if requested != Decimal(formatted):
                raise ValueError(f"maximum {field['precision']} decimal places")
            getattr(self.laser, field["setter"])(float(requested))
        except (InvalidOperation, ValueError) as exc:
            self.status_var.set(f"Invalid {field['label'].lower()} setpoint: {exc}")
            return
        except Exception as exc:
            self.status_var.set(f"Could not set {field['label'].lower()}: {exc}")
            return

        field["display"] = formatted
        field["value"].set(formatted)
        field["entry"].focus_set()
        self._show_selected_digit(field)
        self.status_var.set(f"Sent {field['label'].lower()} setpoint: {formatted}")

    def close(self):
        if self.laser is not None:
            self.laser.close()
            self.laser = None
        self.destroy()


def main():
    parser = argparse.ArgumentParser(description="Control TOPTICA DLC pro setpoints")
    parser.add_argument("--host", default=DEFAULT_HOST, help="DLC pro IP address or host name")
    args = parser.parse_args()
    CTLTopticaControl(args.host).mainloop()


if __name__ == "__main__":
    main()
