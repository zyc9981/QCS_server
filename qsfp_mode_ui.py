"""Shared Tk controls; hardware work runs outside the Tk event loop."""
import json
import queue
import threading
from pathlib import Path
from tkinter import BooleanVar, StringVar, Checkbutton, Label, Entry
from tkinter.ttk import Combobox

from qsfp_mode_control import QSFPModeControl, guard_ticks

SETTINGS = Path(__file__).with_name("qsfp_mode_settings.json")


class QSFPModeUI:
    def __init__(self, app):
        self.app = app
        try:
            settings = json.loads(SETTINGS.read_text())
            mode = settings["mode"]
            guard_ticks(settings["guard_ms"])
            if mode not in ("Continuous", "Burst"):
                raise ValueError("Invalid saved mode")
            guard = str(settings["guard_ms"])
        except (OSError, ValueError, KeyError, TypeError):
            mode, guard = "Continuous", "15.0"
        self.enabled = BooleanVar(master=app.root, value=False)
        self.mode = StringVar(master=app.root, value=mode)
        self.guard = StringVar(master=app.root, value=guard)
        self.status = StringVar(master=app.root, value="QSFP disabled")
        self.control = None
        self.applied = None
        self.applying = False
        self.messages = queue.Queue()
        self.panels = []
        self.closed = False
        app.root.after(100, self.poll)

    def add_panel(self, parent):
        colors = self.app.ui
        bg = parent.cget("bg")
        label_style = dict(bg=bg, fg=colors["text"])
        checkbox = Checkbutton(parent, text="QSFP", variable=self.enabled,
                              command=self.apply, selectcolor=colors["surface"],
                              activebackground=bg, activeforeground=colors["text"],
                              highlightthickness=0, **label_style)
        checkbox.pack(side="left")
        Label(parent, text="Mode:", **label_style).pack(side="left", padx=(12, 4))
        mode = Combobox(parent, textvariable=self.mode, values=("Continuous", "Burst"),
                        state="readonly", width=12)
        mode.pack(side="left")
        mode.bind("<<ComboboxSelected>>", lambda event: self.apply())
        Label(parent, text="Guard:", **label_style).pack(side="left", padx=(12, 4))
        guard = Entry(parent, textvariable=self.guard, width=7)
        guard.pack(side="left")
        guard.bind("<Return>", lambda event: self.apply())
        guard.bind("<FocusOut>", lambda event: self.apply())
        Label(parent, text="ms", **label_style).pack(side="left", padx=(4, 0))
        self.panels.append((checkbox, mode, guard))
        self.refresh()
        return checkbox

    def busy(self):
        return any(bool(getattr(self.app, name, False)) for name in (
            "qutag_timestamp_writing", "qutag_timestamp_preparing_hydraharp",
            "qutag_timestamp_qsfp_active", "optical_capture_active"))

    def refresh(self):
        locked = self.busy() or self.applying or self.closed
        living = []
        for checkbox, mode, guard in self.panels:
            if not checkbox.winfo_exists():
                continue
            living.append((checkbox, mode, guard))
            checkbox.configure(state="disabled" if locked else "normal")
            mode.configure(state="disabled" if locked else "readonly")
            guard.configure(state="normal" if not locked and self.mode.get() == "Burst" else "disabled")
        self.panels = living

    def apply(self):
        if self.busy() or self.applying or self.closed:
            self.refresh()
            return
        try:
            guard_ticks(self.guard.get())
            choice = (self.enabled.get(), self.mode.get().lower(), float(self.guard.get()))
            if choice[1] not in ("continuous", "burst"):
                raise ValueError("Select Continuous or Burst")
            temporary = SETTINGS.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({"mode": self.mode.get(), "guard_ms": choice[2]}, indent=2) + "\n")
            temporary.replace(SETTINGS)
        except (OSError, ValueError, OverflowError) as exc:
            self.status.set(f"QSFP settings error: {exc}")
            self.refresh()
            return
        if choice == self.applied:
            self.refresh()
            return
        if not choice[0] and self.control is None:
            self.applied = choice
            self.status.set("QSFP disabled")
            self.refresh()
            return
        self.applying = True
        self.status.set("Connecting QSFP controls..." if self.control is None else "Applying QSFP mode...")
        self.refresh()

        def work():
            try:
                if self.control is None:
                    self.control = QSFPModeControl()
                self.control.idle(choice[1] if choice[0] else "off")
                self.messages.put(("applied", choice))
            except Exception as exc:
                self.messages.put(("error", str(exc)))
        threading.Thread(target=work, name="qsfp-mode-control", daemon=True).start()

    def capture_options(self):
        guard_ticks(self.guard.get())
        choice = (self.enabled.get(), self.mode.get().lower(), float(self.guard.get()))
        if self.applying:
            raise RuntimeError("QSFP controls are connecting; wait for the status to finish")
        if not choice[0]:
            raise RuntimeError("Enable QSFP first")
        if self.control is None or choice != self.applied:
            self.apply()
            raise RuntimeError("QSFP settings are being applied; retry after the status completes")
        return dict(mode=choice[1], guard_ms=choice[2], control=self.control,
                    status=self.report)

    def report(self, text):
        self.messages.put(("status", text))

    def poll(self):
        if self.closed:
            return
        while True:
            try:
                event, value = self.messages.get_nowait()
            except queue.Empty:
                break
            if event == "applied":
                self.applying = False
                self.applied = value
                self.status.set("Laser on · Continuous" if value[0] and value[1] == "continuous"
                                else "Laser off · Burst ready" if value[0] else "QSFP disabled · Laser off")
            elif event == "error":
                self.applying = False
                self.applied = None
                self.enabled.set(False)
                self.status.set(f"QSFP error: {value.splitlines()[0]} — see TX log")
                self.app.tx_log_print(f"[QSFP] {value}")
                # Recreate a failed connection on the next enable attempt.
                failed_control, self.control = self.control, None
                if failed_control is not None:
                    threading.Thread(target=failed_control.close, daemon=True).start()
            else:
                self.status.set(value)
        self.refresh()
        self.app.root.after(100, self.poll)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.control is not None:
            self.control.close()


def get_qsfp_modes(app):
    if not hasattr(app, "qsfp_modes"):
        app.qsfp_modes = QSFPModeUI(app)
    return app.qsfp_modes
