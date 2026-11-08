"""Persistent Vivado control of the existing simplex burst/continuous image.

DMA data still travels through optical_stream. This connection controls only
the module laser, payload gate, and external FIFOs; it never programs the FPGA.
"""
from __future__ import annotations

import base64
import fcntl
import math
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "FPGA_HH_QSFP/controller_modes_20261018"
DEFAULT_VIVADO = "/home/alzhu/vivado_dl/2025.2/Vivado/bin/vivado"


def guard_ticks(guard_ms):
    value = float(guard_ms)
    if not math.isfinite(value) or not 0 <= value <= 1000:
        raise ValueError("QSFP guard time must be between 0 and 1000 ms")
    return round(value * 200_000)


class QSFPModeControl:
    """One hardware owner, shared by the controller's two acquisition windows."""

    def __init__(self):
        self._lock = threading.RLock()
        self._process = self._socket = self._reader = self._temp = None
        self._log = None
        self._lease = None
        self.log_path = None

    def _start(self):
        if self._process is not None:
            if self._process.poll() is not None:
                raise RuntimeError(f"QSFP control exited; see {self.log_path}")
            return
        vivado = os.environ.get("QSFP_VIVADO", DEFAULT_VIVADO)
        probes = Path(os.environ.get("QSFP_PROBES_FILE", IMAGE_DIR / "hh_qsfp_top.ltx"))
        if not probes.is_file():
            raise FileNotFoundError(f"QSFP mode probes missing: {probes}. See QSFP_MODES.md")
        self._temp = tempfile.TemporaryDirectory(prefix="qcs_qsfp_")
        ready = Path(self._temp.name) / "ready"
        log_dir = Path.home() / ".cache" / "qcs" / "qsfp"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._lease = (log_dir / "control.lock").open("a+")
        try:
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lease.close()
            self._lease = None
            raise RuntimeError("Another controller or pattern test owns the QSFP mode controls") from None
        self.log_path = log_dir / f"control_{time.time_ns()}_{os.getpid()}.log"
        self._log = self.log_path.open("w")
        env = dict(os.environ)
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        try:
            self._process = subprocess.Popen(
                [vivado, "-mode", "batch", "-nojournal", "-nolog", "-source",
                 str(Path(__file__).with_suffix(".tcl")), "-tclargs", str(probes),
                 str(ready), str(os.getpid())],
                cwd=self._temp.name, env=env, stdin=subprocess.DEVNULL,
                stdout=self._log, stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 60
            while not ready.exists():
                if self._process.poll() is not None:
                    self._log.flush()
                    detail = self.log_path.read_text(errors="replace")[-2200:]
                    raise RuntimeError("Cannot attach QSFP mode controls. Load the controller "
                                       f"mode image once (QSFP_MODES.md).\n{detail}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Vivado attachment timed out; see {self.log_path}")
                time.sleep(0.05)
            self._socket = socket.create_connection(("127.0.0.1", int(ready.read_text())), timeout=30)
            self._socket.settimeout(30)
            self._reader = self._socket.makefile("rb")
        except BaseException:
            # Retain the diagnostic log even when initialization fails.
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            if self._lease is not None:
                self._lease.close()
                self._lease = None
            raise

    def _command(self, *words):
        with self._lock:
            self._start()
            self._socket.sendall((" ".join(str(w) for w in words) + "\n").encode("ascii"))
            response = self._reader.readline().decode("ascii").strip()
            if response.startswith("ERR "):
                raise RuntimeError(base64.b64decode(response[4:]).decode("utf-8"))
            if not response.startswith("OK "):
                raise RuntimeError(f"QSFP control disconnected; see {self.log_path}")
            return response[3:]

    def idle(self, mode):
        if mode not in ("burst", "continuous", "off"):
            raise ValueError(f"Unknown QSFP mode: {mode}")
        return self._command("idle", mode)

    def prepare(self, mode, guard_ms):
        if mode not in ("burst", "continuous"):
            raise ValueError(f"Unknown QSFP mode: {mode}")
        return self._command("prepare", mode, guard_ticks(guard_ms) if mode == "burst" else 0)

    def start(self):
        """Called after the first DMA packet is submitted, before waiting for RX."""
        return self._command("start")

    def finish(self, mode):
        return self._command("finish", mode)

    def snapshot(self):
        return int(self._command("status"), 16)

    def times(self):
        """Eight init-clock timestamps; origin is the TX-enable STOP."""
        value = int(self._command("times"), 16)
        return [(value >> (32 * i)) & 0xFFFFFFFF for i in range(8)]

    def close(self):
        with self._lock:
            try:
                if self._socket is not None and self._process.poll() is None:
                    self._command("quit")
            finally:
                if self._reader is not None:
                    self._reader.close()
                if self._socket is not None:
                    self._socket.close()
                if self._process is not None and self._process.poll() is None:
                    try:
                        self._process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        self._process.terminate()
                        self._process.wait(timeout=5)
                if self._log is not None:
                    self._log.close()
                self._reader = self._socket = self._process = None
                if self._temp is not None:
                    self._temp.cleanup()
                    self._temp = None
                if self._lease is not None:
                    self._lease.close()
                    self._lease = None

