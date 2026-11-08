import time

import numpy as np
import serial


def open_arduino(port, baud=115200, to=0.2):
    return serial.Serial(port=port, baudrate=baud, timeout=to)



def readline_str(ser):
    try:
        return ser.readline().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def write_line(ser, text):
    if not text.endswith("\n"):
        text += "\n"
    ser.write(text.encode("utf-8"))


def write_float_line(ser, value):
    ser.write(f"{float(value)}\r\n".encode("utf-8"))
    time.sleep(0.02)


def vsq_linspace(v_start: float, v_stop: float, count: int) -> np.ndarray:
    v_start = float(v_start)
    v_stop = float(v_stop)
    count = int(count)
    if count < 1:
        raise ValueError("N must be >= 1")
    if count == 1:
        return np.asarray([v_start], dtype=np.float64)

    start_squared = v_start * v_start
    stop_squared = v_stop * v_stop
    squared_values = np.linspace(start_squared, stop_squared, count)
    return np.sqrt(np.clip(squared_values, 0.0, None))
