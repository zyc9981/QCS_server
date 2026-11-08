import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---- user options ----
USE_NORMALIZED_DIFF = False   # True -> (PDx - PDy)/(PDx + PDy)
EPS = 1e-12                   # avoid divide-by-zero
VIEW_ELEV = 30
VIEW_AZIM = -60
DOWNSAMPLE_STRIDE = 1         # set to 2 or 3 if arrays are large and plotting is slow


_SCIPY_FIT_UNAVAILABLE_LOGGED = False


def pick_npz_file(parent=None) -> str:
    owns_root = parent is None
    root = Tk() if owns_root else parent
    if owns_root:
        root.withdraw()
        root.attributes("-topmost", True)
    path = askopenfilename(
        parent=root,
        title="Select an NPZ file",
        filetypes=[("NumPy NPZ", "*.npz"), ("All files", "*.*")]
    )
    if owns_root:
        root.destroy()
    return path


def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    print(f"\nLoaded: {path}")
    print("Keys:", data.files)
    return data


def _ensure_shape(Z, expected):
    """If Z is transposed, fix it. Otherwise error."""
    if Z.shape == expected:
        return Z
    if Z.T.shape == expected:
        return Z.T
    raise ValueError(f"Shape mismatch: expected {expected} (or transpose), got {Z.shape}")


def _scalar_from_npz(d, key, default):
    if key not in d.files:
        return default
    value = d[key]
    if isinstance(value, np.ndarray):
        return value.item()
    return value


def get_axis_and_data(d):
    # Axis keys
    if "V1_axis" not in d.files or "V2_axis" not in d.files:
        raise KeyError("NPZ must contain V1_axis and V2_axis")

    v1 = np.array(d["V1_axis"], dtype=float)  # (n1,)
    v2 = np.array(d["V2_axis"], dtype=float)  # (n2,)

    expected = (len(v2), len(v1))  # (n2, n1)
    pd1_label = f"PD{int(_scalar_from_npz(d, 'pd1', 1))}"
    pd2_label = f"PD{int(_scalar_from_npz(d, 'pd2', 2))}"

    # Data keys
    if "PDsum" in d.files and "PDdiff" in d.files:
        PDsum = np.array(d["PDsum"], dtype=float)
        PDdiff = np.array(d["PDdiff"], dtype=float)
        label_sum = f"{pd1_label} + {pd2_label}"
        label_diff = f"{pd1_label} - {pd2_label}"
    elif "PD3" in d.files and "PD4" in d.files:
        PD3 = np.array(d["PD3"], dtype=float)
        PD4 = np.array(d["PD4"], dtype=float)
        PDsum = PD3 + PD4
        PDdiff = PD3 - PD4
        label_sum = "PD3 + PD4"
        label_diff = "PD3 - PD4"
    elif "PD1" in d.files and "PD2" in d.files:
        PD1 = np.array(d["PD1"], dtype=float)
        PD2 = np.array(d["PD2"], dtype=float)
        PDsum = PD1 + PD2
        PDdiff = PD1 - PD2
        label_sum = "PD1 + PD2"
        label_diff = "PD1 - PD2"
    else:
        raise KeyError("NPZ must contain (Zsum & Zdiff) OR (PD3 & PD4) OR (PD1 & PD2)")

    # Fix shape (handle transpose automatically)
    PDsum = _ensure_shape(PDsum, expected)
    PDdiff = _ensure_shape(PDdiff, expected)

    # Optional normalized diff
    if USE_NORMALIZED_DIFF:
        PDdiff = PDdiff / (PDsum + EPS)
        label_diff = f"({label_diff}) / ({label_sum})"

    # Optional labels
    name1 = d["name1"] if "name1" in d.files else "V1"
    name2 = d["name2"] if "name2" in d.files else "V2"
    if isinstance(name1, np.ndarray):
        name1 = name1.item()
    if isinstance(name2, np.ndarray):
        name2 = name2.item()
    name1, name2 = str(name1), str(name2)

    return v1, v2, PDsum, PDdiff, name1, name2, label_sum, label_diff


def get_individual_port_data(d, v1, v2):
    expected = (len(v2), len(v1))
    pd1_label = f"PD{int(_scalar_from_npz(d, 'pd1', 1))}"
    pd2_label = f"PD{int(_scalar_from_npz(d, 'pd2', 2))}"

    if "PD1" in d.files and "PD2" in d.files:
        return [
            (pd1_label, _ensure_shape(np.array(d["PD1"], dtype=float), expected)),
            (pd2_label, _ensure_shape(np.array(d["PD2"], dtype=float), expected)),
        ]
    if "PD3" in d.files and "PD4" in d.files:
        return [
            ("PD3", _ensure_shape(np.array(d["PD3"], dtype=float), expected)),
            ("PD4", _ensure_shape(np.array(d["PD4"], dtype=float), expected)),
        ]
    return []


def _axis_extent(axis_values):
    squared = np.asarray(axis_values, dtype=float) ** 2
    lower = float(squared[0])
    upper = float(squared[-1])
    if lower == upper:
        pad = max(abs(lower) * 0.01, 0.01)
        return lower - pad, upper + pad
    return lower, upper


def plot_3d_surfaces(v1, v2, Zsum, Zdiff, name1, name2, label_sum, label_diff, port_data=None):
    v1_plot = np.square(v1)
    v2_plot = np.square(v2)

    # Downsample for speed if desired
    s = max(int(DOWNSAMPLE_STRIDE), 1)
    v1s = v1_plot[::s]
    v2s = v2_plot[::s]
    Zs_sum = Zsum[::s, ::s]
    Zs_diff = Zdiff[::s, ::s]

    V1, V2 = np.meshgrid(v1s, v2s)  # (n2,n1)

    x0, x1 = _axis_extent(v1)
    y0, y1 = _axis_extent(v2)
    extent = [x0, x1, y0, y1]

    for label, values in port_data or []:
        plt.figure()
        plt.title(label)
        image = plt.imshow(values, origin="lower", aspect="auto", extent=extent)
        plt.colorbar(image, label=label)
        plt.xlabel(name1)
        plt.ylabel(name2)

    plt.figure()
    plt.title(label_sum)
    image = plt.imshow(Zsum, origin="lower", aspect="auto", extent=extent)
    plt.colorbar(image, label=label_sum)
    plt.xlabel(name1)
    plt.ylabel(name2)

    plt.figure()
    plt.title(label_diff)
    image = plt.imshow(Zdiff, origin="lower", aspect="auto", extent=extent)
    plt.colorbar(image, label=label_diff)
    plt.xlabel(name1)
    plt.ylabel(name2)

    plt.show()


def get_1d_axis_and_values(v1, v2, values, name1, name2):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(v1) == 1 and len(v2) == 1:
        return np.square(v1), values.reshape(-1), name1
    if len(v1) == 1:
        return np.square(v2), values[:, 0], name2
    if len(v2) == 1:
        return np.square(v1), values[0, :], name1
    raise ValueError("Data is not one-dimensional.")


def fit_sinusoid(x, y, log_print=print, initial_half_period=300.0):
    global _SCIPY_FIT_UNAVAILABLE_LOGGED

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
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
        if not _SCIPY_FIT_UNAVAILABLE_LOGGED:
            log_print(f"Sinusoid fit skipped: scipy unavailable ({exc})")
            _SCIPY_FIT_UNAVAILABLE_LOGGED = True
        return None

    offset0 = float(np.mean(y))
    amplitude0 = float((np.max(y) - np.min(y)) / 2.0)
    if amplitude0 == 0.0:
        amplitude0 = 1.0

    try:
        params, _ = curve_fit(
            sinusoid,
            x,
            y,
            p0=[offset0, amplitude0, 0.0, float(initial_half_period)],
            bounds=([-np.inf, -np.inf, -2.0 * np.pi, 1e-9], [np.inf, np.inf, 2.0 * np.pi, np.inf]),
            maxfev=20000,
        )
    except Exception as exc:
        log_print(f"Sinusoid fit failed: {exc}")
        return None

    return sinusoid, params


def plot_1d_series(v1, v2, series, name1, name2, title, ylabel, log_print=print):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    xlabel = None
    for label, values in series:
        x, y, xlabel = get_1d_axis_and_values(v1, v2, values, name1, name2)
        (line,) = ax.plot(x, y, marker="o", linestyle="-", label=f"{label} data")

        fit = fit_sinusoid(x, y, log_print=log_print)
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
            log_print(f"Fit {label}: half-period={half_period:.6g}")

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.legend()
    fig.tight_layout()


def plot_1d_curves(v1, v2, Zsum, Zdiff, name1, name2, label_sum, label_diff, port_data=None, log_print=print):
    if port_data:
        plot_1d_series(v1, v2, port_data, name1, name2, "Individual ports", "Counts", log_print=log_print)

    plot_1d_series(v1, v2, [(label_sum, Zsum)], name1, name2, label_sum, label_sum, log_print=log_print)
    plot_1d_series(v1, v2, [(label_diff, Zdiff)], name1, name2, label_diff, label_diff, log_print=log_print)

    plt.show()


def run_scan_2v_analyzer(parent=None, log_print=print):
    path = pick_npz_file(parent=parent)
    if not path:
        log_print("No file selected.")
        return

    try:
        d = load_npz(path)
        v1, v2, PDsum, PDdiff, name1, name2, label_sum, label_diff = get_axis_and_data(d)
        port_data = get_individual_port_data(d, v1, v2)

        max_arr = np.max(PDdiff, axis=0)
        min_arr = np.min(PDdiff, axis=0)
        max_mean = np.mean(max_arr)
        min_mean = np.mean(min_arr)
        log_print(f"Loaded: {path}")
        log_print(f"Mean max/min of diff: {max_mean}, {min_mean}")

        if len(v1) == 1 or len(v2) == 1:
            plot_1d_curves(
                v1,
                v2,
                PDsum,
                PDdiff,
                name1,
                name2,
                label_sum,
                label_diff,
                port_data=port_data,
                log_print=log_print,
            )
        else:
            plot_3d_surfaces(v1, v2, PDsum, PDdiff, name1, name2, label_sum, label_diff, port_data=port_data)
    except Exception as exc:
        if parent is not None:
            messagebox.showerror("2V Analyzer", str(exc), parent=parent)
        else:
            raise


if __name__ == "__main__":
    run_scan_2v_analyzer()
