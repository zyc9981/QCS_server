from qcs_analyzer._implementation import *
import numpy as np
import matplotlib.pyplot as plt


def allan_deviation_estimator_single_channel(
    t_ps,
    ch,
    ch_clk=3,
    nominal_frequency_hz=None,
    number_of_tau_points=30,
    *,
    show=True,
):
    """Measure periodic input stability relative to the raw tagger timebase.

    Timestamps are in picoseconds. Every selected event must represent one
    consecutive pulse of the same periodic train. If ``nominal_frequency_hz``
    is omitted, use the median measured pulse interval as the nominal period.
    Obvious missing/extra pulses are rejected, not interpolated. A fixed
    frequency offset is retained in the time-error series; no drift fit or
    clock correction is applied. The result includes tagger and input noise.
    """
    t_ps = np.asarray(t_ps).ravel()
    ch = np.asarray(ch).ravel()
    if len(t_ps) != len(ch):
        raise ValueError("t_ps and ch must have the same length.")
    if int(number_of_tau_points) != number_of_tau_points or number_of_tau_points < 1:
        raise ValueError("number_of_tau_points must be a positive integer.")

    timestamps = np.sort(t_ps[ch == ch_clk])
    if len(timestamps) < 6:
        raise ValueError("Need at least six pulses on the selected channel.")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("Selected timestamps must be finite.")

    # Difference before converting integer timestamps to float, so a large
    # absolute epoch does not discard picosecond interval information.
    intervals_ps = np.diff(timestamps).astype(np.float64)
    if not np.all(np.isfinite(intervals_ps)) or np.any(intervals_ps <= 0):
        raise ValueError("Selected pulses must have strictly increasing timestamps.")
    if nominal_frequency_hz is None:
        period_ps = float(np.median(intervals_ps))
    else:
        nominal_frequency_hz = float(nominal_frequency_hz)
        if not np.isfinite(nominal_frequency_hz) or nominal_frequency_hz <= 0:
            raise ValueError("nominal_frequency_hz must be finite and positive.")
        period_ps = 1e12 / nominal_frequency_hz
    if not np.isfinite(period_ps) or period_ps <= 0:
        raise ValueError("Nominal pulse period must be finite and positive.")
    if np.any(intervals_ps < 0.5 * period_ps) or np.any(intervals_ps > 1.5 * period_ps):
        raise ValueError(
            "Pulse intervals differ from the nominal period by more than 50%; "
            "check for missing/extra pulses or an incorrect nominal frequency."
        )

    # Equivalent to (t[n] - t[0]) - n*period, without subtracting large floats.
    time_error_ps = np.concatenate(([0.0], np.cumsum(intervals_ps - period_ps)))
    ideal_t_ps = np.arange(len(timestamps), dtype=np.float64) * period_ps
    result = allan_deviation_estimator_tau_input(
        time_error_ps,
        ideal_t_ps,
        number_of_tau_points=number_of_tau_points,
        show=show,
    )
    result["channel"] = ch_clk
    result["nominal_frequency_hz"] = 1e12 / period_ps
    return result


def allan_deviation_estimator(
    t_ps,
    ch,
    ch_clk_0=2,
    ch_clk_1=7,
    number_of_tau_points=30,
):
    t_ps = np.asarray(t_ps).ravel()
    ch = np.asarray(ch).ravel()

    if len(t_ps) == 0 or len(ch) == 0:
        raise ValueError("t_ps and ch must not be empty.")
    if len(t_ps) != len(ch):
        raise ValueError("t_ps and ch must have the same length.")
    if int(number_of_tau_points) < 1:
        raise ValueError("number_of_tau_points must be at least 1.")

    # Keep each channel chronological.
    t_clk_0 = np.sort(t_ps[ch == ch_clk_0])
    t_clk_1 = np.sort(t_ps[ch == ch_clk_1])

    if len(t_clk_0) == 0 or len(t_clk_1) == 0:
        raise ValueError("Both channels need at least one event.")

    N_samples = min(np.size(t_clk_0), np.size(t_clk_1))
    x_samples = t_clk_1[0:N_samples] - t_clk_0[0:N_samples]

    # Estimate the nominal sampling interval from clock 0.  x_samples is a
    # time-error series, so its first difference divided by this interval is
    # the fractional-frequency series used for Allan deviation.
    sample_interval_ps = float(np.median(np.diff(t_clk_0[:N_samples])))
    if not np.isfinite(sample_interval_ps) or sample_interval_ps <= 0:
        raise ValueError("Clock-0 timestamps must have a positive sample interval.")

    fractional_frequency = np.diff(x_samples) / sample_interval_ps
    if len(fractional_frequency) < 2:
        raise ValueError("Need at least three paired clock samples for Allan deviation.")

    # Keep at least five non-overlapping clock-frequency blocks at every tau.
    minimum_blocks = 5
    max_averaging_factor = len(fractional_frequency) // minimum_blocks
    averaging_factors = np.unique(
        np.logspace(
            0,
            np.log10(max_averaging_factor),
            num=int(number_of_tau_points),
            dtype=int,
        )
    )

    tau_s = []
    allan_deviation = []
    for m in averaging_factors:
        number_of_blocks = len(fractional_frequency) // m
        if number_of_blocks < 2:
            continue

        # Non-overlapping averages of fractional frequency over tau = m*tau0.
        block_averages = fractional_frequency[: number_of_blocks * m].reshape(
            number_of_blocks, m
        ).mean(axis=1)
        allan_deviation.append(
            np.sqrt(0.5 * np.mean(np.diff(block_averages) ** 2))
        )
        tau_s.append(m * sample_interval_ps * 1e-12)

    tau_s = np.asarray(tau_s)
    allan_deviation = np.asarray(allan_deviation)

    fig, axis = plt.subplots()
    axis.loglog(tau_s, allan_deviation, "-", linewidth=2.5)
    axis.set_xlabel(r"Averaging time $\tau$ (s)")
    axis.set_ylabel("Allan deviation")
    axis.grid(True, which="both")
    axis.set_ylim(bottom=5e-12)
    fig.tight_layout()
    plt.show()

    return {
        "tau_s": tau_s,
        "allan_deviation": allan_deviation,
        "sample_interval_ps": sample_interval_ps,
        "x_samples_ps": x_samples,
        "figure": fig,
        "axis": axis,
    }


def allan_deviation_estimator_sep(
    t1_ps,
    ch1,
    t2_ps,
    ch2,
    ch_clk_0=2,
    ch_clk_1=7,
    number_of_tau_points=30,
):
    t1_ps = np.asarray(t1_ps).ravel()
    ch1 = np.asarray(ch1).ravel()
    t2_ps = np.asarray(t2_ps).ravel()
    ch2 = np.asarray(ch2).ravel()

    if len(t1_ps) == 0 or len(ch1) == 0:
        raise ValueError("t1_ps and ch1 must not be empty.")
    if len(t1_ps) != len(ch1):
        raise ValueError("t1_ps and ch1 must have the same length.")
    if int(number_of_tau_points) < 1:
        raise ValueError("number_of_tau_points must be at least 1.")

    # Keep each channel chronological.
    t_clk_0 = np.sort(t1_ps[ch1 == ch_clk_0])
    t_clk_1 = np.sort(t2_ps[ch2 == ch_clk_1])

    if len(t_clk_0) == 0 or len(t_clk_1) == 0:
        raise ValueError("Both channels need at least one event.")

    N_samples = min(np.size(t_clk_0), np.size(t_clk_1))
    x_samples = t_clk_1[0:N_samples] - t_clk_0[0:N_samples]

    # Estimate the nominal sampling interval from clock 0.  x_samples is a
    # time-error series, so its first difference divided by this interval is
    # the fractional-frequency series used for Allan deviation.
    sample_interval_ps = float(np.median(np.diff(t_clk_0[:N_samples])))
    if not np.isfinite(sample_interval_ps) or sample_interval_ps <= 0:
        raise ValueError("Clock-0 timestamps must have a positive sample interval.")

    fractional_frequency = np.diff(x_samples) / sample_interval_ps
    if len(fractional_frequency) < 2:
        raise ValueError("Need at least three paired clock samples for Allan deviation.")

    # Keep at least five non-overlapping clock-frequency blocks at every tau.
    minimum_blocks = 5
    max_averaging_factor = len(fractional_frequency) // minimum_blocks
    averaging_factors = np.unique(
        np.logspace(
            0,
            np.log10(max_averaging_factor),
            num=int(number_of_tau_points),
            dtype=int,
        )
    )

    tau_s = []
    allan_deviation = []
    for m in averaging_factors:
        number_of_blocks = len(fractional_frequency) // m
        if number_of_blocks < 2:
            continue

        # Non-overlapping averages of fractional frequency over tau = m*tau0.
        block_averages = fractional_frequency[: number_of_blocks * m].reshape(
            number_of_blocks, m
        ).mean(axis=1)
        allan_deviation.append(
            np.sqrt(0.5 * np.mean(np.diff(block_averages) ** 2))
        )
        tau_s.append(m * sample_interval_ps * 1e-12)

    tau_s = np.asarray(tau_s)
    allan_deviation = np.asarray(allan_deviation)

    fig, axis = plt.subplots()
    axis.loglog(tau_s, allan_deviation, "-", linewidth=2.5)
    axis.set_xlabel(r"Averaging time $\tau$ (s)")
    axis.set_ylabel("Allan deviation")
    axis.grid(True, which="both")
    axis.set_ylim(bottom=5e-12)
    fig.tight_layout()
    plt.show()

    return {
        "tau_s": tau_s,
        "allan_deviation": allan_deviation,
        "sample_interval_ps": sample_interval_ps,
        "x_samples_ps": x_samples,
        "figure": fig,
        "axis": axis,
    }


def allan_deviation_estimator_tau_input(
    tau,
    T,
    number_of_tau_points=30,
    *,
    show=True,
):
    if len(tau) == 0:
        raise ValueError("tau array must not be empty.")

    N_samples = np.size(tau)
    x_samples = tau

    # Estimate the nominal sampling interval from clock 0.  x_samples is a
    # time-error series, so its first difference divided by this interval is
    # the fractional-frequency series used for Allan deviation.
    sample_interval_ps = float(np.median(np.diff(T)))
    if not np.isfinite(sample_interval_ps) or sample_interval_ps <= 0:
        raise ValueError("Clock-0 timestamps must have a positive sample interval.")

    fractional_frequency = np.diff(x_samples) / sample_interval_ps
    if len(fractional_frequency) < 2:
        raise ValueError("Need at least three paired clock samples for Allan deviation.")

    # Keep at least five non-overlapping clock-frequency blocks at every tau.
    minimum_blocks = 5
    max_averaging_factor = len(fractional_frequency) // minimum_blocks
    averaging_factors = np.unique(
        np.logspace(
            0,
            np.log10(max_averaging_factor),
            num=int(number_of_tau_points),
            dtype=int,
        )
    )

    tau_s = []
    allan_deviation = []
    for m in averaging_factors:
        number_of_blocks = len(fractional_frequency) // m
        if number_of_blocks < 2:
            continue

        # Non-overlapping averages of fractional frequency over tau = m*tau0.
        block_averages = fractional_frequency[: number_of_blocks * m].reshape(
            number_of_blocks, m
        ).mean(axis=1)
        allan_deviation.append(
            np.sqrt(0.5 * np.mean(np.diff(block_averages) ** 2))
        )
        tau_s.append(m * sample_interval_ps * 1e-12)

    tau_s = np.asarray(tau_s)
    allan_deviation = np.asarray(allan_deviation)

    fig, axis = plt.subplots()
    axis.loglog(tau_s, allan_deviation, "-", linewidth=2.5)
    axis.set_xlabel(r"Averaging time $\tau$ (s)")
    axis.set_ylabel("Allan deviation")
    axis.grid(True, which="both")
    fig.tight_layout()
    if show:
        plt.show()

    return {
        "tau_s": tau_s,
        "allan_deviation": allan_deviation,
        "sample_interval_ps": sample_interval_ps,
        "x_samples_ps": x_samples,
        "figure": fig,
        "axis": axis,
    }
