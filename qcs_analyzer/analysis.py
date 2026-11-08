"""Time alignment, correlation processing, and peak fitting."""

from ._implementation import (
    estimate_qcs_delay,
    fit_g2,
    fit_g2_curve,
    fit_g2_exponential,
    fit_g2_expo,
    fit_g2_gaussian,
    fit_g2_lorentzian,
    g2_process,
    g2_process_sep,
    g2_process_time_dependent,
    g2_process_tstop_dependent,
    process_qcs_delay,
    timetag_channel_time_difference,
    timetag_correction,
    timetag_correction_sep,
)

__all__ = [
    "estimate_qcs_delay",
    "fit_g2",
    "fit_g2_curve",
    "fit_g2_exponential",
    "fit_g2_expo",
    "fit_g2_gaussian",
    "fit_g2_lorentzian",
    "g2_process",
    "g2_process_sep",
    "g2_process_time_dependent",
    "g2_process_tstop_dependent",
    "process_qcs_delay",
    "timetag_channel_time_difference",
    "timetag_correction",
    "timetag_correction_sep",
]
