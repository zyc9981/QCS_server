"""Public API for QCS timetag analysis.

Import only the part of the workflow you need, for example::

    from qcs_analyzer.io import load_timetag
    from qcs_analyzer.analysis import g2_process
"""

from .analysis import (
    estimate_qcs_delay,
    fit_g2,
    fit_g2_curve,
    fit_g2_exponential,
    fit_g2_expo,
    fit_g2_gaussian,
    fit_g2_lorentzian,
    g2_process,
    g2_process_time_dependent,
    g2_process_tstop_dependent,
    process_qcs_delay,
    timetag_channel_time_difference,
    timetag_correction,
    timetag_correction_sep,
)
from .cli import main
from .config import (
    FileSelectionCancelled,
    get_default_file_dialog_directory,
    set_default_file_dialog_directory,
)
from .io import (
    load_timetag,
    load_timetag_sep,
    timetag_to_h5,
    timetag_to_h5_divider,
    timetag_to_h5_sep,
    timetag_to_h5_TTXonly,
    timetag_to_h5_qutagHH,
    timetag_to_h5_qutag,
    trancate_timetag,
    truncate_timetag,
    truncate_timetag_sep,
)
from .plotting import g2_plot, plot_photon_count, plot_qcs_delay_results
from .results import (
    load_g2_results,
    load_time_dependent_results,
    save_g2_results,
    save_time_dependent_results,
)

__all__ = [
    "FileSelectionCancelled",
    "estimate_qcs_delay",
    "fit_g2",
    "fit_g2_curve",
    "fit_g2_exponential",
    "fit_g2_expo",
    "fit_g2_gaussian",
    "fit_g2_lorentzian",
    "g2_plot",
    "g2_process",
    "g2_process_time_dependent",
    "g2_process_tstop_dependent",
    "get_default_file_dialog_directory",
    "load_g2_results",
    "load_timetag",
    "load_timetag_sep",
    "load_time_dependent_results",
    "main",
    "plot_photon_count",
    "plot_qcs_delay_results",
    "process_qcs_delay",
    "save_g2_results",
    "save_time_dependent_results",
    "set_default_file_dialog_directory",
    "timetag_channel_time_difference",
    "timetag_correction",
    "timetag_correction_sep",
    "timetag_to_h5",
    "timetag_to_h5_divider",
    "timetag_to_h5_sep",
    "timetag_to_h5_TTXonly",
    "timetag_to_h5_qutagHH",
    "timetag_to_h5_qutag",
    "trancate_timetag",
    "truncate_timetag",
    "truncate_timetag_sep",
]
