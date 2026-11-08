"""Saving and loading QCS analysis results."""

from ._implementation import (
    load_g2_results,
    load_time_dependent_results,
    save_g2_results,
    save_time_dependent_results,
    load_dict_npz,
    save_dict_npz,
)

__all__ = [
    "load_g2_results",
    "load_time_dependent_results",
    "save_g2_results",
    "save_time_dependent_results",
    "load_dict_npz"
    "save_dict_npz"
]
