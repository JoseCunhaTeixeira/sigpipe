"""Petrophysical inversion of a line's picked curves: soils, N values and the water table.

PAC's petrophysical inversion (a Silex model on each window's fundamental mode, the curve its
prediction gives back, the rock physics with depth) on each window of a run (window), on the
windows of a line in worker processes (line), and the line's sections (section). Needs sigpipe's silex and santiludo extras; the models bundled with
sigpipe, and what each was trained on, are in algorithms.inversion.rayleigh.petro.silex_catalog,
which needs neither.
"""

from .line import PetroOutcome, invert_line_petro, save_line_sections
from .window import (
    QUANTITIES,
    Quantity,
    fundamental_curve,
    invert_window_petro,
    load_modeled_curve,
    load_petro_model,
    load_profile,
)

__all__ = [
    "QUANTITIES",
    "PetroOutcome",
    "Quantity",
    "fundamental_curve",
    "invert_line_petro",
    "invert_window_petro",
    "load_modeled_curve",
    "load_petro_model",
    "load_profile",
    "save_line_sections",
]
