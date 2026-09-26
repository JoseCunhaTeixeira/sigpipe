"""Inversion of a line's picked curves into layered shear-wave velocity models.

PAC's seismic inversion (sigpipe's MCMC and PAC's output files) on each window, its parameters
derived from the window's own curve (priors), what its files say of the models (measuring), and
the line's velocity section and pseudo-section comparison (section). The parameters are the
MCMC's own (InversionParameters, re-exported here).
"""

from sigpipe.algorithms.inversion.rayleigh.seismic.parameters import (
    InversionParameters,
    ThicknessLayer,
    VsLayer,
)

from .priors import InversionError
from .window import build_inversion_pipeline, invert_window

__all__ = [
    "InversionError",
    "InversionParameters",
    "ThicknessLayer",
    "VsLayer",
    "build_inversion_pipeline",
    "invert_window",
]
