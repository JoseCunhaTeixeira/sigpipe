"""Inversion of a line's picked curves into layered shear-wave velocity models.

The seismic inversion (sigpipe's MCMC and PAC's output files) of each window, its parameters
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
from .window import (
    PARAMETERS_FILE,
    WindowParameters,
    build_inversion_pipeline,
    invert_window,
    load_parameters,
)

__all__ = [
    "PARAMETERS_FILE",
    "InversionError",
    "InversionParameters",
    "ThicknessLayer",
    "VsLayer",
    "WindowParameters",
    "build_inversion_pipeline",
    "invert_window",
    "load_parameters",
]
