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
    IMAGE_FIGURE,
    MARGINALS_FIGURE,
    PARAMETERS_FILE,
    SPREAD_FILE,
    VS_SPREAD_FILE,
    WINDOW_FIGURE,
    Spread,
    VsSpread,
    WindowParameters,
    build_inversion_pipeline,
    draw_figures,
    invert_window,
    load_parameters,
    load_spread,
    load_vs_spread,
    vs_spread,
)

__all__ = [
    "IMAGE_FIGURE",
    "MARGINALS_FIGURE",
    "PARAMETERS_FILE",
    "SPREAD_FILE",
    "VS_SPREAD_FILE",
    "WINDOW_FIGURE",
    "InversionError",
    "InversionParameters",
    "Spread",
    "ThicknessLayer",
    "VsLayer",
    "VsSpread",
    "WindowParameters",
    "build_inversion_pipeline",
    "draw_figures",
    "invert_window",
    "load_parameters",
    "load_spread",
    "load_vs_spread",
    "vs_spread",
]
