from typing import Callable

from matplotlib.figure import Figure

from sigproc.base.dispersion import DispersionCurves, DispersionImage
from sigproc.base.stream import Stream
from sigproc.dataio.dispersion.loading import (
    load_dispersion_curves,
    load_dispersion_image,
)
from sigproc.dataio.dispersion.plotting import (
    plot_dispersion_curves,
    plot_dispersion_image,
)
from sigproc.dataio.dispersion.saving import (
    save_dispersion_curves,
    save_dispersion_image,
)
from sigproc.dataio.dispersion.section import plot_dispersion_curve_section
from sigproc.dataio.stream.loading import (
    load_gero_active,
    load_gero_passive,
    load_segy,
    load_stream,
)
from sigproc.dataio.stream.plotting import plot_stream
from sigproc.dataio.stream.saving import save_stream

SAVE_HANDLERS: dict[type, Callable[..., None]] = {
    DispersionImage: save_dispersion_image,
    DispersionCurves: save_dispersion_curves,
    Stream: save_stream,
}

LOAD_HANDLERS: dict[
    str,
    Callable[..., list[DispersionImage]]
    | Callable[..., list[DispersionCurves]]
    | Callable[..., list[Stream]],
] = {
    "dispersion_image": load_dispersion_image,
    "dispersion_curves": load_dispersion_curves,
    "stream": load_stream,
    "gero_passive": load_gero_passive,
    "gero_active": load_gero_active,
    "segy": load_segy,
}

PLOT_HANDLERS: dict[type, Callable[..., Figure]] = {
    DispersionImage: plot_dispersion_image,
    DispersionCurves: plot_dispersion_curves,
    Stream: plot_stream,
}


SECTION_HANDLERS: dict[type, Callable[..., Figure]] = {
    DispersionCurves: plot_dispersion_curve_section,
}
