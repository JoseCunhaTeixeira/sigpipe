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
    load_obspy,
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
    type | str,
    Callable[..., DispersionImage]
    | Callable[..., DispersionCurves]
    | Callable[..., Stream]
    | Callable[..., list[Stream]],
] = {
    DispersionImage: load_dispersion_image,
    DispersionCurves: load_dispersion_curves,
    Stream: load_stream,
    "gero_passive": load_gero_passive,
    "gero_active": load_gero_active,
    "segy": load_obspy,
    "seg2": load_obspy,
    "su": load_obspy,
}

PLOT_HANDLERS: dict[type, Callable[..., Figure]] = {
    DispersionImage: plot_dispersion_image,
    DispersionCurves: plot_dispersion_curves,
    Stream: plot_stream,
}


SECTION_HANDLERS: dict[type, Callable[..., Figure]] = {
    DispersionCurves: plot_dispersion_curve_section,
}
