from collections.abc import Callable

from sigproc.algorithms.picking.dispersion.curve import pick_curves
from sigproc.algorithms.picking.stream.arrival import pick_arrivals
from sigproc.base.dispersion import DispersionImage
from sigproc.base.stream import Stream

DISPERSION_PICKING_METHODS: dict[str, Callable[..., DispersionImage]] = {"maximum": pick_curves}

STREAM_PICKING_METHODS: dict[str, Callable[..., Stream]] = {
    "maximum": pick_arrivals,
}
