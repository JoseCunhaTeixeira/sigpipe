from typing import Callable

from src.sigproc.algorithms.picking.dispersion.curve import pick_curves
from src.sigproc.algorithms.picking.stream.arrival import pick_arrivals
from src.sigproc.base.dispersion import DispersionImage
from src.sigproc.base.stream import Stream

DISPERSION_PICKING_METHODS: dict[str, Callable[..., DispersionImage]] = {
    "maximum": pick_curves
}

STREAM_PICKING_METHODS: dict[str, Callable[..., Stream]] = {
    "maximum": pick_arrivals,
}
