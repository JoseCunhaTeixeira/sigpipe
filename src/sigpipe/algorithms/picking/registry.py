from collections.abc import Callable

from sigpipe.algorithms.picking.dispersion.curve import pick_curves
from sigpipe.algorithms.picking.dispersion.lasso import pick_lasso
from sigpipe.algorithms.picking.dispersion.tracking import pick_tracking
from sigpipe.algorithms.picking.stream.arrival import pick_arrivals
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.base.stream import Stream

DISPERSION_PICKING_METHODS: dict[str, Callable[..., DispersionImage]] = {
    "maximum": pick_curves,  # within bounds: PAC's box
    "lasso": pick_lasso,  # within a polygon: PAC's lasso
    "tracking": pick_tracking,  # automatic: PACo's picker
}

STREAM_PICKING_METHODS: dict[str, Callable[..., Stream]] = {
    "maximum": pick_arrivals,
}
