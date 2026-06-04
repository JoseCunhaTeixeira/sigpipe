from src.base.dispersion import DispersionImage
from src.base.stream import Stream

from .dispersion.linear import stack_linear as ls_dispersion_image
from .stream.linear import stack_linear as ls_stream
from .stream.phase_weighted import stack_phase_weighted
from .stream.root import stack_root

STREAM_STACKING_METHODS = {
    "linear": ls_stream,
    "root": stack_root,
    "phase_weighted": stack_phase_weighted,
}

DISPERSION_IMAGE_STACKING_METHODS = {
    "linear": ls_dispersion_image,
}

STACKING_HANDLERS = {
    Stream: STREAM_STACKING_METHODS,
    DispersionImage: DISPERSION_IMAGE_STACKING_METHODS,
}
