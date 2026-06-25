from collections.abc import Callable

from sigpipe.algorithms.dispersion.ftan import dispersion_ftan
from sigpipe.algorithms.dispersion.phase_shift import dispersion_phase_shift
from sigpipe.base.dispersion_image import DispersionImage

DISPERSION_METHODS: dict[str, Callable[..., DispersionImage | list[DispersionImage]]] = {
    "group": dispersion_ftan,
    "phase": dispersion_phase_shift,
}
