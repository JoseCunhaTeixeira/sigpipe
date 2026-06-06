from typing import Callable

from base.dispersion import DispersionImage

from .ftan import dispersion_ftan
from .phase_shift import dispersion_phase_shift

DISPERSION_METHODS: dict[
    str, Callable[..., DispersionImage | list[DispersionImage]]
] = {
    "group": dispersion_ftan,
    "phase": dispersion_phase_shift,
}
