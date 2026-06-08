from typing import Callable

from src.algorithms.dispersion.ftan import dispersion_ftan
from src.algorithms.dispersion.phase_shift import dispersion_phase_shift
from src.base.dispersion import DispersionImage

DISPERSION_METHODS: dict[
    str, Callable[..., DispersionImage | list[DispersionImage]]
] = {
    "group": dispersion_ftan,
    "phase": dispersion_phase_shift,
}
