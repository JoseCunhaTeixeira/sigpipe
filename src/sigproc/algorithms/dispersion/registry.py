from typing import Callable

from sigproc.algorithms.dispersion.ftan import dispersion_ftan
from sigproc.algorithms.dispersion.phase_shift import dispersion_phase_shift
from sigproc.base.dispersion import DispersionImage

DISPERSION_METHODS: dict[
    str, Callable[..., DispersionImage | list[DispersionImage]]
] = {
    "group": dispersion_ftan,
    "phase": dispersion_phase_shift,
}
