from .ftan import dispersion_ftan
from .phase_shift import dispersion_phase_shift

DISPERSION_METHODS = {
    "group": dispersion_ftan,
    "phase": dispersion_phase_shift,
}
