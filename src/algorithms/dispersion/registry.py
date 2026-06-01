from .ftan import dispersion_ftan
from .phase_shift import dispersion_phase_shift

DISPERSION_METHODS = {
    "ftan": dispersion_ftan,
    "phase_shift": dispersion_phase_shift,
}
