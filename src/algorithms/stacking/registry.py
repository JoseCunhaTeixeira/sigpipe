from .linear import stack_linear
from .phase_weighted import stack_phase_weighted
from .root import stack_root

STACKING_METHODS = {
    "linear": stack_linear,
    "root": stack_root,
    "phase_weighted": stack_phase_weighted,
}
