from .constant import detrend_constant
from .linear import detrend_linear

DETRENDING_METHODS = {
    "linear": detrend_linear,
    "constant": detrend_constant,
}
