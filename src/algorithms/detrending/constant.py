import numpy as np
from scipy.signal import detrend


def detrend_constant(xt: np.ndarray) -> np.ndarray:
    xt_detrended = detrend(xt, axis=1, type="constant")
    return xt_detrended.astype(np.float32)
