import numpy as np
from scipy.signal import detrend


def detrend_linear(xt: np.ndarray) -> np.ndarray:
    xt_detrended = detrend(xt, axis=1, type="linear")
    return xt_detrended.astype(np.float32)
