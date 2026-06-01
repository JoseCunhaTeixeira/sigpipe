import numpy as np
from scipy.signal import butter, sosfiltfilt


def filter_iir(
    xt: np.ndarray,
    sampling_freq: float,
    *,
    fmin: float,
    fmax: float,
    order: int = 4,
) -> np.ndarray:
    """
    Bandpass filter applied to all traces.
    """
    nyq = 0.5 * sampling_freq
    if not (0 < fmin < fmax < nyq):
        raise ValueError("Require 0 < fmin < fmax < sampling_freq/2")
    sos = butter(order, [fmin / nyq, fmax / nyq], btype="band", output="sos")
    filtered = sosfiltfilt(sos, xt, axis=-1)
    return filtered.astype(np.float32)
