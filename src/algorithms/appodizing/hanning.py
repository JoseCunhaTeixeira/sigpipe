import numpy as np
from scipy.signal.windows import hann


def appodize_hanning(
    xt: np.ndarray,
    *,
    frac: float = 0.1,
) -> np.ndarray:
    """
    Apply a Hanning taper to the edges of each trace.
    """
    _, nt = xt.shape
    n_taper = max(1, int(frac * nt))
    wd_full = hann(2 * n_taper)
    taper_start = wd_full[:n_taper]
    taper_end = wd_full[n_taper:]
    wd = np.ones(nt)
    wd[:n_taper] = taper_start
    wd[-n_taper:] = taper_end
    xt_appodized = xt * wd
    return xt_appodized.astype(np.float32)
