import numpy as np
from scipy.signal.windows import hann

from src.sigproc.base.stream import Stream


def appodize_hanning(
    stream: Stream,
    *,
    frac: float = 0.1,
) -> Stream:
    """
    Apply a Hanning taper to the edges of each trace.
    """
    n_taper = max(1, int(frac * stream.nt))
    wd_full = hann(2 * n_taper)
    taper_start = wd_full[:n_taper]
    taper_end = wd_full[n_taper:]
    wd = np.ones(stream.nt)
    wd[:n_taper] = taper_start
    wd[-n_taper:] = taper_end
    xt_appodized = stream.xt * wd
    return Stream(
        xt=xt_appodized,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
    )
