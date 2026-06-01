import numpy as np
from scipy.signal.windows import hann


def slice(
    xt: np.ndarray,
    ts: np.ndarray,
    *,
    t_slice_start: float,
    t_slice_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice data between t_slice_start and t_slice_end..

    Args:
        xt (np.ndarray): recording data [nx, nt]
        ts (np.ndarray): times [nt]
        t_slice_start (float): starting time
        t_slice_end (float): ending time

    Returns:
        tuple[np.ndarray, np.ndarray]: (ts_slice [nt_slice], xt_slice [nx, nt_slice])
    """
    if t_slice_start >= t_slice_end:
        raise ValueError(
            f"t_slice_start ({t_slice_start}) must be < t_slice_end ({t_slice_end})"
        )
    dt = ts[1] - ts[0]
    i_start = int(round(t_slice_start / dt))
    i_end = int(round(t_slice_end / dt))
    ts_slice = ts[i_start : i_end + 1]
    xt_slice = xt[:, i_start : i_end + 1]
    return ts_slice.astype(np.float32), xt_slice.astype(np.float32)


def compute_nsegments(record_duration, segment_duration, segment_step):
    if record_duration <= 0:
        raise ValueError(f"record_duration must be > 0 (got {record_duration})")
    if segment_duration <= 0:
        raise ValueError(f"segment_duration must be > 0 (got {segment_duration})")
    if segment_duration > record_duration:
        raise ValueError(
            f"segment_duration must be <= record_duration "
            f"(got segment_duration={segment_duration}, record_duration={record_duration})"
        )
    if segment_step < 0:
        raise ValueError(f"segment_step must be >= 0 (got {segment_step})")
    if segment_step == 0:
        return 2
    max_start = record_duration - segment_duration
    nsegments = (int(np.floor(max_start / segment_step)) * 2) + 2
    return nsegments


def appodize(
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
