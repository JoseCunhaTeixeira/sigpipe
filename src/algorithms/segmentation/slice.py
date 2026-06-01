import numpy as np


def segment_slice(
    xt: np.ndarray,
    ts: np.ndarray,
    t_slice_start: float,
    t_slice_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice data between t_slice_start and t_slice_end.

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
