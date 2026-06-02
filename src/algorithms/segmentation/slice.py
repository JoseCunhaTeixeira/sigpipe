from src.base.stream import Stream


def segment_slice(
    stream: Stream,
    t_slice_start: float,
    t_slice_end: float,
) -> Stream:
    """Slice data between t_slice_start and t_slice_end.

    Args:
        stream (Stream): recording data
        t_slice_start (float): starting time
        t_slice_end (float): ending time

    Returns:
        Stream: sliced data
    """
    if t_slice_start >= t_slice_end:
        raise ValueError(
            f"t_slice_start ({t_slice_start}) must be < t_slice_end ({t_slice_end})"
        )
    dt = stream.ts[1] - stream.ts[0]
    i_start = int(round(t_slice_start / dt))
    i_end = int(round(t_slice_end / dt))
    ts_slice = stream.ts[i_start : i_end + 1]
    xt_slice = stream.xt[:, i_start : i_end + 1]
    return Stream(
        xt=xt_slice,
        ts=ts_slice,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
    )
