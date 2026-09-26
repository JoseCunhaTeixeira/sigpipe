from sigpipe.base.stream import Stream


def slice_segments(
    stream: Stream,
    *,
    segment_duration: float,
    segment_step: float,
) -> list[Stream]:
    """The segments of `stream`, `segment_duration` seconds long, one every `segment_step`
    seconds, as long as they fit in the record."""
    if segment_duration <= 0:
        raise ValueError(f"requires segment_duration > 0 s, got {segment_duration} s")
    if segment_step <= 0:
        raise ValueError(f"requires segment_step > 0 s, got {segment_step} s")
    if segment_step > segment_duration:
        raise ValueError(
            f"requires segment_step <= segment_duration, got {segment_step} s and {segment_duration} s"
        )
    record_duration = float(stream.ts[-1])
    if segment_duration > record_duration:
        raise ValueError(
            f"requires segment_duration <= record duration, got {segment_duration} s and {record_duration} s"
        )
    segments: list[Stream] = []
    t = 0.0
    while t + segment_duration <= record_duration + 1e-12:
        segments.append(segment_slice(stream, t, t + segment_duration))
        t += segment_step
    return segments


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
    if not 0 <= t_slice_start < t_slice_end <= stream.ts[-1]:
        raise ValueError(
            f"requires 0 s < t_slice_start < t_slice_end <= record duration, got {t_slice_start}, {t_slice_end}, {stream.ts[-1]}"
        )
    dt = stream.ts[1] - stream.ts[0]
    i_start = round(t_slice_start / dt)
    i_end = round(t_slice_end / dt)
    ts_slice = stream.ts[i_start : i_end + 1]
    xt_slice = stream.xt[:, i_start : i_end + 1].copy()
    return Stream(
        xt=xt_slice,
        ts=ts_slice,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
    )
