from sigpipe.base.stream import Stream


def slice_segments(
    stream: Stream,
    *,
    segment_duration: float,
    segment_step: float,
) -> list[Stream]:
    """The segments of `stream`, `segment_duration` seconds long, one every `segment_step`
    seconds, as long as they fit in the record: each the same number of samples, its start the
    sample nearest its time, so that they stack whatever the step (0.375 s at 500 Hz is 187.5
    samples)."""
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
    dt = float(stream.ts[1] - stream.ts[0])
    length = round(segment_duration / dt)  # samples after a segment's first
    segments: list[Stream] = []
    t = 0.0
    while t + segment_duration <= record_duration + 1e-12:
        start = round(t / dt)
        if start + length >= len(stream.ts):
            break
        segments.append(
            Stream(
                xt=stream.xt[:, start : start + length + 1].copy(),
                ts=stream.ts[start : start + length + 1],
                sampling_freq=stream.sampling_freq,
                acquisition=stream.acquisition,
            )
        )
        t += segment_step
    return segments
