from src.base.stream import Stream


def mute(
    stream: Stream,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
) -> Stream:

    if tmin is None and tmax is None:
        raise ValueError("At least one of tmin or tmax must be provided")

    if tmin is not None and tmax is not None and tmin >= tmax:
        raise ValueError(f"Expected tmin < tmax, got tmin={tmin} and tmax={tmax}")

    xt = stream.xt.copy()

    if tmin is not None:
        xt[:, tmin < stream.ts] = 0.0

    if tmax is not None:
        xt[:, stream.ts < tmax] = 0.0

    return Stream(
        xt=xt,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
    )
