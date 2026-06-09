from typing import Literal

from sigproc.base.stream import Stream


def flip(
    stream: Stream,
    axis: Literal["space", "time"] = "space",
) -> Stream:
    if axis.lower() not in {"space", "time"}:
        raise ValueError(f"axis must be 'space' or 'time, got {axis.lower()!r}")
    xt = stream.xt.copy()
    if axis.lower() == "space":
        xt = xt[::-1, :]
    elif axis.lower() == "time":
        xt = xt[:, ::-1]
    return Stream(
        xt=xt,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
    )
