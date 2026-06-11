from typing import Literal

from sigproc.base.acquisition import Acquisition
from sigproc.base.stream import Stream


def flip(
    stream: Stream,
    axis: Literal["space", "time"] = "space",
    flip_acquisition: bool = False,
) -> Stream:
    if axis.lower() not in {"space", "time"}:
        raise ValueError(f"axis must be 'space' or 'time, got {axis.lower()!r}")
    xt = stream.xt.copy()
    acquisition = stream.acquisition
    if axis.lower() == "space":
        xt = xt[::-1, :]
        if flip_acquisition:
            acquisition = Acquisition(
                source=stream.acquisition.source,
                receivers=tuple(reversed(stream.acquisition.receivers)),
            )
    elif axis.lower() == "time":
        xt = xt[:, ::-1]
    arrivals = None
    if stream.arrivals is not None:
        arrivals = tuple(reversed(stream.arrivals))
    return Stream(
        xt=xt,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=acquisition,
        arrivals=arrivals,
    )
