from enum import StrEnum

from sigproc.base.acquisition import Acquisition
from sigproc.base.stream import Stream


class FlipAxis(StrEnum):
    SPACE = "space"
    TIME = "time"


def flip(
    stream: Stream,
    axis: FlipAxis = FlipAxis.SPACE,
    flip_acquisition: bool = False,
) -> Stream:
    axis = FlipAxis(axis)

    xt = stream.xt.copy()
    acquisition = stream.acquisition

    if axis is FlipAxis.SPACE:
        xt = xt[::-1, :]
        if flip_acquisition:
            acquisition = Acquisition(
                source=stream.acquisition.source,
                receivers=tuple(reversed(stream.acquisition.receivers)),
            )
    elif axis is FlipAxis.TIME:
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
