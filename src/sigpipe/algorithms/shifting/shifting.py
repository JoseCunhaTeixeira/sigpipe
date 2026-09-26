from dataclasses import replace

import numpy as np

from sigpipe.base.arrivals import TraceArrivals
from sigpipe.base.stream import Stream


def shift(
    stream: Stream,
    *,
    t0: float = 0.0,
) -> Stream:
    """
    Move the time origin to `t0` seconds, keeping the record's length: a
    trigger `t0` late drops the first `t0` seconds and pads zeros at the end,
    an early one (`t0` < 0) pads zeros at the start. Arrival times follow.
    """
    samples = round(t0 * stream.sampling_freq)
    if samples == 0:
        return stream
    if abs(samples) >= stream.nt:
        raise ValueError(f"t0 ({t0} s) must be shorter than the record ({stream.nt} samples)")

    xt = np.zeros_like(stream.xt)
    if samples >= 0:
        xt[:, : stream.nt - samples] = stream.xt[:, samples:]
    else:
        xt[:, -samples:] = stream.xt[:, : stream.nt + samples]

    arrivals = None
    if stream.arrivals is not None:
        arrivals = tuple(
            TraceArrivals(
                arrivals=tuple(
                    replace(arrival, time=arrival.time - samples / stream.sampling_freq)
                    for arrival in trace
                )
            )
            for trace in stream.arrivals
        )

    return Stream(
        xt=xt,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
        arrivals=arrivals,
    )
