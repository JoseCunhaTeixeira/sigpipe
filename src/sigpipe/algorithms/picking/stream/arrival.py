from collections.abc import Sequence

import numpy as np

from sigpipe.base.arrivals import Arrival, TraceArrivals
from sigpipe.base.stream import Stream


def pick_arrivals(
    stream: Stream,
    label: str = "",
    traces_to_pick: Sequence[int] | None = None,
    tmins: Sequence[float] | None = None,
    tmaxs: Sequence[float] | None = None,
) -> Stream:

    if traces_to_pick is None:
        traces_to_pick = range(stream.nx)

    n = len(traces_to_pick)

    if tmins is None:
        tmins = [float(stream.ts[0])] * n

    if tmaxs is None:
        tmaxs = [float(stream.ts[-1])] * n

    if not (n == len(tmins) == len(tmaxs)):
        raise ValueError("traces_to_pick, tmins and tmaxs must have the same length")

    arrivals = list(stream.arrivals or (TraceArrivals(),) * stream.nx)

    for trace_idx, tmin, tmax in zip(
        traces_to_pick,
        tmins,
        tmaxs,
        strict=True,
    ):
        i0 = np.searchsorted(stream.ts, tmin, side="left")
        i1 = np.searchsorted(stream.ts, tmax, side="right")

        if i0 >= i1:
            raise ValueError(
                f"Empty picking window for trace {trace_idx}: tmin={tmin}, tmax={tmax}"
            )

        local_peak = np.argmax(stream.xt_envelope[trace_idx, i0:i1])
        k = i0 + local_peak

        old = arrivals[trace_idx]

        arrivals[trace_idx] = TraceArrivals(
            arrivals=(
                *old.arrivals,
                Arrival(
                    label=label,
                    time=float(stream.ts[k]),
                    amplitude=float(stream.xt_envelope[trace_idx, k]),
                ),
            )
        )

    return Stream(
        xt=stream.xt,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
        arrivals=tuple(arrivals),
    )
