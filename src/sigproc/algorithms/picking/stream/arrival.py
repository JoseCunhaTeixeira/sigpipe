from collections.abc import Sequence

import numpy as np
from scipy.signal import hilbert

from sigproc.base.arrivals import Arrival, TraceArrivals
from sigproc.base.stream import Stream


def pick_arrivals(
    stream: Stream,
    label: str = "unknown",
    traces_to_pick: Sequence[int] | None = None,
    tmins: Sequence[float] | None = None,
    tmaxs: Sequence[float] | None = None,
) -> Stream:

    if traces_to_pick is None:
        traces_to_pick = range(stream.nx)

    if tmins is None:
        tmins = [stream.ts[0]] * len(traces_to_pick)

    if tmaxs is None:
        tmaxs = [stream.ts[-1]] * len(traces_to_pick)

    if not (len(traces_to_pick) == len(tmins) == len(tmaxs)):
        raise ValueError("traces_to_pick, tmins and tmaxs must have the same length")

    arrivals = list(stream.arrivals or (TraceArrivals(),) * stream.nx)

    for trace_idx, tmin, tmax in zip(
        traces_to_pick,
        tmins,
        tmaxs,
        strict=True,
    ):
        trace = stream.xt[trace_idx]

        mask = (stream.ts >= tmin) & (stream.ts <= tmax)

        if not np.any(mask):
            raise ValueError(
                f"Empty picking window for trace {trace_idx}: tmin={tmin}, tmax={tmax}"
            )

        trace_window = trace[mask]
        ts_window = stream.ts[mask]

        envelope = np.abs(np.array(hilbert(trace_window)))

        i_peak = np.argmax(envelope)

        time = np.round(ts_window[i_peak], decimals=6)

        # Original trace amplitude at the picked time
        amplitude = trace_window[i_peak]

        old = arrivals[trace_idx]

        arrivals[trace_idx] = TraceArrivals(
            arrivals=(
                *old.arrivals,
                Arrival(
                    label=label,
                    time=float(time),
                    amplitude=float(amplitude),
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
