from dataclasses import replace

import numpy as np

from sigpipe.base.arrivals import TraceArrivals
from sigpipe.base.stream import Stream


def arrival_residual_phase(stream: Stream, *, f0: float) -> Stream:
    """`stream` with each arrival's residual phase at frequency `f0`: the trace's phase at the
    arrival, less the phase a wave of `f0` accumulates by then."""
    if stream.arrivals is None:
        raise ValueError("ComputePhase requires arrivals. Run Pick before ComputePhase.")

    trace_arrivals_new = []
    for itrace, trace_arrivals in enumerate(stream.arrivals):
        arrivals_new = []
        for arrival in trace_arrivals:
            k = np.argmin(np.abs(stream.ts - arrival.time))
            residual_phase = stream.xt_phase[itrace, k] - 2 * np.pi * float(f0) * arrival.time
            arrivals_new.append(replace(arrival, residual_phase=float(residual_phase)))
        trace_arrivals_new.append(TraceArrivals(arrivals=tuple(arrivals_new)))

    return replace(stream, arrivals=tuple(trace_arrivals_new))
