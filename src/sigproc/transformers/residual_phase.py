from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from sigproc.base.arrivals import TraceArrivals
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer


class ArrivalResidualPhase(Transformer[Stream, Stream]):
    def __init__(self, f0: float):
        self.f0 = float(f0)

    def transform(
        self,
        data: Sequence[Stream],
    ) -> list[Stream]:

        out = []

        for stream in data:
            if stream.arrivals is None:
                raise ValueError(
                    "ComputePhase requires arrivals. Run Pick before ComputePhase."
                )

            trace_arrivals_new = []

            for itrace, trace_arrivals in enumerate(stream.arrivals):
                arrivals_new = []

                for arrival in trace_arrivals:
                    k = np.argmin(np.abs(stream.ts - arrival.time))

                    residual_phase = (
                        stream.xt_phase[itrace, k] - 2 * np.pi * self.f0 * arrival.time
                    )

                    arrivals_new.append(
                        replace(
                            arrival,
                            residual_phase=float(residual_phase),
                        )
                    )

                trace_arrivals_new.append(TraceArrivals(arrivals=tuple(arrivals_new)))

            out.append(
                replace(
                    stream,
                    arrivals=tuple(trace_arrivals_new),
                )
            )

        return out
