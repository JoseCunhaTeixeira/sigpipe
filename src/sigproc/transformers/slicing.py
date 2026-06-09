from collections.abc import Sequence

import numpy as np

from sigproc.algorithms.segmentation.slice import segment_slice
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer


class Slice(Transformer):
    """
    Slicing transformer.
    """

    def __init__(
        self,
        segment_duration: float,
        segment_step: float,
        **params,
    ):
        self.segment_duration = segment_duration
        self.segment_step = segment_step
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"expected Sequence[Stream], got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("empty input sequence")

        if not all(isinstance(s, Stream) for s in data):
            raise TypeError("all elements in sequence must be Stream")

        if self.segment_duration <= 0:
            raise ValueError(
                f"requires segment_duration > 0 s, got {self.segment_duration} s"
            )

        if self.segment_step <= 0:
            raise ValueError(f"requires segment_step > 0 s, got {self.segment_step} s")

        if self.segment_step > self.segment_duration:
            raise ValueError(
                f"requires segment_step <= segment_duration, got {self.segment_step} s and {self.segment_duration} s"
            )

        durations = np.array([stream.ts[-1] for stream in data])
        if any(self.segment_duration > durations):
            raise ValueError(
                f"requires segment_duration <= record durations, got {self.segment_duration} s and {durations} s"
            )

        streams_out: list[Stream] = []
        for stream in data:
            record_duration = stream.ts[-1]
            t = 0.0
            while t + self.segment_duration <= record_duration + 1e-12:
                stream_out = segment_slice(
                    stream=stream,
                    t_slice_start=t,
                    t_slice_end=t + self.segment_duration,
                )
                streams_out.append(stream_out)
                t += self.segment_step

        return streams_out
