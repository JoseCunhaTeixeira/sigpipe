from typing import Iterator

from src.algorithms.slincing.slice import slice
from src.base.stream import Stream
from src.base.transformer import Transformer


class SlicingTransformer(Transformer):
    """
    Time normalization transformer.
    """

    def __init__(
        self,
        segment_duration: float,
        segment_step: float,
    ):
        self.segment_duration = segment_duration
        self.segment_step = segment_step

    def transform(self, data: Stream) -> Iterator[Stream]:

        if not isinstance(data, Stream):
            raise TypeError(f"Expected Stream, got {type(data).__name__}")

        if self.segment_duration <= 0:
            raise ValueError("segment_duration must be > 0")

        if self.segment_step <= 0:
            raise ValueError("segment_step must be > 0")

        if self.segment_step > self.segment_duration:
            raise ValueError("segment_step should be <= segment_duration")

        record_duration = data.ts[-1]
        t = 0.0
        while t + self.segment_duration <= record_duration + 1e-12:
            ts_slice, xt_slice = slice(
                xt=data.xt,
                ts=data.ts,
                t_slice_start=t,
                t_slice_end=t + self.segment_duration,
            )

            yield Stream(
                xt=xt_slice,
                ts=ts_slice,
                sampling_freq=data.sampling_freq,
                acquisition=data.acquisition,
            )

            t += self.segment_step
