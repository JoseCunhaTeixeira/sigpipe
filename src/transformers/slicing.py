from collections.abc import Sequence

from src.algorithms.segmentation.slice import segment_slice
from src.base.stream import Stream
from src.base.transformer import Transformer


class SlicingTransformer(Transformer):
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

    def transform(self, data: Sequence[Stream]) -> Sequence[Stream]:

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence[Stream], got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(s, Stream) for s in data):
            raise TypeError("All elements must be Stream")

        if self.segment_duration <= 0:
            raise ValueError("segment_duration must be > 0")

        if self.segment_step <= 0:
            raise ValueError("segment_step must be > 0")

        if self.segment_step > self.segment_duration:
            raise ValueError("segment_step should be <= segment_duration")

        streams_out = []
        for stream in data:
            record_duration = stream.ts[-1]
            t = 0.0
            while t + self.segment_duration <= record_duration + 1e-12:
                ts_slice, xt_slice = segment_slice(
                    xt=stream.xt,
                    ts=stream.ts,
                    t_slice_start=t,
                    t_slice_end=t + self.segment_duration,
                )

                stream_out = Stream(
                    xt=xt_slice,
                    ts=ts_slice,
                    sampling_freq=stream.sampling_freq,
                    acquisition=stream.acquisition,
                )

                streams_out.append(stream_out)

                t += self.segment_step

        return streams_out
