from collections.abc import Sequence

from sigproc.algorithms.flipping.flipping import flip
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer


class Flip(Transformer):
    """
    Flipping transformer.
    """

    def __init__(
        self,
        **params,
    ):
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence[Stream], got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(s, Stream) for s in data):
            raise TypeError("All elements must be Stream")

        streams_out: list[Stream] = []
        for stream in data:
            stream_out = flip(
                stream=stream,
                **self.params,
            )
            streams_out.append(stream_out)

        return streams_out
