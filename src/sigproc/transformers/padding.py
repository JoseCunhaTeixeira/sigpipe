from collections.abc import Sequence

from sigproc.algorithms.padding.padding import pad
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer


class Pad(Transformer):
    """
    Padding transformer.
    """

    def __init__(
        self,
        n: int,
        taper: int = 0,
    ) -> None:
        self.n = n
        self.taper = taper

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence[Stream], got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(s, Stream) for s in data):
            raise TypeError("All elements must be Stream")

        streams_out: list[Stream] = []
        for stream in data:
            stream_out = pad(
                stream=stream,
                n=self.n,
                taper=self.taper,
            )
            streams_out.append(stream_out)

        return streams_out
