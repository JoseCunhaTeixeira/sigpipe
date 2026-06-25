from collections.abc import Sequence

from sigpipe.algorithms.padding.padding import pad
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer


class Pad(Transformer[Stream, Stream]):
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

        self.validate_sequence(data, Stream)

        streams_out: list[Stream] = []
        for stream in data:
            stream_out = pad(
                stream=stream,
                n=self.n,
                taper=self.taper,
            )
            streams_out.append(stream_out)

        return streams_out
