from collections.abc import Sequence

from sigpipe.algorithms.flipping.flipping import FlipAxis, flip
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer


class Flip(Transformer[Stream, Stream]):
    """Flip transformer."""

    def __init__(
        self,
        axis: FlipAxis = FlipAxis.SPACE,
        flip_acquisition: bool = False,
    ) -> None:
        self.axis = axis
        self.flip_acquisition = flip_acquisition

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        self.validate_sequence(data, Stream)

        streams_out: list[Stream] = []
        for stream in data:
            stream_out = flip(
                stream=stream,
                axis=self.axis,
                flip_acquisition=self.flip_acquisition,
            )
            streams_out.append(stream_out)

        return streams_out
