from collections.abc import Sequence
from typing import Literal

from sigpipe.algorithms.selection.registry import STREAM_SELECTION_METHODS
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer


class Selection(Transformer[Stream, Stream]):
    """
    Stream selection transformer.
    """

    def __init__(
        self,
        method: Literal["none", "fk"],
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        self.validate_sequence(data, Stream, DispersionImage)

        if self.method == "none":
            return list(data)

        algorithm = STREAM_SELECTION_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown selection method '{self.method}'. "
                f"Available methods: {list(STREAM_SELECTION_METHODS.keys())}"
            )

        first = data[0]

        if isinstance(first, Stream):
            streams_out: list[Stream] = []
            for stream in data:
                stream_out = algorithm(
                    stream=stream,
                    **self.params,
                )
                if stream_out is not None:
                    streams_out.append(stream_out)
            return streams_out

        raise TypeError(f"No selection handler for {type(first).__name__}")
