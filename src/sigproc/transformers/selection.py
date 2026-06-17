from collections.abc import Sequence
from typing import Literal

from sigproc.algorithms.selection.registy import STREAM_SELECTION_METHODS
from sigproc.base.dispersion import DispersionImage
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer


class Selection(Transformer):
    """
    Stream selection transformer.
    """

    def __init__(
        self,
        method: Literal["none", "fk"],
        **params,
    ):
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        if self.method == "none":
            return list(data)

        algorithm = STREAM_SELECTION_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown normalizing method '{self.method}'. "
                f"Available methods: {list(STREAM_SELECTION_METHODS.keys())}"
            )

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence[Stream], got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(s, (Stream, DispersionImage)) for s in data):
            raise TypeError("All elements must be Stream or DispersionImage")

        first = data[0]

        if isinstance(first, Stream):
            algorithm_stream = STREAM_SELECTION_METHODS.get(self.method)
            if algorithm_stream is None:
                raise ValueError(
                    f"Unknown normalizing method '{self.method}'. "
                    f"Available methods: {list(STREAM_SELECTION_METHODS.keys())}"
                )
            streams_out: list[Stream] = []
            for stream in data:
                stream_out = algorithm(
                    stream=stream,
                    **self.params,
                )
                if stream_out is not None:
                    streams_out.append(stream_out)
            return streams_out

        else:
            raise TypeError(f"No save handler for {type(first).__name__}")
