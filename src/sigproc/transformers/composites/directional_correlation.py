from collections.abc import Sequence
from typing import Literal

from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer
from sigproc.transformers.correlation import Correlate
from sigproc.transformers.flipping import Flip


class DirectionalCorrelation(Transformer):
    """
    Directional correlation depending on the source position for active acquisitions.
    """

    def __init__(
        self,
        method: Literal["none", "cross"],
    ):
        self.method: Literal["none", "cross"] = method

    def transform(
        self,
        data: Sequence[Stream],
    ) -> list[Stream]:

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence[Stream], got {type(data).__name__}")

        if not all(isinstance(s, Stream) for s in data):
            raise TypeError("All elements must be Stream")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if self.method == "none":
            return list(data)

        streams_out = []

        for stream in data:
            if (
                stream.acquisition.source.x <= stream.acquisition.receivers[0].x
            ):  # Source at left
                virtual_source_index = 0
            elif (
                stream.acquisition.source.x >= stream.acquisition.receivers[-1].x
            ):  # Source at right
                virtual_source_index = -1
            else:
                continue

            correl = Correlate(
                method=self.method,
                virtual_source_index=virtual_source_index,
                part="causal",
            ).transform([stream])

            if virtual_source_index == -1:
                correl = Flip(
                    axis="space",
                    flip_acquisition=False,
                ).transform(correl)

            streams_out.extend(correl)

        return streams_out
