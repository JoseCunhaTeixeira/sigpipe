from collections.abc import Sequence
from typing import Literal

from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer
from sigproc.transformers.correlation import Correlate
from sigproc.transformers.flipping import Flip


class BidirectionalCorrelate(Transformer):
    """
    Double correlation transformer for linear arrays.
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

        left = Correlate(
            method=self.method,
            virtual_source_index=0,
            part="causal",
        ).transform(data)

        right = Correlate(
            method=self.method,
            virtual_source_index=-1,
            part="acausal",
        ).transform(data)

        right = Flip(
            axis="space",
            flip_acquisition=False,
        ).transform(right)

        return [
            *left,
            *right,
        ]
