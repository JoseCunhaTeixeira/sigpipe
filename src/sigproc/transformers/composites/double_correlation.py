from collections.abc import Sequence
from typing import Literal

from src.sigproc.base.stream import Stream
from src.sigproc.base.transformer import Transformer
from src.sigproc.transformers.correlation import Correlate
from src.sigproc.transformers.flipping import Flip


class BidirectionalCorrelate(Transformer[Stream, Stream]):
    """
    Double correlation transformer for linear arrays.
    """

    def __init__(
        self,
        method: Literal["cross"],
        **params,
    ):
        self.method: Literal["cross"] = method
        self.params = params

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
        ).transform(right)

        return [
            *left,
            *right,
        ]
