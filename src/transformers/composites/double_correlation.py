from collections.abc import Sequence

from src.base.stream import Stream
from src.base.transformer import Transformer
from src.transformers.correlation import Correlate
from src.transformers.flipping import Flip


class BidirectionalCorrelate(Transformer[Stream, Stream]):
    """
    Double correlation transformer for linear arrays.
    """

    def __init__(
        self,
        method: str,
        **params,
    ):
        self.method = method
        self.params = params

    def transform(
        self,
        data: Sequence[Stream],
    ) -> Sequence[Stream]:

        if not data:
            return []

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
