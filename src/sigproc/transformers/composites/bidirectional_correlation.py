from collections.abc import Sequence
from typing import Literal

from sigproc.algorithms import FlipAxis
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer
from sigproc.transformers.correlation import Correlate
from sigproc.transformers.flipping import Flip


class BidirectionalCorrelate(Transformer[Stream, Stream]):
    """
    Double correlation transformer for linear arrays.
    """

    def __init__(
        self,
        method: Literal["none", "cross"],
    ) -> None:
        self.method: Literal["none", "cross"] = method

    def transform(
        self,
        data: Sequence[Stream],
    ) -> list[Stream]:

        self.validate_sequence(data, Stream)

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
            axis=FlipAxis.SPACE,
            flip_acquisition=False,
        ).transform(right)

        return [
            *left,
            *right,
        ]
