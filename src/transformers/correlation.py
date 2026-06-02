from collections.abc import Sequence

from src.algorithms.correlation.registry import CORRELATION_METHODS
from src.base.stream import Stream
from src.base.transformer import Transformer


class Correlation(Transformer):
    """
    Correlation transformer.
    """

    def __init__(
        self,
        method: str,
        virtual_source_index: int,
        **params,
    ):
        self.method = method
        self.virtual_source_index = virtual_source_index
        self.params = params

    def transform(self, data: Sequence[Stream]) -> Sequence[Stream]:

        algorithm = CORRELATION_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown normalizing method '{self.method}'. "
                f"Available methods: {list(CORRELATION_METHODS.keys())}"
            )

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence[Stream], got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(s, Stream) for s in data):
            raise TypeError("All elements must be Stream")

        streams_out = []
        for stream in data:
            stream_causal, stream_acausal = algorithm(
                stream=stream,
                virtual_source_index=self.virtual_source_index,
                **self.params,
            )
            streams_out.append(stream_causal)
            streams_out.append(stream_acausal)

        return streams_out
