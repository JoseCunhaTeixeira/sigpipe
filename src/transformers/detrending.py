from collections.abc import Sequence

from src.algorithms.detrending.registry import DETRENDING_METHODS
from src.base.stream import Stream
from src.base.transformer import Transformer


class DetrendingTransformer(Transformer):
    """
    Detrending transformer.
    """

    def __init__(
        self,
        method: str,
        **params,
    ):
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> Sequence[Stream]:

        algorithm = DETRENDING_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown normalizing method '{self.method}'. "
                f"Available methods: {list(DETRENDING_METHODS.keys())}"
            )

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence[Stream], got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(s, Stream) for s in data):
            raise TypeError("All elements must be Stream")

        streams_out = []
        for stream in data:
            out_xt = algorithm(
                stream.xt,
                **self.params,
            )

            stream_out = Stream(
                xt=out_xt,
                ts=stream.ts,
                sampling_freq=stream.sampling_freq,
                acquisition=stream.acquisition,
            )

            streams_out.append(stream_out)

        return streams_out
