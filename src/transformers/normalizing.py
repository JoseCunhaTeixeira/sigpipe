from src.algorithms.normalizing.registry import NORMALIZING_METHODS
from src.base.stream import Stream
from src.base.transformer import Transformer


class NormalizingTransformer(Transformer):
    """
    Time normalization transformer.
    """

    def __init__(
        self,
        method: str,
        **params,
    ):
        self.method = method
        self.params = params

    def transform(self, data: Stream) -> Stream:

        algorithm = NORMALIZING_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown normalizing method '{self.method}'. "
                f"Available methods: {list(NORMALIZING_METHODS.keys())}"
            )

        if not isinstance(data, Stream):
            raise TypeError(f"Expected Stream, got {type(data).__name__}")

        out_xt = algorithm(
            data.xt,
            **self.params,
        )

        return Stream(
            xt=out_xt,
            ts=data.ts,
            sampling_freq=data.sampling_freq,
            acquisition=data.acquisition,
        )
