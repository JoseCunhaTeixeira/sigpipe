from collections.abc import Sequence

import numpy as np

from src.algorithms.stacking.registry import STACKING_METHODS
from src.base.stream import Stream
from src.base.transformer import Transformer


class StackingTransformer(Transformer):
    """
    Stacking transformer.
    """

    def __init__(
        self,
        method: str,
        **params,
    ):
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> Sequence[Stream]:

        algorithm = STACKING_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown stacking method '{self.method}'. "
                f"Available methods: {list(STACKING_METHODS.keys())}"
            )

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence[Stream], got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(s, Stream) for s in data):
            raise TypeError("All elements must be Stream")

        ref_shot = data[0]
        nx = ref_shot.nx
        nt = ref_shot.nt
        if any(s.nx != nx or s.nt != nt for s in data):
            raise ValueError("Inconsistent Stream dimensions")

        cube = np.stack([s.xt for s in data], axis=0)
        out_xt = np.zeros((ref_shot.nx, ref_shot.nt))
        for i_receiver in range(ref_shot.nx):
            traces = cube[:, i_receiver, :]
            out_xt[i_receiver, :] = algorithm(traces, **self.params)

        return (
            Stream(
                xt=out_xt,
                ts=ref_shot.ts,
                sampling_freq=ref_shot.sampling_freq,
                acquisition=ref_shot.acquisition,
            ),
        )
