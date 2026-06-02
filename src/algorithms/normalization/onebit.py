import numpy as np

from src.base.stream import Stream


def normalize_onebit(stream: Stream) -> Stream:
    return Stream(
        xt=np.sign(stream.xt),
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
    )
