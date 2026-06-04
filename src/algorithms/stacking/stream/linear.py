import numpy as np

from src.base.stream import Stream


def stack_linear(
    streams: list[Stream],
) -> Stream:
    if not streams:
        raise ValueError("list cannot be empty.")
    reference = streams[0]

    if any(
        stream.nx != reference.nx or stream.nt != reference.nt for stream in streams
    ):
        raise ValueError("Inconsistent Stream dimensions.")

    if any(stream.sampling_freq != reference.sampling_freq for stream in streams):
        raise ValueError("Inconsistent sampling frequencies.")

    if any(not np.allclose(stream.ts, reference.ts) for stream in streams):
        raise ValueError("Inconsistent time axes.")

    cube = np.stack([stream.xt for stream in streams], axis=0)
    xt_stack = np.mean(cube, axis=0)

    return Stream(
        xt=xt_stack,
        ts=reference.ts,
        sampling_freq=reference.sampling_freq,
        acquisition=reference.acquisition,
    )
