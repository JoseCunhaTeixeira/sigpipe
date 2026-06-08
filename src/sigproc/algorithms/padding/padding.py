import numpy as np

from src.sigproc.base.stream import Stream


def pad(
    stream: Stream,
    *,
    n: int,
    taper: int = 0,
) -> Stream:

    if n < 0:
        raise ValueError(f"n ({n}) must be non-negative")

    xt = stream.xt.copy()

    if taper > 0:
        if taper > xt.shape[1]:
            raise ValueError(f"taper ({taper}) cannot exceed nt ({xt.shape[1]})")

        window = 0.5 * (1 + np.cos(np.linspace(0, np.pi, taper)))

        xt[:, -taper:] *= window

    xt = np.pad(
        xt,
        ((0, 0), (0, n)),
        mode="constant",
    )

    nt = xt.shape[1]

    ts = (
        np.arange(
            nt,
            dtype=np.float32,
        )
        / stream.sampling_freq
    )

    if stream.ts.size:
        ts += stream.ts[0]

    return Stream(
        xt=xt,
        ts=ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
    )
