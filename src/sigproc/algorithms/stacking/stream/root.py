import numpy as np

from sigproc.base.stream import Stream


def stack_root(
    streams: list[Stream],
    *,
    n: int = 2,
) -> Stream:
    ref_shot = streams[0]
    ref_nx = ref_shot.nx
    ref_nt = ref_shot.nt
    if any(s.nx != ref_nx or s.nt != ref_nt for s in streams):
        raise ValueError("Inconsistent Stream dimensions")
    cube = np.stack([s.xt for s in streams], axis=0)
    out_xt = np.zeros((ref_shot.nx, ref_shot.nt))
    for i_receiver in range(ref_shot.nx):
        out_xt[i_receiver, :] = rs(
            xt=cube[:, i_receiver, :],
            n=n,
        )
    return Stream(
        xt=out_xt,
        ts=ref_shot.ts,
        sampling_freq=ref_shot.sampling_freq,
        acquisition=ref_shot.acquisition,
    )


def rs(
    xt: np.ndarray,
    *,
    n: int = 2,
) -> np.ndarray:
    """nth-root stack according to Schimmel and Paulssen (1997)."""
    if not isinstance(xt, np.ndarray) or xt.ndim != 2:
        raise TypeError("xt must be a 2D numpy array: [ntraces, nt]")
    if n < 1:
        msg = f"n {n} must be greater or equal to 1"
        raise ValueError(msg)
    r = np.mean(np.sign(xt) * np.abs(xt) ** (1 / n), axis=0)
    return np.asarray(np.sign(r) * np.abs(r) ** n, dtype=xt.dtype)
