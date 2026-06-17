import numpy as np

from sigproc.base.stream import Stream


def stack_root(
    streams: list[Stream],
    *,
    power: int = 2,
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
            power=power,
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
    power: int = 2,
) -> np.ndarray:
    """nth-root stack according to Schimmel and Paulssen (1997)."""
    if not isinstance(xt, np.ndarray) or xt.ndim != 2:
        raise TypeError("xt must be a 2D numpy array: [ntraces, nt]")
    if power < 1:
        msg = f"power {power} must be greater or equal to 1"
        raise ValueError(msg)
    r = np.mean(np.sign(xt) * np.abs(xt) ** (1 / power), axis=0)
    trace_stacked = np.sign(r) * np.abs(r) ** power
    return trace_stacked
