import numpy as np

from src.base.stream import Stream


def stack_linear(
    streams: list[Stream],
) -> Stream:
    ref_shot = streams[0]
    ref_nx = ref_shot.nx
    ref_nt = ref_shot.nt
    if any(s.nx != ref_nx or s.nt != ref_nt for s in streams):
        raise ValueError("Inconsistent Stream dimensions")
    cube = np.stack([s.xt for s in streams], axis=0)
    out_xt = np.zeros((ref_shot.nx, ref_shot.nt))
    for i_receiver in range(ref_shot.nx):
        out_xt[i_receiver, :] = (np.mean(cube[:, i_receiver, :], axis=0),)
    return Stream(
        xt=out_xt,
        ts=ref_shot.ts,
        sampling_freq=ref_shot.sampling_freq,
        acquisition=ref_shot.acquisition,
    )
