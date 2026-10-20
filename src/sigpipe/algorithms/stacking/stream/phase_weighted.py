from collections.abc import Sequence

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import hilbert

from sigpipe.base.stream import Stream


def stack_phase_weighted(
    streams: list[Stream],
    *,
    nu: int = 2,
) -> Stream:
    """The streams' phase-weighted stack, receiver by receiver (pws)."""
    return stack_phase_weighted_given(streams, None, nu=nu)


def stack_phase_weighted_given(
    streams: Sequence[Stream],
    phases: Sequence[np.ndarray] | None,
    *,
    nu: int = 2,
) -> Stream:
    """stack_phase_weighted with each stream's instantaneous phase given (instantaneous_phase of
    its xt), when made before, as for stacks of the same streams kept in several ways; None: each
    measured. Not a stacking method: the stages' parameters are its signature's."""
    ref_shot = streams[0]
    ref_nx = ref_shot.nx
    ref_nt = ref_shot.nt
    if any(s.nx != ref_nx or s.nt != ref_nt for s in streams):
        raise ValueError("Inconsistent Stream dimensions")
    if phases is not None and len(phases) != len(streams):
        raise ValueError(f"requires one phase a stream, got {len(phases)} for {len(streams)}")
    cube = np.stack([s.xt for s in streams], axis=0)
    phase_cube = np.stack(phases, axis=0) if phases is not None else None
    out_xt = np.zeros((ref_shot.nx, ref_shot.nt))
    for i_receiver in range(ref_shot.nx):
        out_xt[i_receiver, :] = pws(
            xt=cube[:, i_receiver, :],
            nu=nu,
            phase=phase_cube[:, i_receiver, :] if phase_cube is not None else None,
        )
    return Stream(
        xt=out_xt,
        ts=ref_shot.ts,
        sampling_freq=ref_shot.sampling_freq,
        acquisition=ref_shot.acquisition,
    )


def instantaneous_phase(xt: np.ndarray) -> np.ndarray:
    """The instantaneous phase of each row of `xt` (a stream's traces, or a trace's streams): its
    analytic signal over its magnitude, as pws weighs it."""
    npts = np.shape(xt)[-1]
    nfft = next_fast_len(npts)
    anal_sig = np.array(hilbert(xt, N=nfft, axis=-1), np.complex64)[..., :npts]
    return anal_sig / (np.abs(anal_sig) + 1e-12)


def pws(
    xt: np.ndarray,
    *,
    nu: int = 2,
    phase: np.ndarray | None = None,
) -> np.ndarray:
    """Phase-weighted stack according to Schimmel and Paulssen (1997); `phase`: the instantaneous
    phase of `xt`'s rows, when made before."""
    if not isinstance(xt, np.ndarray) or xt.ndim != 2:
        raise TypeError("xt must be a 2D numpy array: [ntraces, nt]")
    if nu < 0:
        msg = f"nu {nu} must be greater or equal to 0"
        raise ValueError(msg)
    instant_phase = instantaneous_phase(xt) if phase is None else phase
    phase_stack = np.abs(np.mean(instant_phase, axis=0)) ** nu
    return np.asarray(np.mean(xt, axis=0) * phase_stack, dtype=xt.dtype)
