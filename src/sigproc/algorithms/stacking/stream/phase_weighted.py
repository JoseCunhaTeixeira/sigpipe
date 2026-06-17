import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import hilbert

from sigproc.base.stream import Stream


def stack_phase_weighted(
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
        out_xt[i_receiver, :] = pws(
            xt=cube[:, i_receiver, :],
            power=power,
        )
    return Stream(
        xt=out_xt,
        ts=ref_shot.ts,
        sampling_freq=ref_shot.sampling_freq,
        acquisition=ref_shot.acquisition,
    )


def pws(
    xt: np.ndarray,
    *,
    power: int = 2,
) -> np.ndarray:
    """Phase-weighted stack according to Schimmel and Paulssen (1997)."""
    if not isinstance(xt, np.ndarray) or xt.ndim != 2:
        raise TypeError("xt must be a 2D numpy array: [ntraces, nt]")
    if power < 0:
        msg = f"power {power} must be greater or equal to 0"
        raise ValueError(msg)
    npts = np.shape(xt)[1]
    nfft = next_fast_len(npts)
    anal_sig = np.array(hilbert(xt, N=nfft, axis=-1), np.complex64)[:, :npts]
    instant_phase = anal_sig / (np.abs(anal_sig) + 1e-12)
    phase_stack = np.abs(np.mean(instant_phase, axis=0)) ** power
    trace_stacked = np.mean(xt, axis=0) * phase_stack
    return trace_stacked
