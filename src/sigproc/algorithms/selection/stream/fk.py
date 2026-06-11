import numpy as np
from numpy.fft import fft, fftfreq
from scipy.fft import rfft, rfftfreq
from scipy.fftpack import next_fast_len

from sigproc.base.stream import Stream


def selection_fk(
    stream: Stream,
    threshold: float,
    vmin: float,
    vmax: float,
) -> Stream | None:

    if not 0 <= threshold <= 1:
        raise ValueError(f"requires 0 <= threshold <= 1, got {threshold}")

    nf_fft = next_fast_len(stream.nt)
    fs = rfftfreq(nf_fft, d=1.0 / stream.sampling_freq).astype(np.float32)
    xf = np.array(rfft(stream.xt, n=nf_fft, axis=1), dtype=np.complex64)

    if stream.nx < 2:
        raise ValueError(
            f"At least 2 receivers are required to define a wavenumber axis, got {stream.nx}"
        )
    dx = np.diff(stream.acquisition.offsets)
    if not np.allclose(dx, dx[0]):
        raise ValueError("Non-uniform receiver spacing")
    nk_fft = next_fast_len(stream.nx)
    ks = fftfreq(nk_fft, d=dx[0]).astype(np.float32)
    kf = np.abs(fft(xf, n=nk_fft, axis=0)).astype(np.float32)

    mask_pos = ks > 0
    mask_neg = ks < 0
    K_pos = kf[mask_pos]
    ks_pos = ks[mask_pos]
    K_neg = kf[mask_neg]
    ks_neg = ks[mask_neg]

    # Limit the FK diagrams to the velocity range as in Cheng et al. (2018)
    # Users should avoid wavenumber=frequency/velocity > 1/(2*dx) by
    # limiting frequencies in accordance to the velocities to study
    fs_min_lim_neg = vmin * abs(ks_neg)
    fs_max_lim_neg = vmax * abs(ks_neg)
    for i_k in range(len(ks_neg)):
        if fs[0] <= fs_min_lim_neg[i_k] <= fs[-1]:
            i_flim = np.where(fs >= fs_min_lim_neg[i_k])[0][0]
            K_neg[: i_flim + 1, i_k] = 0
        if fs[0] <= fs_max_lim_neg[i_k] <= fs[-1]:
            i_flim = np.where(fs >= fs_max_lim_neg[i_k])[0][0]
            K_neg[i_flim:, i_k] = 0

    fs_min_lim_pos = vmin * abs(ks_pos)
    fs_max_lim_pos = vmax * abs(ks_pos)
    for i_k in range(len(ks_pos)):
        if fs[0] <= fs_min_lim_pos[i_k] <= fs[-1]:
            i_flim = np.where(fs >= fs_min_lim_pos[i_k])[0][0]
            K_pos[: i_flim + 1, i_k] = 0
        if fs[0] <= fs_max_lim_pos[i_k] <= fs[-1]:
            i_flim = np.where(fs >= fs_max_lim_pos[i_k])[0][0]
            K_pos[i_flim:, i_k] = 0

    eps = 1e-12
    energy_pos = np.sum(K_pos)
    energy_neg = np.sum(K_neg)
    fk_ratio = (energy_pos - energy_neg) / (energy_pos + energy_neg + eps)
    if abs(fk_ratio) > threshold:
        return stream
    return None
