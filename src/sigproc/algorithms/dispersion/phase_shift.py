import numpy as np
from scipy.fft import rfft, rfftfreq

from sigproc.base.dispersion import DispersionImage
from sigproc.base.stream import Stream


def dispersion_phase_shift(
    stream: Stream,
    *,
    fmin: float,
    fmax: float,
    vmin: float,
    vmax: float,
    nv: int = 1_000,
) -> DispersionImage:
    if stream.acquisition.is_unknown:
        raise ValueError("dispersion has unknown acquisition")
    fs, vs, fv_map = phase_shift(
        xt=stream.xt,
        sampling_freq=stream.sampling_freq,
        offsets=stream.acquisition.offsets,
        fmin=fmin,
        fmax=fmax,
        vmin=vmin,
        vmax=vmax,
        nv=nv,
    )
    return DispersionImage(
        fv_map=fv_map,
        fs=fs,
        vs=vs,
        type="phase",
        acquisitions=(stream.acquisition,),
    )


def phase_shift(
    xt: np.ndarray,
    sampling_freq: float,
    offsets: np.ndarray,
    *,
    fmin: float,
    fmax: float,
    vmin: float,
    vmax: float,
    nv: int = 1_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase-shift transform.

    Computes phase-velocity versus frequency energy map
    using a delay and sum technique according to Park et al. (1998).

    """
    if not isinstance(xt, np.ndarray) or xt.ndim != 2:
        raise TypeError("xt must be a 2D numpy array [nx, nt]")
    nx, nt = xt.shape
    if not isinstance(offsets, np.ndarray) or offsets.ndim != 1 or offsets.size != nx:
        raise ValueError("offsets must be a 1D array with length nx")
    if sampling_freq <= 0:
        raise ValueError(f"requires sampling_freq > 0 Hz, got {sampling_freq} Hz")
    if not (0 <= fmin < fmax):
        raise ValueError(f"requires 0 Hz <= fmin < fmax, got {fmin} Hz and {fmax} Hz")
    if not (vmin < vmax):
        raise ValueError(f"requires vmin < vmax, got {vmin} m/s and {vmax} m/s")
    if nv <= 0:
        raise ValueError(f"requires nv > 0, got {nv}")

    xt = xt * np.sqrt(offsets[:, None])  # Compensate geometrical attenuation sqrt(r)

    xf = np.array(rfft(xt, axis=1, n=nt), dtype=np.complex64)
    fs = compute_frequency_vector(nt=nt, sampling_freq=sampling_freq)
    mask = (fs >= fmin) & (fs <= fmax)
    if not np.any(mask):
        raise ValueError(
            f"no frequencies found in the requested band ({fmin} to {fmax} Hz)"
        )
    fs = fs[mask]
    xf = xf[:, mask]
    vs = np.linspace(vmin, vmax, nv)
    if vs.size == 0:
        raise ValueError("velocity range produced an empty array")
    fv_map = np.zeros((len(fs), len(vs)), dtype=np.float32)
    xf_norm = xf / (np.abs(xf) + 1e-12)
    two_pi = 2 * np.pi
    for v_i, v in enumerate(vs):
        dphi = two_pi * offsets[:, None] * fs / (v + 1e-12)
        fv_map[:, v_i] = np.abs(np.sum(xf_norm * np.exp(1j * dphi), axis=0))
    fv_map /= len(np.unique(offsets))
    return (
        fs.astype(np.float32),
        vs.astype(np.float32),
        fv_map.astype(np.float32),
    )


def compute_frequency_vector(nt: int, sampling_freq: float) -> np.ndarray:
    if nt <= 0:
        raise ValueError(f"requires nt > 0, got {nt}")
    if sampling_freq <= 0:
        raise ValueError(f"requires sampling_freq > 0 Hz, got {sampling_freq} Hz")
    return rfftfreq(n=nt, d=1 / sampling_freq)
