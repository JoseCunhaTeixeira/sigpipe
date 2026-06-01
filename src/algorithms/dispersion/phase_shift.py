import numpy as np
from scipy.fft import rfft, rfftfreq

from src.base.dispersion import DispersionImage
from src.base.stream import Stream


def compute_frequency_vector(nt: int, sampling_freq: float) -> np.ndarray:
    if nt <= 0:
        raise ValueError(f"nt ({nt}) must be greather than 0")
    if sampling_freq <= 0:
        raise ValueError(f"sampling_freq ({sampling_freq}) must be greather than 0")
    return rfftfreq(n=nt, d=1 / sampling_freq)


def dispersion_phase_shift(
    stream: Stream,
    *,
    fmin: float,
    fmax: float,
    vmin: float,
    vmax: float,
    nv: int = 1_000,
    vmin_expected: float | None = None,
    vmax_expected: float | None = None,
) -> DispersionImage:
    """Phase-shift transform.

    Computes phase-velocity versus frequency energy map
    using a delay and sum technique according to Park et al. (1998).

    Args:
        stream (Stream): recording data of successive sensors/traces [nx, nt]
        fmin (float): minimum frequency to compute [in Hz]
        fmax (float): maximum frequency to compute [in Hz]
        vmin (float): minimum phase velocity to compute [in m/s]
        vmax (float): maximum phase velocity to compute [in m/s]
        nv (int, optional): number of velocities [samples]. Defaults to 1_000
        vmin_expected (float, optional): minimum expected velocity used to clip the stream-gather
        vmax_expected (float, optional): maximum expected velocity used to clip the stream-gather

    Returns:
        np.tuple[np.ndarray, np.ndarray, np.ndarray]: frequencies [nf], phase velocities [nv], and FV map [nf, nv]
    """
    if not isinstance(stream.xt, np.ndarray) or stream.xt.ndim != 2:
        raise TypeError("xt must be a 2D numpy array [nx, nt]")
    if (
        not isinstance(stream.acquisition.offsets, np.ndarray)
        or stream.acquisition.offsets.ndim != 1
        or stream.acquisition.offsets.size != stream.nx
    ):
        raise ValueError("offsets must be a 1D array with length nx")
    if stream.sampling_freq <= 0:
        raise ValueError(f"sampling_freq {stream.sampling_freq} must be positive")
    if not (0 <= fmin < fmax):
        raise ValueError(f"require 0 Hz <= fmin ({fmin} Hz) < fmax ({fmax} Hz)")
    if not (0 <= vmin < vmax):
        raise ValueError(f"require 0 m/s < vmin ({vmin} m/s) < vmax ({vmax} m/s)")
    if nv <= 0:
        raise ValueError(f"nv {nv} must be positive")

    xt = stream.xt * np.sqrt(
        stream.acquisition.offsets[:, None]
    )  # Compensate geometrical attenuation sqrt(r)

    if vmin_expected is not None:
        if not (vmin <= vmin_expected < vmax):
            raise ValueError(
                f"require vmin ({vmin} m/s) <= vmin_expected ({vmin_expected} m/s) < vmax ({vmax} m/s)"
            )
        tlims = stream.acquisition.offsets / vmin_expected
        for i_trace, (trace, tlim) in enumerate(zip(xt, tlims)):
            xt[i_trace, stream.ts > tlim] = 0.0

    if vmax_expected is not None:
        if not (vmin < vmax_expected <= vmax):
            raise ValueError(
                f"require vmin ({vmin} m/s) < vmax_expected ({vmax_expected} m/s) <= vmax ({vmax} m/s)"
            )
        tlims = stream.acquisition.offsets / vmax_expected
        for i_trace, (trace, tlim) in enumerate(zip(xt, tlims)):
            xt[i_trace, stream.ts < tlim] = 0.0

    xf = np.array(rfft(xt, axis=1, n=stream.nt), dtype=np.complex64)
    fs = compute_frequency_vector(nt=stream.nt, sampling_freq=stream.sampling_freq)
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
        dphi = two_pi * stream.acquisition.offsets[:, None] * fs / (v + 1e-12)
        fv_map[:, v_i] = np.abs(np.sum(xf_norm * np.exp(1j * dphi), axis=0))
    fv_map /= len(np.unique(stream.acquisition.offsets))
    return DispersionImage(
        fv_map=fv_map.astype(np.float32),
        fs=fs.astype(np.float32),
        vs=vs.astype(np.float32),
        type="phase",
        acquisitions=(stream.acquisition,),
    )
