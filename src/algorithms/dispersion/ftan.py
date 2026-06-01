import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import hilbert

from src.base.dispersion import DispersionImage
from src.base.stream import Stream


def dispersion_ftan(
    stream: Stream,
    *,
    fmin: float,
    fmax: float,
    vmin: float,
    vmax: float,
    nf: int = 1_000,
    nv: int = 1_000,
    alpha: float = 10.0,
    normalize: bool = False,
) -> DispersionImage:
    """Frequency-Time Analysis (FTAN) for dispersion imaging.

    Computes group velocity versus frequency energy map from a single trace using
    narrow-band Gaussian filtering and envelope detection according to Dziewonski et al. (1969),
    Levshin et al. (1972), and Bensen et al. (2007).

    Args:
        trace (np.ndarray): input time-domain signal [nt]
        sampling_freq (float): sampling frequency [Hz]
        distance (float): source-receiver distance [in m].
        fmin (float): minimum frequency of analysis [Hz]
        fmax (float): maximum frequency of analysis [Hz]
        vmin (float): minimum velocity to scan [m/s]
        vmax (float): maximum velocity to scan [m/s]
        nf (int, optional): number of frequency samples. Defaults to 1_000
        nv (int, optional): number of velocity samples. Defaults to 1_000
        alpha (float, optional): width parameter of the Gaussian filters. Defaults to 10.0
        normalize (bool, optional): normalize per frequencies. Defaults to False

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: frequencies [N_f], group velocities [N_v], and FV map [n_f, n_v]
    """
    trace = stream.xt[0]  # Unique trace
    distance = stream.acquisition.offsets[0]  # Unique trace
    if not isinstance(trace, np.ndarray) or trace.ndim != 1:
        raise TypeError("trace must be a 1D numpy array [nt]")
    if stream.sampling_freq <= 0:
        raise ValueError("sampling_freq must be positive")
    if distance <= 0:
        raise ValueError("distance must be positive")
    if not (0 < fmin < fmax):
        raise ValueError(f"require 0 Hz < fmin ({fmin} Hz) < fmax ({fmax} Hz)")
    if not (0 < vmin < vmax):
        raise ValueError(f"require 0 m/s < vmin ({vmin} m/s) < vmax ({vmax} m/s)")
    if nf <= 0 or nv <= 0:
        raise ValueError("nf and nv must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    dt = 1 / stream.sampling_freq
    ts = np.arange(stream.nt) * dt
    trace_fft = fft(trace)
    fs = fftfreq(stream.nt, dt)
    f0s = np.logspace(np.log10(fmin), np.log10(fmax), nf)
    vs = np.linspace(vmin, vmax, nv)
    fv_map = np.zeros((nf, nv))
    for i, f0 in enumerate(f0s):
        gaussian = np.exp(-alpha * ((np.abs(fs) - f0) / f0) ** 2)
        filtered = np.array(ifft(trace_fft * gaussian), dtype=np.complex64).real
        envelope = np.abs(np.array(hilbert(filtered), dtype=np.complex64))
        for j, v in enumerate(vs):
            travel_time = distance / v
            fv_map[i, j] = np.interp(travel_time, ts, envelope, left=0, right=0)
    if normalize:
        eps = 1e-12
        fv_map /= np.max(fv_map, axis=1, keepdims=True) + eps
    return DispersionImage(
        fv_map=fv_map.astype(np.float32),
        fs=f0s.astype(np.float32),
        vs=vs.astype(np.float32),
        type="group",
        acquisitions=(stream.acquisition,),
    )
