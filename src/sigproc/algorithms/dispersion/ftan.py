import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import hilbert

from sigproc.base.dispersion_curve import VelocityType
from sigproc.base.dispersion_image import DispersionImage
from sigproc.base.stream import Stream


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
) -> list[DispersionImage]:
    """Frequency-Time Analysis (FTAN) for dispersion imaging.

    Computes group velocity versus frequency energy map from a single trace using
    narrow-band Gaussian filtering and envelope detection according to Dziewonski et al. (1969),
    Levshin et al. (1972), and Bensen et al. (2007).
    """
    if stream.acquisition.is_unknown:
        raise ValueError("dispersion has unknown acquisition")
    dispersion_images = []
    for trace, offset in zip(stream.xt, stream.acquisition.offsets, strict=False):
        f0s, vs, fv_map = ftan(
            trace=trace,
            offset=offset,
            sampling_freq=stream.sampling_freq,
            fmin=fmin,
            fmax=fmax,
            vmin=vmin,
            vmax=vmax,
            nf=nf,
            nv=nv,
            alpha=alpha,
            normalize=normalize,
        )
        dispersion_images.append(
            DispersionImage(
                fv_map=fv_map,
                fs=f0s,
                vs=vs,
                type=VelocityType.GROUP,
                acquisition=stream.acquisition,
            )
        )
    return dispersion_images


def ftan(
    trace: np.ndarray,
    offset: float,
    sampling_freq: float,
    *,
    fmin: float,
    fmax: float,
    vmin: float,
    vmax: float,
    nf: int = 1_000,
    nv: int = 1_000,
    alpha: float = 10.0,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frequency-Time Analysis (FTAN) for dispersion imaging.

    Computes group velocity versus frequency energy map from a single trace using
    narrow-band Gaussian filtering and envelope detection according to Dziewonski et al. (1969),
    Levshin et al. (1972), and Bensen et al. (2007).
    """
    if not isinstance(trace, np.ndarray) or trace.ndim != 1:
        raise TypeError("trace must be a 1D numpy array [nt]")
    if sampling_freq <= 0:
        raise ValueError(f"requires sampling_freq > 0 Hz, got {sampling_freq} Hz")
    if offset < 0:
        raise ValueError(f"requires offset >= 0 m, got {offset} m")
    if not (0 < fmin < fmax):
        raise ValueError(f"requires 0 Hz < fmin < fmax, got {fmin} Hz and {fmax} Hz")
    if not (0 < vmin < vmax):
        raise ValueError(f"requires 0 m/s < vmin < vmax, got {vmin} m/s and {vmax} m/s)")
    if nf <= 0 or nv <= 0:
        raise ValueError(f"requires nf > 0 and nv > 0, got {nf} and {nv}")
    if alpha <= 0:
        raise ValueError(f"requires alpha > 0, got {alpha}")
    dt = 1 / sampling_freq
    nt = len(trace)
    ts = np.arange(nt) * dt
    trace_fft = fft(trace)
    fs = fftfreq(nt, dt)
    f0s = np.logspace(np.log10(fmin), np.log10(fmax), nf)
    vs = np.linspace(vmin, vmax, nv)
    fv_map = np.zeros((nf, nv))
    for i, f0 in enumerate(f0s):
        gaussian = np.exp(-alpha * ((np.abs(fs) - f0) / f0) ** 2)
        filtered = np.array(ifft(trace_fft * gaussian), dtype=np.complex64).real
        envelope = np.abs(np.array(hilbert(filtered), dtype=np.complex64))
        for j, v in enumerate(vs):
            travel_time = offset / v
            fv_map[i, j] = np.interp(travel_time, ts, envelope, left=0, right=0)
    if normalize:
        eps = 1e-12
        fv_map /= np.max(fv_map, axis=1, keepdims=True) + eps
    return f0s.astype(np.float32), vs.astype(np.float32), fv_map.astype(np.float32)
