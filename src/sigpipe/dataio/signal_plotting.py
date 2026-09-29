"""A stream's spectra (a record's, a window's stacked correlations), for the figure a run saves
beside it, the data Visualization draws them from, and PAC's filter preview: each trace's
amplitude spectrum by its receiver's position, and the band the signal checks (G1) found or a
filter keeps."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sigpipe.base.stream import Stream
from sigpipe.dataio.plot_config import CM, DISP_DPI, DOUBLE_COLUMN_CM, SAVING_DPI

# The band drawn over the spectra: PAC's pass green.
_BAND = "#1a9e4b"
_SMALL = 7
# The frequencies a spectrum keeps at most: the largest of each group of them (a figure, a
# preview, shows no more).
SPECTRUM_POINTS = 1000
# The figure and its data, beside the stream (the data's amplitudes in 256 levels: the colour
# map's).
SPECTRA_STEM = "Spectrum"
_LEVELS = 255


@dataclass(frozen=True)
class TraceSpectra:
    """Each trace's amplitude spectrum at its receiver along the line."""

    positions: np.ndarray  # m
    freqs: np.ndarray  # Hz, 0 to Nyquist
    amplitude: np.ndarray  # traces x freqs, 0 to 1 of each trace's largest
    band: tuple[float, float] | None  # the band drawn over them, Hz


def trace_spectra(
    xt: np.ndarray, sampling_freq: float, points: int = SPECTRUM_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """Each trace's amplitude spectrum (its mean removed), from 0 to Nyquist, each scaled to its
    own largest (0 to 1), the largest of each group of frequencies kept (`points` at most):
    frequencies, and amplitudes (traces x frequencies)."""
    data = np.nan_to_num(np.asarray(xt, dtype=float))
    data = data - data.mean(axis=1, keepdims=True)
    amplitude = np.abs(np.fft.rfft(data, axis=1))
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / sampling_freq)
    group = max(1, -(-freqs.size // points))
    n = freqs.size // group * group
    kept = np.concatenate((freqs[:n:group], freqs[n:][:1]))
    grouped = amplitude[:, :n].reshape(amplitude.shape[0], -1, group).max(axis=2)
    if n < freqs.size:
        grouped = np.concatenate((grouped, amplitude[:, n:].max(axis=1, keepdims=True)), axis=1)
    peaks = grouped.max(axis=1, keepdims=True)
    return kept, grouped / np.where(peaks > 0, peaks, 1.0)


def plot_trace_spectra(
    positions: np.ndarray,
    freqs: np.ndarray,
    amplitude: np.ndarray,
    band: tuple[float, float] | None,
) -> Figure:
    """Each trace's amplitude spectrum (`amplitude`, traces x `freqs`, each scaled to its own
    largest) at its receiver's position, bone reversed (white: nothing, black: the trace's
    largest): the whole of it, 0 to Nyquist; `band` (Hz) dashed. No title."""
    fig, ax = plt.subplots(
        figsize=(DOUBLE_COLUMN_CM * CM, 9.0 * CM), dpi=DISP_DPI, layout="constrained"
    )
    mesh = ax.pcolormesh(
        positions, freqs, amplitude.T, shading="nearest", cmap="bone_r", vmin=0.0, vmax=1.0
    )
    bar = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.035)
    bar.set_label("Amplitude [normalized per trace]", fontsize=_SMALL)
    bar.ax.tick_params(labelsize=_SMALL - 1)
    if band is not None:
        for f in band:
            ax.axhline(f, color=_BAND, linestyle=(0, (5, 4)), linewidth=1.0)
        ax.plot(
            [],
            [],
            color=_BAND,
            linestyle=(0, (5, 4)),
            label=f"usable band, {band[0]:.0f}-{band[1]:.0f} Hz",
        )
        ax.legend(fontsize=_SMALL - 1, frameon=True, loc="upper right")
    ax.set_xlabel("Position [m]", fontsize=_SMALL)
    ax.set_ylabel("Frequency [Hz]", fontsize=_SMALL)
    ax.tick_params(labelsize=_SMALL - 1)
    return fig


def stream_spectra(stream: Stream, band: tuple[float, float] | None = None) -> TraceSpectra:
    """`stream`'s spectra (trace_spectra) at its receivers along the line: their x, or their
    offsets when the line runs along y."""
    positions = np.array([receiver.x for receiver in stream.acquisition.receivers], dtype=float)
    if np.ptp(positions) == 0 and positions.size > 1:
        positions = np.asarray(stream.acquisition.offsets, dtype=float)
    freqs, amplitude = trace_spectra(stream.xt, stream.sampling_freq)
    return TraceSpectra(positions=positions, freqs=freqs, amplitude=amplitude, band=band)


def save_spectra(
    stream: Stream, folder: Path, band: tuple[float, float] | None = None, index: int = 0
) -> Path:
    """`stream`'s spectra in `folder`: the figure (`Spectrum_0000.png`, `band` dashed) and its
    data (`Spectrum_0000.npz`), for Visualization to draw them without computing them. Returns
    the figure's path."""
    spectra = stream_spectra(stream, band)
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"{SPECTRA_STEM}_{index:04d}"
    np.savez_compressed(
        stem.with_suffix(".npz"),
        positions=spectra.positions,
        freqs=spectra.freqs.astype(np.float32),
        amplitude=np.round(spectra.amplitude * _LEVELS).astype(np.uint8),
        band=np.asarray(band if band is not None else (), dtype=float),
    )
    figure = plot_trace_spectra(spectra.positions, spectra.freqs, spectra.amplitude, band)
    path = stem.with_suffix(".png")
    figure.savefig(path, bbox_inches="tight", dpi=SAVING_DPI)
    plt.close(figure)
    return path


def load_spectra(folder: Path, index: int = 0) -> TraceSpectra | None:
    """The spectra save_spectra saved in `folder`; None when it saved none."""
    path = folder / f"{SPECTRA_STEM}_{index:04d}.npz"
    if not path.exists():
        return None
    with np.load(path) as saved:
        band = tuple(float(one) for one in saved["band"])
        return TraceSpectra(
            positions=saved["positions"],
            freqs=saved["freqs"].astype(float),
            amplitude=saved["amplitude"].astype(float) / _LEVELS,
            band=(band[0], band[1]) if len(band) == 2 else None,
        )
