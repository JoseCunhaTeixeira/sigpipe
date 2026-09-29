"""A record's spectra, for the figure a run saves beside it and PAC's filter preview: each trace's
amplitude spectrum by its receiver's position, and the band the signal checks (G1) found or a
filter keeps."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sigpipe.dataio.plot_config import CM, DISP_DPI, DOUBLE_COLUMN_CM

# The band drawn over the spectra: PAC's pass green.
_BAND = "#1a9e4b"
_SMALL = 7
# The frequencies a spectrum keeps at most: the largest of each group of them (a figure, a
# preview, shows no more).
SPECTRUM_POINTS = 1000


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
    largest): the whole of it, 0 to Nyquist; `band` (Hz) dashed. No title (the user,
    2026-09-29)."""
    fig, ax = plt.subplots(
        figsize=(DOUBLE_COLUMN_CM * CM, 9.0 * CM), dpi=DISP_DPI, layout="constrained"
    )
    mesh = ax.pcolormesh(
        positions, freqs, amplitude.T, shading="nearest", cmap="bone_r", vmin=0.0, vmax=1.0
    )
    bar = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.035)
    bar.set_label("Amplitude [of each trace's largest]", fontsize=_SMALL)
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
