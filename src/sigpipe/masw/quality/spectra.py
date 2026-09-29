"""A record's spectra, saved beside its preprocessed file with the usable band its signal checks
(G1) found: what PAC's measures and the assistant's G1 draw alike."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sigpipe.base.stream import Stream
from sigpipe.dataio.signal_plotting import plot_trace_spectra, trace_spectra
from sigpipe.transformers import Plot

SPECTRA_FIGURE = "Spectrum_0000.png"


def save_record_spectra(
    stream: Stream,
    folder: Path,
    band: tuple[float, float] | None = None,
) -> Path:
    """The figure of preprocessed record `stream`'s spectra (after its whole preprocessing:
    trigger, detrend, mute and filter, as the windows use it; the user, 2026-09-29) in record
    folder `folder` (SPECTRA_FIGURE): each trace's amplitude spectrum at its receiver, the usable
    band its checks found (`band`) dashed."""
    positions = np.array([receiver.x for receiver in stream.acquisition.receivers], dtype=float)
    if np.ptp(positions) == 0 and positions.size > 1:
        positions = np.asarray(stream.acquisition.offsets, dtype=float)
    freqs, amplitude = trace_spectra(stream.xt, stream.sampling_freq)
    figure = plot_trace_spectra(positions, freqs, amplitude, band)
    path = folder / SPECTRA_FIGURE
    Plot.savefig(path=path, figure=figure)
    plt.close(figure)
    return path
