"""A record's spectra, saved beside its preprocessed file with the usable band its signal checks
(G1) found: what PAC's measures and the assistant's G1 draw alike."""

from pathlib import Path

from sigpipe.base.stream import Stream
from sigpipe.dataio.signal_plotting import SPECTRA_STEM, save_spectra

SPECTRA_FIGURE = f"{SPECTRA_STEM}_0000.png"


def save_record_spectra(
    stream: Stream,
    folder: Path,
    band: tuple[float, float] | None = None,
) -> Path:
    """The spectra of preprocessed record `stream` (after its whole preprocessing: trigger,
    detrend, mute and filter, as the windows use it; the user, 2026-09-29) in record folder
    `folder`: the figure (SPECTRA_FIGURE), each trace's amplitude spectrum at its receiver, the
    usable band its checks found (`band`) dashed, and its data (sigpipe's save_spectra)."""
    return save_spectra(stream, folder, band)
