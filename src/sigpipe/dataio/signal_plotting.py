"""A record's spectrum, for the figure a run saves beside it: the surface waves' window against
the noise window, and the usable band the signal checks (G1) found."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sigpipe.dataio.plot_config import CM, DISP_DPI, DOUBLE_COLUMN_CM

# PAC's colours: the signal (its series), the noise, the band a check keeps (its pass green).
_SIGNAL = "#2a78d6"
_NOISE = "#898781"
_BAND = (26 / 255, 158 / 255, 75 / 255, 0.12)
_SMALL = 7
# The frequencies shown: up to where the signal falls this far under its peak for good, and a
# fifth more.
_SHOWN_DB = -40.0


def plot_record_spectrum(
    freqs: np.ndarray,
    signal: np.ndarray,
    noise: np.ndarray | None,
    band: tuple[float, float] | None,
    title: str,
) -> Figure:
    """A record's mean power spectra in dB of the signal's peak: its surface waves' window
    (`signal`; a passive record's whole length), its noise window (`noise`, none for a passive
    record), the usable band (`band`, Hz) shaded; `title` above them."""
    peak = float(np.max(signal)) or 1.0
    signal_db = 10 * np.log10(signal / peak + 1e-30)
    loud = np.flatnonzero(signal_db > _SHOWN_DB)
    top = float(freqs[loud[-1]]) * 1.2 if loud.size else float(freqs[-1])
    fig, ax = plt.subplots(
        figsize=(DOUBLE_COLUMN_CM * CM, 7.0 * CM), dpi=DISP_DPI, layout="constrained"
    )
    if band is not None:
        ax.axvspan(
            *band, color=_BAND, linewidth=0, label=f"usable band, {band[0]:.0f}-{band[1]:.0f} Hz"
        )
    ax.plot(
        freqs,
        signal_db,
        color=_SIGNAL,
        linewidth=0.9,
        label="surface waves" if noise is not None else "the record",
    )
    if noise is not None:
        ax.plot(
            freqs, 10 * np.log10(noise / peak + 1e-30), color=_NOISE, linewidth=0.9, label="noise"
        )
    ax.set_xlim(0.0, min(top, float(freqs[-1])))
    ax.set_ylim(_SHOWN_DB - 20.0, 5.0)
    ax.set_xlabel("Frequency [Hz]", fontsize=_SMALL)
    ax.set_ylabel("Power [dB of the peak]", fontsize=_SMALL)
    ax.set_title(title, fontsize=_SMALL + 1, loc="left")
    ax.tick_params(labelsize=_SMALL - 1)
    ax.grid(color="#ecebe6", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=_SMALL - 1, frameon=False, loc="upper right")
    return fig
