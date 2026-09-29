"""A passive window's segment selection, for the figure a run saves: each segment's score, kept
or not, around the threshold it was kept by."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sigpipe.dataio.plot_config import CM, DISP_DPI, DOUBLE_COLUMN_CM

# PAC's colours: a segment kept (flipped when its ratio is positive), one left out.
_KEPT = "#2a78d6"
_FLIPPED = "#eb6834"
_REJECTED = "#a9abb2"
_LIMIT = "#898781"


def plot_selection(scores: Sequence[tuple[float, bool]], threshold: float) -> Figure:
    """Each segment's f-k ratio (`scores`: its ratio and whether it was kept, in the order the
    window met them, record after record), from -1 (energy all towards negative wavenumbers) to
    1: kept beyond ±`threshold` (dashed; the band between them shaded), those positive flipped
    so that every kept segment runs one way."""
    ratios = np.array([ratio for ratio, _ in scores], dtype=float)
    kept = np.array([one for _, one in scores], dtype=bool)
    segments = np.arange(1, ratios.size + 1)
    fig, ax = plt.subplots(
        figsize=(DOUBLE_COLUMN_CM * CM, 7.0 * CM), dpi=DISP_DPI, layout="constrained"
    )
    ax.axhspan(-threshold, threshold, color=(0.5, 0.5, 0.55, 0.08), linewidth=0)
    for bound in (-threshold, threshold):
        ax.axhline(bound, color=_LIMIT, linestyle=(0, (4, 3)), linewidth=0.8)
    ax.axhline(0.0, color=_LIMIT, linewidth=0.5)
    size = 10 if ratios.size <= 300 else 4
    ax.scatter(
        segments[~kept],
        ratios[~kept],
        s=size,
        facecolors="none",
        edgecolors=_REJECTED,
        linewidths=0.6,
        label=f"left out ({int((~kept).sum())})",
    )
    ax.scatter(
        segments[kept & (ratios < 0)],
        ratios[kept & (ratios < 0)],
        s=size,
        color=_KEPT,
        label=f"kept ({int((kept & (ratios < 0)).sum())})",
    )
    ax.scatter(
        segments[kept & (ratios > 0)],
        ratios[kept & (ratios > 0)],
        s=size,
        color=_FLIPPED,
        label=f"kept, flipped ({int((kept & (ratios > 0)).sum())})",
    )
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, ratios.size + 1)
    ax.set_xlabel("Segment", fontsize=7)
    ax.set_ylabel("F-k ratio", fontsize=7)
    ax.set_title(
        f"{int(kept.sum())} of {ratios.size} segments kept (|ratio| > {threshold:g})",
        fontsize=8,
        loc="left",
    )
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    return fig
