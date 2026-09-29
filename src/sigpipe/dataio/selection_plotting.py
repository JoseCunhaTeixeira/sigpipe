"""A passive window's segment selection, for the figure a run saves and the data Visualization
shows it from: each segment's score, kept or not, around the threshold it was kept by."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict

from sigpipe.dataio.plot_config import CM, DISP_DPI, DOUBLE_COLUMN_CM, SAVING_DPI

# PAC's colours: a segment kept (flipped when its ratio is positive), one left out.
_KEPT = "#2a78d6"
_FLIPPED = "#eb6834"
_REJECTED = "#a9abb2"
_LIMIT = "#898781"
SELECTION_STEM = "Selection"


class SelectionScores(BaseModel):
    """A window's fk segment selection as saved beside its figure: each segment's f-k ratio and
    whether it was kept, in the order the window met them (record after record), the threshold
    they were kept by, and what that made of them."""

    model_config = ConfigDict(frozen=True)

    threshold: float
    flip: bool  # a kept segment with a positive ratio space-flipped, so that all run one way
    ratios: tuple[float, ...]
    kept: tuple[bool, ...]
    segments: int
    kept_count: int
    flipped_count: int
    kept_share: float  # of the segments, 0 to 1


def selection_scores(
    scores: Sequence[tuple[float, bool]], threshold: float, flip: bool
) -> SelectionScores:
    """The selection's `scores` (each segment's ratio and whether it was kept) as saved."""
    kept_count = sum(1 for _, kept in scores if kept)
    return SelectionScores(
        threshold=threshold,
        flip=flip,
        ratios=tuple(round(float(ratio), 4) for ratio, _ in scores),
        kept=tuple(bool(kept) for _, kept in scores),
        segments=len(scores),
        kept_count=kept_count,
        flipped_count=sum(1 for ratio, kept in scores if kept and flip and ratio > 0),
        kept_share=kept_count / len(scores) if scores else 0.0,
    )


def save_selection(
    scores: Sequence[tuple[float, bool]],
    threshold: float,
    flip: bool,
    folder: Path,
    stem: str = SELECTION_STEM,
) -> Path:
    """The selection in `folder`: its figure (`Selection_0000.png`, plot_selection) and its data
    (`Selection_0000.json`, SelectionScores), for Visualization to show it without measuring it
    again. Returns the figure's path."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}_0000"
    path.with_suffix(".json").write_text(
        selection_scores(scores, threshold, flip).model_dump_json()
    )
    figure = plot_selection(scores, threshold)
    figure.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=SAVING_DPI)
    plt.close(figure)
    return path.with_suffix(".png")


def load_selection(folder: Path, stem: str = SELECTION_STEM) -> SelectionScores | None:
    """The selection save_selection saved in `folder`; None when it saved none."""
    path = folder / f"{stem}_0000.json"
    return SelectionScores.model_validate_json(path.read_text()) if path.exists() else None


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
