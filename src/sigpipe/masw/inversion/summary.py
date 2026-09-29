"""The line's inversions at a glance, for the figure a run saves: per window along the line, how
deep its data inform it, how well its median of the ensemble fits the picks, whether its chains
agree and how many layers its layered median has."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sigpipe.dataio.plot_config import CM, DISP_DPI, DOUBLE_COLUMN_CM
from sigpipe.masw.inversion.measuring import MODELS, informed_depth, saved_measures
from sigpipe.transformers import Plot

LINE_SUMMARY_FIGURE = "SeismicInversion_LineSummary_0000.png"
# PAC's colours: the values, those past their limit, the limits, the models' bottom.
_VALUE = "#2a78d6"
_OFF = "#d63c3c"
_LIMIT = "#898781"
_SMALL = 7


def save_line_summary(
    run_folder: Path, units: Sequence[str], max_misfit: float = 2.0, max_rhat: float = 1.1
) -> Path | None:
    """The figure of the line's inversions at a glance (LINE_SUMMARY_FIGURE, at run folder
    `run_folder`'s root), from the measures saved with the windows of `units`: per window, the
    depth the data inform (its models' bottom beside it), the RMS misfit of the median of the
    ensemble (`max_misfit` dashed), the largest split R-hat of Vs at the depths watched
    (`max_rhat` dashed), and the layered median's number of layers; a value past its limit in
    red. None with fewer than two windows measured."""
    rows: list[tuple[float, float | None, float, float | None, float | None, int]] = []
    for unit in units:
        measures = saved_measures(run_folder / unit)
        if measures is None:
            continue
        fit = next((one for one in measures.fits if one.model == MODELS[0]), None)
        agreement = [
            value
            for name, value in measures.rhat.items()
            if name in measures.watched and value is not None
        ]
        rows.append(
            (
                float(unit.removeprefix("xmid_")),
                informed_depth(measures),
                measures.depth_max_m,
                fit.misfit if fit is not None else None,
                max(agreement) if agreement else None,
                len(measures.vs_layers),
            )
        )
    if len(rows) < 2:
        return None
    rows.sort()
    xs = np.array([row[0] for row in rows])
    informed = np.array([np.nan if row[1] is None else row[1] for row in rows])
    bottoms = np.array([row[2] for row in rows])
    misfits = np.array([np.nan if row[3] is None else row[3] for row in rows])
    rhats = np.array([np.nan if row[4] is None else row[4] for row in rows])
    layers = np.array([row[5] for row in rows], dtype=float)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(DOUBLE_COLUMN_CM * CM, 16.0 * CM),
        dpi=DISP_DPI,
        sharex=True,
        layout="constrained",
    )
    ax_depth, ax_misfit, ax_rhat, ax_layers = axes
    ax_depth.step(
        xs,
        bottoms,
        where="mid",
        color=_LIMIT,
        linewidth=0.8,
        linestyle=(0, (4, 3)),
        label="the models' bottom",
    )
    ax_depth.plot(
        xs, informed, "o-", color=_VALUE, markersize=2.5, linewidth=0.8, label="informed to"
    )
    ax_depth.set_ylim(float(np.nanmax(bottoms)) * 1.05, 0.0)
    ax_depth.set_ylabel("Depth informed [m]", fontsize=_SMALL)
    ax_depth.legend(
        fontsize=_SMALL - 1, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2
    )
    for ax, values, limit, label in (
        (ax_misfit, misfits, max_misfit, "Misfit (RMS)"),
        (ax_rhat, rhats, max_rhat, "R-hat"),
    ):
        off = values > limit
        ax.plot(xs, values, "-", color=_VALUE, linewidth=0.8)
        ax.plot(xs[~off], values[~off], "o", color=_VALUE, markersize=2.5)
        ax.plot(xs[off], values[off], "o", color=_OFF, markersize=3)
        ax.axhline(limit, color=_LIMIT, linestyle=(0, (4, 3)), linewidth=0.8)
        ax.set_ylabel(label, fontsize=_SMALL)
    ax_layers.step(xs, layers, where="mid", color=_VALUE, linewidth=0.9)
    ax_layers.plot(xs, layers, "o", color=_VALUE, markersize=2.5)
    ax_layers.set_ylabel("Layers", fontsize=_SMALL)
    ax_layers.yaxis.get_major_locator().set_params(integer=True)
    ax_layers.set_xlabel("Position [m]", fontsize=_SMALL)
    for ax in axes:
        ax.tick_params(labelsize=_SMALL - 1)
        ax.grid(color="#ecebe6", linewidth=0.5)
        ax.set_axisbelow(True)
    path = run_folder / LINE_SUMMARY_FIGURE
    Plot.savefig(path=path, figure=fig)
    plt.close(fig)
    return path
