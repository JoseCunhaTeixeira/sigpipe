"""A line's sections as PAC's Visualization draws them, for the figures a run saves: panels over
position and elevation (or a pseudo-section's frequency or wavelength), stacked, each with its
colour bar; the depth the data inform, veiled below its dashed line; the water table."""

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sigpipe.dataio.plot_config import CM, DISP_DPI, DOUBLE_COLUMN_CM

# PAC's depth informed (light theme): its line, the veil below it; the water table.
_INFORMED = "#d98a04"
_VEIL = (1.0, 1.0, 1.0, 0.62)
_WATER = "darkblue"
_SMALL = 7


@dataclass(frozen=True, slots=True)
class SectionPanel:
    """One section: `values` (positions x ys, NaN: empty) at `positions` and `ys` (elevations,
    or a pseudo-section's frequencies or wavelengths), coloured by `cmap` over `norm`; with, per
    position, the elevation down to which the data inform it (`informed`, NaN: not known) above
    the lowest one drawn (`floors`), and the water table's elevation (`water_table`)."""

    positions: np.ndarray
    ys: np.ndarray
    values: np.ndarray
    label: str  # its colour bar's
    cmap: str | colors.Colormap
    norm: colors.Normalize | None = None
    # Categories (soils, N): the colour bar's ticks, a value's name.
    ticks: dict[float, str] | None = None
    informed: np.ndarray | None = None
    floors: np.ndarray | None = None
    water_table: np.ndarray | None = None


def plot_sections(
    panels: Sequence[SectionPanel], y_label: str = "Elevation [m]", downward: bool = False
) -> Figure:
    """`panels` stacked, one position axis; their ys labelled `y_label`, increasing downward
    when `downward` (a pseudo-section's wavelengths, as depths); one legend under them for what
    the overlays mean."""
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(DOUBLE_COLUMN_CM * CM, (4.4 * len(panels) + 1.6) * CM),
        dpi=DISP_DPI,
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    legend: dict[str, Artist] = {}
    for ax, panel in zip(axes[:, 0], panels, strict=True):
        mesh = ax.pcolormesh(
            panel.positions,
            panel.ys,
            np.ma.masked_invalid(panel.values).T,
            shading="nearest",
            cmap=panel.cmap,
            norm=panel.norm,
        )
        bar = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.035)
        bar.set_label(panel.label, fontsize=_SMALL)
        if panel.ticks:
            bar.set_ticks(list(panel.ticks), labels=list(panel.ticks.values()))
        bar.ax.tick_params(labelsize=_SMALL - 1)
        if panel.informed is not None:
            floors = (
                panel.floors
                if panel.floors is not None
                else np.full(panel.positions.size, float(np.nanmin(panel.ys)))
            )
            known = ~np.isnan(panel.informed)
            ax.fill_between(
                panel.positions,
                np.where(known, panel.informed, floors),
                floors,
                where=known,
                step="mid",
                color=_VEIL,
                linewidth=0,
            )
            ax.step(panel.positions, panel.informed, where="mid", color="white", linewidth=2.2)
            ax.step(
                panel.positions,
                panel.informed,
                where="mid",
                color=_INFORMED,
                linewidth=1.0,
                linestyle=(0, (5, 4)),
            )
            legend["depth informed"] = Line2D([], [], color=_INFORMED, linestyle=(0, (5, 4)))
            legend["not informed by the data"] = Patch(facecolor="#d9d9d9", edgecolor="none")
        if panel.water_table is not None:
            ax.step(
                panel.positions,
                panel.water_table,
                where="mid",
                color=_WATER,
                linewidth=1.0,
                linestyle=(0, (4, 3)),
            )
            legend["water table"] = Line2D([], [], color=_WATER, linestyle=(0, (4, 3)))
        ax.set_xlim(float(panel.positions[0]), float(panel.positions[-1]))
        low, high = float(np.nanmin(panel.ys)), float(np.nanmax(panel.ys))
        ax.set_ylim((high, low) if downward else (low, high))
        ax.set_ylabel(y_label, fontsize=_SMALL)
        ax.tick_params(labelsize=_SMALL - 1)
    axes[-1, 0].set_xlabel("Position [m]", fontsize=_SMALL)
    if legend:
        fig.legend(
            list(legend.values()),
            list(legend),
            loc="outside lower center",
            ncol=len(legend),
            fontsize=_SMALL - 1,
            frameon=False,
        )
    return fig


def categories(
    names: Sequence[str], colours: Sequence[str]
) -> tuple[colors.ListedColormap, colors.BoundaryNorm, dict[float, str]]:
    """The colour map, norm and ticks of categories `names` (each its colour of `colours`), the
    values their indices."""
    cmap = colors.ListedColormap(list(colours))
    norm = colors.BoundaryNorm(np.arange(len(names) + 1) - 0.5, len(names))
    return cmap, norm, {float(i): name for i, name in enumerate(names)}
