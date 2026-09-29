"""One window's petrophysical inversion as PAC's card shows it, for the figure it saves: the picked
curve against the one the soil column gives back, and the soil column itself."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from sigpipe.base.dispersion_curve import DispersionCurve
from sigpipe.base.petro_model import PetroModel
from sigpipe.dataio.plot_config import CM, DISP_DPI, DOUBLE_COLUMN_CM, SOIL_TYPE_COLORS

# PAC's colours: the picks, the curve a model gives back, the water table.
_PICKED = "#2a78d6"
_MODELLED = "#d63c3c"
_WATER = "darkblue"
_SMALL = 7


def plot_petro_window(
    observed: DispersionCurve, modelled: DispersionCurve | None, model: PetroModel
) -> Figure:
    """Left, the picked fundamental mode with its uncertainties and the curve the soil column
    `model` gives back (`modelled`, none when it gave none back); right, the column to scale:
    each layer in its soil's colour with its N value, the water table dashed."""
    fig = plt.figure(figsize=(DOUBLE_COLUMN_CM * CM, 10.0 * CM), dpi=DISP_DPI)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.5, 0.7), wspace=0.3)
    ax_fit = fig.add_subplot(grid[0])
    ax_column = fig.add_subplot(grid[1])

    order = np.argsort(observed.fs)
    ax_fit.errorbar(
        np.asarray(observed.fs)[order],
        np.asarray(observed.vs)[order],
        yerr=None if observed.vs_err is None else np.asarray(observed.vs_err)[order],
        fmt="o",
        color=_PICKED,
        markersize=1.8,
        elinewidth=0.4,
        capsize=0,
        label="picked, ± its uncertainty",
    )
    if modelled is not None:
        order = np.argsort(modelled.fs)
        ax_fit.plot(
            np.asarray(modelled.fs)[order],
            np.asarray(modelled.vs)[order],
            color=_MODELLED,
            linestyle="--",
            linewidth=1.0,
            label="given back by the soil column",
        )
    ax_fit.set_xlabel("Frequency [Hz]", fontsize=_SMALL)
    ax_fit.set_ylabel("Phase velocity [m/s]", fontsize=_SMALL)
    ax_fit.set_title("Picked and modelled curve", fontsize=_SMALL + 1, loc="left")
    ax_fit.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=_SMALL - 1, frameon=False, ncol=2
    )

    thicknesses = np.asarray(model.thicknesses, dtype=float)
    tops = np.concatenate(([0.0], np.cumsum(thicknesses)[:-1]))
    total = float(thicknesses.sum())
    for top, thickness, soil, n in zip(tops, thicknesses, model.soils, model.Ns, strict=True):
        colour = SOIL_TYPE_COLORS.get(soil, "#cfd4da")
        ax_column.add_patch(
            Rectangle((0.0, top), 1.0, thickness, facecolor=colour, edgecolor="white", linewidth=1)
        )
        if thickness >= 0.035 * total:
            ax_column.text(
                0.5,
                top + thickness / 2,
                f"{soil} · N {n}",
                ha="center",
                va="center",
                fontsize=_SMALL - 1,
                color="#1a1a1a",
            )
    ax_column.axhline(model.water_table_depth, color=_WATER, linestyle=(0, (4, 3)), linewidth=1.0)
    ax_column.text(
        0.98,
        model.water_table_depth,
        f"water table {model.water_table_depth:.1f} m",
        ha="right",
        va="bottom",
        fontsize=_SMALL - 1,
        color=_WATER,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.0},
    )
    ax_column.set_xlim(0.0, 1.0)
    ax_column.set_ylim(total, 0.0)
    ax_column.set_xticks([])
    ax_column.set_ylabel("Depth [m]", fontsize=_SMALL)
    ax_column.set_title("Soil column", fontsize=_SMALL + 1, loc="left")
    for ax in (ax_fit, ax_column):
        ax.tick_params(labelsize=_SMALL - 1)
    ax_fit.grid(color="#ecebe6", linewidth=0.5)
    ax_fit.set_axisbelow(True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.24)
    return fig
