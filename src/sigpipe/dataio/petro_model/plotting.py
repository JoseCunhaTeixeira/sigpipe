import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sigpipe.base.petro_model import PetroModel, SoilType
from sigpipe.dataio.plot_config import (
    CM,
    DISP_DPI,
    HEIGHT_CM,
    SINGLE_COLUMN_CM,
    SOIL_TYPE_COLORS,
    n_value_colors,
)


def _bar_column(ax: Axes, petro_model: PetroModel, colors: list[str]) -> None:
    """Draw one profile as horizontal bands at negative depth (0 at the
    surface, more negative going down) -- plain non-inverted axes then
    already put the surface at the top and read out in negative meters,
    with no invert_yaxis() needed."""
    tops = np.insert(np.asarray(petro_model.depths[:-1], dtype=np.float32), 0, 0.0)
    for top, thickness, color in zip(tops, petro_model.thicknesses, colors, strict=True):
        ax.barh(
            y=-(top + thickness / 2),
            width=1.0,
            height=thickness,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
    ax.axhline(-petro_model.water_table_depth, color="royalblue", linestyle="--", linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([])


def plot_petro_models(
    petro_models: dict[str, PetroModel] | PetroModel,
) -> Figure:
    """
    Plot one or more named profiles (depth on the y-axis, inverted) side by
    side, each as a soil-type column and an N-value (blow count) column, with
    the water table marked on both.

    petro_models maps a label (e.g. "Best", "Median") to the profile to draw
    under that label, or a single PetroModel.
    """
    if isinstance(petro_models, PetroModel):
        petro_models = {"Model": petro_models}

    # N is an open-ended integer, not a fixed enum like SoilType, so its color
    # mapping is built from whatever values are actually present across all
    # profiles being plotted here -- keeps a given N the same color in every
    # column of this figure.
    all_ns = [n for petro_model in petro_models.values() for n in petro_model.Ns]
    n_colors = n_value_colors(all_ns)

    fig, axes = plt.subplots(
        2,
        len(petro_models),
        figsize=(SINGLE_COLUMN_CM * CM * len(petro_models), HEIGHT_CM * CM * 1.6),
        dpi=DISP_DPI,
        sharey="row",
        squeeze=False,
    )
    soil_axes, n_axes = axes[0], axes[1]

    depth_max = max(petro_model.depths[-1] for petro_model in petro_models.values())

    for ax_soil, ax_n, (label, petro_model) in zip(
        soil_axes, n_axes, petro_models.items(), strict=True
    ):
        _bar_column(ax_soil, petro_model, [SOIL_TYPE_COLORS[soil] for soil in petro_model.soils])
        ax_soil.set_title(label)

        _bar_column(ax_n, petro_model, [n_colors[n] for n in petro_model.Ns])

    # sharey="row" only shares within each row (all soil columns together, all
    # N columns together) -- it does NOT share soil's row with N's row, so
    # both need this set explicitly or the N row is left at its own default
    # (non-negative, un-inverted) autoscale and reads upside down.
    soil_axes[0].set_ylim(-depth_max, 0)
    soil_axes[0].set_ylabel("Depth [m]")
    n_axes[0].set_ylim(-depth_max, 0)
    n_axes[0].set_ylabel("Depth [m]")

    # Legends attached to each row's own rightmost axes (rather than floating
    # fig-level legends) so tight_layout accounts for them and they can't
    # overlap the plot or each other.
    soil_handles: list[Patch | Line2D] = [
        Patch(facecolor=color, edgecolor="black", label=soil.value or "none")
        for soil, color in SOIL_TYPE_COLORS.items()
        if soil != SoilType.NONE
    ]
    soil_handles.append(Line2D([0], [0], color="royalblue", linestyle="--", label="Water table"))
    soil_axes[-1].legend(
        handles=soil_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Soil",
        borderaxespad=0,
    )

    n_handles = [
        Patch(facecolor=color, edgecolor="black", label=str(n)) for n, color in n_colors.items()
    ]
    n_axes[-1].legend(
        handles=n_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="N [#]",
        borderaxespad=0,
    )

    fig.tight_layout()

    return fig
