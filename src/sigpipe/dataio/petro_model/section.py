import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure

from sigpipe.base.petro_model import PetroModelsSection, SoilType
from sigpipe.dataio.plot_config import (
    CM,
    DISP_DPI,
    DOUBLE_COLUMN_CM,
    NO_DATA_COLOR,
    SOIL_TYPE_COLORS,
    n_value_colors,
)


def plot_petro_models_section(
    petro_section: PetroModelsSection,
    *,
    dz: float = 0.01,
    dx: float | None = None,
) -> Figure:
    """
    Soil-type section, with SPT N-value and the water table.

    X-axis: position [m]
    Y-axis: elevation [m] (decreasing downward)
    Top: soil type (categorical), water table as a dashed line.
    Bottom: N-value (blow count), also categorical -- N is a small set of
    integers, not a continuous field, so it gets one solid color per class
    rather than a gradient.
    """
    xs, zs, soil_grid, n_grid, water_table_profile = petro_section.to_grid(dz=dz, dx=dx)

    # SoilType.NONE (above a shallower position's local topography) isn't in
    # SOIL_TYPE_COLORS/soil_order at all -- it maps to NaN, which set_bad
    # below renders as plain white with no colorbar entry, rather than being
    # a labeled class of its own.
    soil_order = list(SOIL_TYPE_COLORS.keys())
    soil_to_code = {soil: code for code, soil in enumerate(soil_order)}

    def _soil_code(soil: SoilType) -> float:
        return soil_to_code.get(soil, np.nan)

    # otypes pins the output dtype to float regardless of which element
    # np.vectorize happens to call first -- without it, a real soil (int)
    # landing first infers an integer output array, which then can't hold the
    # NaN from a later SoilType.NONE cell.
    soil_code_grid = np.vectorize(_soil_code, otypes=[np.float32])(soil_grid)

    soil_cmap = ListedColormap(list(SOIL_TYPE_COLORS.values()))
    soil_cmap.set_bad(NO_DATA_COLOR)
    soil_norm = BoundaryNorm(np.arange(-0.5, len(soil_order) + 0.5), soil_cmap.N)

    n_valid = ~np.isnan(n_grid)
    n_values = sorted({int(v) for v in n_grid[n_valid]})
    n_colors = n_value_colors(n_values)
    n_value_to_code = {value: code for code, value in enumerate(n_values)}
    n_code_grid = np.full(n_grid.shape, np.nan, dtype=np.float32)
    n_code_grid[n_valid] = np.vectorize(n_value_to_code.get)(n_grid[n_valid].astype(int))

    n_cmap = ListedColormap([n_colors[value] for value in n_values])
    n_cmap.set_bad(NO_DATA_COLOR)
    n_norm = BoundaryNorm(np.arange(-0.5, len(n_values) + 0.5), n_cmap.N)

    fig, (ax_soil, ax_n) = plt.subplots(
        2, 1, figsize=(DOUBLE_COLUMN_CM * CM, 10 * CM), dpi=DISP_DPI, sharex=True
    )

    pcm_soil = ax_soil.pcolormesh(
        xs, zs, soil_code_grid.T, cmap=soil_cmap, norm=soil_norm, shading="nearest"
    )
    ax_soil.plot(
        xs,
        water_table_profile,
        color="royalblue",
        linewidth=1.5,
        linestyle="--",
        label="Water table",
    )
    # ticks=range(len(soil_order)) + set_yticklabels below: exactly one tick per
    # class, no extra interpolated ticks in between (BoundaryNorm's continuous
    # default locator would otherwise add those).
    cbar_soil = fig.colorbar(pcm_soil, ax=ax_soil, ticks=range(len(soil_order)))
    cbar_soil.ax.set_yticklabels([soil.value for soil in soil_order])
    ax_soil.set_ylabel("Elevation [m]")
    ax_soil.legend(loc="lower right")

    pcm_n = ax_n.pcolormesh(xs, zs, n_code_grid.T, cmap=n_cmap, norm=n_norm, shading="nearest")
    cbar_n = fig.colorbar(pcm_n, ax=ax_n, ticks=range(len(n_values)))
    cbar_n.ax.set_yticklabels([str(value) for value in n_values])
    cbar_n.set_label("N [#]")
    ax_n.set_xlabel("Position [m]")
    ax_n.set_ylabel("Elevation [m]")

    ax_soil.set_xlim(xs[0], xs[-1])
    ax_n.set_xlim(xs[0], xs[-1])

    fig.tight_layout()

    return fig
