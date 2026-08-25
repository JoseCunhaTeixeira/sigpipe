from collections.abc import Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from sigpipe.base.dispersion_curve import VelocityType
from sigpipe.base.petro_model import SoilType

CM = 0.3937008  # cm to inch
SINGLE_COLUMN_CM = 9.0
DOUBLE_COLUMN_CM = 18.0
HEIGHT_CM = 9.0
DISP_DPI = 300
SAVING_DPI = 300

VELOCITY_TYPE_LABELS: dict[VelocityType, str] = {
    VelocityType.PHASE: "Phase velocity [m/s]",
    VelocityType.GROUP: "Group velocity [m/s]",
    VelocityType.UNKNOWN: "Velocity [m/s]",
}

# Fixed order/colors shared by every soil-type plot (single-profile columns and
# sections alike) so the same soil always reads as the same color everywhere.
SOIL_TYPE_COLORS: dict[SoilType, str] = {
    SoilType.NONE: "#FFFFFF",
    SoilType.CLAY: "#8B5A2B",
    SoilType.SILT: "#C2B280",
    SoilType.LOAM: "#6B6B3A",
    SoilType.SAND: "#F2D57E",
}


def n_value_colors(values: Iterable[int]) -> dict[int, str]:
    """Categorical colors for a set of SPT N-values (blow counts), sorted
    ascending so the same set of values always maps to the same colors.

    Unlike SOIL_TYPE_COLORS, N isn't a fixed enum -- it's an open-ended
    integer -- so this has to build the mapping from whatever distinct
    values are actually present, per call, rather than being a constant.
    """
    sorted_values = sorted({int(v) for v in values})
    cmap = plt.get_cmap("tab10" if len(sorted_values) <= 10 else "tab20")
    return {v: to_hex(cmap(i)) for i, v in enumerate(sorted_values)}
