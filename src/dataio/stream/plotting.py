import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.base.stream import Stream
from src.dataio.plot_config import CM, DISP_DPI, HEIGHT_CM, SINGLE_COLUMN_CM


def plot_stream(
    stream: Stream,
    spacing: float = 2.25,
    normalize: str | None = None,
) -> Figure:
    xt = stream.xt.copy()
    nx = stream.nx
    ts = stream.ts.copy()
    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_CM * CM, HEIGHT_CM * CM), dpi=DISP_DPI
    )
    ts *= 1e3  # s to ms
    if normalize is not None:
        normalize = str(normalize).lower()
        match normalize:
            case "global":
                xt /= np.max(np.abs(xt)) + 1e-12
            case "per_trace":
                xt /= np.max(np.abs(xt), axis=1, keepdims=True) + 1e-12
            case _:
                raise ValueError(f"Unknown normalize option: {normalize}")
    for i_trace, trace in enumerate(xt):
        y = i_trace * spacing + trace
        ax.plot(y, ts, color="black", lw=0.5)
        ax.fill_betweenx(
            ts,
            i_trace * spacing,
            y,
            where=(trace > 0),
            color="black",
        )
        if i_trace % 5 == 0:
            ax.text(
                i_trace * spacing,
                ts[0],
                f"{i_trace}",
                ha="center",
                va="bottom",
            )
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylim(ts[0], ts[-1])
    ax.set_xlim(-2, (nx - 1) * spacing + 2)
    ax.set_ylabel("Time [ms]")
    ax.set_xlabel("Sensor [#]")
    ax.xaxis.set_label_coords(0.5, 1.13)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig
