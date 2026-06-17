import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sigproc.base.stream import Stream
from sigproc.dataio.plot_config import CM, DISP_DPI, HEIGHT_CM, SINGLE_COLUMN_CM


def plot_stream(
    stream: Stream,
    spacing: float = 2.25,
    normalize: bool = True,
) -> Figure:
    xt = stream.xt.copy()
    nx = stream.nx
    ts = stream.ts.copy()
    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_CM * CM, HEIGHT_CM * CM), dpi=DISP_DPI
    )
    if normalize:
        scales = np.max(np.abs(xt), axis=1, keepdims=True)
        xt /= scales + 1e-12
    y_positions = np.arange(nx) * spacing
    for i_trace, (trace, y_position) in enumerate(zip(xt, y_positions)):
        ax.plot(y_position + trace, ts, color="black", lw=0.5)
        ax.fill_betweenx(
            ts,
            y_position,
            y_position + trace,
            where=(trace > 1e-6),
            color="black",
        )
        if i_trace % 10 == 0:
            ax.text(
                y_position,
                ts[0],
                f"{i_trace}",
                ha="center",
                va="bottom",
            )
    if stream.arrivals is not None:
        for i_trace, (trace, trace_arrivals, y_position) in enumerate(
            zip(xt, stream.arrivals, y_positions)
        ):
            for arrival in trace_arrivals:
                ax.scatter(
                    y_position,
                    arrival.time,
                    color="red",
                    marker="+",
                    s=40,
                    linewidths=1.5,
                    zorder=10,
                )
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylim(ts[0], ts[-1])
    ax.set_xlim(-2, (nx - 1) * spacing + 2)
    ax.set_ylabel("Time [s]")
    ax.set_xlabel("Receiver [#]")
    ax.xaxis.set_label_coords(0.5, 1.13)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig
