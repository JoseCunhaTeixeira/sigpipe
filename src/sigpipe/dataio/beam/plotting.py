import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sigpipe.base.beamforming import Beam
from sigpipe.base.coordinate import coordinates_to_tuples
from sigpipe.dataio.plot_config import CM, DISP_DPI, HEIGHT_CM, SINGLE_COLUMN_CM


def plot_beamforming(
    beam: Beam,
    show_cbar: bool = False,
) -> Figure:
    stations = np.array(coordinates_to_tuples(beam.acquisition.receivers))

    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_CM * CM, HEIGHT_CM * CM),
        dpi=DISP_DPI,
    )
    pcm = ax.pcolormesh(beam.xs, beam.ys, beam.xy_map.T, cmap="PuOr", vmin=-1, vmax=1)
    ax.scatter(stations[:, 0], stations[:, 1], marker="v", s=25, label="station", ec="k", lw=0.5)
    if show_cbar:
        fig.colorbar(pcm, ax=ax, pad=0.02, aspect=40)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    return fig
