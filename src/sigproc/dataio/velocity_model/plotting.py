import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sigproc.base.velocity_model import VelocityModel
from sigproc.dataio.plot_config import CM, DISP_DPI, HEIGHT_CM, SINGLE_COLUMN_CM


def plot_velocity_models(
    velocity_models: dict[str, VelocityModel] | VelocityModel,
    *,
    show_std: bool = False,
) -> Figure:
    """
    Plot one or more named Vs profiles (depth on the y-axis, inverted).

    velocity_models maps a label (e.g. "Best", "Median", "Ensemble") to the
    profile to draw with that label, or a single VelocityModel.
    """
    if isinstance(velocity_models, VelocityModel):
        velocity_models = {"Model": velocity_models}

    fig, ax_v = plt.subplots(
        figsize=(SINGLE_COLUMN_CM * CM, HEIGHT_CM * CM),
        dpi=DISP_DPI,
    )

    depth_max = 0.0
    for label, velocity_model in velocity_models.items():
        depths = np.insert(np.asarray(velocity_model.depths, dtype=np.float32), 0, 0.0)
        depth_max = max(depth_max, float(depths[-1]))

        vs_s = np.append(velocity_model.vs_s, velocity_model.vs_s[-1])

        # where="pre": each layer's own Vs spans its own depth range (depths[i]
        # to depths[i+1]); where="post" would draw the *next* layer's Vs there
        # instead, collapsing the first layer to a zero-length point at depth 0.
        (line,) = ax_v.step(vs_s, depths, where="pre", label=label, linewidth=1)
        color = line.get_color()

        if show_std:
            vs_s_std = np.append(velocity_model.vs_s_std, velocity_model.vs_s_std[-1])
            for sign in (-1, 1):
                ax_v.step(
                    vs_s + sign * vs_s_std,
                    depths,
                    where="pre",
                    color=color,
                    linewidth=0.5,
                    linestyle="dotted",
                    label="_nolegend_",
                )

    ax_v.invert_yaxis()
    ax_v.set_ylim(depth_max, 0)
    ax_v.set_ylabel("Depth [m]")
    ax_v.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax_v.set_xlabel("$v_S$ [m/s]")
    fig.tight_layout()

    return fig
