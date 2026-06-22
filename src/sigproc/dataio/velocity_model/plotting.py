import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sigproc.base.velocity_model import VelocityModel
from sigproc.dataio.plot_config import CM, DISP_DPI, DOUBLE_COLUMN_CM, HEIGHT_CM


def plot_velocity_models(
    velocity_models: dict[str, VelocityModel] | VelocityModel,
    *,
    show_std: bool = False,
) -> Figure:
    """
    Plot one or more named velocity profiles (depth on the y-axis, inverted).

    Left axis: Vs (solid) and Vp (dashed) together. Right axis: density.

    velocity_models maps a label (e.g. "Best", "Median", "Ensemble") to the
    profile to draw with that label, or a single VelocityModel.
    """
    if isinstance(velocity_models, VelocityModel):
        velocity_models = {"Model": velocity_models}

    fig, (ax_v, ax_rho) = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_CM * CM, HEIGHT_CM * CM),
        dpi=DISP_DPI,
    )

    depth_max = 0.0
    for label, velocity_model in velocity_models.items():
        depths = np.insert(np.asarray(velocity_model.depths, dtype=np.float32), 0, 0.0)
        depth_max = max(depth_max, float(depths[-1]))

        vs_s = np.append(velocity_model.vs_s, velocity_model.vs_s[-1])
        vs_p = np.append(velocity_model.vs_p, velocity_model.vs_p[-1])
        rhos = np.append(velocity_model.rhos, velocity_model.rhos[-1])

        (line,) = ax_v.step(vs_s, depths, where="post", label=f"{label} ($v_S$)", linewidth=1)
        color = line.get_color()
        ax_v.step(
            vs_p,
            depths,
            where="post",
            label=f"{label} ($v_P$)",
            linewidth=1,
            linestyle="dashed",
            color=color,
        )

        if show_std:
            vs_s_std = np.append(velocity_model.vs_s_std, velocity_model.vs_s_std[-1])
            for sign in (-1, 1):
                ax_v.step(
                    vs_s + sign * vs_s_std,
                    depths,
                    where="post",
                    color=color,
                    linewidth=0.5,
                    linestyle="dotted",
                    label="_nolegend_",
                )

        ax_rho.step(rhos, depths, where="post", label=label, linewidth=1, color=color)

    for ax in (ax_v, ax_rho):
        ax.invert_yaxis()
        ax.set_ylim(depth_max, 0)
        ax.set_ylabel("Depth [m]")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))

    ax_v.set_xlabel("$v_S$, $v_P$ [m/s]")
    ax_rho.set_xlabel("Density [kg/m$^3$]")
    fig.tight_layout()

    return fig
