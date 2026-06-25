import matplotlib.pyplot as plt
import numpy as np
from disba import DispersionError
from matplotlib import colors
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from sigpipe.algorithms.inversion.dispersion_curve.rayleigh.forward import fwd_rayleigh_phase
from sigpipe.base.dispersion_curve import DispersionCurves
from sigpipe.base.inversion import InversionResult
from sigpipe.dataio.plot_config import CM, DISP_DPI, DOUBLE_COLUMN_CM

_MODEL_STYLE: dict[str, tuple[str, str]] = {
    "best": ("tab:green", "Best layered model"),
    "smooth_best": ("green", "Smooth best layered model"),
    "median": ("orange", "Median layered model"),
    "smooth_median": ("tab:orange", "Smooth median layered model"),
    "ensemble": ("tab:red", "Median ensemble model"),
}


def _hdi_levels(density: np.ndarray, probs: tuple[float, ...]) -> list[float]:
    """Density thresholds whose enclosed area contains each given probability mass."""
    flat = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(flat)
    cumulative /= cumulative[-1]
    levels = [flat[min(int(np.searchsorted(cumulative, p)), len(flat) - 1)] for p in probs]
    return sorted(levels)


def plot_posterior_marginals(
    samples: dict[str, np.ndarray],
    *,
    hdi_probs: tuple[float, ...] = (0.3, 0.6, 0.9),
    n_grid: int = 80,
    textsize: float = 9,
) -> Figure:
    """
    Corner plot of posterior samples.

    Diagonal: 1D KDE marginal for each parameter.
    Lower triangle: 2D KDE with highest-density-interval contours (hdi_probs).
    """
    names = list(samples.keys())
    n = len(names)

    fig, axs = plt.subplots(n, n, figsize=(2.6 * CM * n, 2.4 * CM * n), dpi=150, squeeze=False)

    grids_1d = {
        name: np.linspace(values.min(), values.max(), n_grid) for name, values in samples.items()
    }

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            ax = axs[i, j]

            if j > i:
                ax.axis("off")
                continue

            if i == j:
                kde = gaussian_kde(samples[name_i])
                xs = grids_1d[name_i]
                ax.plot(xs, kde(xs), color="tab:blue", linewidth=1)
                ax.set_yticks([])
            else:
                kde = gaussian_kde(np.vstack([samples[name_j], samples[name_i]]))
                X, Y = np.meshgrid(grids_1d[name_j], grids_1d[name_i])
                density = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                levels = _hdi_levels(density, hdi_probs)
                ax.contourf(X, Y, density, levels=[*levels, density.max()], cmap="Blues")

            if i == n - 1:
                ax.set_xlabel(name_j, fontsize=textsize)
            else:
                ax.set_xticklabels([])

            if j == 0 and i != 0:
                ax.set_ylabel(name_i, fontsize=textsize)
            elif j == 0:
                ax.set_ylabel("Density", fontsize=textsize)
            if j != 0:
                ax.set_yticklabels([])

    fig.tight_layout()
    return fig


def plot_density_curves(
    result: InversionResult,
    observed_curves: DispersionCurves,
    Vp_Vs_ratio: float,
) -> Figure:
    """
    Dispersion fit (left) and sampled-models cloud (right) for one inversion run.

    Left: observed data, the posterior-predicted-data percentile band, and all
    5 named models (best, smooth_best, median, smooth_median, ensemble), each
    forward-modeled at the observed frequencies.
    Right: every sampled model as a misfit-colored (greyscale) Vs(depth) step
    profile, plus the 5 named models, plus the smooth median's std band.

    Port of the old Streamlit app's `density_curves.png` in `run_inversion.py`.
    """
    models = {name: getattr(result, name) for name in _MODEL_STYLE}
    depth_max = max(float(np.sum(model.thicknesses)) for model in models.values())

    fig, (ax_fit, ax_vs) = plt.subplots(
        1, 2, figsize=(DOUBLE_COLUMN_CM * CM, 12 * CM), dpi=DISP_DPI
    )

    labeled: set[str] = set()
    for i, observed_curve in enumerate(observed_curves):
        mode_number = observed_curve.mode.number
        d_pred = result.dpred.get(mode_number)
        if d_pred is not None:
            p10, p50, p90 = np.percentile(d_pred, (10, 50, 90), axis=0)
            ax_fit.fill_between(
                observed_curve.fs,
                p10,
                p90,
                color="k",
                alpha=0.2,
                label="10th-90th percentiles" if i == 0 else "_nolegend_",
                zorder=1,
            )
            ax_fit.plot(
                observed_curve.fs,
                p50,
                color="k",
                linewidth=0.2,
                linestyle="--",
                label="50th percentile" if i == 0 else "_nolegend_",
                zorder=2,
            )

        # Observed data always renders behind the modeled markers below, so
        # it's never hidden by a model that fits closely on top of it.
        ax_fit.errorbar(
            observed_curve.fs,
            observed_curve.vs,
            yerr=observed_curve.vs_err,
            fmt="o",
            color="tab:blue",
            markersize=1.5,
            capsize=0,
            elinewidth=0.3,
            label="Observed Data" if i == 0 else "_nolegend_",
            zorder=2,
        )

        for name, (color, label) in _MODEL_STYLE.items():
            model = models[name]
            try:
                modeled = fwd_rayleigh_phase(
                    list(model.thicknesses),
                    list(model.vs_s),
                    mode_number,
                    observed_curve.fs,
                    Vp_Vs_ratio,
                )
            except DispersionError:
                continue
            ax_fit.plot(
                modeled.fs,
                modeled.vs,
                "o",
                color=color,
                markersize=1.5,
                label=label if name not in labeled else "_nolegend_",
                zorder=3,
            )
            labeled.add(name)

    ax_fit.set_xlabel("Frequency [Hz]")
    ax_fit.set_ylabel("Phase velocity $v_{R}$ [m/s]")
    ax_fit.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))

    cmap = plt.get_cmap("Greys_r")
    norm = colors.LogNorm(
        vmin=max(float(np.min(result.misfits)), 1e-12), vmax=float(np.max(result.misfits))
    )
    n_finite_layers = result.n_layers - 1
    worst_first = np.argsort(result.misfits)[::-1]  # best (lowest misfit) drawn last, on top
    for s in worst_first:
        vs_vals = np.array([result.samples[f"vs{i + 1}"][s] for i in range(result.n_layers)])
        thick_vals = np.array(
            [result.samples[f"thick{i + 1}"][s] for i in range(n_finite_layers)] + [0.0]
        )
        thick_vals[-1] = (depth_max - np.sum(thick_vals[:-1])) / 2
        depths = np.insert(np.cumsum(thick_vals), 0, 0.0)
        vs_step = np.append(vs_vals, vs_vals[-1])
        # where="pre": each layer's own Vs spans its own depth range (depths[i]
        # to depths[i+1]); where="post" would draw the *next* layer's Vs there
        # instead, collapsing the first layer to a zero-length point at depth 0.
        ax_vs.step(vs_step, depths, where="pre", color=cmap(norm(result.misfits[s])), linewidth=0.5)

    for name, (color, label) in _MODEL_STYLE.items():
        model = models[name]
        depths = np.insert(np.asarray(model.depths, dtype=np.float64), 0, 0.0)
        vs_step = np.append(model.vs_s, model.vs_s[-1])
        ax_vs.step(vs_step, depths, where="pre", color=color, label=label, linewidth=1)
        if name == "smooth_median":
            std_step = np.append(model.vs_s_std, model.vs_s_std[-1])
            ax_vs.step(
                vs_step - std_step,
                depths,
                where="pre",
                color=color,
                linewidth=1,
                linestyle="dotted",
                label="Standard deviation",
            )
            ax_vs.step(
                vs_step + std_step,
                depths,
                where="pre",
                color=color,
                linewidth=1,
                linestyle="dotted",
                label="_nolegend_",
            )

    ax_vs.set_ylim(depth_max, 0)
    ax_vs.set_xlabel("Shear wave velocity $v_{S}$ [m/s]")
    ax_vs.set_ylabel("Depth [m]")
    ax_vs.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))

    fig.tight_layout()
    return fig
