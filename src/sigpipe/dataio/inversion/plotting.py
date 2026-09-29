from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from disba import DispersionError
from matplotlib import colors
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from sigpipe.algorithms.inversion.rayleigh.seismic.forward import fwd_seismic_phase
from sigpipe.base.dispersion_curve import DispersionCurves
from sigpipe.base.inversion import InversionResult, LayeredSamples
from sigpipe.base.velocity_model import VelocityModel
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
    # A name's unit under it: long names side by side do not run into each other.
    labels = {name: name.replace(" [", "\n[") for name in names}

    fig, axs = plt.subplots(n, n, figsize=(3.2 * CM * n, 3.0 * CM * n), dpi=DISP_DPI, squeeze=False)

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

            # Three ticks an axis, whole numbers for a count of layers.
            ax.xaxis.set_major_locator(MaxNLocator(3, integer=name_j == "Layers"))
            if i != j:
                ax.yaxis.set_major_locator(MaxNLocator(3, integer=name_i == "Layers"))
            ax.tick_params(labelsize=textsize - 1)
            if i == n - 1:
                ax.set_xlabel(labels[name_j], fontsize=textsize)
                for tick in ax.get_xticklabels():
                    tick.set_rotation(30)
                    tick.set_horizontalalignment("right")
            else:
                ax.set_xticklabels([])

            if j == 0 and i != 0:
                ax.set_ylabel(labels[name_i], fontsize=textsize)
            elif j == 0:
                ax.set_ylabel("Density", fontsize=textsize)
            if j != 0:
                ax.set_yticklabels([])

    fig.align_labels()
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
            p10, p50, p90 = np.nanpercentile(d_pred, (10, 50, 90), axis=0)
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
                modeled = fwd_seismic_phase(
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
    worst_first = np.argsort(result.misfits)[::-1]  # best (lowest misfit) drawn last, on top
    for s in worst_first if result.profiles is not None else ():
        interfaces, vs_vals = result.profiles.model(int(s))  # pyright: ignore[reportOptionalMemberAccess]
        bottom = max(depth_max, float(interfaces[-1]) if interfaces.size else 0.0)
        depths = np.concatenate(([0.0], interfaces, [bottom]))
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
        if name == "ensemble":  # the kept models' own spread at each depth
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


# The window figure's colours, as PAC draws the same plots.
_PICKED = "#2a78d6"
_MODELLED = "#d63c3c"
_MODELLED_SOFT = (214 / 255, 60 / 255, 60 / 255, 0.2)
_VS = "#2a78d6"
_VS_SOFT = (42 / 255, 120 / 255, 214 / 255, 0.18)
_UNCERTAINTY = "#b23200"  # afmhot_r at 0.65, as U's section
_INTERFACES = "#653e9b"  # Purples at 0.8, as the interfaces' section
_INFORMED = "#d98a04"
_VEIL = (0.06, 0.06, 0.08, 0.06)
# Every kept model under the profile, in the saved figure only: grey, darker the better it fits.
_EXPLORED = "#8a8a8a"
# PAC's chain colours, in its order, and its prior bounds' grey.
_CHAIN_COLOURS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
_LIMIT = "#898781"


def plot_inversion_window(
    observed_curves: DispersionCurves,
    modelled_curves: DispersionCurves | None,
    curve_spreads: dict[int, tuple[np.ndarray, np.ndarray]],
    ensemble: VelocityModel,
    spread: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    informed: float | None,
    interfaces: tuple[float, ...] = (),
    interface_dz: float = 0.5,
    explored: tuple[LayeredSamples, np.ndarray] | None = None,
) -> Figure:
    """One window's inversion as PAC shows it, the median of the ensemble alone.

    Left: the picked curves with their uncertainties, the median of the ensemble's curves
    (`modelled_curves`) and the kept models' 10-90 % at the picked frequencies (`curve_spreads`,
    their 10th and 90th percentiles by mode number). Middle: the median of the ensemble's Vs with
    the kept models' 10-90 % (`spread`: depths, 10th, 90th percentiles, and the relative
    uncertainty U = (P90 - P10) / (2 P50)); right, U on the same depths, then where the kept
    models place interfaces (`interfaces`: the share of them per `interface_dz` m from the
    surface down). Below the depth informed (`informed`, m; None: all of it), veiled.
    Every kept model (`explored`: the models and their misfits) under the profile, thin, grey,
    darker the better it fits, the best on top; the Vs axis framed on their 10-90 %.
    """
    depths, low, high, uncertainty = spread
    bottom = float(depths[-1] + (depths[1] - depths[0]) / 2) if depths.size > 1 else 1.0
    fig = plt.figure(figsize=(DOUBLE_COLUMN_CM * CM, 11.5 * CM), dpi=DISP_DPI)
    # The curve, then the profile with U and the interfaces on its depths: room between the two
    # for the profile's depth axis.
    outer = fig.add_gridspec(1, 2, width_ratios=(1.3, 1.9), wspace=0.2)
    ax_fit = fig.add_subplot(outer[0])
    grid = outer[1].subgridspec(1, 3, width_ratios=(1.0, 0.45, 0.45), wspace=0.08)
    ax_vs = fig.add_subplot(grid[0])
    ax_u = fig.add_subplot(grid[1], sharey=ax_vs)
    ax_i = fig.add_subplot(grid[2], sharey=ax_vs)
    small = 7

    # The curves: the models' band, the median of the ensemble's, the picks over them.
    for i, observed in enumerate(observed_curves):
        order = np.argsort(observed.fs)
        fs = np.asarray(observed.fs)[order]
        band = curve_spreads.get(observed.mode.number)
        if band is not None:
            ax_fit.fill_between(
                fs,
                band[0][order],
                band[1][order],
                color=_MODELLED_SOFT,
                linewidth=0,
                label="10-90 % of the models" if i == 0 else "_nolegend_",
                zorder=1,
            )
        ax_fit.errorbar(
            fs,
            np.asarray(observed.vs)[order],
            yerr=None if observed.vs_err is None else np.asarray(observed.vs_err)[order],
            fmt="o",
            color=_PICKED,
            markersize=1.8,
            elinewidth=0.4,
            capsize=0,
            label="picked, ± its uncertainty" if i == 0 else "_nolegend_",
            zorder=2,
        )
    for i, modelled in enumerate(modelled_curves or ()):
        order = np.argsort(modelled.fs)
        ax_fit.plot(
            np.asarray(modelled.fs)[order],
            np.asarray(modelled.vs)[order],
            color=_MODELLED,
            linestyle="--",
            linewidth=1.0,
            label="median of the ensemble" if i == 0 else "_nolegend_",
            zorder=3,
        )
    ax_fit.set_xlabel("Frequency [Hz]", fontsize=small)
    ax_fit.set_ylabel("Phase velocity [m/s]", fontsize=small)
    ax_fit.set_title("Picked and modelled curve", fontsize=small + 1, loc="left")

    # The profile: the band, the median of the ensemble, what the data inform.
    ax_vs.fill_betweenx(
        depths, low, high, color=_VS_SOFT, linewidth=0, label="10-90 % of the models"
    )
    if explored is not None:
        profiles, misfits = explored
        lines = []
        for index in range(profiles.vs.shape[0]):
            boundaries, vs = profiles.model(index)
            edges = np.concatenate(([0.0], boundaries, [max(bottom, *boundaries, 0.0)]))
            lines.append(
                np.column_stack(
                    (np.repeat(vs, 2), np.column_stack((edges[:-1], edges[1:])).ravel())
                )
            )
        order = np.argsort(misfits)[::-1]  # the best drawn last, on top
        fit = colors.LogNorm(
            vmin=max(float(np.min(misfits)), 1e-12),
            vmax=max(float(np.max(misfits)), float(np.min(misfits)) * 1.001, 1e-12),
        )
        greys = plt.get_cmap("Greys_r")
        ax_vs.add_collection(
            LineCollection(
                [lines[i] for i in order],
                colors=[greys(0.25 + 0.6 * float(fit(misfits[i]))) for i in order],
                linewidths=0.3,
                zorder=0.8,  # over the grid, under the band and the median
            )
        )
        ax_vs.plot([], [], color=_EXPLORED, linewidth=0.8, label="each kept model")
        span = float(np.nanmax(high) - np.nanmin(low)) or 1.0
        ax_vs.set_xlim(float(np.nanmin(low)) - 0.25 * span, float(np.nanmax(high)) + 0.25 * span)
    tops = np.concatenate(([0.0], np.cumsum(ensemble.thicknesses)[:-1]))
    ax_vs.step(
        np.asarray(ensemble.vs_s),
        tops,
        where="post",
        color=_VS,
        linewidth=1.2,
        label="median of the ensemble",
        zorder=3,
    )
    ax_u.plot(100 * uncertainty, depths, color=_UNCERTAINTY, linewidth=1.0, label="uncertainty")
    if interfaces:
        shares = 100 * np.asarray(interfaces, dtype=float)
        edges = np.arange(shares.size + 1) * interface_dz
        ax_i.stairs(
            shares,
            edges,
            orientation="horizontal",
            color=_INTERFACES,
            linewidth=1.0,
            label="interfaces",
        )
    if informed is not None and informed < bottom:
        for ax in (ax_vs, ax_u, ax_i):
            ax.axhspan(informed, bottom, color=_VEIL, linewidth=0, zorder=0)
            ax.axhline(informed, color=_INFORMED, linestyle="--", linewidth=0.8, zorder=4)
        ax_vs.text(
            0.98,
            informed,
            f"informed to {informed:g} m" if informed > 0 else "not informed",
            transform=ax_vs.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=small - 1,
            color=_INFORMED,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.0},
            zorder=4,
        )
        ax_vs.fill_between([], [], color=_VEIL, label="not informed by the data")
    else:  # said too when the data inform all of it: no line drawn is no depth left out
        ax_vs.text(
            0.98,
            bottom,
            "informed to the whole model",
            transform=ax_vs.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=small - 1,
            color=_INFORMED,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.0},
            zorder=4,
        )
    ax_vs.set_ylim(bottom, 0)
    ax_vs.set_xlabel("Vs [m/s]", fontsize=small)
    ax_vs.set_ylabel("Depth [m]", fontsize=small)
    ax_vs.set_title("Vs profile", fontsize=small + 1, loc="left")
    ax_u.set_xlabel("Uncertainty [%]", fontsize=small)
    ax_u.set_xlim(0, max(10.0, float(np.nanmax(100 * uncertainty)) * 1.08))
    ax_u.tick_params(labelleft=False)
    ax_i.set_xlabel("Interfaces [%]", fontsize=small)
    ax_i.set_xlim(0, max(10.0, 100 * max(interfaces, default=0.0)) * 1.08)
    ax_i.tick_params(labelleft=False)

    for ax in (ax_fit, ax_vs, ax_u, ax_i):
        ax.tick_params(labelsize=small - 1)
        ax.grid(color="#ecebe6", linewidth=0.5)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.3)
    # Each legend centred under its plots: the curve's; the profile's, a column each: the profile,
    # its uncertainty and interfaces, the other medians, the best models.
    under = ax_fit.get_position().y0 - 0.12
    ax_fit.legend(
        loc="upper center",
        bbox_to_anchor=(ax_fit.get_position().x0 + ax_fit.get_position().width / 2, under),
        bbox_transform=fig.transFigure,
        fontsize=small - 1,
        frameon=False,
        ncol=2,
    )
    found: dict[str, Any] = {}
    for ax in (ax_vs, ax_u, ax_i):
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            found[label] = handle
    first = ["each kept model", "10-90 % of the models", "median of the ensemble"]
    order = [*first, "not informed by the data", "uncertainty", "interfaces"]
    shown = [label for label in order if label in found]
    fig.legend(
        [found[label] for label in shown],
        shown,
        loc="upper center",
        bbox_to_anchor=((ax_vs.get_position().x0 + ax_i.get_position().x1) / 2, under),
        fontsize=small - 1,
        frameon=False,
        ncol=3 if len(shown) > 5 else 2,
        columnspacing=1.2,
        handlelength=1.6,
    )
    return fig


def plot_chains(traces: dict[str, np.ndarray], priors: dict[str, tuple[float, float]]) -> Figure:
    """Each chain's saved samples of each parameter along the run, as PAC's Chains view draws
    them: a panel a parameter (`traces`, by its label: chains x samples), each chain in its
    colour (chains that agree overlap, a stuck chain stays flat), its prior's bounds dashed
    where in view (`priors`, by label)."""
    labels = list(traces)
    columns = 2 if len(labels) > 1 else 1
    rows = max(1, -(-len(labels) // columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(DOUBLE_COLUMN_CM * CM, (3.4 * rows + 1.4) * CM),
        dpi=DISP_DPI,
        squeeze=False,
    )
    small = 7
    n_chains = max((chains.shape[0] for chains in traces.values()), default=0)
    for ax, label in zip(axes.flat, labels, strict=False):
        chains = np.asarray(traces[label], dtype=float)
        samples = np.arange(chains.shape[1])
        for c, chain in enumerate(chains):
            colour = _CHAIN_COLOURS[c % len(_CHAIN_COLOURS)]
            ax.plot(samples, chain, color=colour, linewidth=0.5, alpha=0.85)
        low, high = float(np.nanmin(chains)), float(np.nanmax(chains))
        pad = (high - low) * 0.06 or abs(high) * 0.05 or 1.0
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlim(0, max(1, chains.shape[1] - 1))
        for bound in priors.get(label, ()):
            ax.axhline(bound, color=_LIMIT, linestyle=(0, (4, 3)), linewidth=0.8)
        ax.set_title(label, fontsize=small, loc="left")
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=small - 1)
        ax.grid(color="#ecebe6", linewidth=0.5)
        ax.set_axisbelow(True)
    for ax in axes.flat[len(labels) :]:
        ax.set_visible(False)
    # The saved sample under each column's lowest panel drawn.
    for column in range(columns):
        drawn = [axes[row, column] for row in range(rows) if axes[row, column].get_visible()]
        if drawn:
            drawn[-1].set_xlabel("Saved sample", fontsize=small)
    handles = [
        Line2D([], [], color=_CHAIN_COLOURS[c % len(_CHAIN_COLOURS)], label=f"chain {c + 1}")
        for c in range(n_chains)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=max(1, n_chains),
        fontsize=small - 1,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.9 * CM / fig.get_figheight(), 1, 1), h_pad=0.8)
    return fig
