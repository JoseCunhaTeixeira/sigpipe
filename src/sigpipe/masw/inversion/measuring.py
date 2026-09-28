"""What a window's inversion files say about its models: the fit of the monitored model (the
kept models' median at each depth, the ensemble) and of the layered median, by band of
wavelength; how the chains agree, how many independent samples they hold and how autocorrelated
they are, on the models' Vs at depths the curve resolves (the one measure of both layerings: a
layer's own values mean nothing when the data choose the layers); how much of the posterior sits
on the prior's bounds; down to which depth the data inform the model; where the kept models put
interfaces; and the monitored model's Vs at given depths. Measurements only: PACo's QC (G5)
judges them."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.stats import norm, rankdata

from sigpipe.algorithms.inversion.rayleigh.seismic.parameters import InversionParameters
from sigpipe.algorithms.inversion.rayleigh.seismic.transdimensional import THINNEST
from sigpipe.base.dispersion_curve import DispersionCurve
from sigpipe.base.inversion import LayeredSamples
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.dataio.velocity_model.loading import load_velocity_models
from sigpipe.masw.inversion.window import (
    PARAMETERS_FILE,
    SAMPLES_FILE,
    load_parameters,
    load_profiles,
    load_samples,
)
from sigpipe.masw.picks import CURVES_FILE

# The monitored model first (PAC's default view: at each depth, the kept models' median Vs), then
# the layered median (the kept model nearest it).
MODELS = ("ensemble", "median")
# The depth bins the kept models' interfaces are counted in (m).
INTERFACE_DZ = 0.5
LOG_FILE = "SeismicInversion_Log_0000.log"
# The acceptance rates in the log of runs saved before 2026-09-27, which their parameters lack.
_RATE = re.compile(r"ACCEPTANCE RATE: \d+/\d+ \(([\d.]+) %\)")
# Depths the chains' agreement is measured at, between a third of the shortest and of the longest
# picked wavelength.
WATCHED = 5
# The draws of a prior model at most, until one keeps the Vs drop allowed.
DRAWS = 200


class BandFit(BaseModel):
    """A model's fit to the picked curve over one band of wavelengths."""

    model_config = ConfigDict(frozen=True)

    wavelength_m: tuple[float, float]
    n_points: int
    misfit: float | None  # RMS of (picked - modelled) / uncertainty; None: no mode there
    residual: float | None  # median (modelled - picked) / modelled: PAC's residual, signed


class ModelFit(BaseModel):
    """A model's forward-modelled M0 against the picked curve."""

    model_config = ConfigDict(frozen=True)

    model: str  # one of MODELS
    misfit: float | None  # over every point the model has a mode at; None: at none
    n_missing: int  # picked points at which the model has no fundamental mode
    bands: tuple[BandFit, ...]  # short wavelengths first
    lowest_missing_hz: float | None = None  # the lowest of those points' frequencies


class BoundShare(BaseModel):
    """How much of the posterior of one parameter lies at one of its prior's bounds."""

    model_config = ConfigDict(frozen=True)

    # vs1, ..., thick1, ... (layers from the top) when the layers are given; when the data chose
    # them: top_vs, half_space_vs, deepest_interface, layers
    parameter: str
    bound: Literal["min", "max"]
    value: float  # the bound
    share: float  # of the samples within the watched edge of the prior's range


# What a useful depth read against `yardstick` says it was read against: the measures saved
# before it hold none, and their useful depth compares with no other window's.
USEFUL_REFERENCE = "curve"


class InversionMeasures(BaseModel):
    """Everything G5 judges, and job_status reports, of one window's inversion."""

    model_config = ConfigDict(frozen=True)

    fits: tuple[ModelFit, ...]  # in the order of MODELS
    # Split R-hat per series: Vs at depths (vs@2.5m, ...), the layers' own values when given
    # (vs1, ..., thick1, ...), the number of layers, the noise factor; None: not measurable.
    rhat: dict[str, float | None]
    watched: tuple[str, ...] = ()  # the series whose agreement judges the chains: Vs at depths
    # Effective samples per parameter over every chain (rank-normalized, split chains), and the
    # chains' mean lag-1 autocorrelation of the saved samples; None: not measurable.
    ess: dict[str, float | None] = {}
    autocorrelation: dict[str, float | None] = {}
    acceptance: tuple[float, ...]  # per chain, %
    # Each parameter's 5th and 95th percentiles over every chain: where its samples lie.
    quantiles: dict[str, tuple[float, float]] = {}
    steps: dict[str, float] = {}  # each parameter's step as the sampler ran it
    samples_per_chain: int
    at_bounds: tuple[BoundShare, ...]  # the most piled first
    useful_depth_m: float | None  # where the posterior's spread reaches the prior's; None: nowhere
    # What the useful depth was read against: USEFUL_REFERENCE (one yardstick for every window);
    # empty in measures from before 2026-09-28, against the run's own prior.
    useful_reference: str = ""
    depth_max_m: float  # the bottom of the models sigpipe builds
    vs_at_depths: tuple[tuple[float, float], ...]  # (depth m, the monitored model's Vs m/s)
    vs_layers: tuple[float, ...]  # the layered median, top down
    interfaces_m: tuple[float, ...]  # the layered median's interface depths
    # Per INTERFACE_DZ from the surface down to the bottom, the share of the kept models with an
    # interface there; empty in measures from before 2026-09-28.
    interfaces: tuple[float, ...] = ()

    def fit(self, model: str) -> ModelFit:
        return next(fit for fit in self.fits if fit.model == model)


def measure_inversion(
    folder: Path,
    parameters: InversionParameters,
    depths: Sequence[float] = (),
    n_bands: int = 3,
    bound_edge: float = 0.02,
    std_ratio: float = 0.5,
    output_folder: Path | None = None,
) -> InversionMeasures:
    """The measures of the inversion saved in window folder `folder` (or in `output_folder`, a
    staging folder, beside the curves `folder` keeps), inverted with
    `parameters`: the fits over `n_bands` bands of the picked curve's wavelengths, the share of
    each parameter's samples within `bound_edge` of its prior's range from each bound, the
    useful depth where the posterior's spread of Vs reaches `std_ratio` of the spread of one
    yardstick for every window (`yardstick`), where the kept models put interfaces, and the
    monitored model's Vs at `depths`."""
    picked = _picked_m0(folder)
    out = output_folder or folder
    fits = tuple(fit_by_band(model, picked, _forward(out, model), n_bands) for model in MODELS)
    ran_file = out / PARAMETERS_FILE
    window = load_parameters(ran_file) if ran_file.exists() else None
    # As the chains ran: the free layering's bounds as found from the curves.
    fs, vs = np.asarray(picked.fs, dtype=float), np.asarray(picked.vs, dtype=float)
    ran = window.parameters if window is not None else parameters.resolved(fs, vs)
    samples, n_chains = load_samples(out / SAMPLES_FILE)
    profiles = load_profiles(out / SAMPLES_FILE)
    depths_watched = convergence_depths(picked, ran.bottom)
    at_depths = profiles.at(np.asarray(depths_watched))
    watched = tuple(f"vs@{depth:g}m" for depth in depths_watched)
    series = {name: at_depths[:, j] for j, name in enumerate(watched)}
    # The convergence of what was sampled: a value fixed has none to measure.
    fixed = ran.fixed()
    series |= {name: values for name, values in samples.items() if name not in fixed}
    chains = {
        name: profiles.per_chain(np.asarray(values, dtype=float)) for name, values in series.items()
    }
    per_chain = profiles.vs.shape[0] // n_chains
    monitored = load_velocity_models([out / f"SeismicInversion_Model_0000_{MODELS[0]}.csv"])[0][0]
    median = load_velocity_models([out / "SeismicInversion_Model_0000_median.csv"])[0][0]
    acceptance = (
        window.acceptance
        if window is not None and window.acceptance
        else acceptance_rates((out / LOG_FILE).read_text())
        if (out / LOG_FILE).exists()
        else ()
    )
    return InversionMeasures(
        fits=fits,
        rhat={name: split_rhat(values) for name, values in chains.items()},
        watched=watched,
        ess={name: effective_sample_size(values) for name, values in chains.items()},
        autocorrelation={name: lag1_autocorrelation(values) for name, values in chains.items()},
        acceptance=acceptance,
        steps=dict(window.steps) if window is not None and window.steps else _steps(ran),
        samples_per_chain=per_chain,
        at_bounds=(
            bound_shares(samples, ran, bound_edge)
            if ran.layering == "fixed"
            else free_bound_shares(profiles, ran, bound_edge)
        ),
        useful_depth_m=useful_depth(profiles, ran, std_ratio, reference=yardstick(fs, vs)),
        useful_reference=USEFUL_REFERENCE,
        quantiles={
            name: (
                round(float(np.percentile(values, 5)), 3),
                round(float(np.percentile(values, 95)), 3),
            )
            for name, values in series.items()
        },
        depth_max_m=ran.bottom,
        vs_at_depths=tuple(
            (float(depth), value)
            for depth, value in zip(
                depths, vs_at(monitored.thicknesses, monitored.vs_s, depths), strict=True
            )
        ),
        vs_layers=tuple(round(float(value), 1) for value in median.vs_s),
        interfaces_m=tuple(round(float(depth), 2) for depth in np.cumsum(median.thicknesses[:-1])),
        interfaces=interface_shares(profiles, ran.bottom),
    )


def interface_shares(
    profiles: LayeredSamples, bottom: float, dz: float = INTERFACE_DZ
) -> tuple[float, ...]:
    """Per `dz` from the surface down to `bottom`: the share of the kept models with an interface
    there (once a model), where the data put the layering."""
    bins = int(np.ceil(bottom / dz))
    depths = np.asarray(profiles.depths, dtype=float)
    rows, columns = np.nonzero(~np.isnan(depths))
    which = np.floor(depths[rows, columns] / dz).astype(int)
    inside = (which >= 0) & (which < bins)
    hit = np.zeros((depths.shape[0], bins), dtype=bool)
    hit[rows[inside], which[inside]] = True
    return tuple(round(float(share), 3) for share in hit.mean(axis=0))


def _picked_m0(folder: Path) -> DispersionCurve:
    """The fundamental mode among window folder `folder`'s picked curves; StopIteration when
    none is."""
    return next(
        curve
        for curve in load_dispersion_curves([folder / CURVES_FILE])[0]
        if curve.mode.number == 0
    )


def convergence_depths(
    picked: DispersionCurve, bottom: float, count: int = WATCHED
) -> tuple[float, ...]:
    """`count` depths (m) evenly spaced in their logarithm, from a third of the shortest to a
    third of the longest picked wavelength (the bottom of the models at most): where the curve
    resolves the models for sure, and so where the chains must agree. Deeper, down to half the
    longest wavelength, the curve still says something, but the deepest interfaces allowed sit
    there, and the Vs of a depth an interface may be above or below takes two values."""
    wavelengths = np.asarray(picked.vs, dtype=float) / np.asarray(picked.fs, dtype=float)
    top = max(0.1, float(np.min(wavelengths)) / 3)
    base = max(top * 1.5, min(float(np.max(wavelengths)) / 3, bottom))
    return tuple(sorted({round(float(depth), 1) for depth in np.geomspace(top, base, count)}))


def fit_by_band(
    model: str, picked: DispersionCurve, modelled: DispersionCurve | None, n_bands: int
) -> ModelFit:
    """`modelled`, the model's M0 at the picked frequencies (None: no mode at all), against
    `picked`, over bands holding equal numbers of points, from the short wavelengths up. A
    point without an uncertainty (NaN, or 0) is left out of the misfits: nothing weighs it."""
    fs, vs = np.asarray(picked.fs, dtype=float), np.asarray(picked.vs, dtype=float)
    errors = np.asarray(picked.vs_err if picked.vs_err is not None else vs, dtype=float)
    predicted = np.full_like(vs, np.nan)
    if modelled is not None:
        predicted = np.interp(
            fs, np.asarray(modelled.fs, dtype=float), np.asarray(modelled.vs, dtype=float)
        )
    order = np.argsort(vs / fs)
    bands = tuple(
        _band(vs[band], predicted[band], errors[band], (vs / fs)[band])
        for band in np.array_split(order, min(n_bands, vs.size))
    )
    known = ~np.isnan(predicted)
    weighed = known & _weighable(errors)
    return ModelFit(
        model=model,
        misfit=_rms(vs[weighed], predicted[weighed], errors[weighed]) if weighed.any() else None,
        n_missing=int((~known).sum()),
        bands=bands,
        lowest_missing_hz=round(float(fs[~known].min()), 2) if (~known).any() else None,
    )


def split_rhat(chains: np.ndarray) -> float | None:
    """Gelman and Rubin's R-hat on `chains` (chains x samples), each chain split in halves: near
    1 when the chains sample the same distribution. None with fewer than 2 samples a half."""
    half = chains.shape[1] // 2
    if half < 2:
        return None
    parts = np.concatenate([chains[:, :half], chains[:, half : 2 * half]])
    within = float(parts.var(axis=1, ddof=1).mean())
    between = half * float(parts.mean(axis=1).var(ddof=1))
    if within == 0:
        return 1.0 if between == 0 else None
    pooled = (half - 1) / half * within + between / half
    return round(float(np.sqrt(pooled / within)), 3)


def effective_sample_size(chains: np.ndarray) -> float | None:
    """The bulk effective sample size of `chains` (chains x samples): the draws rank-normalized
    and each chain split in halves, then Geyer's initial positive sequence of the chains'
    combined autocorrelations (Vehtari et al. 2021, as Stan and ArviZ compute it). About the
    number of samples when they are independent, far fewer when each repeats the one before.
    None with fewer than 4 samples a half."""
    half = chains.shape[1] // 2
    if half < 4:
        return None
    parts = np.concatenate([chains[:, :half], chains[:, half : 2 * half]])
    m, n = parts.shape
    ranks = rankdata(parts, method="average").reshape(m, n)
    z = norm.ppf((ranks - 0.375) / (m * n + 0.25))
    centred = z - z.mean(axis=1, keepdims=True)
    # Each half's autocovariance by FFT, biased (divided by n), lags 0 to n - 1.
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centred, size, axis=1)
    autocov = np.fft.irfft(spectrum * np.conj(spectrum), size, axis=1)[:, :n] / n
    within = float(np.mean(autocov[:, 0] * n / (n - 1)))
    variance = within * (n - 1) / n + float(np.var(z.mean(axis=1), ddof=1)) if m > 1 else within
    if variance <= 0:
        return None
    rho = 1 - (within - autocov.mean(axis=0)) / variance
    rho[0] = 1.0
    # Sums of consecutive pairs while positive, never increasing.
    total, previous = 0.0, np.inf
    for k in range(0, n - 1, 2):
        pair = float(rho[k] + rho[k + 1])
        if pair <= 0:
            break
        pair = min(pair, previous)
        total += pair
        previous = pair
    tau = max(-1 + 2 * total, 1 / np.log10(m * n))
    return round(float(m * n / tau), 1)


def lag1_autocorrelation(chains: np.ndarray) -> float | None:
    """The chains' mean correlation between each saved sample and the next; None with fewer
    than 3 samples a chain or a chain that never moved."""
    if chains.shape[1] < 3:
        return None
    centred = chains - chains.mean(axis=1, keepdims=True)
    variance = (centred**2).sum(axis=1)
    if np.any(variance == 0):
        return None
    lagged = (centred[:, 1:] * centred[:, :-1]).sum(axis=1)
    return round(float(np.mean(lagged / variance)), 3)


def acceptance_rates(log: str) -> tuple[float, ...]:
    """Each chain's acceptance rate in %, from the sampler's statistics in the log."""
    return tuple(float(rate) for rate in _RATE.findall(log))


def bound_shares(
    samples: dict[str, np.ndarray], parameters: InversionParameters, edge: float
) -> tuple[BoundShare, ...]:
    """For each parameter and each bound of its prior, the share of the samples within `edge` of
    the prior's range from that bound; the most piled first."""
    priors = {
        f"vs{i + 1}": (layer.vs_min, layer.vs_max) for i, layer in enumerate(parameters.vs_layers)
    }
    priors |= {
        f"thick{i + 1}": (layer.thickness_min, layer.thickness_max)
        for i, layer in enumerate(parameters.thickness_layers)
    }
    fixed = parameters.fixed()  # not sampled: no prior to pile against
    priors = {name: prior for name, prior in priors.items() if name not in fixed}
    shares: list[BoundShare] = []
    for name, (low, high) in priors.items():
        values = np.asarray(samples[name], dtype=float)
        width = edge * (high - low)
        shares += [
            BoundShare(
                parameter=name,
                bound="min",
                value=low,
                share=round(float((values <= low + width).mean()), 3),
            ),
            BoundShare(
                parameter=name,
                bound="max",
                value=high,
                share=round(float((values >= high - width).mean()), 3),
            ),
        ]
    return tuple(sorted(shares, key=lambda share: -share.share))


def free_bound_shares(
    profiles: LayeredSamples, parameters: InversionParameters, edge: float
) -> tuple[BoundShare, ...]:
    """When the data chose the layers: the shares of the models whose top layer's or
    half-space's Vs lies within `edge` of the Vs prior's range (in its logarithm, as it is
    sampled) from a bound, whose deepest interface lies as close to the deepest allowed, and
    which hold as many layers as allowed; the most piled first."""
    free = parameters.free
    assert free.vs_min is not None and free.vs_max is not None and free.depth_max is not None
    assert free.depth_min is not None
    low, high = np.log(free.vs_min), np.log(free.vs_max)
    width = edge * (high - low)
    count = profiles.layers
    top = np.log(profiles.vs[:, 0])
    half_space = np.log(profiles.vs[np.arange(count.size), count - 1])
    shares = [
        BoundShare(
            parameter=name,
            bound=bound,
            value=round(float(np.exp(limit)), 1),
            share=round(
                float(
                    np.mean(values <= limit + width if bound == "min" else values >= limit - width)
                ),
                3,
            ),
        )
        for name, values in (("top_vs", top), ("half_space_vs", half_space))
        for bound, limit in _ends(low, high)
    ]
    deepest = np.nanmax(
        np.where(np.isnan(profiles.depths), -np.inf, profiles.depths), axis=1, initial=-np.inf
    )
    reach = np.log(free.depth_max) - edge * np.log(free.depth_max / free.depth_min)
    shares.append(
        BoundShare(
            parameter="deepest_interface",
            bound="max",
            value=free.depth_max,
            share=round(float(np.mean(deepest >= np.exp(reach))), 3),
        )
    )
    shares.append(
        BoundShare(
            parameter="layers",
            bound="max",
            value=float(free.max_layers),
            share=round(float(np.mean(count >= free.max_layers)), 3),
        )
    )
    return tuple(sorted(shares, key=lambda share: -share.share))


def _ends(low: float, high: float) -> tuple[tuple[Literal["min", "max"], float], ...]:
    """A range's two bounds, each named."""
    return (("min", low), ("max", high))


def yardstick(fs: np.ndarray, vs: np.ndarray) -> InversionParameters:
    """What a window's useful depth is read against, the same for every window whatever
    layering or bounds it ran with: the prior of the layers chosen by the data, its bounds found
    from the fundamental mode's picks (frequencies Hz, phase velocities m/s): Vs from half the
    slowest pick to three times the fastest, interfaces from a third of the shortest wavelength
    to half the longest. Against a run's own prior, a narrow one made every model look
    uninformed and a wide one every model informed: two windows could not be compared."""
    return InversionParameters().resolved(fs, vs)


def useful_depth(
    profiles: LayeredSamples,
    parameters: InversionParameters,
    ratio: float,
    dz: float = 0.05,
    n_prior: int = 2_000,
    reference: InversionParameters | None = None,
) -> float | None:
    """The depth below which the spread of the sampled Vs stays at least `ratio` of the prior's
    spread there (the prior drawn `n_prior` times, with a fixed seed): below it, the data say
    little. The prior is `reference`'s when given (`yardstick`'s, for measure_inversion), else
    `parameters`' (both resolved); the depths read, `parameters`' models'. The spread is the
    interquartile range, which a minority of samples in another mode does not widen as it does
    the standard deviation. Read from the bottom up, so that a thin top layer the data cannot
    resolve does not end it at the surface. 0 when the data inform no depth, None when they
    inform the models down to their bottom."""
    depth_max = parameters.bottom
    grid = (np.arange(int(np.ceil(depth_max / dz))) + 0.5) * dz
    posterior = profiles.at(grid)
    prior = prior_draws(reference if reference is not None else parameters, n_prior).at(grid)
    informed = np.flatnonzero(_spread(posterior) < ratio * _spread(prior))
    if not informed.size:
        return 0.0
    if informed[-1] == posterior.shape[1] - 1:
        return None
    return round(float((informed[-1] + 1) * dz), 2)


def prior_draws(parameters: InversionParameters, count: int, seed: int = 0) -> LayeredSamples:
    """`count` models of the priors (resolved), as the chains' priors hold them: the layers given
    or chosen, the Vs drop allowed (a draw breaking it drawn again, then its Vs sorted after
    DRAWS draws: close enough to the prior for its spread). Drawn all at once, those breaking the
    drop again together: a window's yardstick is drawn whenever its measures are read again."""
    rng = np.random.default_rng(seed)
    draw = _fixed_draws if parameters.layering == "fixed" else _free_draws
    depths, vs = draw(parameters, count, rng)
    pending = np.flatnonzero(_drops(vs, parameters.least_ratio))
    for _ in range(DRAWS - 1):
        if not pending.size:
            break
        depths[pending], vs[pending] = draw(parameters, pending.size, rng)
        pending = pending[_drops(vs[pending], parameters.least_ratio)]
    vs[pending] = np.sort(vs[pending], axis=1)  # the padding (NaN) stays last
    return LayeredSamples(depths=depths, vs=vs, n_chains=1)


def _drops(vs: np.ndarray, least_ratio: float) -> np.ndarray:
    """Whether each model (a row, NaN past its layers) has a layer whose Vs is under
    `least_ratio` of the one above it."""
    return np.any(vs[:, 1:] < least_ratio * vs[:, :-1], axis=1)


def _fixed_draws(
    parameters: InversionParameters, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """`count` models of the layers given: each thickness and Vs uniform within its bounds."""
    thickness = np.column_stack(
        [rng.uniform(*layer.bounds, count) for layer in parameters.thickness_layers]
    )
    vs = np.column_stack([rng.uniform(*layer.bounds, count) for layer in parameters.vs_layers])
    return np.cumsum(thickness, axis=1), vs


def _free_draws(
    parameters: InversionParameters, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """`count` models of layers chosen by the data (NaN past each one's layers): 1 to max_layers
    layers, their interfaces log-uniform within the depths' bounds, a layer at least THINNEST
    of its top's depth thick (the interfaces drawn again, up to 100 times, when they break it),
    each Vs log-uniform within its bounds."""
    free = parameters.free
    assert free.vs_min is not None and free.vs_max is not None
    assert free.depth_min is not None and free.depth_max is not None
    most = free.max_layers
    layers = rng.integers(1, most + 1, count)
    # Past a model's interfaces: +inf, sorted last and never too close to the one before.
    unused = np.arange(most - 1)[None, :] >= (layers - 1)[:, None]
    low, high = np.log(free.depth_min), np.log(free.depth_max)
    gap = np.log(1 + THINNEST)

    def interfaces(rows: np.ndarray) -> np.ndarray:
        drawn = rng.uniform(low, high, (rows.size, most - 1))
        drawn[unused[rows]] = np.inf
        return np.sort(drawn, axis=1)

    log_depths = interfaces(np.arange(count))
    for _ in range(100):
        # inf - inf, past the interfaces, is NaN: never under the gap.
        with np.errstate(invalid="ignore"):
            close = np.flatnonzero(np.any(np.diff(log_depths, axis=1) < gap, axis=1))
        if not close.size:
            break
        log_depths[close] = interfaces(close)
    log_vs = rng.uniform(np.log(free.vs_min), np.log(free.vs_max), (count, most))
    log_vs[np.arange(most)[None, :] >= layers[:, None]] = np.nan
    depths = np.exp(log_depths)
    depths[unused] = np.nan
    return depths, np.exp(log_vs)


def _steps(parameters: InversionParameters) -> dict[str, float]:
    """Each sampled parameter's step in `parameters`, by name (vs1, ..., thick1, ...): the
    steps runs saved before give in their parameters; none when the data chose the layers."""
    if parameters.layering != "fixed":
        return {}
    steps = {f"vs{i + 1}": layer.vs_perturb_std for i, layer in enumerate(parameters.vs_layers)}
    steps |= {
        f"thick{i + 1}": layer.thickness_perturb_std
        for i, layer in enumerate(parameters.thickness_layers)
    }
    fixed = parameters.fixed()
    return {name: step for name, step in steps.items() if name not in fixed}


def _spread(rasters: np.ndarray) -> np.ndarray:
    """The interquartile range of the samples' Vs at each depth."""
    high, low = np.percentile(rasters, [75, 25], axis=0)
    return high - low


def depth_bottom(parameters: InversionParameters) -> float:
    """The bottom of the models sigpipe builds (the parameters resolved): InversionParameters'
    own."""
    return parameters.bottom


def vs_at(
    thicknesses: Sequence[float], vs: Sequence[float], depths: Sequence[float]
) -> tuple[float, ...]:
    """A layered model's Vs at `depths` (a depth on an interface belongs to the layer below)."""
    bottoms = np.cumsum(np.asarray(thicknesses, dtype=float))
    index = np.minimum(np.searchsorted(bottoms, np.asarray(depths), side="right"), len(vs) - 1)
    return tuple(round(float(vs[i]), 1) for i in index)


def report_depths(longest_wavelengths: Sequence[float], count: int = 5) -> tuple[float, ...]:
    """Round depths, at most `count`, down to half the median of `longest_wavelengths` (the
    depth a line's curves resolve): where a line's models are compared and reported."""
    if not longest_wavelengths:
        return ()
    reach = float(np.median(longest_wavelengths)) / 2
    for step in (0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0):
        if int(reach / step + 1e-9) <= count:
            break
    else:
        step = reach / count
    return tuple(round(step * k, 2) for k in range(1, int(reach / step + 1e-9) + 1))


def _forward(folder: Path, model: str) -> DispersionCurve | None:
    path = folder / f"SeismicInversion_DispersionCurves_0000_{model}.csv"
    if not path.exists():  # disba found no mode of this model at all
        return None
    return load_dispersion_curves([path])[0][0]


def _band(
    picked: np.ndarray, predicted: np.ndarray, errors: np.ndarray, wavelengths: np.ndarray
) -> BandFit:
    known = ~np.isnan(predicted)
    weighed = known & _weighable(errors)
    return BandFit(
        wavelength_m=(round(float(wavelengths.min()), 2), round(float(wavelengths.max()), 2)),
        n_points=int(picked.size),
        misfit=_rms(picked[weighed], predicted[weighed], errors[weighed])
        if weighed.any()
        else None,
        residual=(
            round(float(np.median((predicted[known] - picked[known]) / predicted[known])), 3)
            if known.any()
            else None
        ),
    )


def _weighable(errors: np.ndarray) -> np.ndarray:
    return np.isfinite(errors) & (errors > 0)


def _rms(picked: np.ndarray, predicted: np.ndarray, errors: np.ndarray) -> float:
    return round(float(np.sqrt(np.mean(((picked - predicted) / errors) ** 2))), 3)
