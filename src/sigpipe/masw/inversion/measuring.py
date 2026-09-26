"""What a window's inversion files say about its models: the fit of the monitored smooth median
and of the layered median it comes from, by band of wavelength; how the chains agree; how much
of the posterior sits on the prior's bounds; down to which depth the data inform the model;
and the smooth median's Vs at given depths. Measurements only: PACo's QC (G5) judges them."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from sigpipe.algorithms.inversion.rayleigh.seismic.parameters import InversionParameters
from sigpipe.base.dispersion_curve import DispersionCurve
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.dataio.velocity_model.loading import load_velocity_models
from sigpipe.masw.inversion.window import SAMPLES_FILE, load_samples
from sigpipe.masw.picks import CURVES_FILE

# The monitored model first (PAC's default view), then the layered model it smooths.
MODELS = ("smooth_median", "median")
LOG_FILE = "SeismicInversion_Log_0000.log"
_RATE = re.compile(r"ACCEPTANCE RATE: \d+/\d+ \(([\d.]+) %\)")


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

    parameter: str  # vs1, ..., thick1, ... (layers from the top)
    bound: Literal["min", "max"]
    value: float  # the bound
    share: float  # of the samples within the watched edge of the prior's range


class InversionMeasures(BaseModel):
    """Everything G5 judges, and job_status reports, of one window's inversion."""

    model_config = ConfigDict(frozen=True)

    fits: tuple[ModelFit, ...]  # in the order of MODELS
    rhat: dict[str, float | None]  # split R-hat per parameter; None: not measurable
    acceptance: tuple[float, ...]  # per chain, %
    samples_per_chain: int
    at_bounds: tuple[BoundShare, ...]  # the most piled first
    useful_depth_m: float | None  # where the posterior's spread reaches the prior's; None: nowhere
    depth_max_m: float  # the bottom of the models sigpipe builds
    vs_at_depths: tuple[tuple[float, float], ...]  # (depth m, smooth median Vs m/s)
    vs_layers: tuple[float, ...]  # the layered median, top down
    interfaces_m: tuple[float, ...]  # the layered median's interface depths

    def fit(self, model: str) -> ModelFit:
        return next(fit for fit in self.fits if fit.model == model)


def measure_inversion(
    folder: Path,
    parameters: InversionParameters,
    depths: Sequence[float] = (),
    n_bands: int = 3,
    bound_edge: float = 0.02,
    std_ratio: float = 0.5,
) -> InversionMeasures:
    """The measures of the inversion saved in window folder `folder`, inverted with
    `parameters`: the fits over `n_bands` bands of the picked curve's wavelengths, the share of
    each parameter's samples within `bound_edge` of its prior's range from each bound, the
    useful depth where the posterior's spread of Vs reaches `std_ratio` of the prior's, and the
    smooth median's Vs at `depths`."""
    picked = next(
        curve
        for curve in load_dispersion_curves([folder / CURVES_FILE])[0]
        if curve.mode.number == 0
    )
    fits = tuple(fit_by_band(model, picked, _forward(folder, model), n_bands) for model in MODELS)
    samples, n_chains = load_samples(folder / SAMPLES_FILE)
    per_chain = len(next(iter(samples.values()))) // n_chains
    rhat = {
        name: split_rhat(np.asarray(values)[: per_chain * n_chains].reshape(n_chains, per_chain))
        for name, values in samples.items()
    }
    smooth = load_velocity_models([folder / "SeismicInversion_Model_0000_smooth_median.csv"])[0][0]
    median = load_velocity_models([folder / "SeismicInversion_Model_0000_median.csv"])[0][0]
    depth_max = depth_bottom(parameters)
    return InversionMeasures(
        fits=fits,
        rhat=rhat,
        acceptance=acceptance_rates((folder / LOG_FILE).read_text()),
        samples_per_chain=per_chain,
        at_bounds=bound_shares(samples, parameters, bound_edge),
        useful_depth_m=useful_depth(samples, parameters, std_ratio),
        depth_max_m=depth_max,
        vs_at_depths=tuple(
            (float(depth), vs)
            for depth, vs in zip(
                depths, vs_at(smooth.thicknesses, smooth.vs_s, depths), strict=True
            )
        ),
        vs_layers=tuple(round(float(vs), 1) for vs in median.vs_s),
        interfaces_m=tuple(round(float(depth), 2) for depth in np.cumsum(median.thicknesses[:-1])),
    )


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


def useful_depth(
    samples: dict[str, np.ndarray],
    parameters: InversionParameters,
    ratio: float,
    dz: float = 0.05,
    n_prior: int = 2_000,
) -> float | None:
    """The shallowest depth at which the spread of the sampled Vs reaches `ratio` of the prior's
    spread there (the prior drawn `n_prior` times, with a fixed seed): below it, the data say
    little. None when the posterior stays narrower down to the models' bottom."""
    n_layers = parameters.n_layers
    depth_max = depth_bottom(parameters)
    posterior = _raster(
        np.array([samples[f"vs{i + 1}"] for i in range(n_layers)]),
        np.array([samples[f"thick{i + 1}"] for i in range(n_layers - 1)]),
        dz,
        depth_max,
    )
    rng = np.random.default_rng(0)
    prior = _raster(
        np.array(
            [rng.uniform(layer.vs_min, layer.vs_max, n_prior) for layer in parameters.vs_layers]
        ),
        np.array(
            [
                rng.uniform(layer.thickness_min, layer.thickness_max, n_prior)
                for layer in parameters.thickness_layers
            ]
        ),
        dz,
        depth_max,
    )
    reached = np.flatnonzero(posterior.std(axis=0) >= ratio * prior.std(axis=0))
    return round(float(reached[0] * dz), 2) if reached.size else None


def depth_bottom(parameters: InversionParameters) -> float:
    """The bottom of the models sigpipe builds: the deepest interface the prior allows, plus 1 m."""
    return float(sum(layer.thickness_max for layer in parameters.thickness_layers)) + 1.0


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


def _raster(vs: np.ndarray, thicknesses: np.ndarray, dz: float, depth_max: float) -> np.ndarray:
    """Every sampled model (layers x samples) as Vs on a grid of `dz` down to `depth_max`, the
    half-space filling the rest: sigpipe's ensemble model, before its median."""
    rows = int(np.ceil(depth_max / dz))
    tops = np.cumsum(thicknesses, axis=0) if thicknesses.size else np.empty((0, vs.shape[1]))
    grid = (np.arange(rows) + 0.5) * dz
    # For each depth and sample, the layer it falls in: the number of interfaces above it.
    layer = (grid[None, :, None] >= tops.T[:, None, :]).sum(axis=2)
    return np.take_along_axis(vs.T, layer, axis=1)
