"""The measures of an inverted model (PACo's model QC, G5, judges them; PAC measures its own
runs' alike): its forward curve against the pick by band of wavelength (the fit), how its
chains agree and what they hold (the chains), how deep the data inform it and how distinct its
layers are (the model). Each measure says what it describes and what it covers."""

import itertools
import statistics
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from sigpipe.masw.inversion import InversionParameters
from sigpipe.masw.inversion.measuring import (
    BoundShare,
    InversionMeasures,
    ModelFit,
    informed_depth,
)
from sigpipe.masw.quality.measures import Measure, figure

# The fit's bands of wavelength, by name, as many as n_bands: short to long.
BAND_NAMES = {3: ("short", "middle", "long")}


class ModelLimits(BaseModel):
    """How an inverted model is measured, and its limits (PACo's G5 judges against them, PAC
    measures its own runs' alike): provisional, measured on the demo profiles."""

    # The limits of runs saved before are read too.
    model_config = ConfigDict(frozen=True, extra="ignore")

    max_misfit: float = Field(
        default=2.0,
        gt=0,
        description="RMS of the residuals over the curve's uncertainties, in any band, at most: "
        "about 1 is a fit within the errors.",
    )
    n_bands: int = Field(default=3, ge=1, description="Bands of wavelength the fit is judged in.")
    max_rhat: float = Field(
        default=1.1,
        gt=1,
        description="Split R-hat of the models' Vs at any depth watched, at most: the chains "
        "agree.",
    )
    min_samples_per_chain: int = Field(
        default=100, ge=2, description="Models each chain keeps after the burn-in, at least."
    )
    acceptance_band: tuple[float, float] = Field(
        default=(20.0, 30.0),
        description="%: when the data choose the layers, the chains' median acceptance outside "
        "it is a warning, never a failure (their steps adapt in the burn-in, towards 30 % a "
        "move); the layers given (DREAM, near 5 % by design) report it only.",
    )
    min_ess: float = Field(
        default=200.0,
        gt=0,
        description="Effective samples of the models' Vs at any depth watched, over the chains, "
        "at least: a median and 5 to 95 % band to a few percent (a Vs that jumps between two "
        "values where an interface may be above or below holds fewer than a layer's value).",
    )
    bound_edge: float = Field(
        default=0.02,
        gt=0,
        lt=0.5,
        description="The edge of a prior's range watched at each bound, as a share of the range.",
    )
    max_at_bound: float = Field(
        default=0.1,
        gt=0,
        description="Share of a parameter's samples within the edge of a bound, at most: 5 times "
        "what a flat posterior puts there.",
    )
    useful_uncertainty: float = Field(
        default=0.25,
        gt=0,
        description="The depth informed ends where the kept models' relative uncertainty of Vs, "
        "U(z) = (P90 - P10) / (2 P50), gets above this, from the surface down (sigpipe's "
        "useful_depth).",
    )
    min_useful_share: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description="The depth informed, at least this share of the depth the half-space's top "
        "may reach: shallower, the model is shrunk to what the data inform.",
    )
    min_contrast: float = Field(
        default=0.05,
        ge=0,
        description="Adjacent layers of the layered median whose Vs differ less, relative to "
        "their mean, are one: a model that fits loses a layer.",
    )


@dataclass(frozen=True)
class ModelReport:
    """An inverted model's measures, and what its checks act on."""

    measures: tuple[Measure, ...]
    rhat: float | None  # the worst over the series watched
    ess: float | None  # the fewest over them
    acceptance: float | None  # the chains' median, %
    agree: bool  # the chains agree, with enough samples each
    converged: bool  # ...and enough independent samples
    depth: float  # the deepest the half-space's top may be, m
    useful: float | None  # the depth informed, m (None: all of it)
    informed: bool  # at least min_useful_share of `depth`
    contrast: tuple[float, int] | None  # the least between adjacent layers, the upper's index
    piled: BoundShare | None  # the parameter most at a bound


def measure_model(
    measures: InversionMeasures, limits: ModelLimits, depth: float, free: bool
) -> ModelReport:
    """A window's inversion (`measures`; `depth`, the deepest its half-space's top may be,
    model_depth; `free`, the data chose its layers) measured against `limits`, each measure
    saying what it covers: the monitored model's fit by band and the layered median's (the
    fit), the chains' agreement, samples, acceptance and bounds (the chains), the depth informed
    and the layers' contrast (the model). One definition for PACo's G5 and PAC's views."""
    monitored, layered = measures.fits[0], measures.fits[1]
    names = BAND_NAMES.get(
        len(monitored.bands), tuple(f"band{i + 1}" for i in range(len(monitored.bands)))
    )
    result = [
        Measure(
            name=f"misfit_{name}",
            value=band.misfit,
            threshold=limits.max_misfit,
            bound="max",
            # A band with no point to weigh (no mode, no uncertainty) is not measured: a picked
            # point without a mode is no_mode's to judge.
            passed=band.misfit is None or band.misfit <= limits.max_misfit,
            of="fit",
            over=_band_over(band.n_points, band.wavelength_m, monitored.model),
        )
        for name, band in zip(names, monitored.bands, strict=True)
    ]
    # PAC's residual, (modelled - picked) / modelled in %, by band: reported, never judged (the
    # misfit divides by the Lorentzian uncertainties).
    result += [
        Measure(
            name=f"residual_{name}",
            value=None if band.residual is None else round(100 * band.residual, 1),
            passed=True,
            unit="%",
            of="fit",
            over=_band_over(band.n_points, band.wavelength_m, monitored.model)
            + ": (modelled - picked) / modelled, median",
        )
        for name, band in zip(names, monitored.bands, strict=True)
    ]
    result.append(
        Measure(
            name="misfit_layered",
            value=layered.misfit,
            threshold=limits.max_misfit,
            bound="max",
            passed=fits_within(layered, limits.max_misfit),
            of="fit",
            over=f"the layered median's curve against every picked point, "
            f"{layered.n_missing} without its mode",
        )
    )
    judged = judged_series(measures)
    rhats = [value for name, value in measures.rhat.items() if value is not None and name in judged]
    rhat = max(rhats) if rhats else None
    esses = [value for name, value in measures.ess.items() if value is not None and name in judged]
    ess = min(esses) if esses else None
    correlations = [
        value
        for name, value in measures.autocorrelation.items()
        if value is not None and name in judged
    ]
    piled = measures.at_bounds[0] if measures.at_bounds else None
    acceptance = round(statistics.median(measures.acceptance), 2) if measures.acceptance else None
    # The chains agree on one posterior: what it says of the data can be judged. Converged,
    # they also hold enough independent samples for its uncertainties.
    agree = (
        rhat is not None
        and rhat <= limits.max_rhat
        and measures.samples_per_chain >= limits.min_samples_per_chain
    )
    converged = agree and (ess is None or ess >= limits.min_ess)
    useful = measures.useful_depth_m
    informed = useful is None or useful >= limits.min_useful_share * depth
    contrast = least_contrast(measures.vs_layers)
    chains = len(measures.acceptance)
    watched = f"the models' Vs at the {len(judged)} depths watched, over the {chains} chains"
    result += [
        Measure(
            name="rhat",
            value=rhat,
            threshold=limits.max_rhat,
            bound="max",
            passed=rhat is not None and rhat <= limits.max_rhat,
            of="chains",
            over=f"{watched}: the worst",
        ),
        Measure(
            name="ess",
            value=ess,
            threshold=limits.min_ess,
            bound="min",
            passed=ess is None or ess >= limits.min_ess,
            of="chains",
            over=f"{watched}: the fewest",
        ),
        Measure(
            name="autocorrelation",
            value=max(correlations) if correlations else None,
            passed=True,
            of="chains",
            over=f"{watched}: lag 1, the most",
        ),
        *_acceptance(acceptance, limits.acceptance_band if free else None, chains),
        Measure(
            name="samples_per_chain",
            value=measures.samples_per_chain,
            threshold=limits.min_samples_per_chain,
            bound="min",
            passed=measures.samples_per_chain >= limits.min_samples_per_chain,
            of="chains",
            over=f"each of the {chains} chains, after its burn-in",
        ),
        Measure(
            name="at_bound",
            value=piled.share if piled else None,
            threshold=limits.max_at_bound,
            bound="max",
            passed=piled is None or piled.share <= limits.max_at_bound,
            of="chains",
            over=f"each parameter's samples within {limits.bound_edge:.0%} of a prior's bound"
            + (f": the most, {piled.parameter} at its {piled.bound}" if piled else ""),
        ),
    ]
    if informed_depth(measures) is not None:
        # When the data choose the layers, the deepest allowed is the curve's reach whatever the
        # data inform: reported, the model's depth is not shrunk.
        result.append(
            Measure(
                name="depth_informed",
                value=useful,
                threshold=round(limits.min_useful_share * depth, 2),
                bound="min",
                passed=informed or free,
                unit="m",
                of="model",
                over=f"from the surface down, where the models' Vs spread U stays under "
                f"{limits.useful_uncertainty:.0%}; of the {figure(depth)} m the half-space's top "
                "may reach",
            )
        )
    result.append(
        Measure(
            name="contrast",
            value=None if contrast is None else round(100 * contrast[0], 1),
            threshold=round(100 * limits.min_contrast, 1),
            bound="min",
            passed=contrast is None or contrast[0] >= limits.min_contrast,
            unit="%",
            of="model",
            over=f"the layered median's {len(measures.vs_layers)} layers: adjacent Vs over their "
            "mean, the least",
        )
    )
    return ModelReport(
        tuple(result),
        rhat=rhat,
        ess=ess,
        acceptance=acceptance,
        agree=agree,
        converged=converged,
        depth=depth,
        useful=useful,
        informed=informed,
        contrast=contrast,
        piled=piled,
    )


def model_depth(parameters: InversionParameters) -> float:
    """The deepest the half-space's top may be: every layer given at its thickest; the deepest
    interface allowed when the data choose the layers."""
    if parameters.layering == "free":
        return round(parameters.free.depth_max or 0.0, 2)
    return round(sum(layer.thickness_max for layer in parameters.thickness_layers), 2)


def judged_series(measures: InversionMeasures) -> set[str]:
    """The series the chains are judged on: the models' Vs at the depths watched; every one for
    windows measured before."""
    return set(measures.watched) or set(measures.rhat)


def least_contrast(vs_layers: tuple[float, ...]) -> tuple[float, int] | None:
    """The smallest difference of Vs between adjacent layers, relative to their mean, and the
    upper layer's index; None with fewer than two layers."""
    pairs = [
        (abs(below - above) / ((above + below) / 2), index)
        for index, (above, below) in enumerate(itertools.pairwise(vs_layers))
        if above + below > 0
    ]
    return min(pairs) if pairs else None


def fits_within(fit: ModelFit, max_misfit: float) -> bool:
    """Whether a model fits every point within `max_misfit`: it has a mode at each, and no band
    misfits beyond it."""
    return (
        fit.n_missing == 0
        and fit.misfit is not None
        and all(band.misfit is not None and band.misfit <= max_misfit for band in fit.bands)
    )


def _band_over(n_points: int, wavelengths: tuple[float, float], model: str) -> str:
    return (
        f"the {n_points} picked points at {figure(wavelengths[0])}-{figure(wavelengths[1])} m of "
        f"wavelength, against the {model.replace('_', ' ')} model's curve"
    )


def _acceptance(
    value: float | None, band: tuple[float, float] | None, chains: int
) -> tuple[Measure, ...]:
    """The chains' median acceptance (%): with a band, a floor and a ceiling (one row of two, a
    warning outside it, the user 2026-09-29), else reported."""
    over = f"the {chains} chains' moves accepted, median"
    if band is None:
        return (
            Measure(name="acceptance", value=value, passed=True, unit="%", of="chains", over=over),
        )
    low, high = band
    return (
        Measure(
            name="acceptance",
            value=value,
            threshold=low,
            bound="min",
            passed=value is None or value >= low,
            unit="%",
            of="chains",
            over=over,
        ),
        Measure(
            name="acceptance",
            value=value,
            threshold=high,
            bound="max",
            passed=value is None or value <= high,
            unit="%",
            of="chains",
            over=over,
        ),
    )
