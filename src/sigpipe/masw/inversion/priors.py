"""An inversion's parameters derived from the curve it inverts (the checks before S4 of PACo's QC
workflow, its docs/qc_workflow.md), or checked against it.

The layers chosen by the data (the free layering, the rules' default): Vs from 100 to 2,000 m/s,
widened where the curve needs it (as slow as 0.8 of its slowest velocity, as fast as 1.5 of its
fastest), interfaces from a third of its shortest wavelength (thinner is not resolved) down to
half its longest (deeper is not resolved), at most `max_layers` layers. Values given are kept
when they pass, changed with a note when they do not.

The layers given (the fixed layering). The first inversion starts wide, for
the loop to narrow to what the data inform: Vs 100 to 1,000 m/s for the layers and to 2,000 m/s
for the half-space, widened when the curve needs it (the top layer as slow as the curve's
slowest velocity, the half-space as fast as its fastest: Vs is about 1.09 Vr at the inversion's
Vp/Vs), and every layer 1 to 10 m thick. The curve sets how many layers it resolves (each at
least a third of its shortest wavelength, down to half its longest). Values given by the user
or the loop are kept when they pass, and changed with a note when they do not: no layer thinner
than the first range's or the curve's thinnest, whichever is thinner, and the half-space no
deeper than the longest wavelength reaches.

Values locked (PACo's: those the user gave for a run, which nothing changes) stay as given even
where a check fails, the note saying what the check asks; the fewest layers are kept still."""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sigpipe.algorithms.inversion.rayleigh.seismic.parameters import InversionParameters
from sigpipe.base.dispersion_curve import DispersionCurve

# The default steps against the default ranges: 20 m/s for Vs over 100-1,000 m/s, 1 m for
# thicknesses over 1-10 m. Derived bounds keep the same proportions.
VS_STEP_SHARE = 20 / 900
THICKNESS_STEP_SHARE = 1 / 9
# Vs over Vr for a homogeneous half-space at Vp/Vs 1.77: the least a bound must allow above the
# curve's fastest point.
VS_OVER_VR = 1.09


# The fewest layers an inversion has, the half-space among them.
MIN_LAYERS = 3
# The keys of the fixed layering: given, they mean it.
FIXED_KEYS = ("n_layers", "vs_layers", "thickness_layers")
# A note's verbs for a scale: made, and asked.
SCALES = ("scaled by", "scales by")


class InversionError(ValueError):
    """An inversion cannot start. Messages say what to fix, for a person in PAC or the agent in
    PACo."""


class PriorRules(BaseModel):
    """The rules the bounds follow: provisional, measured on the demo profiles (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vs_min: float = Field(
        default=100.0,
        gt=0,
        description="Lowest Vs of every layer, m/s; for a curve slower than that, vs_low of its "
        "slowest velocity.",
    )
    vs_max: float = Field(
        default=1_000.0,
        gt=0,
        description="Highest Vs of the layers above the half-space, m/s; for a curve too slow at "
        "its slowest for that, vs_high of its slowest velocity.",
    )
    half_space_vs_max: float = Field(
        default=2_000.0,
        gt=0,
        description="Highest Vs of the half-space, m/s; for a curve too fast at its fastest for "
        "that, vs_high of its fastest velocity.",
    )
    vs_low: float = Field(
        default=0.8,
        gt=0,
        description="Lowest Vs where the curve needs it, as a share of its slowest velocity.",
    )
    vs_high: float = Field(
        default=1.5,
        gt=0,
        description="Highest Vs where the curve needs it, as a multiple of its velocity there.",
    )
    min_thickness: float = Field(
        default=1 / 3,
        gt=0,
        description="Thinnest layer, as a share of the shortest wavelength: thinner is not "
        "resolved.",
    )
    thickness_min_m: float = Field(
        default=1.0,
        gt=0,
        description="Thinnest layer the first inversion allows, m: wide, the loop narrows it.",
    )
    thickness_max_m: float = Field(
        default=10.0,
        gt=0,
        description="Thickest layer the first inversion allows, m: wide, the loop narrows it.",
    )
    max_depth: float = Field(
        default=0.5,
        gt=0,
        description="Deepest top of the half-space, as a share of the longest wavelength: deeper "
        "is not resolved.",
    )
    layering: Literal["free", "fixed"] = Field(
        default="free",
        description="free: the data choose the layers, up to max_layers; fixed: n_layers given, "
        "G5 adding or removing one where the model asks.",
    )
    max_layers: int = Field(
        default=8,
        ge=1,
        le=20,
        description="The free layering's most layers, the half-space among them; G5 allows more "
        "where the models pile at it.",
    )
    n_layers: int = Field(
        default=4,
        ge=MIN_LAYERS,
        description="The fixed layering's layers to start with, the half-space among them: never "
        "fewer than 3; G5 adds layers up to what the curve resolves.",
    )


@dataclass(frozen=True)
class Derived:
    """The parameters an inversion runs with, and what the checks changed in the values given."""

    parameters: InversionParameters
    notes: tuple[str, ...]
    reach_m: float  # the deepest the half-space's top may be: the curve's reach
    max_layers: int | None = None  # the most layers the curve resolves


def derive_inversion(
    curve: DispersionCurve,
    rules: PriorRules,
    given: Mapping[str, Any] | None = None,
    locked: Mapping[str, Any] | None = None,
) -> Derived:
    """The parameters to invert `curve` with: `given` (InversionParameters' fields, from the
    user or the loop) where they pass the checks, the rest derived from the curve. The layers
    chosen by the data unless given (or the fixed layering named, or the rules' own). Those of
    `locked` (fields of `given`) stay as given where a check fails, with a note."""
    locked = locked or {}
    if layering_of(given or {}, rules) == "free":
        return _derive_free(curve, rules, given or {}, locked)
    given = broadcast_layers(given or {}, rules.n_layers)
    velocities = np.asarray(curve.vs, dtype=float)
    wavelengths = velocities / np.asarray(curve.fs, dtype=float)
    vr_min, vr_max = float(velocities.min()), float(velocities.max())
    thinnest = rules.min_thickness * float(wavelengths.min())
    deepest = rules.max_depth * float(wavelengths.max())
    notes: list[str] = []

    n_layers = int(given.get("n_layers", rules.n_layers))
    if n_layers < MIN_LAYERS:
        notes.append(f"n_layers {n_layers}: never fewer than {MIN_LAYERS}; set to {MIN_LAYERS}.")
        n_layers = MIN_LAYERS
        _recount(given, n_layers, notes)
    # Layers above the half-space, each at least `thinnest`, all within `deepest`.
    # Rounded first: 5 m over layers of 5/3 m is 3.0000000000000004, and one layer more would
    # leave every layer a range of zero width.
    resolved = max(MIN_LAYERS, math.ceil(round(deepest / thinnest, 6)))
    # Every layer above the half-space keeps a range as the parameters round it: at the limit
    # its maximum rounds to its minimum (5 layers of 1.67 to 1.67 m).
    top = round(thinnest, 2)
    while resolved > MIN_LAYERS and round(deepest / (resolved - 1), 2) <= top:
        resolved -= 1
    if round(deepest / (MIN_LAYERS - 1), 2) <= top:
        raise InversionError(
            f"The curve's wavelengths ({float(wavelengths.min()):.1f} to "
            f"{float(wavelengths.max()):.1f} m) resolve fewer than {MIN_LAYERS} layers of at "
            f"least {top:g} m: a curve reaching longer wavelengths is needed (longer windows)."
        )
    if n_layers > resolved:
        # A count given (by the user, or the loop) is changed with a note; the default is
        # fitted to the curve, as every derived bound is. A count locked, or the layers locked
        # one by one, stays.
        kept = "n_layers" in locked or any(
            isinstance(locked.get(key), list | tuple) and len(locked[key]) > 1
            for key in ("vs_layers", "thickness_layers")
        )
        if "n_layers" in given:
            notes.append(
                f"n_layers {n_layers}: the curve resolves {resolved} (layers of at least "
                f"{thinnest:.2f} m down to {deepest:.2f} m); {_ending(kept, str(resolved))}"
            )
        if not kept:
            n_layers = resolved
            _recount(given, n_layers, notes)

    # The margins where the curve needs them: a Vs given that does not bracket the curve is set
    # to them, and the wide ranges widen to them.
    floor, ceiling = round(rules.vs_low * vr_min), round(rules.vs_high * vr_max)
    low = rules.vs_min if rules.vs_min <= vr_min else floor
    high = rules.vs_max if VS_OVER_VR * vr_min <= rules.vs_max else round(rules.vs_high * vr_min)
    half_space_high = (
        rules.half_space_vs_max if VS_OVER_VR * vr_max <= rules.half_space_vs_max else ceiling
    )
    derived_vs = [_vs_range(low, high)] * (n_layers - 1) + [_vs_range(low, half_space_high)]
    vs_layers = [dict(layer) for layer in derived_vs]
    if given.get("vs_layers"):
        ranges = list(given["vs_layers"])
        # The last range given is the half-space's: a count that does not match n_layers is
        # the parameters' own error, said below.
        vs_layers = [
            {**derived_vs[-1 if i == len(ranges) - 1 else 0], **dict(layer)}
            for i, layer in enumerate(ranges)
        ]
        first, half_space = vs_layers[0], vs_layers[-1]
        kept = "vs_layers" in locked
        # The values given, said in the notes: the user reads what they typed was changed.
        if first["vs_min"] > vr_min:
            notes.append(
                f"vs_min {first['vs_min']:g} m/s of the top layer above the curve's slowest "
                f"velocity ({vr_min:.0f} m/s): {_ending(kept, f'{floor} m/s')}"
            )
            if not kept:
                first["vs_min"] = float(floor)
        if half_space["vs_max"] < VS_OVER_VR * vr_max:
            notes.append(
                f"vs_max {half_space['vs_max']:g} m/s of the half-space below {VS_OVER_VR} times "
                f"the curve's fastest velocity ({vr_max:.0f} m/s): {_ending(kept, f'{ceiling} m/s')}"
            )
            if not kept:
                half_space["vs_max"] = float(ceiling)

    # The first inversion's range, the same for every layer whatever the curve: wide, for the
    # loop to narrow to what the data inform.
    first_min, first_max = rules.thickness_min_m, rules.thickness_max_m
    derived_thickness = {
        "thickness_min": first_min,
        "thickness_max": first_max,
        "thickness_perturb_std": max(
            round((first_max - first_min) * THICKNESS_STEP_SHARE, 2), 0.01
        ),
    }
    thickness_layers = [dict(derived_thickness) for _ in range(n_layers - 1)]
    if "thickness_layers" in given:
        # Given, a layer is no thinner than the first range's or the curve's thinnest.
        floor = min(top, first_min)
        thickness_layers = [
            {**derived_thickness, **dict(layer)} for layer in given["thickness_layers"]
        ]
        kept = "thickness_layers" in locked
        thin = [i for i, layer in enumerate(thickness_layers) if layer["thickness_min"] < floor]
        thin_given = _values(thickness_layers[i]["thickness_min"] for i in thin)
        for index in thin if not kept else ():
            thickness_layers[index]["thickness_min"] = floor
        if thin:
            notes.append(
                f"thickness_min {thin_given} m thinner than the thinnest allowed ({floor:g} m) "
                f"in {_layers(thin)}: {_ending(kept, f'{floor:g} m')}"
            )
        total = sum(float(layer["thickness_max"]) for layer in thickness_layers)
        if round(total, 2) > round(deepest, 2):
            scale = deepest / total
            for layer in thickness_layers if not kept else ():
                layer["thickness_max"] = round(float(layer["thickness_max"]) * scale, 2)
                # A range scaled under its own minimum keeps a width.
                if layer["thickness_min"] >= layer["thickness_max"]:
                    layer["thickness_min"] = round(min(floor, layer["thickness_max"] / 2), 2)
            notes.append(
                f"thickness_max puts the half-space as deep as {total:g} m, below the "
                f"{deepest:.2f} m the curve reaches: {_ending(kept, f'{scale:.2f}', SCALES)}"
            )

    values = {
        **{
            key: value
            for key, value in given.items()
            if key not in ("vs_layers", "thickness_layers")
        },
        "n_layers": n_layers,
        "vs_layers": vs_layers,
        "thickness_layers": thickness_layers,
    }
    try:
        return Derived(
            InversionParameters.model_validate(values), tuple(notes), round(deepest, 2), resolved
        )
    except ValidationError as error:
        problems = "; ".join(str(problem["msg"]) for problem in error.errors())
        raise InversionError(f"The inversion's parameters do not hold: {problems}") from error


def layering_of(given: Mapping[str, Any], rules: PriorRules) -> str:
    """The layering an inversion runs with: the one named, the fixed one when layers are given,
    else the rules'."""
    named = given.get("layering")
    if isinstance(named, str):
        return named
    return "fixed" if any(key in given for key in FIXED_KEYS) else rules.layering


def _derive_free(
    curve: DispersionCurve,
    rules: PriorRules,
    given: Mapping[str, Any],
    locked: Mapping[str, Any],
) -> Derived:
    """The free layering's bounds from the curve, those `given` kept where they pass, and those
    `locked` kept where they fail, with a note."""
    velocities = np.asarray(curve.vs, dtype=float)
    wavelengths = velocities / np.asarray(curve.fs, dtype=float)
    vr_min, vr_max = float(velocities.min()), float(velocities.max())
    floor, ceiling = round(rules.vs_low * vr_min), round(rules.vs_high * vr_max)
    deepest = round(rules.max_depth * float(wavelengths.max()), 2)
    shallowest = round(rules.min_thickness * float(wavelengths.min()), 2)
    derived = {
        "vs_min": float(min(rules.vs_min, floor)),
        "vs_max": float(max(rules.half_space_vs_max, ceiling)),
        "depth_min": shallowest,
        "depth_max": deepest,
        "max_layers": rules.max_layers,
    }
    asked = {
        key: value
        for key, value in dict(cast(Mapping[str, Any], given.get("free") or {})).items()
        if value is not None
    }
    free = {**derived, **asked}
    kept = dict(cast(Mapping[str, Any], locked.get("free") or {}))
    notes: list[str] = []
    if free["vs_min"] > vr_min:
        notes.append(
            f"free.vs_min {free['vs_min']:g} m/s above the curve's slowest velocity "
            f"({vr_min:.0f} m/s): {_ending('vs_min' in kept, f'{floor} m/s')}"
        )
        if "vs_min" not in kept:
            free["vs_min"] = float(floor)
    if free["vs_max"] < VS_OVER_VR * vr_max:
        notes.append(
            f"free.vs_max {free['vs_max']:g} m/s below {VS_OVER_VR} times the curve's fastest "
            f"velocity ({vr_max:.0f} m/s): {_ending('vs_max' in kept, f'{ceiling} m/s')}"
        )
        if "vs_max" not in kept:
            free["vs_max"] = float(ceiling)
    if free["depth_max"] > deepest:
        notes.append(
            f"free.depth_max {free['depth_max']:g} m below the {deepest:g} m the curve reaches: "
            f"{_ending('depth_max' in kept, f'{deepest:g} m')}"
        )
        if "depth_max" not in kept:
            free["depth_max"] = deepest
    if free["depth_min"] >= free["depth_max"] and "depth_min" not in kept:
        free["depth_min"] = round(free["depth_max"] / 2, 2)
    values = {
        **{key: value for key, value in given.items() if key not in ("free", *FIXED_KEYS)},
        "layering": "free",
        "free": free,
    }
    try:
        return Derived(InversionParameters.model_validate(values), tuple(notes), deepest)
    except ValidationError as error:
        problems = "; ".join(str(problem["msg"]) for problem in error.errors())
        raise InversionError(f"The inversion's parameters do not hold: {problems}") from error


def _vs_range(low: float, high: float) -> dict[str, float]:
    """A layer's Vs bounds, with the step in the defaults' proportion."""
    return {
        "vs_min": float(low),
        "vs_max": float(high),
        "vs_perturb_std": round((high - low) * VS_STEP_SHARE, 1),
    }


def broadcast_layers(given: Mapping[str, Any], default_layers: int = 4) -> dict[str, Any]:
    """`given` with a single Vs range (or thickness range) standing for every layer: "Vs between
    100 and 180 m/s" is one range, where the parameters want one per layer. Several Vs ranges
    given without n_layers are that many layers."""
    values = dict(given)
    vs_layers = values.get("vs_layers")
    if (
        "n_layers" not in values
        and isinstance(vs_layers, list | tuple)
        and len(cast(Sequence[Any], vs_layers)) > 1
    ):
        values["n_layers"] = len(cast(Sequence[Any], vs_layers))
    n_layers = int(values.get("n_layers", default_layers))
    if isinstance(vs_layers, list | tuple) and len(cast(Sequence[Any], vs_layers)) == 1:
        values["vs_layers"] = list(cast(Sequence[Any], vs_layers)) * n_layers
    thicknesses = values.get("thickness_layers")
    if (
        isinstance(thicknesses, list | tuple)
        and len(cast(Sequence[Any], thicknesses)) == 1
        and n_layers > 2
    ):
        values["thickness_layers"] = list(cast(Sequence[Any], thicknesses)) * (n_layers - 1)
    return values


def checkable(given: Mapping[str, Any], default_layers: int) -> dict[str, Any]:
    """`given` as an inversion reads it, complete enough to check before it starts: one Vs (or
    thickness) range standing for every layer, and the count the inversion starts from when none
    is given. Checked as given, the one range the card offers would be refused against the
    default of 2 layers. The free layering is checked as given."""
    if given.get("layering") == "free" or (
        "layering" not in given and not any(key in given for key in FIXED_KEYS)
    ):
        return dict(given)
    values = broadcast_layers(given, default_layers)
    count = values.get("n_layers", default_layers)
    if isinstance(count, int) and count >= 2:
        values.setdefault("n_layers", count)
        values.setdefault("vs_layers", [{}] * count)
        values.setdefault("thickness_layers", [{}] * (count - 1))
    return values


def _alike(layers: Sequence[Any]) -> bool:
    """Whether several ranges given are one range repeated."""
    return len(layers) > 1 and all(layer == layers[0] for layer in layers)


def _recount(given: dict[str, Any], n_layers: int, notes: list[str]) -> None:
    """The ranges `given` per layer, for `n_layers` layers once the count changed: a range the
    same for every layer stays every layer's, and so does a Vs range the same for every layer
    above the half-space, the half-space keeping its own; ranges that differ from layer to layer
    are derived again from the curve, said in a note."""
    for key, count in (("vs_layers", n_layers), ("thickness_layers", n_layers - 1)):
        layers = given.get(key)
        if not isinstance(layers, list | tuple) or not layers:
            continue
        ranges = cast(Sequence[Any], layers)
        if len(ranges) == 1 or _alike(ranges):
            given[key] = [ranges[0]] * count
        elif key == "vs_layers" and (len(ranges) == 2 or _alike(ranges[:-1])):
            given[key] = [ranges[0]] * (count - 1) + [ranges[-1]]
        else:
            del given[key]
            notes.append(f"{key}: given layer by layer, derived again for {n_layers} layers.")


def _ending(kept: bool, value: str, verbs: tuple[str, str] = ("set to", "sets")) -> str:
    """How a note on a value that fails a check ends: changed to `value`, or kept as given (it is
    locked), saying what the check asks."""
    made, asks = verbs
    return f"kept as given (the check {asks} {value})." if kept else f"{made} {value}."


def _values(values: Iterable[float]) -> str:
    """The distinct values given, as the notes show them: "180", or "150, 180"."""
    return ", ".join(f"{value:g}" for value in sorted({float(value) for value in values}))


def _layers(indices: list[int]) -> str:
    """Layers named from the top, from 1, as the notes give them."""
    names = [str(index + 1) for index in indices]
    return f"layer {names[0]}" if len(names) == 1 else f"layers {', '.join(names)}"
