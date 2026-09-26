"""An inversion's parameters derived from the curve it inverts (the checks before S4 of PACo's QC
workflow, its docs/qc_workflow.md), or checked against it. Bounds come from the window's own curve (a decision of milestone 13): Vs brackets
the curve's velocities with a margin (Vs is about 1.09 Vr at PAC's Vp/Vs), no layer is thinner
than the shortest wavelength resolves, and the half-space starts no deeper than the longest one
reaches. Values given by the user or the loop are kept when they pass, and changed with a note
when they do not."""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sigpipe.algorithms.inversion.rayleigh.seismic.parameters import InversionParameters
from sigpipe.base.dispersion_curve import DispersionCurve

# PAC's steps against its ranges: 20 m/s for Vs over 100-1,000 m/s, 1 m for thicknesses over
# 1-10 m. Derived bounds keep the same proportions.
VS_STEP_SHARE = 20 / 900
THICKNESS_STEP_SHARE = 1 / 9
# Vs over Vr for a homogeneous half-space at Vp/Vs 1.77: the least a bound must allow above the
# curve's fastest point.
VS_OVER_VR = 1.09


# The fewest layers an inversion has, the half-space among them (the user, 2026-09-25).
MIN_LAYERS = 3


class InversionError(ValueError):
    """An inversion cannot start. Messages say what to fix, for a person in PAC or the agent in
    PACo."""


class PriorRules(BaseModel):
    """The rules the bounds follow: provisional, measured on the demo profiles (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vs_low: float = Field(
        default=0.8, gt=0, description="Lowest Vs, as a share of the curve's slowest velocity."
    )
    vs_high: float = Field(
        default=1.5, gt=0, description="Highest Vs, as a multiple of the curve's fastest velocity."
    )
    min_thickness: float = Field(
        default=1 / 3,
        gt=0,
        description="Thinnest layer, as a share of the shortest wavelength: thinner is not "
        "resolved.",
    )
    max_depth: float = Field(
        default=0.5,
        gt=0,
        description="Deepest top of the half-space, as a share of the longest wavelength: deeper "
        "is not resolved.",
    )
    n_layers: int = Field(
        default=4,
        ge=MIN_LAYERS,
        description="Layers to start with, the half-space among them: never fewer than 3 (the "
        "user, 2026-09-25); G5 adds layers up to what the curve resolves.",
    )


@dataclass(frozen=True)
class Derived:
    """The parameters an inversion runs with, and what the checks changed in the values given."""

    parameters: InversionParameters
    notes: tuple[str, ...]
    reach_m: float  # the deepest the half-space's top may be: the curve's reach
    max_layers: int | None = None  # the most layers the curve resolves


def derive_inversion(
    curve: DispersionCurve, rules: PriorRules, given: Mapping[str, Any] | None = None
) -> Derived:
    """The parameters to invert `curve` with: `given` (InversionParameters' fields, from the
    user or the loop) where they pass the checks, the rest derived from the curve."""
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
        # fitted to the curve, as every derived bound is.
        if "n_layers" in given:
            notes.append(
                f"n_layers {n_layers}: the curve resolves {resolved} (layers of at least "
                f"{thinnest:.2f} m down to {deepest:.2f} m); set to {resolved}."
            )
        n_layers = resolved
        _recount(given, n_layers, notes)

    low, high = round(rules.vs_low * vr_min), round(rules.vs_high * vr_max)
    derived_vs = {
        "vs_min": float(low),
        "vs_max": float(high),
        "vs_perturb_std": round((high - low) * VS_STEP_SHARE, 1),
    }
    vs_layers = [dict(derived_vs) for _ in range(n_layers)]
    if "vs_layers" in given:
        vs_layers = [{**derived_vs, **dict(layer)} for layer in given["vs_layers"]]
        slow = [i for i, layer in enumerate(vs_layers) if layer["vs_min"] > vr_min]
        fast = [i for i, layer in enumerate(vs_layers) if layer["vs_max"] < VS_OVER_VR * vr_max]
        # The values given, said in the notes: the user reads what they typed was changed.
        slow_given = _values(vs_layers[i]["vs_min"] for i in slow)
        fast_given = _values(vs_layers[i]["vs_max"] for i in fast)
        for index in slow:
            vs_layers[index]["vs_min"] = derived_vs["vs_min"]
        for index in fast:
            vs_layers[index]["vs_max"] = derived_vs["vs_max"]
        if slow:
            notes.append(
                f"vs_min {slow_given} m/s above the curve's slowest velocity ({vr_min:.0f} m/s) "
                f"in {_layers(slow)}: set to {low} m/s."
            )
        if fast:
            notes.append(
                f"vs_max {fast_given} m/s below {VS_OVER_VR} times the curve's fastest velocity "
                f"({vr_max:.0f} m/s) in {_layers(fast)}: set to {high} m/s."
            )

    bottom = round(deepest / (n_layers - 1), 2)
    derived_thickness = {
        "thickness_min": top,
        "thickness_max": bottom,
        # At least 1 cm: with as many layers as the curve resolves, each may range over a few
        # centimetres only (6 layers on the demo's xmid 16.50), and a step rounded to 0 fails.
        "thickness_perturb_std": max(round((bottom - top) * THICKNESS_STEP_SHARE, 2), 0.01),
    }
    thickness_layers = [dict(derived_thickness) for _ in range(n_layers - 1)]
    if "thickness_layers" in given:
        thickness_layers = [
            {**derived_thickness, **dict(layer)} for layer in given["thickness_layers"]
        ]
        thin = [i for i, layer in enumerate(thickness_layers) if layer["thickness_min"] < top]
        thin_given = _values(thickness_layers[i]["thickness_min"] for i in thin)
        for index in thin:
            thickness_layers[index]["thickness_min"] = top
        if thin:
            notes.append(
                f"thickness_min {thin_given} m thinner than the curve resolves ({top:g} m) in "
                f"{_layers(thin)}: set to {top:g} m."
            )
        total = sum(float(layer["thickness_max"]) for layer in thickness_layers)
        if round(total, 2) > round(deepest, 2):
            scale = deepest / total
            for layer in thickness_layers:
                layer["thickness_max"] = round(float(layer["thickness_max"]) * scale, 2)
            notes.append(
                f"thickness_max puts the half-space as deep as {total:g} m, below the "
                f"{deepest:.2f} m the curve reaches: scaled by {scale:.2f}."
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


def broadcast_layers(given: Mapping[str, Any], default_layers: int = 4) -> dict[str, Any]:
    """`given` with a single Vs range (or thickness range) standing for every layer: "Vs between
    100 and 180 m/s" is one range, where the parameters want one per layer (Qwen3-8B sent one,
    and needed six calls to find the form). Several Vs ranges given without n_layers are that
    many layers."""
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
    is given. Checked as given, the one range the card offers was refused against PAC's default
    of 2 layers (Qwen3-8B then invented two ranges, 2026-09-26)."""
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
    same for every layer stays every layer's; ranges that differ from layer to layer are derived
    again from the curve, said in a note."""
    for key, count in (("vs_layers", n_layers), ("thickness_layers", n_layers - 1)):
        layers = given.get(key)
        if not isinstance(layers, list | tuple) or not layers:
            continue
        ranges = cast(Sequence[Any], layers)
        if len(ranges) == 1 or _alike(ranges):
            given[key] = [ranges[0]] * count
        else:
            del given[key]
            notes.append(f"{key}: given layer by layer, derived again for {n_layers} layers.")


def _values(values: Iterable[float]) -> str:
    """The distinct values given, as the notes show them: "180", or "150, 180"."""
    return ", ".join(f"{value:g}" for value in sorted({float(value) for value in values}))


def _layers(indices: list[int]) -> str:
    """Layers named from the top, from 1, as the notes give them."""
    names = [str(index + 1) for index in indices]
    return f"layer {names[0]}" if len(names) == 1 else f"layers {', '.join(names)}"
