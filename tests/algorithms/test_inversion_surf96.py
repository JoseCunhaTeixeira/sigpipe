"""The forward model (sigpipe-core's surf96): reference curves of layered models, Rayleigh and
Love waves (surf96_reference.json); every Rayleigh velocity found a root of its period equation
(the Love waves' equation, a product of normalized propagators, may also change sign across a
pole, which the search takes for a root, as surf96 does); a half-space's Rayleigh wave at its
known velocity; and the curves sigpipe draws from it."""

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sigpipe_core

from sigpipe.algorithms.inversion.rayleigh.seismic.forward import fwd_seismic_phase

REFERENCE: dict[str, Any] = json.loads(
    (Path(__file__).parent / "surf96_reference.json").read_text(encoding="utf-8")
)
TOLERANCE = sigpipe_core.TOLERANCE
PERIODS = np.sort(1 / np.geomspace(4.0, 80.0, 30))  # s, increasing

type Model = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _layers(
    thickness_m: Sequence[float] | np.ndarray, vs: Sequence[float] | np.ndarray, vp_vs: float = 1.77
) -> Model:
    """A layered model in km, km/s and g/cm3, Vp following Vs, Gardner's densities."""
    s = np.asarray(vs, dtype=float)
    vp = s * vp_vs
    rho = 0.32 * vp + 770.0
    return np.asarray(thickness_m, dtype=float) / 1_000, vp / 1_000, s / 1_000, rho / 1_000


def _random(rng: np.random.Generator, layers: int) -> Model:
    """A layered model as the samplers may propose one: interfaces down to 40 m, Vs of 80 to
    1500 m/s, each layer down to half or up to two and a half times as fast as the one above
    it, Vp/Vs of 1.6 to 3."""
    depths = np.sort(rng.uniform(0.3, 40.0, layers - 1))
    thickness = np.append(np.diff(np.concatenate(([0.0], depths))), 1_000.0)
    vs = np.empty(layers)
    vs[0] = rng.uniform(80.0, 600.0)
    for i in range(1, layers):
        vs[i] = min(vs[i - 1] * rng.uniform(0.5, 2.5), 1_500.0)
    return _layers(thickness, vs, rng.uniform(1.6, 3.0))


@pytest.mark.parametrize("entry", REFERENCE["models"], ids=lambda entry: entry["name"])
def test_the_curves_are_the_reference_curves(entry: dict[str, Any]) -> None:
    model = _layers(entry["thickness_m"], entry["vs"], REFERENCE["vp_vs"])
    periods = np.array(REFERENCE["periods"])
    for curve in entry["curves"]:
        found, velocity = sigpipe_core.phase_dispersion(
            periods, *model, mode=curve["mode"], wave=curve["wave"]
        )
        assert found.tolist() == curve["periods"]
        np.testing.assert_allclose(velocity, curve["velocity_km_s"], rtol=2 * TOLERANCE, atol=0.0)


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_every_rayleigh_velocity_found_is_a_root_of_its_period_equation(mode: int) -> None:
    rng = np.random.default_rng(mode)
    for layers in range(1, 9):
        for _ in range(15):
            model = _random(rng, layers)
            try:
                found, velocity = sigpipe_core.phase_dispersion(PERIODS, *model, mode)
            except sigpipe_core.DispersionError:
                continue
            for period, c in zip(found, velocity, strict=True):
                below, above = (
                    sigpipe_core.period_equation(
                        float(period), float(c) * (1 + side * 4 * TOLERANCE), *model
                    )
                    for side in (-1, 1)
                )
                assert np.sign(below) != np.sign(above), (mode, layers, period, c)


def test_a_half_space_rayleigh_wave_travels_at_its_known_velocity_at_every_period() -> None:
    # A Poisson solid, Vp = sqrt(3) Vs: its Rayleigh wave travels at sqrt(2 - 2 / sqrt(3)) of its
    # Vs, whatever the period.
    model = _layers([1_000.0], [300.0], math.sqrt(3))
    found, velocity = sigpipe_core.phase_dispersion(PERIODS, *model)

    assert found.size == PERIODS.size
    np.testing.assert_allclose(
        velocity, math.sqrt(2 - 2 / math.sqrt(3)) * 0.3, rtol=2 * TOLERANCE, atol=0.0
    )


def test_the_periods_must_increase_and_the_fundamental_mode_exist() -> None:
    layered = _layers([4.0, 1_000.0], [200.0, 450.0])
    with pytest.raises(ValueError, match="sorted"):
        sigpipe_core.phase_dispersion(np.ascontiguousarray(PERIODS[::-1]), *layered)
    # A Love wave needs a slower layer over a faster one: a half-space carries none.
    with pytest.raises(sigpipe_core.DispersionError):
        sigpipe_core.phase_dispersion(PERIODS, *_layers([1_000.0], [300.0]), wave="love")


def test_the_drawn_curve_is_the_reference_curve() -> None:
    entry = next(entry for entry in REFERENCE["models"] if entry["name"] == "three layers")
    reference = next(
        curve for curve in entry["curves"] if curve["wave"] == "rayleigh" and curve["mode"] == 0
    )
    fs = 1 / np.array(reference["periods"])[::-1]

    drawn = fwd_seismic_phase(entry["thickness_m"], entry["vs"], 0, fs, REFERENCE["vp_vs"])

    expected = np.array(reference["velocity_km_s"])[::-1] * 1_000
    np.testing.assert_allclose(drawn.vs, expected, rtol=2 * TOLERANCE, atol=0.0)
