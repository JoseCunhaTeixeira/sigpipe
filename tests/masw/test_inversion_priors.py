"""The checks before S4: bounds derived from the curve, values given kept when they pass and
changed with a note when they do not."""

import numpy as np
import pytest

from sigpipe.base import DispersionCurve, Mode, VelocityType
from sigpipe.base.acquisition import UNKNOWN_ACQUISITION
from sigpipe.masw.inversion import InversionError
from sigpipe.masw.inversion.priors import PriorRules, broadcast_layers, derive_inversion

RULES = PriorRules()
# 150 m/s at 30 m, rising to 300 m/s at 3 m: wavelengths 3 to 30 m, velocities 150 to 300 m/s.
WAVELENGTHS = np.array([30.0, 15.0, 9.0, 6.0, 3.0])
VELOCITIES = np.array([150.0, 180.0, 220.0, 260.0, 300.0])
CURVE = DispersionCurve(
    fs=VELOCITIES / WAVELENGTHS,
    vs=VELOCITIES,
    mode=Mode("M", 0),
    type=VelocityType.PHASE,
    acquisition=UNKNOWN_ACQUISITION,
)


def test_the_bounds_come_from_the_curve() -> None:
    derived = derive_inversion(CURVE, RULES)

    parameters = derived.parameters
    assert derived.notes == ()
    assert parameters.n_layers == 4  # never 2
    # Vs wide: 100 to 1,000 m/s for the layers, to 2,000 m/s for the half-space, steps in the
    # defaults' ratio.
    *layers, half_space = parameters.vs_layers
    for layer in layers:
        assert (layer.vs_min, layer.vs_max, layer.vs_perturb_std) == (100.0, 1000.0, 20.0)
    assert (half_space.vs_min, half_space.vs_max) == (100.0, 2000.0)
    assert half_space.vs_perturb_std == pytest.approx(1900 * 20 / 900, abs=0.05)
    # No layer thinner than a third of 3 m; the half-space no deeper than half of 30 m, shared
    # by the three layers above it.
    assert [
        (layer.thickness_min, layer.thickness_max) for layer in parameters.thickness_layers
    ] == [(1.0, 5.0)] * 3
    # The default effort, with its burn-in.
    assert (parameters.n_iterations, parameters.n_burnin_iterations, parameters.n_chains) == (
        100_000,
        10_000,
        5,
    )


def test_more_layers_share_the_depth_the_curve_reaches() -> None:
    parameters = derive_inversion(CURVE, RULES, {"n_layers": 6}).parameters

    assert parameters.n_layers == 6
    assert [layer.thickness_max for layer in parameters.thickness_layers] == [3.0] * 5


def test_never_fewer_than_three_layers() -> None:
    derived = derive_inversion(CURVE, RULES, {"n_layers": 2})

    assert derived.parameters.n_layers == 3
    assert derived.notes == ("n_layers 2: never fewer than 3; set to 3.",)
    # Two different Vs ranges are two layers, the top one's and the half-space's: raised the same
    # way, the layer added takes the range of the layers above the half-space.
    two = {"vs_layers": [{"vs_min": 100.0, "vs_max": 800.0}, {"vs_min": 120.0, "vs_max": 900.0}]}
    raised = derive_inversion(CURVE, RULES, two)
    assert raised.parameters.n_layers == 3
    assert raised.notes == ("n_layers 2: never fewer than 3; set to 3.",)
    assert [(layer.vs_min, layer.vs_max) for layer in raised.parameters.vs_layers] == [
        (100.0, 800.0),
        (100.0, 800.0),
        (120.0, 900.0),
    ]
    # Layers above the half-space with ranges of their own: derived again, with a note.
    three = {
        "n_layers": 30,
        "vs_layers": [{"vs_min": 100.0, "vs_max": 300.0 + 100 * i} for i in range(3)],
    }
    assert derive_inversion(CURVE, RULES, three).notes[-1] == (
        "vs_layers: given layer by layer, derived again for 15 layers."
    )
    # The same range for both stays every layer's: "Vs between 100 and 180 m/s" sent for two
    # layers keeps the user's bounds, and the half-space's 180 m/s is changed with a note.
    same = derive_inversion(CURVE, RULES, {"vs_layers": [{"vs_min": 100.0, "vs_max": 180.0}] * 2})
    assert same.parameters.n_layers == 3
    assert [(layer.vs_min, layer.vs_max) for layer in same.parameters.vs_layers] == [
        (100.0, 180.0),
        (100.0, 180.0),
        (100.0, 450.0),
    ]
    assert same.notes == (
        "n_layers 2: never fewer than 3; set to 3.",
        "vs_max 180 m/s of the half-space below 1.09 times the curve's fastest velocity (300 "
        "m/s): set to 450 m/s.",
    )


def test_values_given_are_kept_when_they_pass() -> None:
    given = {
        "vs_layers": [{"vs_min": 100.0, "vs_max": 800.0}] * 3,
        "thickness_layers": [{"thickness_min": 2.0, "thickness_max": 7.0}] * 2,
        "n_iterations": 20_000,
    }
    derived = derive_inversion(CURVE, RULES, given)

    assert derived.notes == ()
    parameters = derived.parameters
    assert [(layer.vs_min, layer.vs_max) for layer in parameters.vs_layers] == [(100.0, 800.0)] * 3
    assert parameters.thickness_layers[0].thickness_max == 7.0
    # The burn-in follows the iterations given.
    assert (parameters.n_iterations, parameters.n_burnin_iterations) == (20_000, 2_000)


def test_values_given_that_fail_are_changed_with_a_note() -> None:
    given = {
        "vs_layers": [{"vs_min": 200.0, "vs_max": 300.0}]
        + [{"vs_min": 100.0, "vs_max": 1000.0}] * 2,
        "thickness_layers": [{"thickness_min": 0.5, "thickness_max": 30.0}] * 2,
    }
    derived = derive_inversion(CURVE, RULES, given)

    assert derived.notes == (
        # The values given are named: the user reads what they typed was changed.
        "vs_min 200 m/s of the top layer above the curve's slowest velocity (150 m/s): set to "
        "120 m/s.",
        "thickness_min 0.5 m thinner than the curve resolves (1 m) in layers 1, 2: set to 1 m.",
        "thickness_max puts the half-space as deep as 60 m, below the 15.00 m the curve reaches: "
        "scaled by 0.25.",
    )
    # Only the top layer must reach the curve's slowest velocity, and only the half-space its
    # fastest: a slow range below is kept.
    first, second, third = derived.parameters.vs_layers
    assert (first.vs_min, first.vs_max) == (120.0, 300.0)
    assert (second.vs_min, second.vs_max) == (third.vs_min, third.vs_max) == (100.0, 1000.0)
    slow_half_space = derive_inversion(
        CURVE, RULES, {"vs_layers": [{"vs_min": 100.0, "vs_max": 300.0}] * 3}
    )
    assert [layer.vs_max for layer in slow_half_space.parameters.vs_layers] == [300, 300, 450]
    # 60 m scaled to the 15 m the curve reaches, the minimum raised to what it resolves.
    assert [
        (layer.thickness_min, layer.thickness_max) for layer in derived.parameters.thickness_layers
    ] == [(1.0, 7.5)] * 2


def test_a_value_at_the_limit_within_rounding_passes() -> None:
    # The curve reaches 15 m: 7.5 for each of two layers is the limit itself, not deeper.
    given = {"n_layers": 3, "thickness_layers": [{"thickness_max": 7.5}]}
    derived = derive_inversion(CURVE, RULES, given)

    assert derived.notes == ()
    assert derived.reach_m == 15.0


def test_a_single_range_stands_for_every_layer() -> None:
    given = {"vs_layers": [{"vs_min": 100.0, "vs_max": 800.0}], "n_layers": 3}

    parameters = derive_inversion(CURVE, RULES, given).parameters

    assert [(layer.vs_min, layer.vs_max) for layer in parameters.vs_layers] == [(100.0, 800.0)] * 3
    assert broadcast_layers({"vs_layers": [{"vs_max": 300.0}]}) == {
        "vs_layers": [{"vs_max": 300.0}] * 4
    }
    assert broadcast_layers({"thickness_layers": [{"thickness_max": 4.0}]}) == {
        "thickness_layers": [{"thickness_max": 4.0}] * 3
    }


def test_more_layers_than_the_curve_resolves_are_refused_with_a_note() -> None:
    derived = derive_inversion(CURVE, RULES, {"n_layers": 30})

    # 15 m of depth in layers of at least 1 m: 15 layers.
    assert derived.parameters.n_layers == 15
    assert derived.notes == (
        "n_layers 30: the curve resolves 15 (layers of at least 1.00 m down to 15.00 m); set to "
        "15.",
    )


def _two_points(longest: float, shortest: float = 5.0) -> DispersionCurve:
    velocities = np.array([200.0, 250.0])
    return DispersionCurve(
        fs=velocities / np.array([longest, shortest]),
        vs=velocities,
        mode=Mode("M", 0),
        type=VelocityType.PHASE,
        acquisition=UNKNOWN_ACQUISITION,
    )


def test_every_layer_keeps_a_range_at_the_count_the_curve_resolves() -> None:
    # 8.35 m of depth over layers of at least 1.67 m: 5.01 of them. Six layers would leave each
    # of the five above the half-space 1.67 to 1.67 m once rounded; five keep 1.67 to 2.09 m.
    derived = derive_inversion(_two_points(16.7), RULES, {"n_layers": 6})

    assert (derived.parameters.n_layers, derived.max_layers) == (5, 5)
    layer = derived.parameters.thickness_layers[0]
    assert (layer.thickness_min, layer.thickness_max, layer.thickness_perturb_std) == (
        1.67,
        2.09,
        0.05,
    )
    # A hair deeper, six fit, each over a centimetre: the step stays 1 cm, never 0.
    six = derive_inversion(_two_points(16.75), RULES, {"n_layers": 6}).parameters
    assert six.n_layers == 6
    assert (
        six.thickness_layers[0].thickness_max,
        six.thickness_layers[0].thickness_perturb_std,
    ) == (
        1.68,
        0.01,
    )
    # A curve too short for three layers is said so, for the agent.
    with pytest.raises(InversionError, match=r"\(5.0 to 6.6 m\) resolve fewer than 3 layers"):
        derive_inversion(_two_points(6.6), RULES)


def test_the_default_count_is_fitted_to_the_curve_without_a_note() -> None:
    # Wavelengths of 5 to 10 m: 5 m of depth in layers of at least 1.67 m resolve 3 layers, one
    # fewer than the default 4; nobody gave the count, so nothing to report.
    short = DispersionCurve(
        fs=np.array([200.0, 250.0]) / np.array([10.0, 5.0]),
        vs=np.array([200.0, 250.0]),
        mode=Mode("M", 0),
        type=VelocityType.PHASE,
        acquisition=UNKNOWN_ACQUISITION,
    )

    derived = derive_inversion(short, RULES)

    assert (derived.parameters.n_layers, derived.max_layers, derived.notes) == (3, 3, ())


def test_values_that_cannot_hold_are_an_error_for_the_agent() -> None:
    with pytest.raises(InversionError, match="Extra inputs are not permitted"):
        derive_inversion(CURVE, RULES, {"iterations": 5})
    with pytest.raises(InversionError, match="must exceed n_burnin_iterations"):
        derive_inversion(CURVE, RULES, {"n_iterations": 1_000, "n_burnin_iterations": 900})


def test_the_ranges_widen_where_the_curve_needs_it() -> None:
    # A curve from 80 to 1,900 m/s: slower than 100 m/s, faster than a half-space of 2,000 m/s
    # makes (1.09 x 1,900).
    velocities = np.array([1900.0, 400.0, 80.0])
    curve = DispersionCurve(
        fs=velocities / np.array([30.0, 10.0, 3.0]),
        vs=velocities,
        mode=Mode("M", 0),
        type=VelocityType.PHASE,
        acquisition=UNKNOWN_ACQUISITION,
    )

    *layers, half_space = derive_inversion(curve, RULES).parameters.vs_layers

    # 0.8 x 80 m/s at the lowest, 1.5 x 1,900 m/s for the half-space.
    assert {(layer.vs_min, layer.vs_max) for layer in layers} == {(64.0, 1000.0)}
    assert (half_space.vs_min, half_space.vs_max) == (64.0, 2850.0)
    # A curve too fast at its slowest for layers of 1,000 m/s: 1.5 times its slowest velocity.
    fast = DispersionCurve(
        fs=np.array([1200.0, 950.0]) / np.array([30.0, 3.0]),
        vs=np.array([1200.0, 950.0]),
        mode=Mode("M", 0),
        type=VelocityType.PHASE,
        acquisition=UNKNOWN_ACQUISITION,
    )
    assert derive_inversion(fast, RULES).parameters.vs_layers[0].vs_max == 1425.0


def test_the_rules_are_configurable() -> None:
    rules = PriorRules(vs_min=50.0, vs_max=600.0, half_space_vs_max=900.0, max_depth=1.0 / 3)
    parameters = derive_inversion(CURVE, rules).parameters

    assert (parameters.vs_layers[0].vs_min, parameters.vs_layers[0].vs_max) == (50.0, 600.0)
    assert parameters.vs_layers[-1].vs_max == 900.0
    # A third of 30 m for the half-space's top, shared by the three layers above it.
    assert parameters.thickness_layers[0].thickness_max == 3.33
    assert PriorRules.model_validate_json(rules.model_dump_json()) == rules
