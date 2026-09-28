"""The seismic inversion's samplers on curves with a known answer: the layers given (DREAM) and
the layers chosen by the data (reversible jump), the prior they sample without data, and the
parameters they refuse."""

import math
from typing import Any

import numpy as np
import pytest

from sigpipe.algorithms.inversion.rayleigh.seismic.data import (
    MAX_POINTS,
    NOISE_BOUNDS,
    Curve,
    allowed,
    curves_of,
)
from sigpipe.algorithms.inversion.rayleigh.seismic.forward import fwd_seismic_phase
from sigpipe.algorithms.inversion.rayleigh.seismic.mcmc import inversion_mcmc
from sigpipe.algorithms.inversion.rayleigh.seismic.parameters import (
    SAVE_EVERY,
    InversionParameters,
    ThicknessLayer,
    VsLayer,
)
from sigpipe.algorithms.inversion.rayleigh.seismic.transdimensional import (
    THINNEST,
    Settings,
    Space,
    run_chain,
)
from sigpipe.base import DispersionCurve, DispersionCurvesImage, Mode
from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.inversion import InversionResult

# A soft layer over a stiffer one, and its M0 from 10 to 40 Hz with 2 % noise.
TRUTH_THICKNESS, TRUTH_VS = 4.0, (200.0, 450.0)
FS = np.geomspace(10.0, 40.0, 16)


def _truth(depths: np.ndarray) -> np.ndarray:
    return np.where(depths < TRUTH_THICKNESS, TRUTH_VS[0], TRUTH_VS[1])


def _picked(acquisition: LinearAcquisition, n: int = FS.size) -> DispersionCurve:
    fs = np.geomspace(10.0, 40.0, n)
    clean = fwd_seismic_phase([TRUTH_THICKNESS, 1000.0], list(TRUTH_VS), 0, fs, 1.77).vs
    noisy = clean * (1 + 0.02 * np.random.default_rng(3).standard_normal(n))
    return DispersionCurve(
        fs=fs,
        vs=noisy,
        mode=Mode("M", 0),
        acquisition=acquisition,
        vs_err=0.25 * noisy,  # as wide as an array's resolving power makes them
    )


def _invert(acquisition: LinearAcquisition, **parameters: Any) -> InversionResult:  # noqa: ANN401
    return inversion_mcmc(
        DispersionCurvesImage(dispersion_curves=(_picked(acquisition),)),
        position=acquisition.mid_position,
        seed=1,
        **parameters,
    )


FIXED = {
    "n_layers": 2,
    "vs_layers": (VsLayer(vs_min=100.0, vs_max=1000.0),) * 2,
    "thickness_layers": (ThicknessLayer(thickness_min=1.0, thickness_max=10.0),),
}


def test_a_run_that_keeps_no_model_is_refused(linear_acquisition: LinearAcquisition) -> None:
    with pytest.raises(
        ValueError,
        match=r"n_iterations \(2000\) must exceed n_burnin_iterations \(10000\) by at least 150",
    ):
        _invert(linear_acquisition, n_iterations=2_000, n_burnin_iterations=10_000)
    with pytest.raises(ValueError, match="by at least 150"):
        _invert(
            linear_acquisition,
            n_iterations=2 * SAVE_EVERY - 1,
            n_burnin_iterations=SAVE_EVERY,
        )


def test_the_parameters_are_checked_before_sampling(
    linear_acquisition: LinearAcquisition,
) -> None:
    # Layers given as plain dicts too, as a form or a pipeline's JSON sends them.
    with pytest.raises(ValueError, match=r"vs_layers must have length n_layers \(3\)"):
        _invert(linear_acquisition, **(FIXED | {"n_layers": 3}))
    with pytest.raises(ValueError, match="vs_max must be greater than vs_min"):
        _invert(
            linear_acquisition,
            **(FIXED | {"vs_layers": [{"vs_min": 500.0, "vs_max": 400.0}, {}]}),
        )
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _invert(linear_acquisition, Vs_mins=(100.0, 100.0))
    with pytest.raises(ValueError, match="depth_max must be greater than depth_min"):
        _invert(linear_acquisition, free={"depth_min": 5.0, "depth_max": 2.0})
    with pytest.raises(ValueError, match="chain_jobs must be at least 1"):
        inversion_mcmc(
            DispersionCurvesImage(dispersion_curves=(_picked(linear_acquisition),)),
            position=linear_acquisition.mid_position,
            chain_jobs=0,
        )


def test_runs_saved_before_read_as_their_fixed_layers() -> None:
    # Runs saved before 2026-09-27: no layering, their steps' trial runs.
    saved = InversionParameters.model_validate(
        {"n_layers": 2, "vs_layers": [{}, {}], "thickness_layers": [{}], "tune_steps": True}
    )

    assert saved.layering == "fixed"
    assert InversionParameters().layering == "free"
    # Only the iterations given: a quarter of them burn in.
    assert InversionParameters(n_iterations=2_000).n_burnin_iterations == 500


def test_the_layers_given_are_found(linear_acquisition: LinearAcquisition) -> None:
    result = _invert(linear_acquisition, n_iterations=8_000, n_chains=3, **FIXED)

    assert result.profiles is not None
    # The truth, at the depths the curve resolves.
    at = np.array([1.5, 3.0, 6.0])
    median = np.median(result.profiles.at(at), axis=0)
    assert np.all(np.abs(np.log(median / _truth(at))) < 0.15)
    # The named values; the noise factor at its floor, a third of the picks' 25 %, for 2 % of
    # noise: no further, the models' curves stay spread within the uncertainties.
    assert set(result.samples) == {"vs1", "vs2", "thick1", "noise"}
    assert NOISE_BOUNDS[0] <= float(np.median(result.samples["noise"])) < 0.4
    assert len(result.acceptance) == 3 and set(result.steps) == {"vs1", "vs2", "thick1"}
    # As many kept from each chain; every kept model's curve at the picked frequencies.
    kept = (8_000 - 2_000) // SAVE_EVERY
    assert result.profiles.vs.shape == (3 * kept, 2)
    assert result.dpred[0].shape == (3 * kept, FS.size)
    assert result.parameters["layering"] == "fixed"


def test_the_data_choose_the_layers(linear_acquisition: LinearAcquisition) -> None:
    result = _invert(linear_acquisition, n_iterations=12_000, n_chains=2, chain_jobs=2)

    assert result.profiles is not None
    at = np.array([1.5, 3.0, 6.0])
    median = np.median(result.profiles.at(at), axis=0)
    assert np.all(np.abs(np.log(median / _truth(at))) < 0.15)
    # A few layers at most: two do.
    assert float(np.median(result.samples["layers"])) <= 4
    # The bounds left out, found from the picks.
    free = result.parameters["free"]
    wavelengths = _picked(linear_acquisition).vs / FS
    assert free["depth_max"] == round(float(wavelengths.max()) / 2, 2)
    assert 0 < free["vs_min"] < free["vs_max"]
    assert result.n_layers == len(result.median.vs_s)
    assert len(result.acceptance) == 2
    # How the chains moved: each move's step after the burn-in and its acceptance, and the
    # exchanges between tempered copies (the chains' medians).
    assert set(result.steps) == {"interface", "vs", "noise", "shift", "stretch"}
    assert all(step > 0 for step in result.steps.values())
    assert {"birth", "death", "vs", "noise"} <= set(result.moves)
    assert all(0 <= rate <= 100 for rate in result.moves.values())
    assert result.exchanges is not None and 0 < result.exchanges <= 100
    # Two chains of their own, as many kept each.
    first, second = np.split(result.profiles.vs[:, 0], 2)
    assert first.size == second.size > 1 and not np.array_equal(first, second)


def test_a_value_fixed_is_not_sampled(linear_acquisition: LinearAcquisition) -> None:
    fixed_half_space = (VsLayer(vs_min=100.0, vs_max=1000.0), VsLayer(vs_fixed=450.0))
    result = _invert(
        linear_acquisition,
        n_iterations=3_000,
        n_chains=2,
        **(FIXED | {"vs_layers": fixed_half_space}),
    )

    assert np.all(result.samples["vs2"] == 450.0)
    assert np.ptp(result.samples["vs1"]) > 0 and np.ptp(result.samples["thick1"]) > 0
    assert set(result.steps) == {"vs1", "thick1"}


def test_a_model_with_every_value_fixed_is_refused(linear_acquisition: LinearAcquisition) -> None:
    every = {
        "vs_layers": (VsLayer(vs_fixed=200.0), VsLayer(vs_fixed=400.0)),
        "thickness_layers": (ThicknessLayer(thickness_fixed=3.0),),
    }
    with pytest.raises(ValueError, match="nothing is left to sample"):
        _invert(linear_acquisition, **(FIXED | every))


def test_without_data_the_chains_sample_the_prior() -> None:
    # Picks no model can miss (uncertainties of 10^9 m/s), Vs increasing: the number of layers
    # follows the prior's, uniform before the constraints (sorted Vs: 1/k!; layers thick enough:
    # the gaps' share), which the reversible jumps' acceptance must keep exactly.
    fs = np.array([10.0, 20.0, 30.0])
    by_period = np.argsort(1 / fs)
    curve = Curve(
        mode=0,
        periods=(1 / fs)[by_period],
        observed=np.full(3, 300.0),
        sigma=np.full(3, 1e9),
        order=np.argsort(by_period),
    )
    space = Space(
        curves=(curve,),
        vs_bounds=(100.0, 1000.0),
        depth_bounds=(1.0, 30.0),
        max_layers=4,
        least_ratio=1.0,
        vp_vs=1.77,
    )
    chain = run_chain(space, Settings(60_000, 5_000, 10, 1, 1.0), seed=5)

    span, gap = math.log(30.0), math.log(1 + THINNEST)
    expected = np.array(
        [((span - max(0, k - 2) * gap) / span) ** (k - 1) / math.factorial(k) for k in range(1, 5)]
    )
    shares = np.bincount(chain.layers, minlength=5)[1:] / chain.layers.size
    assert np.allclose(shares, expected / expected.sum(), atol=0.04)


def test_a_dense_curve_is_inverted_on_fewer_points(linear_acquisition: LinearAcquisition) -> None:
    # 90 picks: averaged by bands of wavelength, their uncertainties kept (errors that go
    # together do not average out).
    picked = _picked(linear_acquisition, n=90)
    (curve,) = curves_of([picked])

    assert curve.observed.size <= MAX_POINTS
    assert np.all(np.diff(curve.periods) > 0)
    assert picked.vs_err is not None
    assert float(curve.sigma.mean()) == pytest.approx(float(np.mean(picked.vs_err)), rel=0.1)
    # Every pick, in the picked order, when asked.
    (whole,) = curves_of([picked], max_points=None)
    assert np.allclose(whole.observed[whole.order], picked.vs)


def test_a_stiff_layer_over_a_soft_one_is_refused_beyond_the_drop_allowed() -> None:
    assert allowed(np.array([300.0, 250.0, 400.0]), 0.8)
    assert not allowed(np.array([700.0, 200.0, 400.0]), 0.8)
    assert allowed(np.array([700.0, 200.0]), 0.0)
