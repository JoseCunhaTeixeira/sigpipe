import numpy as np
import pytest
from bayesbay.parameterization import ParameterSpace
from bayesbay.prior import UniformPrior

from sigpipe.algorithms.inversion.rayleigh.seismic.mcmc import (
    CustomParametrization,
    _saved_with_predictions,
    inversion_mcmc,
)
from sigpipe.algorithms.inversion.rayleigh.seismic.parameters import (
    SAVE_EVERY,
    ThicknessLayer,
    VsLayer,
)
from sigpipe.base import DispersionCurve, DispersionCurvesImage, Mode
from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.inversion import InversionResult


def _invert(
    acquisition: LinearAcquisition, n_iterations: int, n_burnin: int, **parameters: object
) -> InversionResult:
    """A two-layer inversion of one smooth M0 curve, with a single chain."""
    n_points = 9
    curve = DispersionCurve(
        fs=np.linspace(10.0, 50.0, n_points),
        vs=np.linspace(300.0, 200.0, n_points),
        mode=Mode("M", 0),
        acquisition=acquisition,
        vs_err=np.full(n_points, 10.0),
    )
    return inversion_mcmc(
        DispersionCurvesImage(dispersion_curves=(curve,)),
        position=acquisition.mid_position,
        **{
            "n_layers": 2,
            "vs_layers": (VsLayer(vs_min=100.0, vs_max=1000.0, vs_perturb_std=20.0),) * 2,
            "thickness_layers": (ThicknessLayer(thickness_min=1.0, thickness_max=10.0),),
            "n_iterations": n_iterations,
            "n_burnin_iterations": n_burnin,
            "n_chains": 1,
            **parameters,
        },
    )


def test_a_run_that_keeps_no_model_is_refused(linear_acquisition: LinearAcquisition) -> None:
    # 2,000 iterations with the default burn-in of 10,000 keep no model.
    with pytest.raises(
        ValueError,
        match=r"n_iterations \(2000\) must exceed n_burnin_iterations \(10000\) by at least 150",
    ):
        _invert(linear_acquisition, n_iterations=2_000, n_burnin=10_000)

    # One iteration short of the first kept model.
    with pytest.raises(ValueError, match="by at least 150"):
        _invert(linear_acquisition, n_iterations=2 * SAVE_EVERY - 1, n_burnin=SAVE_EVERY)


def test_the_parameters_are_checked_before_sampling(
    linear_acquisition: LinearAcquisition,
) -> None:
    # The MCMC's parameters are InversionParameters', layers given as plain dicts too, as a
    # form or a pipeline's JSON sends them.
    with pytest.raises(ValueError, match=r"vs_layers must have length n_layers \(3\)"):
        _invert(linear_acquisition, n_iterations=2_000, n_burnin=200, n_layers=3)
    with pytest.raises(ValueError, match="vs_max must be greater than vs_min"):
        _invert(
            linear_acquisition,
            n_iterations=2_000,
            n_burnin=200,
            vs_layers=[{"vs_min": 500.0, "vs_max": 400.0}, {}],
        )
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _invert(linear_acquisition, n_iterations=2_000, n_burnin=200, Vs_mins=(100.0, 100.0))


def test_the_shortest_run_keeps_one_model(linear_acquisition: LinearAcquisition) -> None:
    result = _invert(linear_acquisition, n_iterations=2 * SAVE_EVERY, n_burnin=SAVE_EVERY)

    assert result.samples["vs1"].size == 1


def test_every_chain_starts_inside_each_layers_prior() -> None:
    # Bounds that differ from layer to layer: the values drawn, then sorted, could land in
    # another layer's range, outside their own prior, and such a chain may never move.
    priors = [
        UniformPrior(name="vs1", vmin=100.0, vmax=400.0, perturb_std=10.0),
        UniformPrior(name="vs2", vmin=150.0, vmax=300.0, perturb_std=10.0),
        UniformPrior(name="vs3", vmin=250.0, vmax=900.0, perturb_std=10.0),
        UniformPrior(name="thick1", vmin=1.0, vmax=10.0, perturb_std=1.0),
        UniformPrior(name="thick2", vmin=0.5, vmax=2.0, perturb_std=1.0),
    ]
    space = ParameterSpace(name="space", n_dimensions=1, parameters=priors)
    parameterization = CustomParametrization(space, [0], [np.linspace(10.0, 50.0, 9)])
    bounds = {prior.name: prior.get_vmin_vmax(None) for prior in priors}

    for _ in range(200):
        state = parameterization.initialize_param_space(space)
        for name, (vmin, vmax) in bounds.items():
            assert vmin <= float(state[name][0]) <= vmax, name


def test_saves_without_a_predicted_curve_are_left_out() -> None:
    # Chain 0 computed from its first save; chain 1 stayed on its starting model for two saves;
    # chain 2 never moved: its saves carry no predicted curve.
    per_chain = {
        "space.vs1": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0]],
        "rayleigh_M0.dpred": [["a", "b", "c"], ["f"]],
    }
    per_chain["rayleigh_M0.dpred"].append([])

    kept, left_out = _saved_with_predictions(per_chain, ["rayleigh_M0.dpred"])

    assert kept == {"space.vs1": [1.0, 2.0, 3.0, 6.0], "rayleigh_M0.dpred": ["a", "b", "c", "f"]}
    assert left_out == 4
    # No chain moved: nothing to build a model from.
    with pytest.raises(ValueError, match="No chain moved from its starting model"):
        _saved_with_predictions({"space.vs1": [[1.0], [2.0]]}, ["rayleigh_M0.dpred"])
