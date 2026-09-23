import numpy as np
import pytest

from sigpipe.algorithms.inversion.rayleigh.seismic.mcmc import SAVE_EVERY, inversion_mcmc
from sigpipe.base import DispersionCurve, DispersionCurvesImage, Mode
from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.inversion import InversionResult


def _invert(acquisition: LinearAcquisition, n_iterations: int, n_burnin: int) -> InversionResult:
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
        n_layers=2,
        thicknesses_min=(1.0,),
        thicknesses_max=(10.0,),
        thickness_perturbations=(1.0,),
        Vs_mins=(100.0, 100.0),
        Vs_maxs=(1000.0, 1000.0),
        Vs_perturbations=(20.0, 20.0),
        n_iterations=n_iterations,
        n_burnin=n_burnin,
        n_chains=1,
    )


def test_a_run_that_keeps_no_model_is_refused(linear_acquisition: LinearAcquisition) -> None:
    # 2,000 iterations with PAC's default burn-in of 10,000: it used to fail after sampling,
    # with KeyError: 'space.vs1'.
    with pytest.raises(
        ValueError, match=r"n_iterations \(2000\) must exceed n_burnin \(10000\) by at least 150"
    ):
        _invert(linear_acquisition, n_iterations=2_000, n_burnin=10_000)

    # One iteration short of the first kept model.
    with pytest.raises(ValueError, match="by at least 150"):
        _invert(linear_acquisition, n_iterations=2 * SAVE_EVERY - 1, n_burnin=SAVE_EVERY)


def test_the_shortest_run_keeps_one_model(linear_acquisition: LinearAcquisition) -> None:
    result = _invert(linear_acquisition, n_iterations=2 * SAVE_EVERY, n_burnin=SAVE_EVERY)

    assert result.samples["vs1"].size == 1
