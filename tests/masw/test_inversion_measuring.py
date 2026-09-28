"""What G5 measures of an inversion, each measure on inputs with a known answer: chains that agree
or not, samples piled at a bound, a half-space the data do not inform, one yardstick for every
window, fits by band, the sampler's log."""

import numpy as np
import pytest

from sigpipe.base import DispersionCurve, Mode, VelocityType
from sigpipe.base.acquisition import UNKNOWN_ACQUISITION
from sigpipe.base.inversion import LayeredSamples
from sigpipe.masw.inversion import InversionParameters, ThicknessLayer, VsLayer
from sigpipe.masw.inversion.measuring import (
    acceptance_rates,
    bound_shares,
    depth_bottom,
    effective_sample_size,
    fit_by_band,
    interface_shares,
    lag1_autocorrelation,
    report_depths,
    split_rhat,
    useful_depth,
    vs_at,
    yardstick,
)

RNG = np.random.default_rng(7)
PARAMETERS = InversionParameters.model_validate(
    {
        "vs_layers": [{"vs_min": 100.0, "vs_max": 500.0}] * 2,
        "thickness_layers": [{"thickness_min": 1.0, "thickness_max": 5.0}],
    }
)


def _profiles(samples: dict[str, np.ndarray]) -> LayeredSamples:
    """Two-layer samples, as the samplers keep them."""
    return LayeredSamples(
        depths=samples["thick1"][:, None],
        vs=np.column_stack([samples["vs1"], samples["vs2"]]),
        n_chains=1,
    )


def _curve(vs: np.ndarray, fs: np.ndarray, errors: np.ndarray | None = None) -> DispersionCurve:
    return DispersionCurve(
        fs=fs,
        vs=vs,
        mode=Mode("M", 0),
        type=VelocityType.PHASE,
        acquisition=UNKNOWN_ACQUISITION,
        vs_err=errors,
    )


def test_chains_sampling_the_same_distribution_agree() -> None:
    chains = RNG.normal(200.0, 10.0, size=(5, 600))

    rhat = split_rhat(chains)
    assert rhat is not None and rhat < 1.01


def test_chains_stuck_apart_do_not() -> None:
    chains = RNG.normal(200.0, 10.0, size=(5, 600)) + np.arange(5)[:, None] * 30.0

    rhat = split_rhat(chains)
    assert rhat is not None and rhat > 2.0


def test_a_chain_drifting_within_itself_is_caught_by_the_split() -> None:
    # Each chain climbs from 150 to 250: its two halves disagree, though all chains agree.
    chains = np.tile(np.linspace(150.0, 250.0, 600), (5, 1)) + RNG.normal(0.0, 5.0, (5, 600))

    rhat = split_rhat(chains)
    assert rhat is not None and rhat > 1.5


def test_too_few_samples_or_no_spread_say_what_they_can() -> None:
    assert split_rhat(RNG.normal(size=(5, 3))) is None
    assert split_rhat(np.full((5, 10), 200.0)) == 1.0


def test_acceptance_rates_are_read_from_the_samplers_log() -> None:
    log = (
        "Chain ID: 0\nTEMPERATURE: 1\nEXPLORED MODELS: 2000\n"
        "ACCEPTANCE RATE: 257/2000 (12.85 %)\nPARTIAL ACCEPTANCE RATES:\n"
        "Chain ID: 1\nTEMPERATURE: 1\nEXPLORED MODELS: 2000\n"
        "ACCEPTANCE RATE: 1400/2000 (70.00 %)\n"
    )

    assert acceptance_rates(log) == (12.85, 70.0)


def test_a_flat_posterior_puts_its_share_at_each_bound() -> None:
    samples = {
        "vs1": RNG.uniform(100.0, 500.0, 20_000),
        "vs2": RNG.uniform(100.0, 500.0, 20_000),
        "thick1": RNG.uniform(1.0, 5.0, 20_000),
    }

    shares = bound_shares(samples, PARAMETERS, 0.02)
    assert len(shares) == 6
    assert all(share.share == pytest.approx(0.02, abs=0.005) for share in shares)


def test_a_posterior_piled_at_a_bound_is_found_first() -> None:
    samples = {
        "vs1": RNG.normal(250.0, 10.0, 3_000),
        # The half-space wants to be slower than the prior allows.
        "vs2": np.clip(RNG.normal(90.0, 20.0, 3_000), 100.0, 500.0),
        "thick1": RNG.normal(3.0, 0.2, 3_000),
    }

    first = bound_shares(samples, PARAMETERS, 0.02)[0]
    assert (first.parameter, first.bound, first.value) == ("vs2", "min", 100.0)
    assert first.share > 0.5


def test_the_useful_depth_ends_where_the_data_stop_informing_vs() -> None:
    # A well-resolved layer of 3 m over a half-space the data say nothing about.
    samples = {
        "vs1": RNG.normal(250.0, 5.0, 3_000),
        "thick1": RNG.normal(3.0, 0.05, 3_000),
        "vs2": RNG.uniform(100.0, 500.0, 3_000),
    }

    depth = useful_depth(_profiles(samples), PARAMETERS, 0.5)
    assert depth is not None and 2.8 <= depth <= 3.1


def test_a_thin_top_layer_the_data_cannot_resolve_does_not_end_the_useful_depth() -> None:
    # The top 0.5 to 1 m spans the prior; the layer below it is well resolved down to 3 m.
    samples = {
        "vs1": RNG.uniform(100.0, 500.0, 3_000),
        "thick1": RNG.uniform(1.0, 1.05, 3_000),
        "vs2": RNG.normal(250.0, 5.0, 3_000),
    }
    parameters = PARAMETERS.model_copy(
        update={
            "thickness_layers": (
                PARAMETERS.thickness_layers[0].model_copy(
                    update={"thickness_min": 1.0, "thickness_max": 1.05}
                ),
            )
        }
    )

    # Informed below the top layer, to the models' bottom.
    assert useful_depth(_profiles(samples), parameters, 0.5) is None
    # Nothing informed at all: 0.
    samples["vs2"] = RNG.uniform(100.0, 500.0, 3_000)
    assert useful_depth(_profiles(samples), parameters, 0.5) == 0.0


def test_a_model_informed_to_its_bottom_has_no_useful_depth() -> None:
    samples = {
        "vs1": RNG.normal(250.0, 5.0, 3_000),
        "thick1": RNG.normal(3.0, 0.05, 3_000),
        "vs2": RNG.normal(350.0, 5.0, 3_000),
    }

    assert useful_depth(_profiles(samples), PARAMETERS, 0.5) is None


# Picks from 400 m/s at 5 Hz to 200 m/s at 40 Hz: wavelengths from 5 to 80 m.
PICKED_FS = np.array([5.0, 10.0, 20.0, 40.0])
PICKED_VS = np.array([400.0, 300.0, 250.0, 200.0])
# Run with the half-space's Vs between 300 and 700 m/s, or far wider.
NARROW = PARAMETERS.model_copy(
    update={"vs_layers": (VsLayer(vs_min=100.0, vs_max=500.0), VsLayer(vs_min=300.0, vs_max=700.0))}
)
WIDE = PARAMETERS.model_copy(
    update={
        "vs_layers": (VsLayer(vs_min=100.0, vs_max=500.0), VsLayer(vs_min=100.0, vs_max=1_500.0))
    }
)


def _resolved_over(half_space: np.ndarray) -> LayeredSamples:
    """A layer resolved to 3 m over a half-space whose Vs spreads as `half_space`."""
    return _profiles(
        {
            "vs1": RNG.normal(250.0, 5.0, half_space.size),
            "thick1": RNG.normal(3.0, 0.05, half_space.size),
            "vs2": half_space,
        }
    )


def test_the_yardstick_is_the_prior_the_curve_alone_gives() -> None:
    stick = yardstick(PICKED_FS, PICKED_VS)

    # The layers chosen by the data: Vs from half the slowest pick to three times the fastest,
    # interfaces from a third of the shortest wavelength to half the longest.
    assert stick.layering == "free"
    assert (stick.free.vs_min, stick.free.vs_max) == (100.0, 1_200.0)
    assert (stick.free.depth_min, stick.free.depth_max) == (1.67, 40.0)


def test_one_yardstick_reads_a_posterior_alike_whatever_its_prior() -> None:
    # The half-space spreads over its narrow prior: against each run's own prior, the narrow
    # run looks informed to 3 m only, the wide one to its bottom.
    profiles = _resolved_over(RNG.uniform(300.0, 700.0, 3_000))
    own = [useful_depth(profiles, run, 0.5) for run in (NARROW, WIDE)]
    assert own[0] is not None and 2.8 <= own[0] <= 3.1 and own[1] is None

    # Against the yardstick, one reading: a spread under half of what the curve alone allows.
    stick = yardstick(PICKED_FS, PICKED_VS)
    assert [useful_depth(profiles, run, 0.5, reference=stick) for run in (NARROW, WIDE)] == [
        None,
        None,
    ]
    # A half-space as open as the curve alone leaves it: informed to 3 m, whatever the run.
    open_below = _resolved_over(np.exp(RNG.uniform(np.log(100.0), np.log(1_200.0), 3_000)))
    read = [useful_depth(open_below, run, 0.5, reference=stick) for run in (NARROW, WIDE)]
    assert read[0] == read[1] and read[0] is not None and 2.8 <= read[0] <= 3.1


def test_the_fit_is_judged_by_band_of_wavelength() -> None:
    fs = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    picked = _curve(np.array([300.0, 260.0, 240.0, 230.0, 225.0, 220.0]), fs, np.full(6, 10.0))
    # The model is right at long wavelengths, 20 m/s too fast at the short ones, and 10 m/s at
    # 5.75 m.
    predicted = _curve(np.array([300.0, 260.0, 240.0, 240.0, 245.0, 240.0]), fs)

    fit = fit_by_band("smooth_median", picked, predicted, 3)

    assert fit.n_missing == 0
    # Short wavelengths first: 3.67 and 4.5 m, then 5.75 and 8 m, then 13 and 30 m.
    assert [band.misfit for band in fit.bands] == [2.0, 0.707, 0.0]
    assert fit.bands[0].wavelength_m == (3.67, 4.5)
    assert fit.bands[0].residual == pytest.approx(0.0825, abs=0.001)  # 20/240 and 20/245
    assert fit.misfit == pytest.approx(np.sqrt((4 + 4 + 1) / 6), abs=0.001)


def test_points_without_a_mode_are_counted_not_fitted() -> None:
    fs = np.array([10.0, 20.0, 30.0])
    picked = _curve(np.array([300.0, 260.0, 240.0]), fs, np.full(3, 10.0))
    predicted = _curve(np.array([300.0, 260.0, np.nan]), fs)

    fit = fit_by_band("median", picked, predicted, 3)
    assert (fit.n_missing, fit.misfit) == (1, 0.0)
    assert fit.bands[0].misfit is None  # 240 m/s at 8 m, the shortest wavelength
    assert fit_by_band("median", picked, None, 3).misfit is None


def test_points_without_an_uncertainty_weigh_nothing() -> None:
    # A pick's end point can carry no uncertainty, which would make the whole band's misfit NaN.
    fs = np.array([10.0, 20.0, 30.0])
    picked = _curve(np.array([300.0, 260.0, 240.0]), fs, np.array([10.0, 10.0, np.nan]))
    predicted = _curve(np.array([310.0, 260.0, 250.0]), fs)

    fit = fit_by_band("petro", picked, predicted, 3)

    assert (fit.n_missing, fit.misfit) == (0, pytest.approx(np.sqrt(1 / 2), abs=0.001))
    assert fit.bands[0].misfit is None and fit.bands[0].residual is not None


def test_a_layered_model_is_read_at_depths() -> None:
    # A depth on an interface belongs to the layer below, as in sigpipe's models.
    assert vs_at((2.0, 1000.0), (200.0, 400.0), (1.0, 2.0, 3.0)) == (200.0, 400.0, 400.0)


@pytest.mark.parametrize(
    ("longest", "depths"),
    [
        ([11.0, 11.0, 10.0], (1.0, 2.0, 3.0, 4.0, 5.0)),
        ([7.0], (1.0, 2.0, 3.0)),
        ([21.0], (2.0, 4.0, 6.0, 8.0, 10.0)),
        ([3.0], (0.5, 1.0, 1.5)),
        ([], ()),
    ],
)
def test_report_depths_are_round_and_within_reach(
    longest: list[float], depths: tuple[float, ...]
) -> None:
    assert report_depths(longest) == depths


def test_the_effective_sample_size_counts_independent_samples() -> None:
    rng = np.random.default_rng(3)
    independent = rng.normal(size=(4, 500))
    # Each sample 0.9 times the one before, and noise: about 1/19 of them are independent.
    correlated = np.zeros((4, 500))
    for t in range(1, 500):
        correlated[:, t] = 0.9 * correlated[:, t - 1] + rng.normal(size=4)

    many = effective_sample_size(independent)
    assert many is not None and 1_600 <= many <= 2_400
    few = effective_sample_size(correlated)
    assert few is not None and 50 <= few <= 200
    assert lag1_autocorrelation(correlated) == pytest.approx(0.9, abs=0.05)
    assert abs(lag1_autocorrelation(independent) or 0) < 0.05
    assert effective_sample_size(independent[:, :6]) is None


def test_values_fixed_are_left_out_of_the_measures() -> None:
    parameters = InversionParameters(
        n_layers=2,
        vs_layers=(VsLayer(), VsLayer(vs_fixed=400.0)),
        thickness_layers=(ThicknessLayer(thickness_fixed=3.0),),
    )
    samples = {
        "vs1": np.linspace(150.0, 250.0, 50),
        "vs2": np.full(50, 400.0),
        "thick1": np.full(50, 3.0),
    }

    # Not sampled: no prior to pile against; the models end under the fixed thickness.
    assert {share.parameter for share in bound_shares(samples, parameters, 0.02)} == {"vs1"}
    assert depth_bottom(parameters) == 4.0
    assert parameters.fixed() == {"vs2": 400.0, "thick1": 3.0}


def test_the_interfaces_are_counted_once_a_model_by_depth() -> None:
    # Four models: interfaces near 3 m in three, one at 8 m, two in one bin in the last.
    depths = np.array(
        [[3.1, np.nan], [3.2, 8.0], [2.9, np.nan], [6.1, 6.3]],
    )
    profiles = LayeredSamples(depths=depths, vs=np.full((4, 3), 300.0), n_chains=1)

    shares = interface_shares(profiles, bottom=10.0, dz=0.5)

    assert len(shares) == 20
    assert (shares[5], shares[6], shares[12], shares[16]) == (0.25, 0.5, 0.25, 0.25)
    assert sum(shares) == 1.25
