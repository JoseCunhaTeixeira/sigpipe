"""The bundled Silex models, and the range each takes, read without keras."""

import numpy as np
import pytest

from sigpipe.algorithms.inversion.rayleigh.petro.silex_catalog import (
    bundled_silex_model_dir,
    list_bundled_silex_models,
    load_silex_card,
    range_gaps,
    resampled_velocities,
)
from sigpipe.base.acquisition import UNKNOWN_ACQUISITION
from sigpipe.base.dispersion_curve import DispersionCurve, Mode

GRAND_EST = "grand_est_15-50hz_193-415mps"


def _curve(f_start: float, f_end: float, v_high: float, v_low: float) -> DispersionCurve:
    fs = np.linspace(f_start, f_end, 30)
    return DispersionCurve(
        fs=fs,
        vs=np.linspace(v_high, v_low, 30),
        mode=Mode("M", 0),
        acquisition=UNKNOWN_ACQUISITION,
    )


def test_a_model_says_what_it_was_trained_on() -> None:
    assert GRAND_EST in list_bundled_silex_models()

    card = load_silex_card(bundled_silex_model_dir(GRAND_EST))

    assert (card.name, card.min_freq, card.max_freq) == (GRAND_EST, 15.0, 50.0)
    assert card.min_vel == pytest.approx(192.854) and card.max_vel == pytest.approx(414.835)
    assert card.soils == ("clay", "loam", "silt", "sand")
    assert (card.max_layers, card.max_depth, card.water_table) == (4, 20.0, (1.0, 10.0))
    assert card.under_layers.splitlines()[-1].startswith("0 ")  # the half-space


def test_a_curve_is_checked_against_the_trained_range() -> None:
    card = load_silex_card(bundled_silex_model_dir(GRAND_EST))

    covered = resampled_velocities(card, _curve(14.0, 48.0, 400.0, 220.0))

    assert covered.shape == (card.n_freqs,)
    # 20% of the 35 Hz band on each end: a curve must reach 43 Hz.
    with pytest.raises(ValueError, match=r"frequency range \[14, 40\] Hz does not cover"):
        resampled_velocities(card, _curve(14.0, 40.0, 400.0, 220.0))
    with pytest.raises(ValueError, match="velocity range"):
        resampled_velocities(card, _curve(14.0, 48.0, 900.0, 600.0))
    with pytest.raises(ValueError, match="Unknown bundled Silex model 'nowhere'"):
        bundled_silex_model_dir("nowhere")


def test_the_gaps_say_how_a_curve_falls_outside() -> None:
    card = load_silex_card(bundled_silex_model_dir(GRAND_EST))

    assert card.band_needed == (22.0, 43.0)
    assert range_gaps(card, _curve(14.0, 48.0, 400.0, 220.0)) == ()
    assert range_gaps(card, _curve(25.0, 37.0, 400.0, 220.0)) == ("starts_late", "ends_early")
    assert range_gaps(card, _curve(14.0, 48.0, 300.0, 120.0)) == ("too_slow",)
    assert range_gaps(card, _curve(14.0, 48.0, 700.0, 300.0)) == ("too_fast",)
