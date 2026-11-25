import numpy as np
import pytest

from sigpipe.algorithms.picking.dispersion.curve import (
    longest_reached_wavelength,
    max_resolvable_wavelength,
    min_resolvable_wavelength,
    pick_curves,
    resample_wavelength,
    shortest_picked_wavelength,
)
from sigpipe.algorithms.picking.dispersion.lasso import pick_lasso
from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.dispersion_curve import DispersionCurve, Mode, VelocityType
from sigpipe.base.dispersion_image import DispersionImage

FS = np.linspace(5.0, 50.0, 46, dtype=np.float32)
VS = np.linspace(50.0, 800.0, 751, dtype=np.float32)
TRUE_VS = 150.0 + 450.0 * np.exp(-(FS - 5.0) / 10.0)


@pytest.fixture
def image() -> DispersionImage:
    acquisition = LinearAcquisition(
        source=Coordinate(0.0, 0.0, 0.0),
        receivers=tuple(Coordinate(float(i), 0.0, 0.0) for i in range(1, 25)),
    )
    fv_map = np.exp(-(((VS[None, :] - TRUE_VS[:, None]) / 20.0) ** 2))
    return DispersionImage(
        fv_map=fv_map.astype(np.float32),
        fs=FS,
        vs=VS,
        type=VelocityType.PHASE,
        acquisition=acquisition,
    )


def test_picking_a_mode_again_replaces_its_curve(image: DispersionImage) -> None:
    both = pick_curves(image, fmaxs=[30.0, None], labels=["R", "R"], modes=[0, 1])
    again = pick_curves(both, fmaxs=[20.0], labels=["R"], modes=[0])
    assert again.dispersion_curves is not None
    curves = {curve.mode: curve for curve in again.dispersion_curves}
    assert set(curves) == {Mode("R", 0), Mode("R", 1)}
    assert float(curves[Mode("R", 0)].fs.max()) == 20.0
    assert float(curves[Mode("R", 1)].fs.max()) == 50.0


@pytest.mark.parametrize("n_points", [2, 3, 4, 5])
def test_short_curves_are_picked(image: DispersionImage, n_points: int) -> None:
    picked = pick_curves(image, fmins=[5.0], fmaxs=[5.0 + n_points - 1])
    assert picked.dispersion_curves is not None
    (curve,) = picked.dispersion_curves
    assert len(curve.fs) == n_points
    np.testing.assert_allclose(curve.vs, TRUE_VS[:n_points], atol=20.0)


def test_the_curve_ends_are_not_flattened(image: DispersionImage) -> None:
    picked = pick_curves(image)
    assert picked.dispersion_curves is not None
    (curve,) = picked.dispersion_curves
    # Zero-padded, the median filter would take the two lowest frequencies for outliers.
    assert abs(float(curve.vs[0]) - TRUE_VS[0]) < 20.0


# A steep branch and its knee, as a soft layer over a stiff one gives: 225 m/s at 4.7 Hz, 105 at
# 6, flat at 82 above 9. Each column a broad ridge (a short spread's), peaking at the curve.
KNEE_FS = np.arange(3.0, 20.05, 0.1, dtype=np.float32)
KNEE_VS = np.arange(40.0, 400.0, 1.0, dtype=np.float32)


def knee(frequencies: np.ndarray | float | np.floating) -> np.ndarray:
    return 82.0 + 143.0 * np.exp(-(np.asarray(frequencies, dtype=float) - 4.7) / 0.85)


@pytest.fixture
def steep() -> DispersionImage:
    centre = knee(KNEE_FS)[:, None]
    fv_map = 0.05 + 0.9 * np.exp(-0.5 * ((KNEE_VS[None, :] - centre) / (0.15 * centre)) ** 2)
    acquisition = LinearAcquisition(
        source=Coordinate(0.0, 0.0, 0.0),
        receivers=tuple(Coordinate(5.0 + 5.0 * k, 0.0, 0.0) for k in range(6)),
    )
    return DispersionImage(
        fv_map=fv_map.astype(np.float32),
        fs=KNEE_FS,
        vs=KNEE_VS,
        type=VelocityType.PHASE,
        acquisition=acquisition,
    )


def _band(lower: float, upper: float, f_lo: float, f_hi: float) -> list[tuple[float, float]]:
    """A lasso around the knee: from its low-frequency top along `upper` times the curve, back
    along `lower` times it."""
    fs = np.linspace(f_lo, f_hi, 60)
    top = [(float(f), float(upper * knee(f))) for f in fs]
    bottom = [(float(f), float(lower * knee(f))) for f in fs[::-1]]
    return top + bottom


def _m0(picked: DispersionImage) -> tuple[np.ndarray, np.ndarray]:
    assert picked.dispersion_curves is not None
    (curve,) = picked.dispersion_curves
    return curve.fs.astype(float), curve.vs.astype(float)


@pytest.mark.parametrize("start", [0, 25, 60, 100])
def test_a_lasso_follows_a_steep_branch_and_its_knee_wherever_it_starts(
    steep: DispersionImage, start: int
) -> None:
    # Drawn from any of its points: the lasso is closed as drawn, never moved.
    polygon = _band(0.8, 1.25, 4.8, 16.0)
    polygon = polygon[start:] + polygon[:start]

    fs, vs = _m0(pick_lasso(steep, polygon, Mode("M", 0)))

    # From where the lasso holds the ridge's peak, along its steepest stretch and its knee, to
    # its last whole metre of wavelength (6 m, 13.7 Hz).
    assert fs.min() < 5.0 and fs.max() > 13.5
    np.testing.assert_allclose(vs, knee(fs), rtol=0.01)


def test_where_a_lasso_cuts_the_ridge_its_columns_give_no_pick(steep: DispersionImage) -> None:
    # Its upper edge 5 % under the ridge between 5 and 6 Hz: there the brightest cell inside is
    # the edge, on the ridge's flank, not a peak.
    upper = [
        (float(f), float(knee(f)) * (0.95 if 5.0 <= f <= 6.0 else 1.25))
        for f in KNEE_FS
        if 4.8 <= f <= 16.0
    ]
    lower = [(f, 0.8 * float(knee(f))) for f, _ in upper[::-1]]

    fs, vs = _m0(pick_lasso(steep, upper + lower, Mode("M", 0)))

    assert not ((fs > 5.05) & (fs < 5.95)).any()
    np.testing.assert_allclose(vs, knee(fs), rtol=0.01)


def test_a_box_whose_bound_cuts_the_ridge_picks_nothing_there(steep: DispersionImage) -> None:
    # Under 200 m/s: below 5 Hz the ridge lies above it, and the box holds only its flank.
    fs, vs = _m0(pick_curves(steep, fmins=[4.0], fmaxs=[16.0], vmaxs=[200.0]))

    assert fs.min() > 4.85
    np.testing.assert_allclose(vs, knee(fs), rtol=0.01)


def test_a_resampled_box_makes_nothing_up_across_a_gap(steep: DispersionImage) -> None:
    # From 6 to 7.5 Hz the columns only rise to the top of the image: no peak, no pick, and the
    # resampling over wavelength draws no chord across.
    gap = (KNEE_FS >= 6.0) & (KNEE_FS <= 7.5)
    fv_map = steep.fv_map.copy()
    fv_map[gap] = np.linspace(0.1, 0.9, KNEE_VS.size, dtype=np.float32)
    image = DispersionImage(
        fv_map=fv_map, fs=steep.fs, vs=steep.vs, type=steep.type, acquisition=steep.acquisition
    )

    fs, vs = _m0(pick_curves(image, fmins=[4.0], fmaxs=[16.0], resample_over_wavelength=True))

    assert not ((fs > 6.05) & (fs < 7.45)).any()
    assert fs.min() < 6.0 and fs.max() > 7.5
    np.testing.assert_allclose(vs, knee(fs), rtol=0.01)


def test_a_curve_whose_wavelength_turns_back_is_resampled_along_it() -> None:
    # Flat at 200 m/s to 15 Hz, then rising as f^1.5 to 20 Hz, its wavelength rising again from
    # 13.3 to 15.4 m, then flat: each wavelength met on its way, no point between two stretches.
    def course(frequencies: np.ndarray) -> np.ndarray:
        return 200.0 * (np.clip(frequencies, 15.0, 20.0) / 15.0) ** 1.5

    fs = np.arange(10.0, 30.01, 0.25)
    curve = DispersionCurve(
        fs=fs.astype(np.float32),
        vs=course(fs).astype(np.float32),
        mode=Mode("M", 0),
        acquisition=LinearAcquisition(
            source=Coordinate(0.0, 0.0, 0.0),
            receivers=tuple(Coordinate(float(k), 0.0, 0.0) for k in range(1, 25)),
        ),
    )

    resampled = resample_wavelength(curve, step=0.5, along=True)

    np.testing.assert_allclose(resampled.vs, course(resampled.fs.astype(float)), rtol=0.01)
    # 13.5 to 15 m met on the flat stretch under 15 Hz, on the rise, and above 20 Hz.
    wavelengths = resampled.vs / resampled.fs
    assert np.count_nonzero(np.isclose(wavelengths, 14.0, atol=1e-3)) == 3
    # Not along it (the tracker's resampling, as PACo's checks were set on): one point a
    # wavelength, as before.
    sorted_once = resample_wavelength(curve, step=0.5)
    assert np.unique(np.round(sorted_once.vs / sorted_once.fs, 6)).size == sorted_once.fs.size


def test_resolvable_wavelengths_follow_the_ground() -> None:
    # Receivers 2 m apart, then 1 m, the last one 1 m higher: lengths along the ground.
    receivers = (
        Coordinate(0.0, 0.0, 0.0),
        Coordinate(2.0, 0.0, 0.0),
        Coordinate(3.0, 0.0, 0.0),
        Coordinate(4.0, 0.0, 1.0),
    )
    acquisition = LinearAcquisition(source=Coordinate(-1.0, 0.0, 0.0), receivers=receivers[::-1])
    assert min_resolvable_wavelength(acquisition) == pytest.approx(2.0)
    assert max_resolvable_wavelength(acquisition) == pytest.approx(3.0 + np.sqrt(2.0))
    # The picker's floor, one spacing; the checks' reach, three window lengths.
    assert shortest_picked_wavelength(acquisition) == pytest.approx(1.0)
    assert longest_reached_wavelength(acquisition) == pytest.approx(3 * (3.0 + np.sqrt(2.0)))


def test_mode_labels() -> None:
    assert Mode("M", 0).label == "M0"
    assert Mode.from_label("R12") == Mode("R", 12)
    for label in ("M", "12", ""):
        with pytest.raises(ValueError, match="wave then a number"):
            Mode.from_label(label)
