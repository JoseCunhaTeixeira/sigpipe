import numpy as np
import pytest

from sigpipe.algorithms.picking.dispersion.curve import (
    max_resolvable_wavelength,
    min_resolvable_wavelength,
    pick_curves,
)
from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.dispersion_curve import Mode, VelocityType
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


def test_mode_labels() -> None:
    assert Mode("M", 0).label == "M0"
    assert Mode.from_label("R12") == Mode("R", 12)
    for label in ("M", "12", ""):
        with pytest.raises(ValueError, match="wave then a number"):
            Mode.from_label(label)
