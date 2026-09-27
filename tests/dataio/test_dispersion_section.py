import numpy as np
import pytest

from sigpipe.base import DispersionCurve, Mode
from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.dispersion_curve import DispersionCurvesSection
from sigpipe.dataio.dispersion.section import pseudo_section_comparison_grids

FS = np.array([10.0, 20.0, 30.0])


def _section(velocity: float) -> DispersionCurvesSection:
    """Two windows, each a curve of one velocity at every frequency."""
    curves = []
    for start in (0.0, 5.0):
        acquisition = LinearAcquisition(
            source=Coordinate(start, 0.0, 0.0),
            receivers=tuple(Coordinate(start + i, 0.0, 0.0) for i in range(1, 5)),
        )
        curves.append(
            DispersionCurve(
                fs=FS, vs=np.full(3, velocity), mode=Mode("M", 0), acquisition=acquisition
            )
        )
    return DispersionCurvesSection(dispersion_curves=tuple(curves))


@pytest.mark.parametrize("along", ["frequency", "wavelength"])
def test_a_model_10_percent_faster_shows_the_same_residual_along_either_axis(along: str) -> None:
    picked, modelled = _section(300.0), _section(330.0)

    positions, ys, observed, predicted, residual = pseudo_section_comparison_grids(
        picked,
        modelled,
        along=along,  # pyright: ignore[reportArgumentType]
    )

    assert positions.size == 2 and np.all(np.diff(ys) > 0)
    # As many rows either way: 1 Hz steps over 10-30 Hz.
    assert ys.size == 21
    both = ~np.isnan(observed) & ~np.isnan(predicted)
    assert both.any()
    np.testing.assert_allclose(residual[both], (330 - 300) / 330 * 100, rtol=1e-5)
    if along == "wavelength":
        # From the picks' shortest wavelength (300 m/s at 30 Hz) to the model's longest.
        assert ys[0] == pytest.approx(10.0) and ys[-1] == pytest.approx(33.0)
        np.testing.assert_allclose(observed[~np.isnan(observed)], 300.0)
