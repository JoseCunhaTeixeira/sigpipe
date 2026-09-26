"""The measures of a dispersion image and of an M0 pick on synthetic images, and the lateral
consistency of windows along a line."""

import math

import numpy as np

from sigpipe.algorithms.picking.dispersion.tracking import pick_modes
from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.dispersion_curve import VelocityType
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.masw.quality.image import (
    aliased,
    coherent_columns,
    competing_ridges,
    edge_peaks,
    noise_floor,
)
from sigpipe.masw.quality.line import Series, neighbourhoods, spread
from sigpipe.masw.quality.pick import measure_pick

ACQUISITION = LinearAcquisition(
    source=Coordinate(0.0, 0.0, 0.0),
    receivers=tuple(Coordinate(2.0 + k, 0.0, 0.0) for k in range(24)),
)
FS = np.arange(5.0, 60.5, 0.5)
VS = np.arange(10.0, 1_001.0, 1.0)
FLOOR = 1 / math.sqrt(24)


def _ridge(velocities: np.ndarray, height: float = 0.8) -> np.ndarray:
    centres = velocities[:, None]
    return height * np.exp(-0.5 * ((VS - centres) / (0.05 * centres)) ** 2)


def _image(fv_map: np.ndarray) -> DispersionImage:
    return DispersionImage(
        fv_map=FLOOR + fv_map, fs=FS, vs=VS, type=VelocityType.PHASE, acquisition=ACQUISITION
    )


M0 = 150 + 250 * np.exp(-FS / 15)


def test_a_ridge_is_coherent_where_it_rises_above_the_noise_floor() -> None:
    heights = np.where(FS < 30, 0.8, 0.05)
    image = _image(_ridge(M0) * heights[:, None])

    assert noise_floor(image) == FLOOR
    coherent = coherent_columns(image, level=0.3)
    np.testing.assert_array_equal(coherent, FS < 30)


def test_peaks_on_the_grids_top_count_only_where_the_window_resolves_velocity() -> None:
    # From 40 Hz the ridge sits just past the grid's top: its energy peaks on the edge. The
    # window (23 m) tells 1,000 m/s from infinity only above 1,000 / 23 = 43.5 Hz.
    image = _image(_ridge(np.where(FS >= 40, 1_010.0, M0)))
    coherent = coherent_columns(image, level=0.3)

    low, high = edge_peaks(image, coherent, edge_share=0.02)

    assert low == 0
    assert high == int(((FS >= 40) & (FS * 23 > 1_000)).sum())


def test_a_second_ridge_competes() -> None:
    image = _image(np.maximum(_ridge(M0), _ridge(1.6 * M0)))
    coherent = coherent_columns(image, level=0.3)

    competing = competing_ridges(image, coherent, ratio=0.7, separation=0.15)

    assert competing[coherent].all()
    assert not competing_ridges(_image(_ridge(M0)), coherent, ratio=0.7, separation=0.15).any()


def test_a_ridge_below_twice_the_spacing_in_wavelength_is_aliased() -> None:
    # M0 / 10 is 18 m/s at 30 Hz: a 0.6 m wavelength, under the 2 m the 1 m spacing resolves.
    slow = _image(_ridge(M0 / 10))
    fast = _image(_ridge(M0))

    assert aliased(slow, coherent_columns(slow, 0.3))[FS > 30].all()
    assert not aliased(fast, coherent_columns(fast, 0.3)).any()


def test_a_clean_ridges_pick_scores_as_a_plane_wave() -> None:
    image = _image(_ridge(M0))
    (m0,) = pick_modes(image)

    measures = measure_pick(image, m0)

    assert measures.n_points > 10
    assert measures.band_hz is not None
    assert measures.on_data is not None and measures.on_data > 0.9
    assert measures.constant_wavelength is not None and measures.constant_wavelength < 0.2
    assert measure_pick(image, None).n_points == 0
    assert measure_pick(image, None).sharpness is None


def test_an_isolated_window_is_an_outlier_and_a_shared_change_is_geology() -> None:
    x = np.linspace(1.0, 20.0, 20)

    def series(xmid: float, scale: float) -> Series:
        return Series(unit=f"xmid_{xmid:.2f}", xmid=xmid, x=x, values=200.0 * scale + 0 * x)

    # Six windows agree, the fourth is 50 % faster; then the last three change together.
    ordered = [series(float(i), 1.5 if i == 3 else 1.0) for i in range(7)]
    ordered += [series(float(i), 1.5) for i in range(7, 10)]

    found = neighbourhoods(ordered, neighbours=4, max_misfit=0.1, min_shared=5)

    assert found[3].standing == "outlier"
    assert found[3].where == "on both sides"
    assert found[7].standing == "shared_change"
    assert found[0].standing == "fits"
    assert spread([1.0, 1.0, 1.0]) == 0.0


def test_misfits_in_the_values_own_unit_and_neighbours_close_enough() -> None:
    def table(xmid: float, depth: float) -> Series:
        return Series(f"xmid_{xmid:.2f}", xmid, np.array([0.0]), np.array([depth]))

    # A water table 1 m deeper than its neighbours' 1.5 m: 67 % relative, one 1 m step.
    ordered = [table(x, 2.5 if x == 3 else 1.5) for x in (1.0, 2.0, 3.0, 4.0, 5.0)]

    relative = neighbourhoods(ordered, neighbours=4, max_misfit=0.5, min_shared=1)
    metres = neighbourhoods(ordered, 4, max_misfit=1.0, min_shared=1, relative=False)
    far = [table(x, 2.5 if x == 30 else 1.5) for x in (1.0, 2.0, 30.0, 58.0, 59.0)]
    apart = neighbourhoods(far, 4, max_misfit=0.5, min_shared=1, max_distance=3.0)

    assert relative[2].standing == "outlier"
    assert metres[2].standing == "fits" and metres[2].worst == 1.0
    # 28 m from any other window: nothing to compare it with.
    assert apart[2].standing == "no_neighbours" and not apart[2].sides
