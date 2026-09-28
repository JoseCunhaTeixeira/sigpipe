"""The Vs section on a grid, as PAC shows it: each window's column from its own ground down,
smoothed along the line over a share of a window's length or not."""

import numpy as np

from sigpipe.base.coordinate import Coordinate
from sigpipe.base.velocity_model import VelocityModel, VelocityModelsSection
from sigpipe.masw.inversion.section import informed_levels, velocity_grid

XS = np.arange(6) * 1.5  # the windows' middles


def _ground(x: float) -> float:
    """The ground steps down 4 m from the third window on."""
    return 10.0 if x < 3.0 else 6.0


def _model(x: float, vs: float | None = None, ground: float | None = None) -> VelocityModel:
    """Two layers, 5 m then 15 m thick, under the ground at `x`."""
    top = 200.0 + 10.0 * x if vs is None else vs
    return VelocityModel(
        vs_s=(top, 2 * top),
        vs_p=(1.8 * top, 3.6 * top),
        rhos=(1_800.0, 1_900.0),
        vs_s_std=(10.0, 20.0),
        thicknesses=(5.0, 15.0),
        position=Coordinate(x=float(x), y=0.0, z=_ground(x) if ground is None else ground),
    )


def test_each_window_starts_at_its_ground_smoothed_or_not() -> None:
    section = VelocityModelsSection(velocity_models=tuple(_model(x) for x in XS))
    grounds = [_ground(float(x)) for x in XS]
    for smoothing in (False, True):
        grid = velocity_grid(section, smoothing, window_m=3.0)
        for x, vs, std in zip(grid.positions, grid.vs, grid.vs_std, strict=True):
            # Smoothed, the ground runs straight from a window's middle to the next.
            ground = float(np.interp(x, XS, grounds))
            above = grid.elevations > ground + 0.2
            below = grid.elevations < ground - 0.2
            assert np.isnan(vs[above]).all() and np.isnan(std[above]).all(), (smoothing, x)
            assert not np.isnan(vs[below]).any(), (smoothing, x)


def _line(tops: np.ndarray) -> VelocityModelsSection:
    """Windows 1.5 m apart on flat ground, their top layer's Vs `tops`."""
    xs = np.arange(tops.size) * 1.5
    return VelocityModelsSection(
        velocity_models=tuple(_model(x, vs, 0.0) for x, vs in zip(xs, tops, strict=True))
    )


def test_an_alternation_from_window_to_window_evens_out() -> None:
    # Alternately 200 and 400 m/s from a window to the next.
    section = _line(np.where(np.arange(40) % 2, 400.0, 200.0))

    plain = velocity_grid(section)
    grid = velocity_grid(section, lateral_smoothing=True, window_m=12.0)

    assert np.ptp(plain.vs[:, 0]) == 200.0
    # Smoothed over a third of 12 m windows: their mean (away from the line's ends).
    inner = (grid.positions > 9.0) & (grid.positions < 49.5)
    np.testing.assert_allclose(grid.vs[inner, 0], 300.0, atol=10.0)


def test_one_odd_model_does_not_spread() -> None:
    section = _line(np.where(np.arange(40) == 20, 2_000.0, 300.0))

    grid = velocity_grid(section, lateral_smoothing=True, window_m=12.0)

    np.testing.assert_allclose(grid.vs[:, 0], 300.0)


def test_the_depth_informed_is_smoothed_as_the_section() -> None:
    section = VelocityModelsSection(velocity_models=tuple(_model(x) for x in XS))
    # Informed 5 m deep, but 12 m at the third window; the fifth's depth not known.
    depths = (5.0, 5.0, 12.0, 5.0, None, 5.0)
    windows = [(float(x), _ground(x), depth) for x, depth in zip(XS, depths, strict=True)]

    plain = informed_levels(velocity_grid(section), windows)
    grid = velocity_grid(section, lateral_smoothing=True, window_m=3.0)
    smoothed = informed_levels(grid, windows, lateral_smoothing=True, window_m=3.0)

    np.testing.assert_allclose(plain, [5.0, 5.0, -6.0, 1.0, np.nan, 1.0])
    nearest = np.abs(grid.positions[:, None] - XS[None, :]).argmin(axis=1)
    unknown = nearest == 4
    assert np.isnan(smoothed[unknown]).all() and not np.isnan(smoothed[~unknown]).any()
    grounds = np.interp(grid.positions, XS, [_ground(float(x)) for x in XS])
    informed = (grounds - smoothed)[~unknown]
    # The deep window softened along the line, as the section's Vs; never above the ground.
    assert informed.min() >= 5.0 and 5.5 < informed.max() < 12.0
