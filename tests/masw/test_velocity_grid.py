"""The Vs section on a grid, as PAC shows it: each window's column from its own ground down,
smoothed along the line over a share of a window's length or not."""

import numpy as np

from sigpipe.base.coordinate import Coordinate
from sigpipe.base.velocity_model import VelocityModel, VelocityModelsSection
from sigpipe.masw.inversion.section import (
    correlation_grid,
    informed_levels,
    interface_grid,
    uncertainty_grid,
    velocity_grid,
)
from sigpipe.masw.inversion.window import VsSpread

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


def test_each_column_runs_from_its_ground_to_its_models_depth_smoothed_or_not() -> None:
    section = VelocityModelsSection(velocity_models=tuple(_model(x) for x in XS))
    grounds = [_ground(float(x)) for x in XS]
    # The models built to 12 to 22 m: each column ends there, the half-space carried no deeper
    # (smoothed, where the depths smoothed as the Vs put it).
    depths = [12.0, 22.0, 15.0, 20.0, 12.0, 18.0]
    for smoothing in (False, True):
        grid = velocity_grid(section, smoothing, window_m=3.0, depths=depths)
        for x, vs, std, floor in zip(grid.positions, grid.vs, grid.vs_std, grid.floor, strict=True):
            # Smoothed, the ground runs straight from a window's middle to the next.
            ground = float(np.interp(x, XS, grounds))
            above = grid.elevations > ground + 0.2
            under = grid.elevations < floor - 0.2
            inside = (grid.elevations < ground - 0.2) & (grid.elevations > floor + 0.2)
            assert np.isnan(vs[above | under]).all() and np.isnan(std[above | under]).all()
            assert not np.isnan(vs[inside]).any(), (smoothing, x)
        plain_floor = [_ground(float(x)) - depth for x, depth in zip(XS, depths, strict=True)]
        if not smoothing:
            np.testing.assert_allclose(grid.floor, plain_floor)
        else:  # within the windows' own floors
            assert grid.floor.min() >= min(plain_floor) - 0.01
            assert grid.floor.max() <= max(plain_floor) + 0.01


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


def test_the_interfaces_are_drawn_from_each_ground() -> None:
    section = VelocityModelsSection(velocity_models=tuple(_model(x) for x in XS))
    # Every model an interface 2 to 2.5 m deep; the fifth window's shares not known.
    shares = tuple(1.0 if i == 4 else 0.0 for i in range(40))
    windows = [(float(x), _ground(x), () if i == 4 else shares) for i, x in enumerate(XS)]

    plain_grid = velocity_grid(section)
    plain = interface_grid(plain_grid, windows)
    grid = velocity_grid(section, lateral_smoothing=True, window_m=3.0)
    smoothed = interface_grid(grid, windows, lateral_smoothing=True, window_m=3.0)

    for column, x in zip(plain, XS, strict=True):
        depth = _ground(float(x)) - plain_grid.elevations
        if x == XS[4]:
            assert np.isnan(column).all()
            continue
        assert (column[(depth >= 2.0) & (depth < 2.5)] == 1.0).all()
        # Down to the shares' reach (40 of 0.5 m); nothing above the ground, nor below.
        reached = (depth >= 0.0) & (depth < 20.0)
        assert (column[reached & ((depth < 2.0) | (depth >= 2.5))] == 0.0).all()
        assert np.isnan(column[(depth < 0.0) | (depth >= 20.0)]).all()
    nearest = np.abs(grid.positions[:, None] - XS[None, :]).argmin(axis=1)
    assert np.isnan(smoothed[nearest == 4]).all()
    grounds = np.interp(grid.positions, XS, [_ground(float(x)) for x in XS])
    above = grid.elevations[None, :] > grounds[:, None] + 1e-3
    assert np.isnan(smoothed[above]).all() and np.nanmax(smoothed) <= 1.0


def _spread(narrow_to: float, bottom: float = 20.0) -> VsSpread:
    """An uncertainty U of 10 % down to `narrow_to` m, 75 % below it."""
    depths = (np.arange(int(bottom / 0.05)) + 0.5) * 0.05
    middle = np.full(depths.size, 300.0)
    half = np.where(depths < narrow_to, 30.0, 225.0)
    return VsSpread(depths, middle - half, middle, middle + half, np.full(depths.size, 1.0))


def test_the_uncertainty_is_drawn_from_each_ground_and_smoothed_as_vs() -> None:
    section = VelocityModelsSection(velocity_models=tuple(_model(x) for x in XS))
    grounds = [_ground(float(x)) for x in XS]
    spreads = [_spread(4.0 + i) for i in range(len(XS))]
    windows = [(float(x), g, s) for x, g, s in zip(XS, grounds, spreads, strict=True)]
    # The fourth window's spread not known: its column left without.
    windows[3] = (windows[3][0], windows[3][1], None)

    grid = velocity_grid(section, depths=[20.0] * len(XS))
    plain = uncertainty_grid(grid, windows)
    for i, ground in enumerate(grounds):
        column = plain[i]
        depth = ground - grid.elevations
        if i == 3:
            assert np.isnan(column).all()
            continue
        inside = ~np.isnan(column)
        # U of 10 % above the window's depth informed, 75 % below, from its ground.
        np.testing.assert_allclose(column[inside & (depth < 3.9 + i)], 0.1)
        np.testing.assert_allclose(column[inside & (depth > 4.1 + i)], 0.75)
        assert np.isnan(column[depth < 0]).all()

    smooth_grid = velocity_grid(section, True, window_m=3.0, depths=[20.0] * len(XS))
    smooth = uncertainty_grid(smooth_grid, windows, True, window_m=3.0)
    assert np.isnan(smooth[smooth_grid.outside()]).all()
    held = smooth[~smooth_grid.outside()]
    assert np.nanmin(held) >= 0.1 - 1e-6 and np.nanmax(held) <= 0.75 + 1e-6
    # The correlation length drawn alike: 1 m in every window's column.
    lengths = correlation_grid(grid, windows)
    assert np.nanmin(lengths) == np.nanmax(lengths) == 1.0 and np.isnan(lengths[3]).all()
