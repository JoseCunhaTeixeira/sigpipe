"""Categories along the line, as PAC shows a petrophysical section's soils smoothed: each
category's signed distance to its edges taken along the line as a velocity section's Vs before
its Gaussian, the largest winning."""

import numpy as np

from sigpipe.masw.inversion.section import along_line, smoothed_categories, smoothed_positions

XS = np.arange(6) * 1.5  # the windows' middles
DZ = 0.1
DEPTHS = np.arange(200) * DZ  # each column 20 m, a label every 10 cm
SAND, CLAY = 0, 1


def _columns(boundaries: list[float]) -> np.ndarray:
    """Sand down to each window's boundary, clay under it."""
    return np.array([np.where(depth > DEPTHS, SAND, CLAY) for depth in boundaries])


def _boundary(labels: np.ndarray) -> np.ndarray:
    """Each column's first clay depth."""
    return np.array([DEPTHS[np.argmax(column == CLAY)] for column in labels])


def test_a_boundary_runs_straight_from_a_window_to_the_next() -> None:
    labels = _columns([3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    positions = smoothed_positions(XS)

    smoothed = smoothed_categories(labels, XS, positions, DZ)

    # Between the windows at 3 and 4.5 m, halfway between their 5 and 6 m: not a step at the
    # middle, one window's depth or the other's.
    middle = np.argmin(np.abs(positions - 3.75))
    assert abs(_boundary(smoothed)[middle] - 5.5) <= 2 * DZ
    # Down the line, deeper and deeper, never back up.
    assert np.all(np.diff(_boundary(smoothed)) >= 0)


def test_a_soil_one_window_alone_holds_does_not_spread() -> None:
    labels = _columns([6.0] * 6)
    labels[3, 40:60] = 2  # a lens 4 to 6 m down, in the fourth window only

    smoothed = smoothed_categories(labels, XS, smoothed_positions(XS), DZ)

    assert not (smoothed == 2).any()
    assert np.all(np.abs(_boundary(smoothed) - 6.0) <= DZ)


def test_a_soil_two_windows_hold_stays_and_ends_just_past_them() -> None:
    labels = _columns([6.0] * 6)
    labels[2:4, 40:60] = 2
    positions = smoothed_positions(XS)

    smoothed = smoothed_categories(labels, XS, positions, DZ)

    lens = (smoothed == 2).any(axis=1)
    assert lens[np.argmin(np.abs(positions - 3.0))] and lens[np.argmin(np.abs(positions - 4.5))]
    assert not lens[np.argmin(np.abs(positions - 1.5))]
    assert not lens[np.argmin(np.abs(positions - 6.0))]


def test_outside_the_columns_no_label() -> None:
    # A window 4 m shorter: under its column, none; the smoothed columns end between.
    labels = _columns([3.0] * 6)
    labels[5, 160:] = -1
    elevations = -DEPTHS
    reach = np.array([20.0] * 5 + [16.0])
    frame = along_line(XS, np.zeros(6), reach - DZ, elevations, 12.0)

    smoothed = smoothed_categories(labels, XS, frame.positions, DZ, frame.empty)

    assert np.all(smoothed[frame.empty] == -1)
    assert np.all(smoothed[~frame.empty] >= 0)
    # The last column ends between its window's 16 m and its neighbours' 20 m, smoothed as Vs.
    assert -20.0 < frame.floor[-1] < -16.0


def test_no_soil_shows_between_two_windows_that_neither_holds_there() -> None:
    # Clay then loam down to 7 m, over sand: between the third window and the fourth, at 5.8 m,
    # clay gives way to loam, and the sand 1.2 m below both never rises in between.
    labels = np.array([np.where(DEPTHS < 7.0, CLAY if x < 4.0 else 2, SAND) for x in XS])
    positions = smoothed_positions(XS, at_least=100)

    smoothed = smoothed_categories(labels, XS, positions, DZ)

    row = smoothed[:, 58]
    between = (positions > 3.0) & (positions < 4.5)
    assert set(row[between].tolist()) == {CLAY, 2}
