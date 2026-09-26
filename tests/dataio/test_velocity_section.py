import numpy as np

from sigpipe.dataio.velocity_model.section import smooth_laterally


def test_lateral_smoothing_keeps_an_edge_in_place() -> None:
    # A step from 200 to 400 m/s between positions 4 and 5, at every depth.
    grid = np.repeat(np.where(np.arange(10) < 5, 200.0, 400.0)[:, None], 3, axis=1)
    column = smooth_laterally(grid)[:, 0]
    np.testing.assert_allclose(column, [200, 200, 200, 200, 250, 350, 400, 400, 400, 400])


def test_lateral_smoothing_skips_missing_values() -> None:
    grid = np.full((6, 2), 300.0)
    grid[2, 0] = np.nan
    np.testing.assert_allclose(smooth_laterally(grid), 300.0)
