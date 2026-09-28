"""The Vs section on a grid, as PAC shows it: each window's column from its own ground down,
smoothed across windows or not."""

import numpy as np

from sigpipe.base.coordinate import Coordinate
from sigpipe.base.velocity_model import VelocityModel, VelocityModelsSection
from sigpipe.masw.inversion.section import velocity_grid

XS = np.arange(6) * 1.5  # the windows' middles


def _ground(x: float) -> float:
    """The ground steps down 4 m from the third window on."""
    return 10.0 if x < 3.0 else 6.0


def _model(x: float) -> VelocityModel:
    """Two layers, 5 m then 15 m thick, under the ground at `x`."""
    vs = 200.0 + 10.0 * x
    return VelocityModel(
        vs_s=(vs, 2 * vs),
        vs_p=(1.8 * vs, 3.6 * vs),
        rhos=(1_800.0, 1_900.0),
        vs_s_std=(10.0, 20.0),
        thicknesses=(5.0, 15.0),
        position=Coordinate(x=float(x), y=0.0, z=_ground(x)),
    )


def test_each_window_starts_at_its_ground_smoothed_or_not() -> None:
    section = VelocityModelsSection(velocity_models=tuple(_model(x) for x in XS))
    for smoothing in (False, True):
        grid = velocity_grid(section, smoothing)
        for x, vs, std in zip(grid.positions, grid.vs, grid.vs_std, strict=True):
            # The window a column shows, as the smoothed grid picks it: the nearest.
            ground = _ground(float(XS[np.abs(XS - x).argmin()]))
            above = grid.elevations > ground + 0.2
            below = grid.elevations < ground - 0.2
            assert np.isnan(vs[above]).all() and np.isnan(std[above]).all(), (smoothing, x)
            assert not np.isnan(vs[below]).any(), (smoothing, x)
