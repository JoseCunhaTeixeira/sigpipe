"""A dispersion image as PAC shows it: on a figure of its own, or on axes given, a panel of a
larger figure."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh

from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.dispersion_curve import VelocityType
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.dataio.dispersion.plotting import draw_dispersion_image, plot_dispersion_image

FS = np.arange(5.0, 50.5, 0.5, dtype=np.float32)
VS = np.arange(50.0, 801.0, 1.0, dtype=np.float32)
RIDGE = 150.0 + 300.0 * np.exp(-(FS - 5.0) / 10.0)


def test_an_image_drawn_on_given_axes_as_on_its_own_figure() -> None:
    # A short window's image: a ridge over a high floor, its peak under 1.
    fv_map = 0.45 + 0.35 * np.exp(-(((VS[None, :] - RIDGE[:, None]) / 60.0) ** 2))
    image = DispersionImage(
        fv_map=fv_map.astype(np.float32),
        fs=FS,
        vs=VS,
        type=VelocityType.PHASE,
        acquisition=LinearAcquisition(
            source=Coordinate(0.0, 0.0, 0.0),
            receivers=tuple(Coordinate(2.0 + k, 0.0, 0.0) for k in range(5)),
        ),
    )
    figure, (panel, other) = plt.subplots(1, 2)
    own = plot_dispersion_image(image, normalize=True, lbmin=2.0, lbmax=12.0)

    mesh = draw_dispersion_image(panel, image, normalize=True, lbmin=2.0, lbmax=12.0)

    (drawn,) = [one for one in own.axes[0].collections if isinstance(one, QuadMesh)]
    for values in (mesh.get_array(), drawn.get_array()):
        rows = np.asarray(values).reshape(len(VS), len(FS)).T
        # Each frequency's row from 0 to 1: the floor dark, the ridge at the top of the scale.
        np.testing.assert_allclose(rows.min(axis=1), 0.0, atol=1e-6)
        np.testing.assert_allclose(rows.max(axis=1), 1.0, atol=1e-6)
    assert len(panel.lines) == len(own.axes[0].lines) == 2
    assert panel.get_xlim() == own.axes[0].get_xlim()
    assert panel.get_ylim() == own.axes[0].get_ylim()
    assert not other.collections
    plt.close(figure)
    plt.close(own)
