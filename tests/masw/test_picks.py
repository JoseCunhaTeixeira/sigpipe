"""A window's picked curves in PAC's layout, whichever picker made them, and PAC's lasso."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from sigpipe.algorithms.picking.dispersion.lasso import pick_lasso
from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.dispersion_curve import DispersionCurve, Mode, VelocityType
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.masw.picks import (
    CURVES_FILE,
    FIGURE_FILE,
    load_curves,
    remove_pick,
    save_pick,
)
from sigpipe.transformers import Pick

ACQUISITION = LinearAcquisition(
    source=Coordinate(0.0, 0.0, 0.0),
    receivers=tuple(Coordinate(2.0 + k, 0.0, 0.0) for k in range(24)),
)
FS = np.arange(5.0, 50.5, 0.5, dtype=np.float32)
VS = np.arange(50.0, 801.0, 1.0, dtype=np.float32)
TRUE_VS = 150.0 + 300.0 * np.exp(-(FS - 5.0) / 10.0)


@pytest.fixture
def image() -> DispersionImage:
    fv_map = np.exp(-(((VS[None, :] - TRUE_VS[:, None]) / 15.0) ** 2))
    return DispersionImage(
        fv_map=fv_map.astype(np.float32),
        fs=FS,
        vs=VS,
        type=VelocityType.PHASE,
        acquisition=ACQUISITION,
    )


def _curve(mode: Mode, scale: float = 1.0) -> DispersionCurve:
    return DispersionCurve(
        fs=FS[10:40],
        vs=TRUE_VS[10:40] * scale,
        mode=mode,
        acquisition=ACQUISITION,
        type=VelocityType.PHASE,
    )


def test_a_pick_replaces_its_modes_curve_and_keeps_the_others(
    image: DispersionImage, tmp_path: Path
) -> None:
    assert load_curves(tmp_path) is None

    assert not save_pick(tmp_path, image, _curve(Mode("M", 0)))
    assert not save_pick(tmp_path, image, _curve(Mode("M", 1), scale=1.5))
    assert save_pick(tmp_path, image, _curve(Mode("M", 0), scale=1.1))

    curves = load_curves(tmp_path)
    assert curves is not None
    by_mode = {curve.mode: curve for curve in curves}
    assert set(by_mode) == {Mode("M", 0), Mode("M", 1)}
    np.testing.assert_allclose(by_mode[Mode("M", 0)].vs, TRUE_VS[10:40] * 1.1, rtol=1e-5)
    assert (tmp_path / CURVES_FILE).exists() and (tmp_path / FIGURE_FILE).exists()


def test_removing_the_last_curve_removes_the_file(image: DispersionImage, tmp_path: Path) -> None:
    save_pick(tmp_path, image, _curve(Mode("M", 0)))
    save_pick(tmp_path, image, _curve(Mode("M", 1), scale=1.5))

    left = remove_pick(tmp_path, image, Mode("M", 1))
    assert left is not None and [curve.mode for curve in left] == [Mode("M", 0)]
    assert remove_pick(tmp_path, image, Mode("M", 0)) is None
    assert not (tmp_path / CURVES_FILE).exists()
    with pytest.raises(ValueError, match="No curve labelled 'M0'"):
        remove_pick(tmp_path, image, Mode("M", 0))


def test_the_lasso_picks_the_ridge_inside_it(image: DispersionImage) -> None:
    # Around the ridge from 10 to 40 Hz, 60 m/s on either side.
    upper = [(float(f), float(v) + 60) for f, v in zip(FS[10:61], TRUE_VS[10:61], strict=True)]
    lower = [(float(f), float(v) - 60) for f, v in zip(FS[10:61], TRUE_VS[10:61], strict=True)]
    polygon = [*lower, *reversed(upper)]

    picked = pick_lasso(image, polygon, Mode("M", 0))

    assert picked.dispersion_curves is not None
    (curve,) = picked.dispersion_curves
    assert curve.mode == Mode("M", 0)
    expected = np.interp(curve.fs, FS, TRUE_VS)
    np.testing.assert_allclose(curve.vs, expected, rtol=0.05)
    # Picked again, the mode's curve is replaced.
    again = pick_lasso(picked, polygon, Mode("M", 0))
    assert again.dispersion_curves is not None and len(again.dispersion_curves) == 1


def test_a_lasso_too_small_is_refused(image: DispersionImage) -> None:
    with pytest.raises(ValueError, match="at least 3 points"):
        pick_lasso(image, [(10.0, 200.0), (20.0, 300.0)], Mode("M", 0))
    with pytest.raises(ValueError, match="too small"):
        pick_lasso(image, [(10.0, 200.0), (10.0, 300.0), (10.0, 250.0)], Mode("M", 0))


def test_every_picker_runs_as_a_pick_method(image: DispersionImage) -> None:
    upper = [(float(f), float(v) + 60) for f, v in zip(FS[10:61], TRUE_VS[10:61], strict=True)]
    lower = [(float(f), float(v) - 60) for f, v in zip(FS[10:61], TRUE_VS[10:61], strict=True)]

    cases: list[tuple[Literal["maximum", "lasso", "tracking"], dict[str, object]]] = [
        ("maximum", {"fmins": [10.0], "fmaxs": [40.0], "labels": ["M"], "modes": [0]}),
        ("lasso", {"polygon": [*lower, *reversed(upper)], "mode": Mode("M", 0)}),
        ("tracking", {}),
    ]
    for method, params in cases:
        (picked,) = Pick(method=method, **params).transform([image])
        assert picked.dispersion_curves is not None, method
        (curve,) = [one for one in picked.dispersion_curves if one.mode.number == 0]
        expected = np.interp(curve.fs, FS, TRUE_VS)
        np.testing.assert_allclose(curve.vs, expected, rtol=0.1, err_msg=method)
