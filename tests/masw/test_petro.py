"""The petrophysical inversion of a line's windows and its sections, on three windows whose
picked fundamental mode lies in the bundled Silex model's range. Needs the silex and santiludo
extras."""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

# Found, not imported: keras needs the backend silex.py sets before its import.
if not all(importlib.util.find_spec(name) for name in ("santiludo", "keras", "keras_nlp")):
    pytest.skip("needs sigpipe's silex and santiludo extras", allow_module_level=True)

from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.dispersion_curve import DispersionCurve, Mode, VelocityType
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.masw.petro import (
    QUANTITIES,
    invert_line_petro,
    invert_window_petro,
    load_modeled_curve,
    load_petro_model,
    load_profile,
    save_line_sections,
)
from sigpipe.masw.petro.measuring import measure_petro
from sigpipe.masw.petro.section import (
    COMPARISON_STEM,
    ROCK_PHYSICS_FIGURE,
    SECTION_FIGURE,
    SECTION_FILE,
    comparison_grids,
    petro_grid,
    petro_models,
    rock_physics_grid,
    save_petro_section,
    save_petro_sections_file,
    save_rock_physics_file,
    save_rock_physics_section,
    smoothed_name,
)
from sigpipe.masw.petro.window import WINDOW_FIGURE
from sigpipe.masw.picks import save_pick

GRAND_EST = "grand_est_15-50hz_193-415mps"
FS = np.arange(10.0, 52.0, 1.0, dtype=np.float32)
VS = np.arange(100.0, 601.0, 2.0, dtype=np.float32)


def _window(run: Path, x: float, slowness: float, fs: np.ndarray = FS) -> str:
    """A window centred on `x` whose image, over `fs`, holds one ridge, picked as M0."""
    acquisition = LinearAcquisition(
        source=Coordinate(x - 6.0, 0.0, 0.0),
        receivers=tuple(Coordinate(x - 5.5 + k, 0.0, 0.0) for k in range(12)),
    )
    velocities = 210.0 + 190.0 * np.exp(-(fs - 10.0) / slowness)
    image = DispersionImage(
        fv_map=np.exp(-(((VS[None, :] - velocities[:, None]) / 15.0) ** 2)).astype(np.float32),
        fs=fs,
        vs=VS,
        type=VelocityType.PHASE,
        acquisition=acquisition,
    )
    curve = DispersionCurve(
        fs=fs,
        vs=velocities.astype(np.float32),
        mode=Mode("M", 0),
        acquisition=acquisition,
        vs_err=np.full(fs.size, 10.0, dtype=np.float32),
    )
    folder = run / f"xmid_{x:.2f}"
    folder.mkdir(parents=True)
    save_pick(folder, image, curve)
    return folder.name


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[str]]:
    """Three windows, each inverted."""
    run_folder = tmp_path_factory.mktemp("run")
    units = [
        _window(run_folder, x, slowness) for x, slowness in ((6.0, 8.0), (8.0, 10.0), (10.0, 12.0))
    ]
    for unit in units:
        invert_window_petro(run_folder / unit, GRAND_EST)
    return run_folder, units


def test_a_window_holds_its_model_curve_and_rock_physics(run: tuple[Path, list[str]]) -> None:
    run_folder, units = run
    folder = run_folder / units[0]

    model = load_petro_model(folder)
    modeled = load_modeled_curve(folder)
    profile = load_profile(folder, "vs")

    assert model is not None and model.position.x == pytest.approx(6.0)
    assert {str(soil) for soil in model.soils} <= {"clay", "loam", "silt", "sand"}
    assert modeled is not None and modeled.mode == Mode("R", 0)
    assert np.array_equal(modeled.fs, FS)
    assert profile is not None
    elevations, values = profile
    assert elevations[0] > elevations[-1]  # deepest last
    assert np.all(values > 0)
    assert load_profile(folder, "shear_modulus") is not None
    # Its figure, as PAC's card shows it: the curves and the soil column.
    assert (folder / WINDOW_FIGURE).stat().st_size > 10_000


def test_the_line_gives_every_section(run: tuple[Path, list[str]]) -> None:
    run_folder, units = run

    section = petro_models(run_folder, units)
    assert section is not None
    grid = petro_grid(section)
    assert grid.positions.tolist() == [6.0, 8.0, 10.0]
    assert grid.n_grid.shape == (3, grid.elevations.size) and grid.elevations.size <= 200
    # Smoothed along the line: more columns, from the first window's middle to the last's.
    smooth = petro_grid(section, lateral_smoothing=True, window_m=12.0)
    assert smooth.positions[0] == 6.0 and smooth.positions[-1] == 10.0
    assert len(smooth.soil_grid) == smooth.positions.size > 3
    assert {str(soil) for column in smooth.soil_grid for soil in column} <= {
        "",
        *(str(soil) for column in grid.soil_grid for soil in column),
    }
    assert smooth.n_grid.shape == (smooth.positions.size, smooth.elevations.size)
    # Refocused on the depths with data: its bottom smoothed shallower than the deepest window's.
    assert smooth.elevations.size <= grid.elevations.size
    assert np.isfinite(smooth.n_grid[:, -1]).any()
    assert smooth.water_table_elevations.shape == smooth.positions.shape
    assert save_petro_section(run_folder, units) == run_folder / SECTION_FIGURE
    # And smoothed along the line, as Visualization's switch shows it.
    assert (run_folder / smoothed_name(SECTION_FIGURE)).exists()
    assert save_petro_sections_file(run_folder, units) == run_folder / SECTION_FILE
    with h5py.File(run_folder / SECTION_FILE) as file:
        assert {"x", "z", "soil", "N", "water_table_elevation"} <= set(file.keys())

    for quantity, found in QUANTITIES.items():
        rock = rock_physics_grid(run_folder, units, quantity)
        assert rock is not None and rock.values.shape == (3, rock.elevations.size)
        smooth_rock = rock_physics_grid(run_folder, units, quantity, True, window_m=12.0)
        assert smooth_rock is not None and smooth_rock.positions.size > 3
        assert smooth_rock.values.shape == (
            smooth_rock.positions.size,
            smooth_rock.elevations.size,
        )
        assert np.isfinite(smooth_rock.values[:, -1]).any()
        assert save_rock_physics_file(run_folder, units, quantity) == (
            run_folder / found.section_file
        )
    # Every quantity in one figure, as the windows' columns and smoothed.
    assert save_rock_physics_section(run_folder, units) == run_folder / ROCK_PHYSICS_FIGURE
    assert (run_folder / smoothed_name(ROCK_PHYSICS_FIGURE)).exists()

    comparison = comparison_grids(run_folder, units)
    assert comparison is not None and comparison.positions.tolist() == [6.0, 8.0, 10.0]


def test_a_windows_inversion_is_measured(run: tuple[Path, list[str]]) -> None:
    run_folder, units = run

    measures = measure_petro(run_folder / units[0], GRAND_EST, (0.5, 1.0, 2.0, 100.0))

    assert measures.silex_model == GRAND_EST
    assert measures.fit.misfit is not None and len(measures.fit.bands) == 3
    assert measures.soils and len(measures.soils) == len(measures.thicknesses_m)
    assert len(measures.ns) == len(measures.soils) and measures.water_table_m > 0
    # 100 m is below the soil column: left out.
    assert [depth for depth, _ in measures.vs_at_depths] == [0.5, 1.0, 2.0]
    assert all(vs > 0 for _, vs in measures.vs_at_depths)


def test_one_window_makes_no_section(run: tuple[Path, list[str]]) -> None:
    run_folder, units = run

    assert petro_models(run_folder, units[:1]) is None
    assert rock_physics_grid(run_folder, units[:1], "vs") is None
    assert comparison_grids(run_folder, units[:1]) is None


def test_a_curve_outside_the_models_range_is_refused(tmp_path: Path) -> None:
    # 10 to 35 Hz: the model needs the curve to reach 43 Hz.
    unit = _window(tmp_path, 6.0, 8.0, fs=np.arange(10.0, 36.0, 1.0, dtype=np.float32))

    with pytest.raises(ValueError, match="does not cover Silex model grand_est"):
        invert_window_petro(tmp_path / unit, GRAND_EST)
    assert load_petro_model(tmp_path / unit) is None


def test_a_line_is_inverted_window_by_window(tmp_path: Path) -> None:
    units = [_window(tmp_path, x, 10.0) for x in (6.0, 8.0)]
    # Its curve ends at 35 Hz, short of the model's range: this window fails alone.
    units.append(_window(tmp_path, 10.0, 10.0, fs=np.arange(10.0, 36.0, 1.0, dtype=np.float32)))
    progress: list[tuple[int, int]] = []

    outcomes = invert_line_petro(
        tmp_path,
        units,
        GRAND_EST,
        workers=2,
        on_window=lambda done, total, _: progress.append((done, total)),
    )

    assert [outcome.unit for outcome in outcomes] == units
    assert [outcome.model is not None for outcome in outcomes] == [True, True, False]
    failed = outcomes[2]
    assert failed.error_type == "ValueError" and "does not cover" in (failed.message or "")
    assert progress == [(1, 3), (2, 3), (3, 3)]
    saved = save_line_sections(tmp_path, units)
    assert {path.name for path in saved} == {
        SECTION_FIGURE,
        SECTION_FILE,
        ROCK_PHYSICS_FIGURE,
        f"{COMPARISON_STEM}.png",
        *(found.section_file for found in QUANTITIES.values()),
    }
    # Beside them, each smoothed along the line, and the comparison by wavelength.
    assert {
        smoothed_name(SECTION_FIGURE),
        smoothed_name(ROCK_PHYSICS_FIGURE),
        f"{COMPARISON_STEM}_wavelength.png",
    } <= {path.name for path in tmp_path.iterdir()}
    # One window with a model: no section, and nothing raised.
    assert save_line_sections(tmp_path, units[:1]) == ()
