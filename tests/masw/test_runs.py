"""A synthetic line from its records to its velocity section: profiles, presets fitted to them,
runs in each mode, picks, and the inversion of two windows."""

from pathlib import Path

import numpy as np
import pytest
from synthetic import N_RECEIVERS, SAMPLING, SOURCES

from sigpipe.algorithms.picking.dispersion.tracking import pick_modes
from sigpipe.masw.inversion import InversionParameters, invert_window
from sigpipe.masw.inversion.section import SECTION_FIGURE, save_comparison, save_section
from sigpipe.masw.picks import save_pick
from sigpipe.masw.presets import PresetError, make_preset, resolve_preset
from sigpipe.masw.profiles import ProfileError, ProfileKind, list_profiles, load_profile
from sigpipe.masw.runs import RunError, find_run, load_image, load_manifest, run_processing
from sigpipe.masw.workspace import Folders

WINDOWS = {"masw": {"length": 6, "step": 3}}


def test_profiles_are_read_from_their_folders(workspace: Folders) -> None:
    assert list_profiles(workspace) == ["noise", "shots"]
    shots = load_profile("shots", workspace)
    assert shots.kind == ProfileKind.ACTIVE
    assert [record.path.name for record in shots.records] == sorted(SOURCES)
    assert len(shots.receivers) == N_RECEIVERS
    assert shots.sampling_rate_hz == SAMPLING
    assert load_profile("noise", workspace).kind == ProfileKind.PASSIVE
    with pytest.raises(ProfileError, match="Unknown profile 'nope'"):
        load_profile("nope", workspace)


def test_a_preset_is_fitted_to_its_profile(workspace: Folders) -> None:
    shots = load_profile("shots", workspace)

    preset = resolve_preset(
        make_preset(
            "passive-active", {"muting": {"method": "mute"}, "filtering": {"method": "iir"}}
        ),
        shots,
    )

    values = preset.model_dump()
    assert values["muting"]["tmax"] == pytest.approx(1.0, abs=0.01)  # the record's length
    assert values["filtering"]["fmax"] == pytest.approx(0.95 * SAMPLING / 2)
    assert values["correlation_window"]["taper"] == 50  # 50 ms at 1,000 Hz
    with pytest.raises(PresetError, match="does not fit passive profile"):
        resolve_preset(make_preset("active"), load_profile("noise", workspace))


@pytest.mark.parametrize(
    ("profile", "mode"), [("shots", "active"), ("shots", "passive-active"), ("noise", "passive")]
)
def test_every_mode_runs_into_pacs_layout(workspace: Folders, profile: str, mode: str) -> None:
    manifest = run_processing(profile, mode, WINDOWS, workspace, packages=())

    assert manifest.preset.mode == mode
    assert set(manifest.versions) == {"sigpipe"}
    assert [window.status for window in manifest.windows] == ["succeeded"] * 3
    assert [window.xmid for window in manifest.windows] == [2.5, 5.5, 8.5]
    folder = find_run(manifest.run_id, workspace)
    assert folder == workspace.output_dir / profile / manifest.run_id
    assert load_manifest(manifest.run_id, workspace) == manifest
    for window in manifest.windows:
        assert (folder / window.folder / "DispersionImage_0000.hdf5").exists()
    assert all(
        (folder / record.folder / "Stream_0000.hdf5").exists() for record in manifest.records
    )


def test_unknown_runs_are_refused(workspace: Folders) -> None:
    for run_id in ("20260923-000000-0000", "../shots", "*"):
        with pytest.raises(RunError, match="Unknown run"):
            find_run(run_id, workspace)


def test_the_shots_wave_is_picked_and_inverted_into_a_section(
    workspace: Folders, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # sigpipe's Pipeline.run writes a logs/ folder in the working directory.
    monkeypatch.chdir(tmp_path)
    manifest = run_processing("shots", "active", WINDOWS, workspace)
    folder = find_run(manifest.run_id, workspace)
    units = [window.folder for window in manifest.windows[:2]]
    for unit in units:
        image = load_image(folder / unit)
        (m0,) = pick_modes(image)
        assert m0.curve is not None
        # The wave moves out at 200 m/s at every frequency: the pick finds it.
        assert np.median(m0.curve.vs) == pytest.approx(200.0, rel=0.1)
        save_pick(folder / unit, image, m0.curve)

    parameters = InversionParameters.model_validate(
        {
            "n_layers": 2,
            "vs_layers": [{"vs_min": 100, "vs_max": 400}] * 2,
            "thickness_layers": [{"thickness_min": 1, "thickness_max": 5}],
            "n_iterations": 1_000,
            "n_chains": 2,
        }
    )
    for unit in units:
        result = invert_window(folder / unit, parameters)
        assert 100 <= result.median.vs_s[0] <= 400
        assert (folder / unit / "SeismicInversion_Model_0000_smooth_median.csv").exists()

    assert save_section(folder, units) == folder / SECTION_FIGURE
    assert save_comparison(folder, units) is not None
