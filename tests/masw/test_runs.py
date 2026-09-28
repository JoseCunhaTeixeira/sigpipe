"""A synthetic line from its records to its velocity section: profiles, presets fitted to them,
runs in each mode, picks, and the inversion of two windows."""

import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest
from obspy import Stream as ObspyStream
from obspy import Trace
from obspy.core.util import AttribDict
from pydantic import ValidationError
from synthetic import N_RECEIVERS, SAMPLING, SOURCES

from sigpipe.algorithms.picking.dispersion.tracking import pick_modes
from sigpipe.dataio.stream import loading as stream_loading
from sigpipe.masw.inversion import (
    PARAMETERS_FILE,
    InversionParameters,
    WindowParameters,
    invert_window,
    load_parameters,
    load_spread,
)
from sigpipe.masw.inversion.measuring import measure_inversion
from sigpipe.masw.inversion.section import SECTION_FIGURE, save_comparison, save_section
from sigpipe.masw.picks import save_pick
from sigpipe.masw.pipelines.common import stage_kwargs
from sigpipe.masw.pipelines.preprocessing import build_preprocessing_pipeline
from sigpipe.masw.presets import PresetError, make_preset, resolve_preset
from sigpipe.masw.profiles import ProfileError, ProfileKind, list_profiles, load_profile
from sigpipe.masw.runs import (
    RunError,
    RunManifest,
    Stopped,
    find_run,
    load_image,
    load_manifest,
    run_processing,
)
from sigpipe.masw.runs.processing import preprocess_records
from sigpipe.masw.workspace import Folders
from sigpipe.transformers import Shift

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


def test_a_profile_whose_records_differ_is_refused(input_dir: Path, tmp_path: Path) -> None:
    # A line's records share their sampling rate and its receivers: one at another rate, or with
    # a trace fewer than the receivers, and the profile is refused, saying which.
    def record(path: Path, n_traces: int, sampling: float) -> None:
        traces = [
            Trace(
                np.zeros(100, dtype=np.float32),
                header={"sampling_rate": sampling, "station": f"R{i:02d}"},
            )
            for i in range(n_traces)
        ]
        ObspyStream(traces).write(str(path), format="MSEED")

    root = tmp_path / "input"
    for name, (n_traces, sampling) in {
        "rates": (N_RECEIVERS, SAMPLING / 2),
        "traces": (N_RECEIVERS - 1, SAMPLING),
    }.items():
        shutil.copytree(input_dir / "shots", root / name)
        record(root / name / "2.mseed", n_traces, sampling)
    workspace = Folders(input_dir=root, output_dir=tmp_path / "output", workers=1)

    with pytest.raises(ProfileError, match="share one sampling rate, found 500 Hz, 1000 Hz"):
        load_profile("rates", workspace)
    with pytest.raises(ProfileError, match=f"2.mseed \\(it has {N_RECEIVERS - 1} traces\\)"):
        load_profile("traces", workspace)


def test_a_preset_is_fitted_to_its_profile(workspace: Folders) -> None:
    shots = load_profile("shots", workspace)

    preset = resolve_preset(
        make_preset(
            "passive-active",
            {
                "muting": {"method": "mute", "vmin": 80.0, "vmax": 1500.0},
                "filtering": {"method": "iir"},
            },
        ),
        shots,
    )

    values = preset.model_dump()
    # A bound left out is none: no stand-in value in its place. The width, one sample.
    assert values["muting"]["tmin"] is None and values["muting"]["tmax"] is None
    assert values["muting"]["width"] == pytest.approx(1 / SAMPLING)
    assert values["filtering"]["fmax"] == pytest.approx(0.95 * SAMPLING / 2)
    assert "correlation_window" not in values  # removed: the muting's velocities cut the same
    # On, with no bound and no trigger (the synthetic files have none), it would cut nothing.
    with pytest.raises(PresetError, match="keeps everything"):
        resolve_preset(make_preset("passive-active", {"muting": {"method": "mute"}}), shots)
    # A width under one sample would empty the trace at the shot.
    narrow = {"muting": {"method": "mute", "vmax": 1500.0, "width": 0.0}}
    with pytest.raises(PresetError, match=r"muting\.width \(0 s\) must be at least one sample"):
        resolve_preset(make_preset("passive-active", narrow), shots)
    with pytest.raises(PresetError, match="does not fit passive profile"):
        resolve_preset(make_preset("active"), load_profile("noise", workspace))


# Settings that would stop a run midway, or leave a stage nothing: refused before it starts, each
# with what to change (the synthetic records: 12 receivers, 1 s at 1 kHz, no trigger in files).
REFUSED = [
    pytest.param(
        "shots",
        "active",
        {"trigger": {"t0": -0.01}},
        r"trigger\.t0: must be >= 0",
        id="a trigger before the record",
    ),
    pytest.param(
        "shots",
        "active",
        {"muting": {"method": "mute", "vmax": 0.0}},
        r"muting\.vmax: must be > 0",
        id="a fastest velocity of 0",
    ),
    pytest.param(
        "shots",
        "active",
        {"muting": {"method": "mute", "tmin": 0.5, "tmax": 0.2}},
        r"muting: tmax \(0\.2\) must be greater than tmin \(0\.5\)",
        id="a time window upside down",
    ),
    pytest.param(
        "shots",
        "active",
        {"muting": {"method": "mute", "vmin": 300.0, "vmax": 200.0}},
        r"muting: vmax \(200\) must be greater than vmin \(300\)",
        id="velocities upside down",
    ),
    pytest.param(
        "shots",
        "active",
        {"muting": {"method": "mute", "vmax": 1500.0, "width": 0.6}, "trigger": {"t0": 0.5}},
        r"The trigger \(0\.5 s\) and muting\.width \(0\.6 s\) exceed record 1\.mseed",
        id="the trigger and the width past the record",
    ),
    pytest.param(
        "shots",
        "passive-active",
        {"muting": {"method": "mute", "tmin": 0.9}, "trigger": {"t0": 0.2}},
        r"muting\.tmin \(0\.9 s\) is past the end of record 1\.mseed's data once moved by its "
        r"trigger \(0\.80 s\)",
        id="a window starting after the data",
    ),
    pytest.param(
        "shots",
        "active",
        {"masw": {"length": 13}},
        r"masw\.length \(13 receivers\) must not exceed the 12 receivers of profile 'shots'",
        id="a window longer than the line",
    ),
    pytest.param(
        "shots",
        "active",
        {"dispersion": {"fmin": 60.0, "fmax": 30.0}},
        r"dispersion: fmax \(30\) must be greater than fmin \(60\)",
        id="an image's band upside down",
    ),
    pytest.param(
        "shots",
        "active",
        {"filtering": {"method": "iir", "fmin": 5.0, "fmax": 20.0}, "dispersion": {"fmin": 30.0}},
        r"dispersion's band \(30-100 Hz\) is outside the 5-20 Hz the filtering and the whitening "
        r"keep",
        id="an image outside the filter's band",
    ),
    pytest.param(
        "noise",
        "passive",
        {"muting": {"method": "mute", "tmin": 1.0}},
        r"muting: not a stage of preset 'passive'",
        id="a passive line muted",
    ),
    pytest.param(
        "noise",
        "passive",
        {"selection": {"method": "fk", "vmin": 500.0, "vmax": 100.0}},
        r"selection: vmax \(100\) must be greater than vmin \(500\)",
        id="a selection's velocities upside down",
    ),
    pytest.param(
        "noise",
        "passive",
        {"whitening": {"method": "onebit_apod", "fmin": 50.0, "fmax": 20.0}},
        r"whitening: fmax \(20\) must be greater than fmin \(50\)",
        id="a whitening's band upside down",
    ),
]


@pytest.mark.parametrize(("profile", "mode", "overrides", "said"), REFUSED)
def test_settings_that_would_leave_nothing_are_refused(
    workspace: Folders, profile: str, mode: str, overrides: dict[str, object], said: str
) -> None:
    with pytest.raises(PresetError, match=said):
        resolve_preset(make_preset(mode, overrides), load_profile(profile, workspace))


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
    assert not list(folder.rglob(".partial"))  # every task's outputs moved into place
    assert all(
        (folder / record.folder / "Stream_0000.hdf5").exists() for record in manifest.records
    )


def test_a_records_trigger_comes_from_its_file(monkeypatch: pytest.MonkeyPatch) -> None:
    # A Geometrics SEG-2 file: its first sample 20 ms before the shot, on every trace.
    traces = [Trace(np.zeros(100, dtype=np.float32)) for _ in range(3)]
    for trace in traces:
        trace.stats.sampling_rate = 1000.0
        trace.stats.seg2 = AttribDict({"DELAY": "-0.020"})
    monkeypatch.setattr(stream_loading, "read_obspy", lambda *_, **__: ObspyStream(traces))

    header = stream_loading.read_seismic_header(Path("1.dat"))

    assert (header.n_traces, header.n_samples, header.trigger_s) == (3, 100, pytest.approx(0.02))
    traces[1].stats.seg2 = AttribDict({"DELAY": "0.000"})  # traces that disagree: none
    assert stream_loading.read_seismic_header(Path("1.dat")).trigger_s is None


def test_the_trigger_moves_a_shots_origin_with_the_muting_only(
    workspace: Folders, tmp_path: Path
) -> None:
    profile = load_profile("shots", workspace)
    record = profile.records[0].model_copy(update={"trigger_s": 0.02})  # as its file said
    windows = {"masw": {"length": 6, "step": 3}}

    def shift(overrides: dict[str, object]) -> float:
        preset = resolve_preset(make_preset("active", windows | overrides), profile)
        built = build_preprocessing_pipeline(preset, record, profile, tmp_path)
        (step,) = [step for step in built.steps if isinstance(step, Shift)]
        return float(step.params["t0"])

    # Off: the record as recorded; on, by its own trigger; a t0 given wins.
    assert shift({}) == 0.0
    assert shift({"muting": {"method": "mute", "vmax": 1500.0}}) == pytest.approx(0.02)
    mute = {"method": "mute", "vmax": 1500.0}
    assert shift({"muting": mute, "trigger": {"t0": 0.005}}) == pytest.approx(0.005)
    # A bound left out reaches sigpipe as none; a value the profile derives never does.
    preset = resolve_preset(make_preset("active", windows | {"muting": mute}), profile)
    assert stage_kwargs(preset, "muting")["tmin"] is None


@pytest.mark.parametrize(
    ("profile", "mode", "removed"),
    [
        # As an earlier sigpipe wrote a passive-active run: its surface-wave mute before
        # correlating.
        (
            "shots",
            "passive-active",
            {"correlation_window": {"method": "mute", "vmin": 80.0, "vmax": 1500.0, "taper": 50}},
        ),
        # And a passive run: its muting and its trigger.
        (
            "noise",
            "passive",
            {"muting": {"method": "mute", "tmin": 1.0, "width": 0.001}, "trigger": {"t0": None}},
        ),
    ],
)
def test_a_run_made_with_a_removed_stage_still_loads(
    workspace: Folders, profile: str, mode: str, removed: dict[str, object]
) -> None:
    manifest = run_processing(profile, mode, WINDOWS, workspace, packages=())
    path = find_run(manifest.run_id, workspace) / "run.json"
    older = json.loads(path.read_text())
    older["preset"] |= removed
    path.write_text(json.dumps(older))

    assert load_manifest(manifest.run_id, workspace) == manifest


def test_one_chain_is_refused_and_a_window_inverted_with_one_still_reads() -> None:
    # Two chains at least: one chain's halves agree even where two would settle apart.
    with pytest.raises(ValidationError, match="n_chains"):
        InversionParameters(n_chains=1)
    # A window inverted with one chain before 2026-09-28: its parameters read as they were.
    saved = WindowParameters.model_validate({"parameters": {"n_chains": 1}, "acceptance": [30.0]})
    assert saved.parameters.n_chains == 1


def test_a_stopped_run_keeps_the_windows_that_finished_and_nothing_half_written(
    input_dir: Path, tmp_path: Path
) -> None:
    workspace = Folders(input_dir=input_dir, output_dir=tmp_path / "output", workers=1)
    stop = threading.Event()

    # Stopped as the first window finishes: with one worker, the next is killed or never starts.
    with pytest.raises(Stopped) as stopped:
        run_processing(
            "shots",
            "active",
            {"masw": {"length": 6, "step": 1}},
            workspace,
            on_progress=lambda done, _: stop.set() if done >= 1 else None,
            stop=stop,
        )

    manifest = stopped.value.kept
    assert isinstance(manifest, RunManifest) and manifest.stopped
    assert load_manifest(manifest.run_id, workspace) == manifest
    folder = find_run(manifest.run_id, workspace)
    kept = {window.folder for window in manifest.windows}
    assert 1 <= len(kept) < manifest.n_positions
    assert {path.name for path in folder.glob("xmid_*")} == kept  # the others undone
    assert not list(folder.rglob(".partial"))
    assert all((folder / name / "DispersionImage_0000.hdf5").exists() for name in kept)


def test_a_run_stopped_before_any_window_leaves_nothing(workspace: Folders) -> None:
    stop = threading.Event()
    stop.set()

    with pytest.raises(Stopped) as stopped:
        run_processing("shots", "active", WINDOWS, workspace, stop=stop)

    assert stopped.value.kept is None
    assert not list(workspace.output_dir.glob("shots/*"))


def test_records_stopped_are_undone(workspace: Folders, tmp_path: Path) -> None:
    profile = load_profile("shots", workspace)
    preset = resolve_preset(make_preset("active", WINDOWS), profile)
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    stop = threading.Event()
    stop.set()

    with pytest.raises(Stopped) as stopped:
        preprocess_records(preset, profile, run_folder, 2, stop=stop)

    assert stopped.value.kept == ()
    assert not list((run_folder / "records").iterdir())


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
        # What the chains ran with, next to the models: each chain's acceptance, each value's
        # typical move.
        ran = load_parameters(folder / unit / PARAMETERS_FILE)
        assert ran.steps == pytest.approx(result.steps, rel=0.01)
        assert (ran.parameters.layering, len(ran.acceptance)) == ("fixed", 2)
        # The kept models' curves at the picked frequencies, as the density figure's band: their
        # 10th, 50th and 90th percentiles, for PAC's plot.
        (spread,) = load_spread(folder / unit).values()
        predicted = result.dpred[0]
        assert spread.fs.size == predicted.shape[1]
        assert spread.middle == pytest.approx(np.nanpercentile(predicted, 50, axis=0), rel=1e-5)
        assert (spread.low <= spread.middle).all() and (spread.middle <= spread.high).all()
        # The chains' agreement measured on Vs at the depths the curve resolves.
        measures = measure_inversion(folder / unit, parameters)
        assert measures.watched and set(measures.watched) <= set(measures.rhat)
        assert {"vs1", "vs2", "thick1", "noise"} <= set(measures.rhat)

    # The data choosing the layers: the bounds left out found from the curve, saved as run.
    free = InversionParameters(n_iterations=1_500, n_chains=2)
    invert_window(folder / units[0], free)
    window = load_parameters(folder / units[0] / PARAMETERS_FILE)
    ran = window.parameters
    assert ran.layering == "free" and ran.free.depth_max is not None
    # How its chains moved, saved with it: each move's step and acceptance, the exchanges.
    assert {"vs", "noise"} <= set(window.steps) and {"birth", "death"} <= set(window.moves)
    assert window.exchanges is not None
    measures = measure_inversion(folder / units[0], free)
    assert {"layers", "top_vs", "half_space_vs", "deepest_interface"} <= {
        share.parameter for share in measures.at_bounds
    }
    assert measures.depth_max_m == ran.bottom
    assert {"layers", "noise"} <= set(measures.rhat)

    assert save_section(folder, units) == folder / SECTION_FIGURE
    assert save_comparison(folder, units) is not None
    # A window inverted before the spread was saved has none.
    assert load_spread(tmp_path) == {}
