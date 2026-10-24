"""A window's stage done again from outside the QC loop: nothing older of it left in place, kept
aside in its replaced/ folder and in the log's past (the log only grows), and the other
windows' and stages' history kept."""

import json
from datetime import UTC, datetime
from pathlib import Path

from sigpipe.masw.runs.history import (
    ATTEMPTS_FOLDER,
    CHANGES_FILE,
    LOG_FILE,
    LOG_VERSION,
    REPLACED_FOLDER,
    Change,
    forget,
    log_entries,
    read_changes,
    record_change,
    set_aside,
)
from sigpipe.masw.runs.stopping import commit, staging


def _run(tmp_path: Path) -> Path:
    """A run whose window xmid_5.00 was picked, inverted twice (the first attempt archived) and
    given a soil column, as the assistant leaves one; and a neighbour, xmid_6.50, inverted."""
    run = tmp_path / "run"
    for unit in ("xmid_5.00", "xmid_6.50"):
        window = run / unit
        window.mkdir(parents=True)
        (window / "DispersionImage_0000.hdf5").write_text("image")
        (window / "DispersionCurves_0000.csv").write_text("picks")
        (window / "quality.json").write_text("{}")
        (window / "SeismicInversion_Samples_0000.npz").write_text("models")
        (window / "SeismicInversion_Model_0000_best.csv").write_text("best")
    window = run / "xmid_5.00"
    (window / "PetroInversion_Model_0000.csv").write_text("soil")
    archive = window / ATTEMPTS_FOLDER / "1_inversion"
    archive.mkdir(parents=True)
    (archive / "SeismicInversion_Samples_0000.npz").write_text("first models")
    (run / "SeismicInversion_VelocitySection_0000.png").write_text("section")
    (run / "seismic_inversion_config.json").write_text("{}")
    lines = [
        {"unit": "xmid_5.00", "stage": "picking", "attempt": 1},
        {"unit": "xmid_5.00", "stage": "inversion", "attempt": 1},
        {"unit": "xmid_5.00", "stage": "inversion", "attempt": 2},
        {"unit": "xmid_5.00", "stage": "inversion", "attempt": 2, "results": {"G5": "pass"}},
        {"unit": "xmid_5.00", "stage": "petro_inversion", "attempt": 1},
        {"unit": "xmid_6.50", "stage": "inversion", "attempt": 1},
    ]
    (run / LOG_FILE).write_text("".join(json.dumps(line) + "\n" for line in lines))
    return run


def _logged(run: Path) -> list[tuple[str, str, int]]:
    """The attempts the log says now."""
    return [
        (entry["unit"], entry["stage"], entry["attempt"])
        for entry in log_entries(run)
        if "attempt" in entry
    ]


def _replaced(window: Path) -> Path:
    """The one folder `forget` set the window's replaced files aside in."""
    (folder,) = (window / REPLACED_FOLDER).iterdir()
    return folder


def test_an_inversion_done_again_forgets_the_windows_earlier_ones(tmp_path: Path) -> None:
    run = _run(tmp_path)
    window = run / "xmid_5.00"
    before = (run / LOG_FILE).read_text()

    forget(run, "xmid_5.00", "inversion", results=False, actor="agent")

    # Its archived attempt and its log lines forgotten; its picks, soil column (the
    # petrophysical inversion reads the picks, not the models) and its neighbour's untouched.
    assert not (window / ATTEMPTS_FOLDER).exists()
    assert _logged(run) == [
        ("xmid_5.00", "picking", 1),
        ("xmid_5.00", "petro_inversion", 1),
        ("xmid_6.50", "inversion", 1),
    ]
    assert (window / "SeismicInversion_Samples_0000.npz").exists()
    assert (run / "SeismicInversion_VelocitySection_0000.png").exists()
    # Kept all the same: the archive set aside, the log's lines there, a reset after them.
    archived = _replaced(window) / ATTEMPTS_FOLDER / "1_inversion"
    assert (archived / "SeismicInversion_Samples_0000.npz").read_text() == "first models"
    log = (run / LOG_FILE).read_text()
    assert log.startswith(before)
    reset = json.loads(log.removeprefix(before))
    assert {key: reset[key] for key in ("event", "version", "unit", "stages", "actor")} == {
        "event": "reset",
        "version": LOG_VERSION,
        "unit": "xmid_5.00",
        "stages": ["inversion"],
        "actor": "agent",
    }


def test_picks_changed_by_hand_erase_what_was_made_of_the_old_ones(tmp_path: Path) -> None:
    run = _run(tmp_path)
    window = run / "xmid_5.00"

    # The picks just saved kept; the picking's other results, and every later stage's.
    forget(run, "xmid_5.00", "picking", keep=("DispersionCurves_*.csv",))

    assert (window / "DispersionCurves_0000.csv").exists()
    assert (window / "DispersionImage_0000.hdf5").exists()
    left = {path.name for path in window.iterdir()}
    assert left == {"DispersionImage_0000.hdf5", "DispersionCurves_0000.csv", REPLACED_FOLDER}
    # What was made of the old picks, set aside.
    assert {path.name for path in _replaced(window).iterdir()} == {
        ATTEMPTS_FOLDER,
        "quality.json",
        "SeismicInversion_Samples_0000.npz",
        "SeismicInversion_Model_0000_best.csv",
        "PetroInversion_Model_0000.csv",
    }
    # The line's section made of its old model, gone; the run's configuration kept.
    assert not (run / "SeismicInversion_VelocitySection_0000.png").exists()
    assert (run / "seismic_inversion_config.json").exists()
    assert _logged(run) == [("xmid_6.50", "inversion", 1)]
    assert (run / "xmid_6.50" / "SeismicInversion_Samples_0000.npz").exists()


def test_new_results_replace_every_old_file_of_their_kind(tmp_path: Path) -> None:
    run = _run(tmp_path)
    window = run / "xmid_5.00"
    output = staging(window)
    (output / "SeismicInversion_Samples_0000.npz").write_text("new models")

    commit(window, replacing=("SeismicInversion_*",))

    # The old best model, which the new run did not write, gone with the rest.
    assert (window / "SeismicInversion_Samples_0000.npz").read_text() == "new models"
    assert not (window / "SeismicInversion_Model_0000_best.csv").exists()
    assert (window / "DispersionCurves_0000.csv").exists()


def test_the_log_says_what_came_after_each_reset(tmp_path: Path) -> None:
    run = _run(tmp_path)
    forget(run, "xmid_5.00", "inversion", results=False)
    with (run / LOG_FILE).open("a") as log:
        log.write(json.dumps({"unit": "xmid_5.00", "stage": "inversion", "attempt": 1}) + "\n")
        log.write('{"unit": "xmid_5.00", "stage": "inv')  # the last line, while written

    # The attempt logged after the reset is the window's inversion now.
    assert _logged(run) == [
        ("xmid_5.00", "picking", 1),
        ("xmid_5.00", "petro_inversion", 1),
        ("xmid_6.50", "inversion", 1),
        ("xmid_5.00", "inversion", 1),
    ]
    # Another kind of line is read as it is.
    with (run / LOG_FILE).open("a") as log:
        log.write("\n" + json.dumps({"event": "given", "family": "picking"}) + "\n")
    assert log_entries(run)[-1] == {"event": "given", "family": "picking"}


def test_a_change_in_pac_sets_aside_what_it_replaces(tmp_path: Path) -> None:
    run = _run(tmp_path)
    window = run / "xmid_5.00"

    # A curve picked again: the curves file copied aside (it holds every mode), still in place.
    curves = set_aside(window, "picking", copy=True)
    assert curves is not None and curves.startswith(f"{REPLACED_FOLDER}/")
    assert curves.endswith("_picking")
    assert (window / curves / "DispersionCurves_0000.csv").read_text() == "picks"
    assert (window / curves / "quality.json").exists()
    assert (window / "DispersionCurves_0000.csv").exists()
    # A window inverted again: its models moved aside, the archived attempts left to forget.
    models = set_aside(window, "inversion")
    assert models is not None and models.endswith("_inversion")
    assert {path.name for path in (window / models).iterdir()} == {
        "SeismicInversion_Samples_0000.npz",
        "SeismicInversion_Model_0000_best.csv",
    }
    assert not any(window.glob("SeismicInversion_*"))
    assert (window / ATTEMPTS_FOLDER / "1_inversion").is_dir()
    # Nothing to set aside: none.
    assert set_aside(window, "inversion") is None


def test_each_windows_changes_are_its_own_record(tmp_path: Path) -> None:
    run = _run(tmp_path)
    window = run / "xmid_5.00"
    lasso = Change(
        at=datetime(2026, 10, 24, 12, 4, tzinfo=UTC),
        stage="picking",
        how="lasso",
        mode="M0",
        inputs={"polygon": [[5.0, 300.0], [30.0, 300.0], [30.0, 200.0]]},
        result={"n_points": 12, "band_hz": [5.0, 30.0]},
        aside="replaced/x_picking",
        later=("inversion",),
    )
    inverted = Change(
        at=datetime(2026, 10, 24, 12, 9, tzinfo=UTC), stage="inversion", how="inversion"
    )
    record_change(window, lasso)
    record_change(window, inverted)
    with (window / CHANGES_FILE).open("a") as file:
        file.write('{"at": "2026-10-24T12:10')  # the last line, while written

    assert read_changes(window) == (lasso, inverted)
    assert read_changes(run / "xmid_6.50") == ()
    assert read_changes(window)[0].actor == "user"


def test_new_results_can_set_aside_the_old_ones(tmp_path: Path) -> None:
    run = _run(tmp_path)
    window = run / "xmid_5.00"
    output = staging(window)
    (output / "SeismicInversion_Samples_0000.npz").write_text("new models")

    kept = commit(window, replacing=("SeismicInversion_*",), aside="inversion")

    assert (window / "SeismicInversion_Samples_0000.npz").read_text() == "new models"
    assert not (window / "SeismicInversion_Model_0000_best.csv").exists()
    # The old models, kept where the history says.
    assert kept is not None and kept.endswith("_inversion")
    assert (window / kept / "SeismicInversion_Samples_0000.npz").read_text() == "models"
    assert (window / kept / "SeismicInversion_Model_0000_best.csv").exists()
    # Nothing to commit: nothing set aside.
    assert commit(window, aside="inversion") is None
