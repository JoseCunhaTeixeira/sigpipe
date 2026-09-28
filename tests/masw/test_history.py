"""A window's stage done again from outside the QC loop: nothing older of it left, and the other
windows' and stages' history kept."""

import json
from pathlib import Path

from sigpipe.masw.runs.history import ATTEMPTS_FOLDER, LOG_FILE, forget
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
    return [
        (entry["unit"], entry["stage"], entry["attempt"])
        for entry in map(json.loads, (run / LOG_FILE).read_text().splitlines())
    ]


def test_an_inversion_done_again_forgets_the_windows_earlier_ones(tmp_path: Path) -> None:
    run = _run(tmp_path)
    window = run / "xmid_5.00"

    forget(run, "xmid_5.00", "inversion", results=False)

    # Its archived attempt and its log lines gone; its picks, soil column (the petrophysical
    # inversion reads the picks, not the models) and its neighbour's untouched.
    assert not (window / ATTEMPTS_FOLDER).exists()
    assert _logged(run) == [
        ("xmid_5.00", "picking", 1),
        ("xmid_5.00", "petro_inversion", 1),
        ("xmid_6.50", "inversion", 1),
    ]
    assert (window / "SeismicInversion_Samples_0000.npz").exists()
    assert (run / "SeismicInversion_VelocitySection_0000.png").exists()


def test_picks_changed_by_hand_erase_what_was_made_of_the_old_ones(tmp_path: Path) -> None:
    run = _run(tmp_path)
    window = run / "xmid_5.00"

    # The picks just saved kept; the picking's other results, and every later stage's.
    forget(run, "xmid_5.00", "picking", keep=("DispersionCurves_*.csv",))

    assert (window / "DispersionCurves_0000.csv").exists()
    assert (window / "DispersionImage_0000.hdf5").exists()
    left = {path.name for path in window.iterdir()}
    assert left == {"DispersionImage_0000.hdf5", "DispersionCurves_0000.csv"}
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
