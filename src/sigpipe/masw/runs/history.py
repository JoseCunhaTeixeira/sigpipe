"""A window's history in a run: each stage's results at the top of its folder, in PAC's layout;
the results of the stage's earlier attempts (the assistant's retries) in
attempts/<n>_<stage>/; and the QC log's lines about them, one per state of an attempt, in the
run's qc_log.jsonl (PACo's). A stage done again from outside the QC loop, by hand in PAC or by
the assistant asked to, starts the window's history of it afresh: `forget` sets aside what came
before, so that nothing of an older result is left beside the new one, and nothing is lost.

The log only grows: a line is never changed nor removed. Forgetting appends a `reset` event,
and the log's readers (`log_entries`) leave out the lines of a unit's stage logged before its
latest reset."""

import fcntl
import json
import shutil
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

type Stage = Literal["preprocessing", "phase_shift", "picking", "inversion", "petro_inversion"]
STAGES: tuple[Stage, ...] = (
    "preprocessing",
    "phase_shift",
    "picking",
    "inversion",
    "petro_inversion",
)
ATTEMPTS_FOLDER = "attempts"
# Where `forget` sets aside what a stage done afresh replaces, in a window's folder (or a
# record's): replaced/<time>_<stage>/.
REPLACED_FOLDER = "replaced"
LOG_FILE = "qc_log.jsonl"
# The version of the log's lines: 1, an attempt's line alone (lines without a version); 2, every
# line with its `event` and `version`, and the reset events.
LOG_VERSION = 2
type Actor = Literal["agent", "gate", "user"]
# Held while the QC log is written, by every writer.
LOCK_FILE = ".qc_log.lock"

# What each stage writes in a window's folder; the picking redraws the image's figure.
STAGE_FILES: dict[Stage, tuple[str, ...]] = {
    "preprocessing": (),  # per record, in records/<record>/: nothing in a window's folder
    "phase_shift": (
        "DispersionImage_0000.hdf5",
        "DispersionImage_0000.png",
        "Stream_*.png",
        "Stream_*.hdf5",
        "Spectrum_*",
        "Selection_*",
        "error.log",
    ),
    "picking": ("DispersionCurves_*.csv", "quality.json"),
    "inversion": ("SeismicInversion_*", "inversion_error.log"),
    "petro_inversion": ("PetroInversion_*",),
}
# What a stage's results make of the whole line, in the run's folder: its sections and figures,
# made again at the end of the stage's next run.
LINE_FILES: dict[Stage, tuple[str, ...]] = {
    "inversion": ("SeismicInversion_*",),
    "petro_inversion": ("PetroInversion_*",),
}


def downstream(stage: Stage) -> tuple[Stage, ...]:
    """`stage` and every stage after it that uses its results: the petrophysical inversion reads
    the picks, not the seismic inversion's models."""
    later = STAGES[STAGES.index(stage) :]
    if stage == "inversion":
        return tuple(one for one in later if one != "petro_inversion")
    return later


@contextmanager
def log_lock(run_folder: Path) -> Generator[None]:
    """The run's QC log held for one writer at a time, across processes: appending a line and
    rewriting the log without some never interleave."""
    with (run_folder / LOCK_FILE).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def forget(
    run_folder: Path,
    unit: str,
    stage: Stage,
    *,
    later: bool = True,
    results: bool = True,
    keep: Sequence[str] = (),
    folder: Path | None = None,
    actor: Actor = "user",
) -> None:
    """Window `unit`'s history at `stage`, and with `later` at the stages after it that use its
    results, forgotten by `actor`: the results of its earlier attempts (attempts/<n>_<stage>/)
    and with `results` the results at the top of its folder (but those matching `keep`) set
    aside in its replaced/<time>_<stage>/; the QC log's lines of them left behind by a reset
    event; the line's figures and sections made from them removed (made again from the windows).
    `folder`: the unit's, when not a window's (a record's, in records/)."""
    stages = downstream(stage) if later else (stage,)
    window = folder if folder is not None else run_folder / unit
    archives = window / ATTEMPTS_FOLDER
    aside = window / REPLACED_FOLDER / f"{datetime.now(UTC):%Y%m%d-%H%M%S-%f}_{stage}"
    for one in stages:
        for archive in sorted(archives.glob(f"*_{one}")):
            _set_aside(archive, aside / ATTEMPTS_FOLDER / archive.name)
        if results:
            for pattern in STAGE_FILES[one]:
                for path in sorted(window.glob(pattern)):
                    if not any(path.match(kept) for kept in keep):
                        _set_aside(path, aside / path.name)
            for pattern in LINE_FILES.get(one, ()):
                for path in run_folder.glob(pattern):
                    if path.is_file():
                        path.unlink(missing_ok=True)
    if archives.is_dir() and not any(archives.iterdir()):
        archives.rmdir()
    _reset(run_folder, unit, stages, actor)


def log_entries(run_folder: Path) -> list[dict[str, Any]]:
    """The QC log's lines as the run stands, in order: the lines of a unit's attempts at a stage
    logged before its latest reset left out; the resets and the lines of other kinds kept. A
    line that does not read (the last, while it is written) is skipped. A version 1 line (an
    attempt's, without `event`) reads as it did."""
    path = run_folder / LOG_FILE
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        entry = cast(dict[str, Any], entry)
        if entry.get("event") == "reset":
            unit, stages = entry.get("unit"), set(entry.get("stages") or ())
            entries = [
                one
                for one in entries
                if not (_attempt(one) and one.get("unit") == unit and one.get("stage") in stages)
            ]
        entries.append(entry)
    return entries


def _attempt(entry: dict[str, Any]) -> bool:
    """Whether a log line is an attempt's (one of its states: its stage run, a gate's verdict on
    it): it numbers the attempt."""
    return "attempt" in entry


def _reset(run_folder: Path, unit: str, stages: Sequence[Stage], actor: Actor) -> None:
    """A reset of `unit`'s `stages` appended to the QC log, when the run has one."""
    path = run_folder / LOG_FILE
    if not path.exists():
        return
    event = {
        "event": "reset",
        "version": LOG_VERSION,
        "unit": unit,
        "stages": list(stages),
        "at": datetime.now(UTC).isoformat(),
        "actor": actor,
    }
    with log_lock(run_folder), path.open("a") as log:
        log.write(json.dumps(event) + "\n")


def _set_aside(path: Path, target: Path) -> None:
    """`path` moved to `target`, its folders made."""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(path, target)
