"""A window's history in a run: each stage's results at the top of its folder, in PAC's layout;
the results of the stage's earlier attempts (the assistant's retries) in
attempts/<n>_<stage>/; and the QC log's lines about them, one per state of an attempt, in the
run's qc_log.jsonl (PACo's). A stage done again from outside the QC loop, by hand in PAC or by
the assistant asked to, starts the window's history of it afresh: `forget` erases what came
before, so that nothing of an older result is left beside the new one."""

import fcntl
import json
import shutil
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

type Stage = Literal["preprocessing", "phase_shift", "picking", "inversion", "petro_inversion"]
STAGES: tuple[Stage, ...] = (
    "preprocessing",
    "phase_shift",
    "picking",
    "inversion",
    "petro_inversion",
)
ATTEMPTS_FOLDER = "attempts"
LOG_FILE = "qc_log.jsonl"
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
        "Selection_*.png",
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
) -> None:
    """Window `unit`'s history at `stage`, and with `later` at the stages after it that use its
    results, erased: the results of its earlier attempts (attempts/<n>_<stage>/), the QC log's
    lines of them, and with `results` the results at the top of its folder (but those matching
    `keep`) with the line's figures and sections made from them. `folder`: the unit's, when not
    a window's (a record's, in records/)."""
    stages = downstream(stage) if later else (stage,)
    window = folder if folder is not None else run_folder / unit
    archives = window / ATTEMPTS_FOLDER
    for one in stages:
        for archive in sorted(archives.glob(f"*_{one}")):
            shutil.rmtree(archive, ignore_errors=True)
        if results:
            for pattern in STAGE_FILES[one]:
                for path in window.glob(pattern):
                    if not any(path.match(kept) for kept in keep):
                        path.unlink(missing_ok=True)
            for pattern in LINE_FILES.get(one, ()):
                for path in run_folder.glob(pattern):
                    if path.is_file():
                        path.unlink(missing_ok=True)
    if archives.is_dir() and not any(archives.iterdir()):
        archives.rmdir()
    _forget_lines(run_folder, unit, set(stages))


def _forget_lines(run_folder: Path, unit: str, stages: set[str]) -> None:
    """The QC log without the lines of `unit`'s attempts at `stages`, rewritten whole (a crash
    leaves either log, never half of one)."""
    path = run_folder / LOG_FILE
    if not path.exists():
        return
    with log_lock(run_folder):
        lines = path.read_text().splitlines(keepends=True)
        kept = [line for line in lines if not _of(line, unit, stages)]
        if len(kept) == len(lines):
            return
        partial = path.with_name(path.name + ".partial")
        partial.write_text("".join(kept))
        partial.replace(path)


def _of(line: str, unit: str, stages: set[str]) -> bool:
    """Whether the log's `line` is about `unit`'s attempts at one of `stages`."""
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return False
    return isinstance(entry, dict) and entry.get("unit") == unit and entry.get("stage") in stages
