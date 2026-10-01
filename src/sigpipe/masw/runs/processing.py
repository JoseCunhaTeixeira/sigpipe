"""Processing a profile with a preset, in worker processes: each record preprocessed once, then
one sigpipe pipeline per MASW window on the preprocessed records."""

import hashlib
import json
import os
import secrets
import shutil
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import distribution, version
from pathlib import Path

import matplotlib

from sigpipe.masw.pipelines import build_image_pipeline, build_preprocessing_pipeline, record_folder
from sigpipe.masw.presets import ActivePreset, PassivePreset, make_preset, resolve_preset
from sigpipe.masw.profiles import Profile, Record, load_profile, summarize
from sigpipe.masw.profiles.loading import RECEIVER_POSITIONS_FILE, SOURCE_POSITIONS_FILE
from sigpipe.masw.runs.models import (
    InputFile,
    RecordOutcome,
    RunError,
    RunManifest,
    WindowOutcome,
)
from sigpipe.masw.runs.stopping import Stopped, check, commit, finished, staging, undo
from sigpipe.masw.runs.writing import write_atomic
from sigpipe.masw.windows import (
    Exclusions,
    Geometry,
    MASWParameters,
    MASWWindow,
    apply_exclusions,
    build_windows,
)
from sigpipe.masw.workspace import Workspace
from sigpipe.workers import one_thread_each

# Called with (windows done, windows in the run).
type ProgressCallback = Callable[[int, int], None]

RECORDS_FOLDER = "records"  # inside the run folder: one folder per preprocessed record
# How each mode's windows share their receivers once G1's exclusions are applied.
GEOMETRIES: dict[str, Geometry] = {
    "active": "per_record",
    "passive-active": "shared",
    "passive": "union",
}


def run_processing(
    profile: str,
    preset: str,
    overrides: Mapping[str, object] | None,
    workspace: Workspace,
    on_progress: ProgressCallback | None = None,
    packages: Sequence[str] = (),
    stop: threading.Event | None = None,
) -> RunManifest:
    """Process `profile` with `preset` and `overrides`, and write the run to disk: its manifest,
    which records the versions of sigpipe and of `packages` (the application's).

    Unknown names and invalid overrides raise before anything is written. A record or a window
    that fails does not stop the run: its error goes to run.json and to its folder's error.log,
    and a window whose record failed fails with it.

    `stop`, once set, stops the run at once (see stopping): a run stopped before any window
    succeeded is removed, and Stopped raised; otherwise the run keeps the windows that finished,
    its manifest says it was stopped, and Stopped carries the manifest.

    Records and windows run in worker processes, which Python 3.14 starts with forkserver: a
    script calling this function needs an `if __name__ == "__main__":` guard.
    """
    loaded = load_profile(profile, workspace)
    resolved = resolve_preset(make_preset(preset, overrides), loaded)
    windows = build_windows(loaded, resolved.masw)
    n_positions = count_positions(loaded, resolved.masw)
    if not windows:
        raise RunError(
            f"No window of profile '{profile}' has a valid shot: all {n_positions} positions "
            "were skipped. Widen masw.distance_min and masw.distance_max, or change masw.length."
        )

    check(stop)
    run_id, run_folder = new_run_folder(workspace.output_dir / profile)
    started_at = datetime.now(UTC)
    try:
        records = preprocess_records(resolved, loaded, run_folder, workspace.workers, stop=stop)
    except Stopped:
        shutil.rmtree(run_folder, ignore_errors=True)  # no window yet: nothing to keep
        raise Stopped() from None
    try:
        outcomes = process_windows(
            resolved, windows, records, run_folder, workspace.workers, on_progress, stop=stop
        )
    except Stopped as stopped:
        done: tuple[WindowOutcome, ...] = stopped.kept if isinstance(stopped.kept, tuple) else ()
        if not any(window.status == "succeeded" for window in done):
            shutil.rmtree(run_folder, ignore_errors=True)
            raise Stopped() from None
        manifest = write_manifest(
            run_id,
            run_folder,
            loaded,
            resolved,
            started_at,
            records,
            done,
            packages=packages,
            stopped=True,
        )
        raise Stopped(manifest) from None
    return write_manifest(
        run_id,
        run_folder,
        loaded,
        resolved,
        started_at,
        records,
        outcomes,
        packages=packages,
    )


def count_positions(profile: Profile, masw: MASWParameters) -> int:
    """The window positions along `profile`, before any is skipped for want of a shot."""
    return len(range(0, len(profile.receivers) - masw.length + 1, masw.step))


def write_manifest(
    run_id: str,
    run_folder: Path,
    profile: Profile,
    preset: ActivePreset | PassivePreset,
    started_at: datetime,
    records: tuple[RecordOutcome, ...],
    windows: tuple[WindowOutcome, ...],
    exclusions: Exclusions | None = None,
    packages: Sequence[str] = (),
    stopped: bool = False,
    inputs: Sequence[InputFile] = (),
) -> RunManifest:
    """The run's manifest, run.json: what was processed, with what, and how it went, with the
    versions of sigpipe and of `packages` (the application's) and the profile's files it read
    (`inputs`, as the run recorded them first; none, read now); `stopped`: on request, with the
    windows that had finished."""
    manifest = RunManifest(
        run_id=run_id,
        profile=summarize(profile),
        preset=preset,
        versions=package_versions(packages),
        inputs=tuple(inputs) or input_files(profile),
        started_at=started_at,
        finished_at=datetime.now(UTC),
        n_positions=count_positions(profile, preset.masw),
        records=records,
        windows=windows,
        exclusions=exclusions or Exclusions(),
        stopped=stopped,
    )
    write_atomic(run_folder / "run.json", manifest.model_dump_json(indent=2))
    return manifest


def input_files(profile: Profile) -> tuple[InputFile, ...]:
    """The files of `profile` a run reads, with their size and SHA-256: its records, then its
    position files."""
    folder = profile.folder
    paths = [record.path for record in profile.records] + [
        folder / name
        for name in (RECEIVER_POSITIONS_FILE, SOURCE_POSITIONS_FILE)
        if (folder / name).exists()
    ]
    return tuple(_input_file(path, folder) for path in paths)


def _input_file(path: Path, folder: Path) -> InputFile:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    name = path.relative_to(folder).as_posix() if path.is_relative_to(folder) else path.name
    return InputFile(name=name, bytes=path.stat().st_size, sha256=digest.hexdigest())


def new_run_folder(profile_folder: Path) -> tuple[str, Path]:
    """A new run ID, e.g. 20260923-142501-a3f9 (UTC time and a random suffix), and its folder."""
    while True:
        run_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        folder = profile_folder / run_id
        try:
            folder.mkdir(parents=True)
        except FileExistsError:
            continue  # same second and same suffix: draw again
        return run_id, folder


def preprocess_records(
    preset: ActivePreset | PassivePreset,
    profile: Profile,
    run_folder: Path,
    workers: int,
    presets: Mapping[str, ActivePreset | PassivePreset] | None = None,
    stop: threading.Event | None = None,
) -> tuple[RecordOutcome, ...]:
    """Every record of `profile` preprocessed once, into <run_folder>/records/<record>/; or, with
    `presets` (by record file name), those records only, each with its own preset (the signal
    QC's changes: a record's trigger delay is its own).

    `stop`, once set, stops them at once: the records not finished undone, Stopped raised with
    those that finished."""
    outcomes: dict[int, RecordOutcome] = {}
    one_thread_each()  # the workers are the cores the run takes
    with ProcessPoolExecutor(
        max_workers=workers, initializer=start_worker, initargs=(run_folder,)
    ) as executor:
        futures: dict[Future[float], tuple[int, Record, Path, bool]] = {}
        for index, record in enumerate(profile.records):
            if presets is not None and record.path.name not in presets:
                continue
            chosen = presets[record.path.name] if presets is not None else preset
            folder = record_folder(run_folder / RECORDS_FOLDER, record)
            created = not folder.exists()
            folder.mkdir(parents=True, exist_ok=presets is not None)
            future = executor.submit(_preprocess_record, chosen, record, profile, staging(folder))
            futures[future] = (index, record, folder, created)

        try:
            for future in finished(executor, futures, stop):
                index, record, folder, _ = futures.pop(future)
                commit(folder)
                duration_s, error = _finish(future, folder)
                outcomes[index] = RecordOutcome(
                    name=record.path.name,
                    folder=f"{RECORDS_FOLDER}/{folder.name}",
                    status="succeeded" if error is None else "failed",
                    duration_s=duration_s,
                    error=error,
                )
        except Stopped:
            for _, _, folder, created in futures.values():
                undo(folder, created)
            raise Stopped(tuple(outcomes[index] for index in sorted(outcomes))) from None
    return tuple(outcomes[index] for index in sorted(outcomes))


def process_windows(
    preset: ActivePreset | PassivePreset,
    windows: list[MASWWindow],
    records: tuple[RecordOutcome, ...],
    run_folder: Path,
    workers: int,
    on_progress: ProgressCallback | None = None,
    records_folder: Path | None = None,
    exclusions: Exclusions | None = None,
    stop: threading.Event | None = None,
) -> tuple[WindowOutcome, ...]:
    """One image pipeline per window, into `run_folder`, on the preprocessed records of
    `records_folder` (those of `run_folder` by default: trial windows read a run's records),
    without the records and traces of `exclusions`.

    A window that uses a record that failed fails at once, with the record's error. `stop`,
    once set, stops them at once: the windows not finished undone (a folder this call made,
    removed), Stopped raised with the outcomes of those that finished.
    """
    records_folder = records_folder or run_folder / RECORDS_FOLDER
    exclusions = exclusions or Exclusions()
    failed = {record.name: record.error for record in records if record.status == "failed"}
    outcomes: list[WindowOutcome] = []
    one_thread_each()  # the workers are the cores the run takes
    with ProcessPoolExecutor(
        max_workers=workers, initializer=start_worker, initargs=(run_folder,)
    ) as executor:
        futures: dict[Future[float], tuple[float, Path, bool]] = {}
        for built in windows:
            folder = run_folder / f"xmid_{built.xmid:.2f}"  # PAC's window folder name
            created = not folder.exists()
            folder.mkdir(exist_ok=True)  # exists when the stage is done again (PACo's QC)
            window = apply_exclusions(built, exclusions, GEOMETRIES[preset.mode])
            if window is None:
                error = "every record, or all but 2 of its receivers, excluded by the signal QC"
                (folder / "error.log").write_text(error + "\n")
                outcomes.append(
                    WindowOutcome(xmid=built.xmid, folder=folder.name, status="failed", error=error)
                )
                continue
            write_atomic(folder / "window.json", window.model_dump_json(indent=2))
            if missing := [path.name for path in window.selected_files if path.name in failed]:
                error = f"record {missing[0]} failed preprocessing: {failed[missing[0]]}"
                (folder / "error.log").write_text(error + "\n")
                outcomes.append(
                    WindowOutcome(
                        xmid=window.xmid, folder=folder.name, status="failed", error=error
                    )
                )
                continue
            future = executor.submit(
                _process_window, preset, window, records_folder, staging(folder)
            )
            futures[future] = (window.xmid, folder, created)

        if on_progress is not None:
            on_progress(len(outcomes), len(windows))
        try:
            for future in finished(executor, futures, stop):
                xmid, folder, _ = futures.pop(future)
                commit(folder)
                duration_s, error = _finish(future, folder)
                outcomes.append(
                    WindowOutcome(
                        xmid=xmid,
                        folder=folder.name,
                        status="succeeded" if error is None else "failed",
                        duration_s=duration_s,
                        error=error,
                    )
                )
                if on_progress is not None:
                    on_progress(len(outcomes), len(windows))
        except Stopped:
            for _, folder, created in futures.values():
                undo(folder, created)
            raise Stopped(tuple(sorted(outcomes, key=lambda outcome: outcome.xmid))) from None

    return tuple(sorted(outcomes, key=lambda outcome: outcome.xmid))


def _finish(future: Future[float], folder: Path) -> tuple[float | None, str | None]:
    """A finished task's duration, or its error, whose traceback goes to <folder>/error.log."""
    try:
        return future.result(), None
    except Exception as exc:
        (folder / "error.log").write_text("".join(traceback.format_exception(exc)))
        return None, f"{type(exc).__name__}: {exc}"


def start_worker(run_folder: Path) -> None:
    """Set up a worker process of a run: in the run folder, with no GUI backend."""
    # sigpipe's Pipeline.run creates a logs/ folder in the working directory and resets a global
    # logger: in a worker whose working directory is the run folder, both stay inside the run.
    os.chdir(run_folder)
    # Figures are only written to files: never start a GUI backend in a worker.
    matplotlib.use("Agg")


def _preprocess_record(
    preset: ActivePreset | PassivePreset, record: Record, profile: Profile, folder: Path
) -> float:
    """Runs in a worker: the record's preprocessing, returning its duration in seconds."""
    start = time.perf_counter()
    build_preprocessing_pipeline(preset, record, profile, folder).run(show_log=False)
    return time.perf_counter() - start


def _process_window(
    preset: ActivePreset | PassivePreset, window: MASWWindow, records_folder: Path, folder: Path
) -> float:
    """Runs in a worker: the window's image pipeline, returning its duration in seconds."""
    start = time.perf_counter()
    build_image_pipeline(preset, window, records_folder, folder).run(show_log=False)
    return time.perf_counter() - start


def package_versions(packages: Sequence[str] = ()) -> dict[str, str]:
    """The versions of sigpipe and of `packages`."""
    versions = {name: version(name) for name in ("sigpipe", *packages)}
    # sigpipe is installed from git, so its commit says more than its version number.
    direct_url = json.loads(distribution("sigpipe").read_text("direct_url.json") or "{}")
    if commit := direct_url.get("vcs_info", {}).get("commit_id"):
        versions["sigpipe"] += f" ({commit[:7]})"
    return versions
