"""The petrophysical inversion of a line: each window in a worker process, a window that fails
kept to itself, then every section of the line. Stoppable: see sigpipe.masw.runs.stopping."""

import functools
import logging
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from sigpipe.base.petro_model import PetroModel
from sigpipe.masw.petro.section import (
    save_petro_section,
    save_petro_sections_file,
    save_rock_physics_file,
    save_rock_physics_section,
)
from sigpipe.masw.petro.window import QUANTITIES, invert_window_petro
from sigpipe.masw.runs import start_worker
from sigpipe.masw.runs.stopping import Stopped, commit, finished, staging, undo

logger = logging.getLogger(__name__)

# Called with (windows done, windows to invert, the outcome of the one just done).
type OnWindow = Callable[[int, int, "PetroOutcome"], None]


@dataclass(frozen=True, slots=True)
class PetroOutcome:
    """What the petrophysical inversion of one window folder gave."""

    unit: str
    model: PetroModel | None  # None when it failed
    duration_s: float | None = None
    error_type: str | None = None
    message: str | None = None
    traceback: str | None = None


def invert_line_petro(
    run_folder: Path,
    units: Sequence[str],
    model_name: str,
    workers: int,
    on_window: OnWindow | None = None,
    stop: threading.Event | None = None,
) -> tuple[PetroOutcome, ...]:
    """Each window folder of `units` inverted with the bundled Silex model `model_name`, in up to
    `workers` processes, in the order given. A window's files are replaced only once its
    inversion succeeded: one that fails keeps those it had.

    `stop`, once set, stops them at once: the windows not finished keep what they had, and
    Stopped is raised with the outcomes of those that finished."""
    outcomes: dict[str, PetroOutcome] = {}
    with ProcessPoolExecutor(
        max_workers=max(1, min(workers, len(units))),
        initializer=start_worker,
        initargs=(run_folder,),
    ) as executor:
        futures: dict[Future[tuple[PetroModel, float]], str] = {
            executor.submit(
                _invert_timed, run_folder / unit, model_name, staging(run_folder / unit)
            ): unit
            for unit in units
        }
        try:
            for future in finished(executor, futures, stop):
                unit = futures.pop(future)
                try:
                    model, duration_s = future.result()
                    commit(run_folder / unit)
                    outcome = PetroOutcome(unit, model, duration_s)
                except Exception as error:
                    undo(run_folder / unit, created=False)
                    outcome = PetroOutcome(
                        unit,
                        None,
                        error_type=type(error).__name__,
                        message=str(error),
                        traceback="".join(traceback.format_exception(error)),
                    )
                outcomes[unit] = outcome
                if on_window is not None:
                    on_window(len(outcomes), len(units), outcome)
        except Stopped:
            for unit in futures.values():
                undo(run_folder / unit, created=False)
            raise Stopped(tuple(outcomes[unit] for unit in units if unit in outcomes)) from None
    return tuple(outcomes[unit] for unit in units)


def _invert_timed(folder: Path, model_name: str, output_folder: Path) -> tuple[PetroModel, float]:
    """Runs in a worker: one window's inversion into `output_folder`, and its duration in
    seconds."""
    start = time.perf_counter()
    model = invert_window_petro(folder, model_name, output_folder)
    return model, time.perf_counter() - start


def save_line_sections(run_folder: Path, units: Sequence[str]) -> tuple[Path, ...]:
    """Every section of the line over the window folders of `units` holding a model: the soils
    and N values, the shear modulus and Vs, each a figure and an HDF5 file. Best effort: none
    with fewer than two windows, and one that fails is logged and left out."""
    saves: list[Callable[[], Path | None]] = [
        functools.partial(save_petro_section, run_folder, units),
        functools.partial(save_petro_sections_file, run_folder, units),
    ]
    for quantity in QUANTITIES:
        saves.append(functools.partial(save_rock_physics_section, run_folder, units, quantity))
        saves.append(functools.partial(save_rock_physics_file, run_folder, units, quantity))
    saved: list[Path] = []
    for save in saves:
        try:
            path = save()
        except Exception:
            logger.exception("A petrophysical section of %s could not be saved", run_folder)
            continue
        if path is not None:
            saved.append(path)
    return tuple(saved)
