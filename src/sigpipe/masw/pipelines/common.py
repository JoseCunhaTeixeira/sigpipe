"""Pieces shared by the preprocessing and image pipelines."""

from pathlib import Path
from typing import Any

from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.masw.presets import ActivePreset, PassivePreset
from sigpipe.masw.profiles import Profile, Record
from sigpipe.masw.windows import MASWWindow
from sigpipe.transformers import Load

# A preprocessed record, as sigpipe's Save names a single stream.
PREPROCESSED = "Stream_0000.hdf5"


def record_folder(records_folder: Path, record: Record) -> Path:
    """Where a record's preprocessed stream and figure go: <records_folder>/<its file's stem>."""
    return records_folder / record.path.stem


def load_record(record: Record, profile: Profile) -> Load:
    """Load a whole record: every receiver of the profile."""
    # A passive record has no source: the first receiver stands in, as in build_windows.
    source = record.source if record.source is not None else profile.receivers[0]
    acquisition = LinearAcquisition(source=source, receivers=profile.receivers)
    return Load(file_paths=[record.path], acquisitions=[acquisition], data_type="seismic")


def load_preprocessed(window: MASWWindow, records_folder: Path) -> Load:
    """The window's records, preprocessed, cut to the window's receivers (each record its own,
    when some of its traces are excluded): what Load does on raw records with
    receivers_to_load."""
    paths = [records_folder / path.stem / PREPROCESSED for path in window.selected_files]
    return Load(
        file_paths=paths,
        data_type="stream",
        acquisitions=window.acquisitions,
        receivers_to_load=window.record_receivers or window.receiver_indices,
    )


def stage_kwargs(preset: ActivePreset | PassivePreset, stage: str) -> dict[str, Any]:
    """Keyword arguments of the sigpipe transformer running `stage` of `preset`.

    Values derived from the profile must have been filled in by resolve_preset: a None left over
    would otherwise reach sigpipe as a missing argument, deep inside a run.
    """
    kwargs: dict[str, Any] = getattr(preset, stage).model_dump()
    if missing := [name for name, value in kwargs.items() if value is None]:
        raise ValueError(
            f"{stage}.{missing[0]} is not set: resolve the preset against its profile "
            "(sigpipe.masw.presets.resolve_preset) before building pipelines."
        )
    return kwargs
