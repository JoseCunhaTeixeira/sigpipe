"""Reading a profile folder from disk into a validated Profile."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from sigpipe.base.coordinate import Coordinate
from sigpipe.dataio.stream.loading import read_seismic_header
from sigpipe.masw.profiles.models import Profile, ProfileKind, Record
from sigpipe.masw.workspace import Workspace

RECEIVER_POSITIONS_FILE = "receiver_positions.yaml"
SOURCE_POSITIONS_FILE = "source_positions.yaml"

# Files with these suffixes are metadata, not records.
_NON_RECORD_SUFFIXES = {".yaml", ".json"}

_MAX_LISTED = 5


class ProfileError(ValueError):
    """A profile is unknown or malformed. Messages say what to fix, for a person in PAC or the
    agent in PACo."""


def list_profiles(workspace: Workspace) -> list[str]:
    """Names of the profile folders in the input directory, sorted."""
    return sorted(
        path.name
        for path in workspace.input_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def load_profile(name: str, workspace: Workspace) -> Profile:
    available = list_profiles(workspace)
    # Also rejects names such as "../x": only direct sub-folders of input_dir are profiles.
    if name not in available:
        raise ProfileError(
            f"Unknown profile '{name}'. Available profiles: {', '.join(available) or 'none'}."
        )
    folder = workspace.input_dir / name

    record_paths = sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix not in _NON_RECORD_SUFFIXES
    )
    if not record_paths:
        raise ProfileError(
            f"Profile '{name}' has no record files (any file that is not .yaml or .json)."
        )

    receivers = _read_receivers(folder / RECEIVER_POSITIONS_FILE, name)
    sources = _read_sources(folder / SOURCE_POSITIONS_FILE, record_paths, name)

    records = tuple(
        _read_record(path, None if sources is None else sources[path.name], receivers, name)
        for path in record_paths
    )

    sampling_rates = sorted({record.sampling_rate_hz for record in records})
    if len(sampling_rates) > 1:
        raise ProfileError(
            f"Profile '{name}': all records must share one sampling rate, "
            f"found {', '.join(f'{rate:g} Hz' for rate in sampling_rates)}."
        )

    return Profile(
        name=name,
        kind=ProfileKind.PASSIVE if sources is None else ProfileKind.ACTIVE,
        folder=folder,
        records=records,
        receivers=receivers,
    )


def _read_record(
    path: Path,
    source: Coordinate | None,
    receivers: tuple[Coordinate, ...],
    profile: str,
) -> Record:
    # Its header, as sigpipe's loader reads the record (one read of the file), with its trigger.
    try:
        header = read_seismic_header(path)
    except Exception as exc:
        raise ProfileError(
            f"Profile '{profile}': sigpipe cannot load record {path.name} ({exc}). "
            f"{RECEIVER_POSITIONS_FILE} lists {len(receivers)} receivers."
        ) from exc
    if header.n_traces != len(receivers):
        raise ProfileError(
            f"Profile '{profile}': sigpipe cannot load record {path.name} (it has "
            f"{header.n_traces} traces). {RECEIVER_POSITIONS_FILE} lists {len(receivers)} "
            "receivers."
        )

    return Record(
        path=path,
        n_traces=header.n_traces,
        sampling_rate_hz=header.sampling_freq,
        duration_s=(header.n_samples - 1) / header.sampling_freq,
        source=source,
        trigger_s=header.trigger_s,
    )


def _read_receivers(path: Path, profile: str) -> tuple[Coordinate, ...]:
    if not path.exists():
        raise ProfileError(f"Profile '{profile}' has no {RECEIVER_POSITIONS_FILE}.")

    raw = _load_yaml(path, profile)
    if not isinstance(raw, list):
        raise ProfileError(
            f"Profile '{profile}': {RECEIVER_POSITIONS_FILE} must be a list of {{x, z}} entries."
        )

    receivers = tuple(_parse_coordinate(entry, RECEIVER_POSITIONS_FILE, profile) for entry in raw)

    if len(receivers) < 2:
        raise ProfileError(
            f"Profile '{profile}': {RECEIVER_POSITIONS_FILE} needs at least 2 receivers, "
            f"found {len(receivers)}."
        )
    xs = [receiver.x for receiver in receivers]
    if xs != sorted(xs):
        raise ProfileError(
            f"Profile '{profile}': receivers in {RECEIVER_POSITIONS_FILE} must be sorted by x."
        )
    return receivers


def _read_sources(
    path: Path, record_paths: Sequence[Path], profile: str
) -> dict[str, Coordinate] | None:
    # No source file means a passive profile.
    if not path.exists():
        return None

    raw = _load_yaml(path, profile)
    if not isinstance(raw, dict):
        raise ProfileError(
            f"Profile '{profile}': {SOURCE_POSITIONS_FILE} must map each record file name "
            "to its {x, z} source position."
        )

    record_names = {record_path.name for record_path in record_paths}
    listed_names = {str(key) for key in raw}
    if listed_names != record_names:
        problems = []
        if missing := sorted(record_names - listed_names):
            problems.append(f"no source position for {_short_list(missing)}")
        if unknown := sorted(listed_names - record_names):
            problems.append(f"positions for unknown records {_short_list(unknown)}")
        raise ProfileError(
            f"Profile '{profile}': {SOURCE_POSITIONS_FILE} does not match the record files: "
            f"{'; '.join(problems)}."
        )

    return {
        str(key): _parse_coordinate(value, f"{SOURCE_POSITIONS_FILE} entry '{key}'", profile)
        for key, value in raw.items()
    }


def _load_yaml(path: Path, profile: str) -> Any:  # noqa: ANN401
    try:
        with path.open(encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ProfileError(f"Profile '{profile}': {path.name} is not valid YAML ({exc}).") from exc


def _parse_coordinate(entry: object, where: str, profile: str) -> Coordinate:
    # Profiles are 2D lines in (x, z): y is ignored and set to 0.
    try:
        return Coordinate(x=float(entry["x"]), y=0.0, z=float(entry["z"]))  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as exc:
        raise ProfileError(
            f"Profile '{profile}': {where} needs numeric 'x' and 'z', got {entry!r}."
        ) from exc


def _short_list(names: Sequence[str]) -> str:
    shown = ", ".join(names[:_MAX_LISTED])
    extra = len(names) - _MAX_LISTED
    return f"{shown} and {extra} more" if extra > 0 else shown
