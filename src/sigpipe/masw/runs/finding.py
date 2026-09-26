"""Finding a run from its ID, and reading what it holds, for the tools that work on a run once
it is processed."""

import re
from pathlib import Path

from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.masw.runs.models import RunError, RunManifest
from sigpipe.masw.workspace import Workspace
from sigpipe.transformers import Load

# Only names shaped like run IDs are looked up, so an ID can never reach outside the output
# directory (e.g. "../x").
_RUN_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4}")
_LISTED_RUNS = 5
IMAGE_FILE = "DispersionImage_0000.hdf5"


def find_run(run_id: str, workspace: Workspace) -> Path:
    """The folder of run `run_id`, in `<output_dir>/<profile>/<run_id>/`."""
    matches = (
        sorted(workspace.output_dir.glob(f"*/{run_id}/run.json"))
        if _RUN_ID.fullmatch(run_id)
        else []
    )
    if not matches:
        latest = sorted(
            workspace.output_dir.glob("*/*/run.json"), key=lambda path: path.parent.name
        )
        listed = ", ".join(
            f"{path.parent.name} ({path.parent.parent.name})"
            for path in reversed(latest[-_LISTED_RUNS:])
        )
        raise RunError(f"Unknown run '{run_id}'. Latest runs: {listed or 'none'}.")
    return matches[0].parent


def list_runs(workspace: Workspace) -> list[str]:
    """Every run of the output directory, as <profile>/<run_id>, the newest first."""
    manifests = workspace.output_dir.glob("*/*/run.json")
    runs = [path.parent for path in manifests if _RUN_ID.fullmatch(path.parent.name)]
    runs.sort(key=lambda run: run.name, reverse=True)
    return [f"{run.parent.name}/{run.name}" for run in runs]


def window_folders(run_folder: Path) -> list[str]:
    """The window folders of a run, named xmid_<x> as PAC names them, sorted by position."""
    names = [
        path.name
        for path in run_folder.iterdir()
        if path.is_dir() and path.name.startswith("xmid_")
    ]
    return sorted(names, key=xmid_of)


def xmid_of(folder: str) -> float:
    """The position of window folder `folder` (xmid_<x>)."""
    return float(folder.removeprefix("xmid_"))


def load_manifest(run_id: str, workspace: Workspace) -> RunManifest:
    return RunManifest.model_validate_json((find_run(run_id, workspace) / "run.json").read_text())


def load_image(folder: Path) -> DispersionImage:
    """The dispersion image a window's pipeline saved in `folder`."""
    path = folder / IMAGE_FILE
    (image,) = Load(file_paths=[path], data_type="dispersion_image").transform()
    if not isinstance(image, DispersionImage):
        raise TypeError(f"{path} did not load as a dispersion image")
    return image
