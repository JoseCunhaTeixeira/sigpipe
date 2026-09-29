"""Processing runs: a profile processed with a preset, each record preprocessed once, then one
sigpipe pipeline per MASW window.

A run writes PAC's layout under <output_dir>/<profile>/<run_id>/: one xmid_<x>/ folder per
window, as PAC's UI expects, plus records/ with the preprocessed records, and run.json with
everything needed to understand or reproduce it.
"""

from .finding import (
    find_run,
    list_runs,
    load_image,
    load_manifest,
    window_folders,
    window_length,
    xmid_of,
)
from .models import RecordOutcome, RunError, RunManifest, WindowOutcome
from .processing import package_versions, run_processing, start_worker
from .stopping import Stopped

__all__ = [
    "RecordOutcome",
    "RunError",
    "RunManifest",
    "Stopped",
    "WindowOutcome",
    "find_run",
    "list_runs",
    "load_image",
    "load_manifest",
    "package_versions",
    "run_processing",
    "start_worker",
    "window_folders",
    "window_length",
    "xmid_of",
]
