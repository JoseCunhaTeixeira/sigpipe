"""MASW windows: sliding sub-arrays of receivers, each with the records that illuminate it.

PAC's windows (its adapters/windows.py): the same windows, the same shot selection; and the
records and traces a run leaves out (PACo's signal QC excludes them).
"""

from .building import Geometry, apply_exclusions, build_windows
from .models import Exclusions, MASWParameters, MASWWindow

__all__ = [
    "Exclusions",
    "Geometry",
    "MASWParameters",
    "MASWWindow",
    "apply_exclusions",
    "build_windows",
]
