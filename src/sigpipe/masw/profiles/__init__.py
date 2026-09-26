"""Profile discovery and inspection.

A profile is a folder of the input directory holding seismic records, a receiver_positions.yaml
and, for active profiles only, a source_positions.yaml. As in PAC, every file that is not .yaml
or .json is a record. Records are read with sigpipe's Load, the same reader the processing
pipelines use, so any format obspy can read is accepted. Positions become sigpipe Coordinates.
"""

from .loading import (
    RECEIVER_POSITIONS_FILE,
    SOURCE_POSITIONS_FILE,
    ProfileError,
    list_profiles,
    load_profile,
)
from .models import MODES, ProcessingMode, Profile, ProfileKind, ProfileSummary, Record
from .summary import inspect_profile, summarize

__all__ = [
    "MODES",
    "RECEIVER_POSITIONS_FILE",
    "SOURCE_POSITIONS_FILE",
    "ProcessingMode",
    "Profile",
    "ProfileError",
    "ProfileKind",
    "ProfileSummary",
    "Record",
    "inspect_profile",
    "list_profiles",
    "load_profile",
    "summarize",
]
