"""Processing presets: the one schema of the processing settings, for the active, passive and
passive-active modes.

The preset models are generated from sigpipe's functions and restricted to what the pipelines
use (stages.py). A preset is picked by name and a few of its values overridden (make_preset), by
a person in PAC or the agent in PACo; the preset is then fitted to a profile, which fills the
values derived from the data (resolve_preset).
"""

from .explaining import explain_parameters
from .making import PRESETS, apply_overrides, make_preset
from .models import (
    ActivePreset,
    PassiveActivePreset,
    PassivePreset,
    Preset,
    PresetBase,
    PresetError,
)
from .resolving import IIR_FMAX_NYQUIST_FRACTION, STAGES, method_defaults, resolve_preset
from .schemas import override_schema, schema_size, without_titles

__all__ = [
    "IIR_FMAX_NYQUIST_FRACTION",
    "PRESETS",
    "STAGES",
    "ActivePreset",
    "PassiveActivePreset",
    "PassivePreset",
    "Preset",
    "PresetBase",
    "PresetError",
    "apply_overrides",
    "explain_parameters",
    "make_preset",
    "method_defaults",
    "override_schema",
    "resolve_preset",
    "schema_size",
    "without_titles",
]
