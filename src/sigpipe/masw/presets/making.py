"""Picking a preset by name and applying overrides to it."""

from collections.abc import Mapping

from pydantic import ValidationError

from sigpipe.masw.presets.explaining import explain
from sigpipe.masw.presets.models import (
    ActivePreset,
    PassiveActivePreset,
    PassivePreset,
    PresetError,
)
from sigpipe.masw.profiles import ProcessingMode

PRESETS: dict[str, type[ActivePreset] | type[PassivePreset]] = {
    ProcessingMode.ACTIVE: ActivePreset,
    ProcessingMode.PASSIVE: PassivePreset,
    ProcessingMode.PASSIVE_ACTIVE: PassiveActivePreset,
}


def make_preset(
    name: str, overrides: Mapping[str, object] | None = None
) -> ActivePreset | PassivePreset:
    """The named preset with `overrides` applied.

    Overrides are a partial preset, e.g. {"filtering": {"method": "iir", "fmin": 5}}: stages
    and fields left out keep their defaults. Invalid overrides raise a PresetError listing each
    problem on its own line, with pydantic's ValidationError as its cause.
    """
    preset = PRESETS.get(name)
    if preset is None:
        raise PresetError(f"Unknown preset '{name}'. Available presets: {', '.join(PRESETS)}.")
    try:
        return preset.model_validate(dict(overrides or {}))
    except ValidationError as error:
        raise PresetError(explain(error, name, PRESETS)) from error


def apply_overrides[P: ActivePreset | PassivePreset](
    preset: P, overrides: Mapping[str, object]
) -> P:
    """`preset` with `overrides` applied on top of its values, for a stage done again.

    A stage given with another method is replaced whole; otherwise its fields are merged. The
    result is not resolved: derive its values against the profile again.
    """
    values = preset.model_dump()
    for stage, given in overrides.items():
        current = values.get(stage)
        if (
            isinstance(given, Mapping)
            and isinstance(current, dict)
            and given.get("method", current.get("method")) == current.get("method")
        ):
            values[stage] = {**current, **given}
        else:
            values[stage] = given
    try:
        return type(preset).model_validate(values)
    except ValidationError as error:
        raise PresetError(explain(error, preset.mode, PRESETS)) from error
