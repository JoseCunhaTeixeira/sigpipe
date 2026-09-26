"""Preset models, generated from sigpipe's functions and completed by the stages (stages.py).

A preset is a model built without arguments, from its defaults, and overrides are a partial dict
validated by the same model. Values left to None are derived from the profile by resolve_preset.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, create_model

from sigpipe.masw.presets.generation import StrictModel, stage_type
from sigpipe.masw.presets.stages import ACTIVE_STAGES, PASSIVE_ACTIVE_STAGES, PASSIVE_STAGES, Stage
from sigpipe.masw.profiles import ProcessingMode
from sigpipe.masw.windows import MASWParameters


class PresetError(ValueError):
    """A preset is unknown or does not fit a profile. Messages say what to fix, for a person in
    PAC or the agent in PACo."""


class PresetBase(StrictModel):
    """What code can rely on in any preset. Stages are generated: reach them by name."""

    mode: ProcessingMode
    masw: MASWParameters = MASWParameters()


# One type per stage, shared by the presets that use it.
_STAGE_TYPES = {
    name: stage_type(name, stage)
    for name, stage in (ACTIVE_STAGES | PASSIVE_STAGES | PASSIVE_ACTIVE_STAGES).items()
}


def _preset(
    name: str, mode: tuple[Any, ProcessingMode], stages: Mapping[str, Stage]
) -> type[PresetBase]:
    fields: dict[str, Any] = {"mode": mode}
    fields |= {stage: _STAGE_TYPES[stage] for stage in stages}
    return create_model(name, __base__=PresetBase, **fields)


if TYPE_CHECKING:
    # pyright cannot see generated fields: statically, every preset is a PresetBase.
    ActivePreset = PassivePreset = PassiveActivePreset = Preset = PresetBase
else:
    ActivePreset = _preset(
        "ActivePreset", (Literal[ProcessingMode.ACTIVE], ProcessingMode.ACTIVE), ACTIVE_STAGES
    )
    PassivePreset = _preset(
        "PassivePreset",
        (Literal[ProcessingMode.PASSIVE], ProcessingMode.PASSIVE),
        PASSIVE_STAGES,
    )
    PassiveActivePreset = _preset(
        "PassiveActivePreset",
        (Literal[ProcessingMode.PASSIVE_ACTIVE], ProcessingMode.PASSIVE_ACTIVE),
        PASSIVE_ACTIVE_STAGES,
    )
    Preset = Annotated[
        ActivePreset | PassivePreset | PassiveActivePreset, Field(discriminator="mode")
    ]
