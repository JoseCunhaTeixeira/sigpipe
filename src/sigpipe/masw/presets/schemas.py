"""The JSON Schema of each preset's overrides, as PACo's agent reads it."""

import json
from typing import Any

from sigpipe.masw.presets.making import PRESETS
from sigpipe.masw.presets.models import PresetError


def override_schema(name: str) -> dict[str, Any]:
    """The overrides that may be sent for preset `name`.

    The preset's JSON Schema, without what only costs tokens: pydantic's titles repeat the field
    names, and `mode` comes from the preset name.
    """
    preset = PRESETS.get(name)
    if preset is None:
        raise PresetError(f"Unknown preset '{name}'. Available presets: {', '.join(PRESETS)}.")
    schema = without_titles(preset.model_json_schema())
    del schema["properties"]["mode"]
    return schema


def schema_size(schema: dict[str, Any]) -> int:
    """Characters of `schema` written compactly, as it travels to the model."""
    return len(json.dumps(schema, separators=(",", ":")))


def without_titles(node: Any) -> Any:  # noqa: ANN401
    """`node`, a JSON Schema or part of one, without pydantic's titles."""
    if isinstance(node, dict):
        # A title is a string; a property that happened to be named "title" would be a schema.
        return {
            key: without_titles(value)
            for key, value in node.items()
            if not (key == "title" and isinstance(value, str))
        }
    if isinstance(node, list):
        return [without_titles(value) for value in node]
    return node
