"""Validation errors rewritten for their sender, a person in PAC or the agent in PACo: one line
per problem, saying what to send instead."""

import difflib
import json
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

_BOUNDS = {
    "greater_than": (">", "gt"),
    "greater_than_equal": (">=", "ge"),
    "less_than": ("<", "lt"),
    "less_than_equal": ("<=", "le"),
}
_TYPES = {
    "int_parsing": "an integer",
    "int_from_float": "an integer",
    "int_type": "an integer",
    "float_parsing": "a number",
    "float_type": "a number",
    "bool_parsing": "true or false",
    "bool_type": "true or false",
    "string_type": "a string",
    "list_type": "a list",
    "tuple_type": "a list",
}
_NOT_AN_OBJECT = {"model_type", "model_attributes_type", "dict_type"}
# Keys no override sets: `mode` comes from the preset name, `method` is listed as a method.
_HIDDEN_KEYS = {"mode", "method"}


@dataclass(frozen=True, slots=True)
class _Stages:
    """Overrides for preset `name`: its top-level keys are stages, some of other presets only."""

    name: str
    presets: Mapping[str, type[BaseModel]]


def explain(error: ValidationError, name: str, presets: Mapping[str, type[BaseModel]]) -> str:
    """`error`, raised while validating overrides for preset `name`, as its sender reads it."""
    stages = _Stages(name, presets)
    lines = [_explain(details, presets[name], "", stages) for details in error.errors()]
    return f"Invalid overrides for preset '{name}':\n" + "\n".join(f"- {line}" for line in lines)


def explain_parameters(error: ValidationError, model: type[BaseModel], argument: str) -> list[str]:
    """`error`, raised while validating a tool's `argument` as a `model`: one line per problem,
    as its sender reads it, each path starting with the argument's name."""
    return [_explain(details, model, argument, None) for details in error.errors()]


def _explain(
    error: ErrorDetails, root: type[BaseModel], argument: str, stages: _Stages | None
) -> str:
    relative, parent = _locate(root, error["loc"])
    path = ".".join(part for part in (argument, relative) if part)
    key = relative.rpartition(".")[2]
    # An item of a list (`vs_layers[0]`) is described by its list's field.
    field = parent.model_fields.get(key.partition("[")[0]) if parent is not None else None
    context: dict[str, Any] = error.get("ctx", {})
    kind = error["type"]
    got = _show(error["input"])

    if kind == "extra_forbidden":
        return _unknown_key(path, key, parent, stages)
    if kind in ("union_tag_invalid", "union_tag_not_found") and field is not None:
        methods = list(_methods(field.annotation))
        if kind == "union_tag_not_found":
            return f'{path}: missing "method". Methods: {", ".join(methods)}.'
        tag = str(context["tag"])
        return (
            f"{path}.method: unknown method '{tag}'. Allowed: {', '.join(methods)}."
            f"{_did_you_mean(tag, methods)}"
        )
    if kind in _BOUNDS:
        sign, bound = _BOUNDS[kind]
        return f"{path}: must be {sign} {context[bound]:g} (got {got})."
    if kind in _TYPES:
        return f"{path}: must be {_TYPES[kind]} (got {got})."
    if kind in _NOT_AN_OBJECT:
        return f"{path}: must be an object (got {got}).{_contents(field)}"
    if kind == "literal_error" and key == "mode" and stages is not None:
        return (
            f"mode: set by the preset name ('{root.model_fields['mode'].default}'); leave it out."
        )
    if kind == "missing":
        return f"{path}: required."
    if kind == "value_error":
        return f"{path}: {error['msg'].removeprefix('Value error, ')}."
    return f"{path}: {error['msg']}."


def _unknown_key(
    path: str, key: str, parent: type[BaseModel] | None, stages: _Stages | None
) -> str:
    fields = parent.model_fields if parent is not None else {}
    allowed = [field for field in fields if field not in _HIDDEN_KEYS]
    if stages is not None and parent is stages.presets[stages.name]:
        # A stage of another preset: say so, rather than suggest a look-alike name.
        owners = [other for other, preset in stages.presets.items() if key in preset.model_fields]
        if owners:
            return (
                f"{path}: not a stage of preset '{stages.name}', only of {', '.join(owners)}. "
                f"Stages: {', '.join(allowed)}."
            )
        return f"{path}: unknown stage. Allowed: {', '.join(allowed)}.{_did_you_mean(key, allowed)}"
    if not allowed and parent is not None:
        method = parent.model_fields["method"].default
        return f"{path}: unknown parameter; method '{method}' takes no parameters."
    return f"{path}: unknown parameter. Allowed: {', '.join(allowed)}.{_did_you_mean(key, allowed)}"


def _locate(
    root: type[BaseModel], loc: tuple[int | str, ...]
) -> tuple[str, type[BaseModel] | None]:
    """The path as its sender wrote it, and the model its last key belongs to.

    pydantic's locations include the method of a stage with a choice of methods
    (`filtering.iir.fmaxx`); the sender never wrote that part, so it is left out of the path. An
    item of a list is written `vs_layers[0]`.
    """
    keys: list[str] = []
    owner: type[BaseModel] | None = root
    parent: type[BaseModel] | None = root
    items: type[BaseModel] | None = None  # after a list of models: the model of its items
    index = 0
    while index < len(loc):
        part = loc[index]
        index += 1
        if isinstance(part, int) and keys:
            keys[-1] += f"[{part}]"
            owner, items = items, None
            continue
        key = str(part)
        keys.append(key)
        parent = owner
        field = owner.model_fields.get(key) if owner is not None else None
        methods = _methods(field.annotation) if field is not None else {}
        items = _items(field.annotation) if field is not None else None
        if index < len(loc) and loc[index] in methods:
            owner = methods[str(loc[index])]
            index += 1
        elif field is not None and _is_model(field.annotation):
            owner = field.annotation
        else:
            owner = None
    return ".".join(keys), parent


def _methods(annotation: Any) -> dict[str, type[BaseModel]]:  # noqa: ANN401
    """The models of a stage with a choice of methods, by method name; empty for anything else."""
    return {
        member.model_fields["method"].default: member
        for member in typing.get_args(annotation)
        if _is_model(member) and "method" in member.model_fields
    }


def _items(annotation: Any) -> type[BaseModel] | None:  # noqa: ANN401
    """The model of the items of a list or tuple of models; None for anything else."""
    if typing.get_origin(annotation) not in (list, tuple):
        return None
    members = {member for member in typing.get_args(annotation) if member is not Ellipsis}
    if len(members) != 1:
        return None
    (member,) = members
    return member if _is_model(member) else None


def _contents(field: Any) -> str:  # noqa: ANN401
    """What an object at `field` holds: its methods, or its parameters (of each item, for a
    list)."""
    if field is None:
        return ""
    if methods := _methods(field.annotation):
        return f" Methods: {', '.join(methods)}."
    model = field.annotation if _is_model(field.annotation) else _items(field.annotation)
    if model is not None:
        names = [name for name in model.model_fields if name not in _HIDDEN_KEYS]
        return f" Parameters: {', '.join(names)}."
    return ""


def _did_you_mean(word: str, options: list[str]) -> str:
    matches = difflib.get_close_matches(word, options, n=1, cutoff=0.6)
    return f" Did you mean {matches[0]}?" if matches else ""


def _show(value: Any) -> str:  # noqa: ANN401
    text = json.dumps(value) if isinstance(value, dict | list) else repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _is_model(annotation: Any) -> bool:  # noqa: ANN401
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)
