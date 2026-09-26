"""Pydantic models generated from sigpipe's functions, completed by the stage definitions."""

import inspect
import operator
import typing
from collections.abc import Callable, Iterable
from functools import reduce
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from sigpipe.masw.presets.stages import Parameter, Stage

# Parameters that carry no value to describe: the data (or self), and **params.
_SKIPPED_KINDS = {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}


class StrictModel(BaseModel):
    # Unknown keys are errors, not silently dropped: a typo in an override must reach its sender.
    model_config = ConfigDict(frozen=True, extra="forbid")


def stage_type(name: str, stage: Stage) -> tuple[Any, StrictModel]:
    """The field type of a preset stage and its default value.

    A stage with a choice of methods is a union of one model per method, told apart by `method`;
    a stage whose method the pipeline fixes is a single model with no `method` field.
    """
    models = {
        method: _method_model(name, method, function, stage)
        for method, function in stage.functions.items()
    }
    if not stage.selectable:
        (model,) = models.values()
        return model, model()

    if stage.none:
        none = create_model(
            f"{_title(name)}None", __base__=StrictModel, method=(Literal["none"], "none")
        )
        models = {"none": _register(none), **models}
    union = Annotated[reduce(operator.or_, models.values()), Field(discriminator="method")]
    return union, models[stage.default]()


def _method_model(
    stage_name: str, method: str, function: Callable[..., object], stage: Stage
) -> type[StrictModel]:
    signature = inspect.signature(function)
    hints = typing.get_type_hints(function)
    names = [
        parameter.name
        for parameter in list(signature.parameters.values())[1:]
        if parameter.kind not in _SKIPPED_KINDS and parameter.name not in stage.fixed
    ]
    known = stage.parameters.get(method, {})
    if unknown := sorted(set(known) - set(names)):
        raise TypeError(
            f"sigpipe's {stage_name} method '{method}' has no parameter {', '.join(unknown)}: "
            "update sigpipe/masw/presets/stages.py"
        )

    fields: dict[str, Any] = {}
    if stage.selectable:
        fields["method"] = (Literal[method], method)
    for name in names:
        fields[name] = _field(hints[name], signature.parameters[name].default, known.get(name))

    title = _title(stage_name) + (_title(method) if stage.selectable else "Parameters")
    return _register(
        create_model(title, __base__=StrictModel, __validators__=_order_checks(names), **fields)
    )


def _field(annotation: Any, sigpipe_default: Any, parameter: Parameter | None) -> tuple[Any, Any]:  # noqa: ANN401
    parameter = parameter or Parameter()
    if parameter.derived:
        annotation, default = annotation | None, None
    elif parameter.default is not None:
        default = parameter.default
    elif sigpipe_default is not inspect.Parameter.empty:
        default = sigpipe_default
    else:
        default = ...  # required: neither sigpipe nor the stage gives a default
    # The description is what a form or an agent shows of a value: its unit, and why it may be
    # null.
    notes = [parameter.unit, "null: from the profile" if parameter.derived else ""]
    return annotation, Field(
        default,
        ge=parameter.ge,
        gt=parameter.gt,
        le=parameter.le,
        description="; ".join(note for note in notes if note) or None,
    )


def _order_checks(names: Iterable[str]) -> dict[str, Any]:
    """A check that each <x>max is above its <x>min, for the pairs a method has."""
    present = set(names)
    pairs = [(f"{x}min", f"{x}max") for x in "tfv" if {f"{x}min", f"{x}max"} <= present]
    if not pairs:
        return {}

    def check_order(model: BaseModel) -> BaseModel:
        for low, high in pairs:
            low_value, high_value = getattr(model, low), getattr(model, high)
            if low_value is not None and high_value is not None and high_value <= low_value:
                raise ValueError(
                    f"{high} ({high_value:g}) must be greater than {low} ({low_value:g})"
                )
        return model

    return {"check_order": model_validator(mode="after")(check_order)}


def _register[M: type[BaseModel]](model: M) -> M:
    """Make a generated model a name of this module, as a written class would be.

    pickle finds classes by module and name, and run workers receive their preset pickled.
    """
    if globals().setdefault(model.__name__, model) is not model:
        raise TypeError(f"two generated models are named {model.__name__}")
    return model


def _title(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))
