import inspect
import json
import pickle
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from sigpipe.algorithms import (
    DISPERSION_METHODS,
    FILTERING_METHODS,
    STREAM_SELECTION_METHODS,
    WHITENING_METHODS,
)
from sigpipe.masw.presets import PresetError, generation, make_preset, override_schema
from sigpipe.masw.presets.generation import stage_type
from sigpipe.masw.presets.stages import WHITENING, Parameter, Stage, pac_methods


@pytest.fixture(autouse=True)
def forget_generated_models() -> Iterator[None]:
    """Models a test generates are registered in sigpipe.masw.presets.generation: remove them after."""
    before = set(vars(generation))
    yield
    for name in set(vars(generation)) - before:
        delattr(generation, name)


# Stand-ins for sigpipe functions: the data first, then keyword parameters.


def _fake(stream: object, *, alpha: float, beta: int = 3, **params: object) -> None:
    """A method with a required parameter, a defaulted one and **params."""


def _fake_band(
    stream: object, *, fmin: float, fmax: float, taper: float = 1_000.0, flip: bool = False
) -> None:
    """A method with a frequency band, like sigpipe's filters and whitenings."""


BAND = Stage(
    functions={"band": _fake_band},
    parameters={
        "band": {
            "fmin": Parameter("Hz", default=0.0, ge=0),
            "fmax": Parameter("Hz", derived=True, gt=0),
            "taper": Parameter("Hz", default=5.0, gt=0, le=50),
        }
    },
    fixed=frozenset({"flip"}),
)


def _validator(name: str, stage: Stage) -> TypeAdapter[Any]:
    field_type, _ = stage_type(name, stage)
    return TypeAdapter(field_type)


# ---------------------------------------------------------------- generation mechanics


def test_fields_come_from_the_signature() -> None:
    fake = _validator("fake", Stage(functions={"fake": _fake}))

    model = fake.validate_python({"method": "fake", "alpha": 1.5})

    # The data parameter and **params are skipped; beta keeps the function's own default.
    assert model.model_dump() == {"method": "fake", "alpha": 1.5, "beta": 3}
    with pytest.raises(ValidationError, match=r"alpha\n  Field required"):
        fake.validate_python({"method": "fake"})
    with pytest.raises(ValidationError, match=r"beta\n  Input should be a valid integer"):
        fake.validate_python({"method": "fake", "alpha": 1.5, "beta": "three"})


def test_stage_definitions_complete_the_signature() -> None:
    band = _validator("band", BAND).validate_python({"method": "band"})
    properties = type(band).model_json_schema()["properties"]

    # PAC's default replaces the function's, a derived value starts at None, a fixed one is hidden.
    assert band.model_dump() == {"method": "band", "fmin": 0.0, "fmax": None, "taper": 5.0}
    assert "flip" not in properties
    expected = {"description": "Hz", "exclusiveMinimum": 0, "maximum": 50}
    assert properties["taper"].items() >= expected.items()


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"taper": 60}, r"taper\n  Input should be less than or equal to 50"),
        ({"fmin": -1}, r"fmin\n  Input should be greater than or equal to 0"),
        ({"fmin": 10, "fmax": 5}, r"fmax \(5\) must be greater than fmin \(10\)"),
    ],
)
def test_bounds_and_min_max_order_are_checked(values: dict[str, Any], message: str) -> None:
    band = _validator("band", BAND)

    with pytest.raises(ValidationError, match=message):
        band.validate_python({"method": "band", **values})


def test_min_max_order_waits_for_derived_values() -> None:
    # fmax is still None: resolve_preset derives it later, and checks it then.
    band = _validator("band", BAND).validate_python({"method": "band", "fmin": 10})

    assert band.model_dump()["fmax"] is None


def test_a_stage_with_a_choice_of_methods_is_a_union_starting_at_none() -> None:
    field_type, default = stage_type("band", BAND)
    band = TypeAdapter(field_type)

    assert default.model_dump() == {"method": "none"}
    with pytest.raises(ValidationError, match="expected tags: 'none', 'band'"):
        band.validate_python({"method": "bnad"})
    with pytest.raises(ValidationError, match=r"none\.fmin\n  Extra inputs are not permitted"):
        band.validate_python({"method": "none", "fmin": 5})


def test_a_stage_with_a_fixed_method_has_no_method_field() -> None:
    stage = Stage(functions={"fake": _fake}, default="fake", none=False, selectable=False)
    fixed = replace(stage, parameters={"fake": {"alpha": Parameter(default=2.0)}})

    field_type, default = stage_type("fixed", fixed)

    assert default.model_dump() == {"alpha": 2.0, "beta": 3}
    assert "method" not in field_type.model_fields


# ---------------------------------------------------------------- drift from sigpipe


def test_a_parameter_sigpipe_renamed_fails_generation() -> None:
    renamed = replace(
        WHITENING, parameters={"onebit_apod": {"taper_width": Parameter("Hz", default=5.0)}}
    )

    with pytest.raises(
        TypeError,
        match=r"'onebit_apod' has no parameter taper_width: update sigpipe/masw/presets/stages\.py",
    ):
        # Its own name: the real whitening models are already registered.
        stage_type("drifted_whitening", renamed)


def test_a_method_sigpipe_dropped_fails_generation() -> None:
    with pytest.raises(
        TypeError,
        match=r"no method onebit_apd \(it has .*onebit_apod.*\): update sigpipe/masw/presets/stages\.py",
    ):
        pac_methods(WHITENING_METHODS, "onebit", "onebit_apd")


def test_two_generated_models_cannot_share_a_name() -> None:
    stage_type("band", BAND)

    with pytest.raises(TypeError, match="two generated models are named BandBand"):
        stage_type("band", BAND)


# ---------------------------------------------------------------- the presets PACo uses

PACS_METHODS = {
    "muting": {"none", "mute"},
    "filtering": {"none", "iir"},
    "selection": {"none", "fk"},
    "whitening": {"none", "onebit", "onebit_apod"},
    "normalization": {"none", "onebit"},
    "stacking": {"linear", "phase_weighted", "root"},
}


@pytest.mark.parametrize(
    ("name", "stages"),
    [
        ("active", ["muting", "filtering"]),
        (
            "passive",
            ["muting", "filtering", "selection", "whitening", "normalization", "stacking"],
        ),
    ],
)
def test_the_agent_sees_only_pacs_methods(name: str, stages: list[str]) -> None:
    # What the agent reads: each stage's choices are the keys of its discriminator mapping.
    schema = override_schema(name)
    offered = {
        stage: set(schema["properties"][stage]["discriminator"]["mapping"]) for stage in stages
    }

    assert offered == {stage: PACS_METHODS[stage] for stage in stages}
    text = json.dumps(schema).lower()
    for hidden in ("group", "ftan", "beamform", "savgol", "stft", "entropy"):
        assert hidden not in text


def test_pacs_defaults_replace_sigpipes_where_they_differ() -> None:
    def sigpipe_default(function: Any, parameter: str) -> object:  # noqa: ANN401
        return inspect.signature(function).parameters[parameter].default

    passive = make_preset(
        "passive", {"filtering": {"method": "iir"}, "whitening": {"method": "onebit_apod"}}
    ).model_dump()

    # sigpipe's taper comes from its ultrasonic use; PAC's form uses 5 Hz.
    assert sigpipe_default(WHITENING_METHODS["onebit_apod"], "taper_width_Hz") == 1_000
    assert passive["whitening"]["taper_width_Hz"] == 5.0
    # Where PAC gives no value, sigpipe's own default stays.
    assert passive["filtering"]["order"] == sigpipe_default(FILTERING_METHODS["iir"], "order")
    assert passive["dispersion"]["nv"] == sigpipe_default(DISPERSION_METHODS["phase"], "nv")


def test_parameters_the_pipeline_sets_are_hidden() -> None:
    fk = STREAM_SELECTION_METHODS["fk"]
    selection = make_preset("passive", {"selection": {"method": "fk"}}).model_dump()["selection"]

    assert "flip_negatives" in inspect.signature(fk).parameters
    assert "flip_negatives" not in selection
    with pytest.raises(
        PresetError,
        match=r"selection\.flip_negatives: unknown parameter\. Allowed: threshold, vmin, vmax\.",
    ):
        make_preset("passive", {"selection": {"method": "fk", "flip_negatives": False}})


def test_generated_presets_survive_pickle() -> None:
    # Run workers receive their preset pickled.
    preset = make_preset(
        "passive", {"whitening": {"method": "onebit_apod"}, "stacking": {"method": "root"}}
    )

    assert pickle.loads(pickle.dumps(preset)) == preset
