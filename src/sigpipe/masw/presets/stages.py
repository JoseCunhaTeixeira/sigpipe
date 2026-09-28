"""The stages of each preset, on top of sigpipe's registries.

Method names, parameter names and parameter types come from the sigpipe functions themselves
(see generation.py). What their signatures cannot say lives here: which methods are offered,
defaults, bounds, units, and the parameters a pipeline sets on its own.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from sigpipe.algorithms import (
    DISPERSION_IMAGE_STACKING_METHODS,
    DISPERSION_METHODS,
    FILTERING_METHODS,
    MUTTING_METHODS,
    NORMALIZATION_METHODS,
    SEGMENTATION_METHODS,
    SHIFTING_METHODS,
    STREAM_SELECTION_METHODS,
    STREAM_STACKING_METHODS,
    WHITENING_METHODS,
)


@dataclass(frozen=True, slots=True)
class Parameter:
    """What a sigpipe signature cannot say about one parameter."""

    unit: str = ""
    default: float | None = None  # None keeps sigpipe's own default
    derived: bool = False  # left to None, then filled from the profile by resolve_preset
    # Left to None, what None means (no bound; each record's own): the value may stay None.
    null: str = ""
    ge: float | None = None
    gt: float | None = None
    le: float | None = None


@dataclass(frozen=True, slots=True)
class Stage:
    """One tunable stage of a preset, run by one sigpipe transformer."""

    functions: Mapping[str, Callable[..., object]]  # exposed method -> sigpipe function it runs
    parameters: Mapping[str, Mapping[str, Parameter]] = field(default_factory=dict)  # per method
    default: str = "none"  # the method presets start with
    none: bool = True  # "none", sigpipe's pass-through, is offered
    selectable: bool = True  # False: one method, set by the pipeline, and no `method` field
    fixed: frozenset[str] = frozenset()  # parameters the pipeline sets itself


def pac_methods(
    registry: Mapping[str, Callable[..., object]], *methods: str
) -> dict[str, Callable[..., object]]:
    """The methods exposed from a sigpipe registry. A method sigpipe drops fails here, at import."""
    if missing := [method for method in methods if method not in registry]:
        raise TypeError(
            f"sigpipe's registry has no method {', '.join(missing)} "
            f"(it has {', '.join(registry)}): update sigpipe/masw/presets/stages.py"
        )
    return {method: registry[method] for method in methods}


# The record's time origin moved to the shot, part of the muting (the user, 2026-09-28): applied
# with the muting on only. Left to None, each record's own trigger, from its file's header.
TRIGGER = Stage(
    functions=pac_methods(SHIFTING_METHODS, "shift"),
    parameters={"shift": {"t0": Parameter("s", null="each record's own, from its file")}},
    default="shift",
    none=False,
    selectable=False,
)

MUTING = Stage(
    functions=pac_methods(MUTTING_METHODS, "mute"),
    parameters={
        "mute": {
            # Each bound may be left out: no stand-in values (the user, 2026-09-28). The bounds
            # are those older runs' manifests hold (0 m/s, 100,000 m/s stood for none).
            "tmin": Parameter("s", null="none", ge=0),
            "tmax": Parameter("s", null="none", ge=0),
            "vmin": Parameter("m/s", null="none", ge=0),
            "vmax": Parameter("m/s", null="none", ge=0),
            # Kept after the slowest arrival: at the source, the shot's pulse. By default one
            # sample (the user, 2026-09-28): the window never empty at the shot.
            "width": Parameter("s", derived=True, ge=0),
            "taper": Parameter("samples", ge=0),
        }
    },
)

FILTERING = Stage(
    functions=pac_methods(FILTERING_METHODS, "iir"),
    parameters={
        "iir": {
            "fmin": Parameter("Hz", default=0.0, ge=0),
            "fmax": Parameter("Hz", derived=True, gt=0),
            "order": Parameter(gt=0),
        }
    },
)

# 2 s segments, whitened and normalized one-bit: a 0.1 s segment has a 10 Hz frequency step, and
# noise bursts outweigh the rest unless whitened.
SLICING = Stage(
    functions=pac_methods(SEGMENTATION_METHODS, "slice"),
    parameters={
        "slice": {
            "segment_duration": Parameter("s", default=2.0, gt=0),
            "segment_step": Parameter("s", default=2.0, gt=0),
        }
    },
    default="slice",
    none=False,
    selectable=False,
)

SELECTION = Stage(
    functions=pac_methods(STREAM_SELECTION_METHODS, "fk"),
    parameters={
        "fk": {
            "threshold": Parameter(default=0.1, ge=0, le=1),
            "vmin": Parameter("m/s", null="none", ge=0),
            "vmax": Parameter("m/s", null="none", gt=0),
        }
    },
    fixed=frozenset({"flip_negatives"}),
)

WHITENING = Stage(
    functions=pac_methods(WHITENING_METHODS, "onebit", "onebit_apod"),
    parameters={
        "onebit_apod": {
            "fmin": Parameter("Hz", default=0.0, ge=0),
            "fmax": Parameter("Hz", derived=True, gt=0),
            # sigpipe's default, 1000 Hz, comes from its ultrasonic use.
            "taper_width_Hz": Parameter("Hz", default=5.0, gt=0),
        }
    },
    default="onebit",  # see SLICING
)

NORMALIZATION = Stage(
    functions=pac_methods(NORMALIZATION_METHODS, "onebit"),
    default="onebit",  # see SLICING
)

STACKING = Stage(
    functions=pac_methods(STREAM_STACKING_METHODS, "linear", "phase_weighted", "root"),
    # sigpipe: nu >= 0, n >= 1.
    parameters={"phase_weighted": {"nu": Parameter(ge=0)}, "root": {"n": Parameter(ge=1)}},
    default="linear",
    none=False,
)

DISPERSION = Stage(
    functions=pac_methods(DISPERSION_METHODS, "phase"),
    parameters={
        "phase": {
            "fmin": Parameter("Hz", default=0.0, ge=0),
            "fmax": Parameter("Hz", default=100.0, gt=0),
            "vmin": Parameter("m/s", default=1.0, gt=0),
            "vmax": Parameter("m/s", default=1_000.0, gt=0),
            "nv": Parameter(gt=0),
        }
    },
    default="phase",
    none=False,
    selectable=False,
)

# The active mode's shot images of a window, stacked. The nth root keeps what every shot sees and
# damps what only a few do: on p2, long windows (24 receivers) passed G3 more often (17 against 13
# of 27), short ones (11) kept shorter wavelengths (30 against 35 m). Linear stays the default.
IMAGE_STACKING = Stage(
    functions=pac_methods(DISPERSION_IMAGE_STACKING_METHODS, "linear", "root"),
    # As sigpipe checks it: n of at least 1.
    parameters={"root": {"n": Parameter(ge=1)}},
    default="linear",
    none=False,
)

# In pipeline order.
ACTIVE_STAGES = {
    "trigger": TRIGGER,
    "muting": MUTING,
    "filtering": FILTERING,
    "dispersion": DISPERSION,
    "image_stacking": IMAGE_STACKING,
}
# The passive-active mode: an active profile's shots, each gather cross-correlated with the
# receiver nearest its shot, the correlations stacked.
PASSIVE_ACTIVE_STAGES = {
    "trigger": TRIGGER,
    "muting": MUTING,
    "filtering": FILTERING,
    "stacking": STACKING,
    "dispersion": DISPERSION,
}
PASSIVE_STAGES = {
    "muting": MUTING,
    "filtering": FILTERING,
    "slicing": SLICING,
    "selection": SELECTION,
    "whitening": WHITENING,
    "normalization": NORMALIZATION,
    "stacking": STACKING,
    "dispersion": DISPERSION,
}
