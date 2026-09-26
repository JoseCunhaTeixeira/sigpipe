"""The stages of each preset: PAC's choices on top of sigpipe's registries.

Method names, parameter names and parameter types come from the sigpipe functions themselves
(see generation.py). What their signatures cannot say lives here: which methods PAC uses, PAC's
form defaults, bounds, units, and the parameters a pipeline sets on its own.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from sigpipe.algorithms import (
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
    default: float | None = None  # PAC's form value; None keeps sigpipe's own default
    derived: bool = False  # left to None, then filled from the profile by resolve_preset
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
    """The methods PAC uses from a sigpipe registry. A method sigpipe drops fails here, at import."""
    if missing := [method for method in methods if method not in registry]:
        raise TypeError(
            f"sigpipe's registry has no method {', '.join(missing)} "
            f"(it has {', '.join(registry)}): update sigpipe/masw/presets/stages.py"
        )
    return {method: registry[method] for method in methods}


# Not in PAC: the correction of a trigger delay G1 measures (2026-09-24); 0 leaves the record
# as it is.
TRIGGER = Stage(
    functions=pac_methods(SHIFTING_METHODS, "shift"),
    parameters={"shift": {"t0": Parameter("s", default=0.0)}},
    default="shift",
    none=False,
    selectable=False,
)

MUTING = Stage(
    functions=pac_methods(MUTTING_METHODS, "mute"),
    parameters={
        "mute": {
            "tmin": Parameter("s", default=0.0, ge=0),
            "tmax": Parameter("s", derived=True, ge=0),
            "vmin": Parameter("m/s", default=0.0, ge=0),
            "vmax": Parameter("m/s", default=100_000.0, ge=0),
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

# PACo's passive defaults, not PAC's form's (0.1 s segments, no whitening, no normalization),
# which gave no curve on a real ambient-noise line (passive_p2: 0 of 27 trial windows of 11
# receivers passed G3; 2 s segments whitened and normalized one-bit, 19 of 27; 0.5, 1 and 5 s,
# 13 to 15; one of the two alone, 11 and 15; phase-weighted stacking, 10: 2026-09-26). A 0.1 s
# segment has a 10 Hz frequency step, and noise bursts outweigh the rest unless whitened.
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
            "vmin": Parameter("m/s", default=0.0, ge=0),
            "vmax": Parameter("m/s", default=100_000.0, gt=0),
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
    default="onebit",  # PACo's: see SLICING
)

NORMALIZATION = Stage(
    functions=pac_methods(NORMALIZATION_METHODS, "onebit"),
    default="onebit",  # PACo's: see SLICING
)

STACKING = Stage(
    functions=pac_methods(STREAM_STACKING_METHODS, "linear", "phase_weighted", "root"),
    # sigpipe: nu >= 0, n >= 1 (PAC's form allowed n = 0, which sigpipe rejects).
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

# In pipeline order.
ACTIVE_STAGES = {
    "trigger": TRIGGER,
    "muting": MUTING,
    "filtering": FILTERING,
    "dispersion": DISPERSION,
}
# Not in PAC: before a shot is correlated (passive-active), its surface-wave window only, the
# window of PACo's signal QC (2026-09-26): correlated whole, the demo's records gave images
# peaking at the grid's top velocity, noise common to every trace; the mute of `muting` would
# also blank the QC's noise window. The window follows the velocities alone, with no tmin or tmax.
CORRELATION_WINDOW = Stage(
    functions=pac_methods(MUTTING_METHODS, "mute"),
    parameters={
        "mute": {
            "vmin": Parameter("m/s", default=80.0, gt=0),
            "vmax": Parameter("m/s", default=1_500.0, gt=0),
            # Ramps of CORRELATION_TAPER_S (resolving.py), in the profile's samples.
            "taper": Parameter("samples", derived=True, ge=0),
        }
    },
    default="mute",
    fixed=frozenset({"tmin", "tmax"}),
)

# PAC's passive-active mode: an active profile's shots, each gather cross-correlated with the
# receiver nearest its shot, the correlations stacked (PAC's adapters/passive_active.py).
PASSIVE_ACTIVE_STAGES = {
    "trigger": TRIGGER,
    "muting": MUTING,
    "filtering": FILTERING,
    "correlation_window": CORRELATION_WINDOW,
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
