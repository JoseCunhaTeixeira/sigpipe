"""Fitting a preset to a profile: the values derived from the acquisition, and the rules sigpipe
checks inside every window, checked once against the profile before any work."""

import contextlib
from typing import Any

from sigpipe.masw.presets.making import make_preset
from sigpipe.masw.presets.models import ActivePreset, PassivePreset, PresetError
from sigpipe.masw.presets.stages import ACTIVE_STAGES, PASSIVE_ACTIVE_STAGES, PASSIVE_STAGES, Stage
from sigpipe.masw.profiles import MODES, ProcessingMode, Profile

# A switched-on IIR filter without fmax stops just below the Nyquist frequency: sigpipe's filter
# requires fmax < Nyquist.
IIR_FMAX_NYQUIST_FRACTION = 0.95


def resolve_preset[P: ActivePreset | PassivePreset](preset: P, profile: Profile) -> P:
    """`preset` with every value left to None derived from `profile`, and checked against it.

    Every problem is reported at once, one line each, in a single PresetError.
    """
    if preset.mode not in MODES[profile.kind]:
        modes = " or ".join(f"'{mode}'" for mode in MODES[profile.kind])
        raise PresetError(
            f"Preset '{preset.mode}' does not fit {profile.kind} profile '{profile.name}': use "
            f"{modes}."
        )

    values = preset.model_dump()
    problems = _derive_values(values, profile) + _sigpipe_rules(values, profile)
    if problems:
        raise PresetError(
            f"Preset '{preset.mode}' does not fit profile '{profile.name}':\n"
            + "\n".join(f"- {problem}" for problem in problems)
        )

    # Validating again runs every other check on the completed values.
    return type(preset).model_validate(values)


STAGES: dict[ProcessingMode, dict[str, Stage]] = {
    ProcessingMode.ACTIVE: ACTIVE_STAGES,
    ProcessingMode.PASSIVE: PASSIVE_STAGES,
    ProcessingMode.PASSIVE_ACTIVE: PASSIVE_ACTIVE_STAGES,
}


def method_defaults(mode: str, profile: Profile) -> dict[str, dict[str, dict[str, Any]]]:
    """For each stage of preset `mode` with a choice of methods, each method's values for
    `profile` (those the profile derives filled in): what a form offers when its method
    changes. A method whose derived values clash with the profile keeps its own defaults; one
    the preset cannot start without more values is left out. The rules a run must meet (a
    muting that cuts something) are not a form's: its values are filled in all the same."""
    methods: dict[str, dict[str, dict[str, Any]]] = {}
    for name, stage in STAGES[ProcessingMode(mode)].items():
        if not stage.selectable:
            continue
        methods[name] = {}
        for method in [*(["none"] if stage.none and stage.optional else []), *stage.functions]:
            try:
                preset = make_preset(mode, {name: {"method": method}})
            except PresetError:
                continue
            derived = preset.model_dump()
            if not _derive_values(derived, profile):
                with contextlib.suppress(ValueError):
                    preset = type(preset).model_validate(derived)
            values = dict(preset.model_dump(mode="json")[name])
            values.pop("method")
            methods[name][method] = values
    return methods


def _derive_values(values: dict[str, Any], profile: Profile) -> list[str]:
    """Fill the values left to None; report those that clash with an override."""
    nyquist = profile.nyquist_hz
    problems: list[str] = []

    muting = values.get("muting")  # the modes of shots
    if muting is not None and muting["method"] == "mute" and muting["width"] is None:
        muting["width"] = 1 / profile.sampling_rate_hz  # one sample

    filtering = values["filtering"]
    if filtering["method"] == "iir" and filtering["fmax"] is None:
        fmax = IIR_FMAX_NYQUIST_FRACTION * nyquist
        problems += _derive(values, "filtering", "fmin", "fmax", fmax, "Hz", profile)

    whitening = values.get("whitening")  # passive only
    if whitening is not None and whitening["method"] == "onebit_apod" and whitening["fmax"] is None:
        problems += _derive(values, "whitening", "fmin", "fmax", nyquist, "Hz", profile)

    return problems


def _derive(
    values: dict[str, Any],
    stage: str,
    lower: str,
    upper: str,
    value: float,
    unit: str,
    profile: Profile,
) -> list[str]:
    """Set `stage.upper` to its derived `value`, which must stay above `stage.lower`."""
    params = values[stage]
    params[upper] = value
    # sigpipe lets some lower bounds be None, meaning no bound (e.g. muting.tmin).
    if params[lower] is not None and params[lower] >= value:
        return [
            f"{stage}.{lower} ({params[lower]:g} {unit}) must be below {stage}.{upper}, which "
            f"defaults to {value:g} {unit} for profile '{profile.name}'. "
            f"Lower {stage}.{lower} or set {stage}.{upper}."
        ]
    return []


def _sigpipe_rules(values: dict[str, Any], profile: Profile) -> list[str]:
    """The rules a run must meet, in pipeline order: those sigpipe checks in every window and
    that depend on the profile, and those without which a stage would leave nothing (a window
    past the records' data, an image outside the band kept). PAC's fields say the same before a
    run is asked for. Each <x>max above its <x>min is the models' own check (generation.py).

    A dispersion fmax above Nyquist is not one: sigpipe lowers it to Nyquist, with a warning.
    """
    name, nyquist = profile.name, profile.nyquist_hz
    problems: list[str] = _masw_rules(values["masw"], profile) + _muting_rules(values, profile)
    mode = ProcessingMode(values["mode"])
    problems += [
        f"{stage_name} cannot be switched off in a {mode.value} run: set its parameters instead."
        for stage_name, stage in STAGES[mode].items()
        if not stage.optional and values[stage_name].get("method") == "none"
    ]

    filtering = values["filtering"]
    kept: tuple[float, float] = (0.0, nyquist)  # the band the filter and the whitening keep
    if filtering["method"] == "iir":
        if filtering["fmax"] >= nyquist:
            problems.append(
                f"filtering.fmax ({filtering['fmax']:g} Hz) must be below the Nyquist frequency "
                f"of profile '{name}' ({nyquist:g} Hz)."
            )
        kept = (filtering["fmin"], filtering["fmax"])

    if (slicing := values.get("slicing")) is not None:  # passive only
        problems += _slicing_rules(slicing, profile)
        whitening = values["whitening"]
        if whitening["method"] == "onebit_apod":
            problems += _whitening_rules(whitening, slicing, profile)
            kept = (max(kept[0], whitening["fmin"]), min(kept[1], whitening["fmax"]))

    dispersion = values["dispersion"]
    if dispersion["fmin"] >= nyquist:
        problems.append(
            f"dispersion.fmin ({dispersion['fmin']:g} Hz) must be below the Nyquist "
            f"frequency of profile '{name}' ({nyquist:g} Hz)."
        )
    # The image within the band the filter and the whitening keep: outside, noise alone.
    elif kept != (0.0, nyquist) and not (
        dispersion["fmin"] < kept[1] and dispersion["fmax"] > kept[0]
    ):
        problems.append(
            f"dispersion's band ({dispersion['fmin']:g}-{dispersion['fmax']:g} Hz) is outside the "
            f"{kept[0]:g}-{kept[1]:g} Hz the filtering and the whitening keep: the image would "
            "hold noise alone."
        )

    return problems


def _masw_rules(masw: dict[str, Any], profile: Profile) -> list[str]:
    """The windows on the line: none longer than it, nor a step past it."""
    receivers = len(profile.receivers)
    return [
        f"masw.{key} ({masw[key]} receivers) must not exceed the {receivers} receivers of "
        f"profile '{profile.name}'."
        for key in ("length", "step")
        if masw[key] > receivers
    ]


def _muting_rules(values: dict[str, Any], profile: Profile) -> list[str]:
    """A muting that cuts something: a bound, or the trigger it applies (a shot's time origin,
    given or from each record's file); what each record keeps once moved by its trigger, the
    signal width at the shot at least and a window that starts within it."""
    muting = values.get("muting")  # the modes of shots
    if muting is None or muting["method"] != "mute":
        return []
    problems: list[str] = []
    trigger = values.get("trigger")
    t0 = trigger["t0"] if trigger is not None else None
    # Each record's shift: the t0 given, else its file's trigger (0 when it says none).
    shifts = {
        record.path.name: t0 if t0 is not None else record.trigger_s or 0.0
        for record in profile.records
    }
    if all(muting[bound] is None for bound in ("tmin", "tmax", "vmin", "vmax")) and not any(
        shifts.values()
    ):
        problems.append(
            "muting is on but keeps everything: give it a time or a velocity, or switch it off."
        )
    # At least one sample kept after the slowest arrival: the trace at the shot never emptied.
    sample = 1 / profile.sampling_rate_hz
    width = muting["width"]
    if width is not None and width < sample * (1 - 1e-9):
        problems.append(
            f"muting.width ({width:g} s) must be at least one sample of profile "
            f"'{profile.name}' ({sample:g} s): the trace at the shot would keep nothing."
        )
    # The record whose data, once moved by its trigger, ends first: every record keeps its part.
    record = min(profile.records, key=lambda one: one.duration_s - shifts[one.path.name])
    shift = shifts[record.path.name]
    end = record.duration_s - shift
    if width is not None and width > end * (1 + 1e-9):
        problems.append(
            f"The trigger ({shift:g} s) and muting.width ({width:g} s) exceed record "
            f"{record.path.name} ({record.duration_s:.2f} s): the pulse at the shot would not fit. "
            "Lower the trigger or the width."
        )
    if muting["tmin"] is not None and muting["tmin"] >= end:
        problems.append(
            f"muting.tmin ({muting['tmin']:g} s) is past the end of record {record.path.name}'s "
            f"data once moved by its trigger ({end:.2f} s): nothing would be kept."
        )
    return problems


def _slicing_rules(slicing: dict[str, Any], profile: Profile) -> list[str]:
    duration, step = slicing["segment_duration"], slicing["segment_step"]
    shortest = min(record.duration_s for record in profile.records)
    problems: list[str] = []
    if step > duration:
        problems.append(
            f"slicing.segment_step ({step:g} s) must not exceed "
            f"slicing.segment_duration ({duration:g} s)."
        )
    if duration > shortest:
        problems.append(
            f"slicing.segment_duration ({duration:g} s) must not exceed the shortest record "
            f"of profile '{profile.name}' ({shortest:g} s)."
        )
    return problems


def _whitening_rules(
    whitening: dict[str, Any], slicing: dict[str, Any], profile: Profile
) -> list[str]:
    if whitening["method"] != "onebit_apod" or whitening["fmin"] >= whitening["fmax"]:
        return []

    band = whitening["fmax"] - whitening["fmin"]
    taper = whitening["taper_width_Hz"]
    # Whitening runs on each segment. sigpipe's frequency step there: sampling rate / samples,
    # with round(duration / dt) + 1 samples in a segment.
    duration = slicing["segment_duration"]
    df = profile.sampling_rate_hz / (round(duration * profile.sampling_rate_hz) + 1)
    problems: list[str] = []
    if band <= taper:
        problems.append(
            f"whitening.taper_width_Hz ({taper:g} Hz) must be smaller than the band "
            f"fmax - fmin ({band:g} Hz)."
        )
    if int(band / df) < 2:
        problems.append(
            f"whitening: the band fmax - fmin ({band:g} Hz) must span at least 2 frequency steps "
            f"of the {duration:g} s segments ({2 * df:.3g} Hz). Widen the band or lengthen "
            "slicing.segment_duration."
        )
    return problems
