"""Fitting a preset to a profile: the values PAC's forms derive from the acquisition, and the rules
sigpipe checks inside every window, checked once against the profile before any work."""

import contextlib
from typing import Any

from sigpipe.masw.presets.making import make_preset
from sigpipe.masw.presets.models import ActivePreset, PassivePreset, PresetError
from sigpipe.masw.presets.stages import ACTIVE_STAGES, PASSIVE_ACTIVE_STAGES, PASSIVE_STAGES, Stage
from sigpipe.masw.profiles import MODES, ProcessingMode, Profile

# A switched-on IIR filter without fmax stops just below the Nyquist frequency. PAC's form puts it
# at Nyquist, which sigpipe's filter rejects (it requires fmax < Nyquist).
IIR_FMAX_NYQUIST_FRACTION = 0.95
# The ramps of the passive-active correlation window, as the signal QC pads its own window.
CORRELATION_TAPER_S = 0.05


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
    the preset cannot start without more values is left out."""
    methods: dict[str, dict[str, dict[str, Any]]] = {}
    for name, stage in STAGES[ProcessingMode(mode)].items():
        if not stage.selectable:
            continue
        methods[name] = {}
        for method in [*(["none"] if stage.none else []), *stage.functions]:
            try:
                preset = make_preset(mode, {name: {"method": method}})
            except PresetError:
                continue
            with contextlib.suppress(PresetError):
                preset = resolve_preset(preset, profile)
            values = dict(preset.model_dump(mode="json")[name])
            values.pop("method")
            methods[name][method] = values
    return methods


def _derive_values(values: dict[str, Any], profile: Profile) -> list[str]:
    """Fill the values left to None, as PAC's forms do; report those that clash with an override."""
    nyquist = profile.nyquist_hz
    problems: list[str] = []

    if values["muting"]["method"] == "mute" and values["muting"]["tmax"] is None:
        # As PAC's form shows it: the longest record, rounded to 10 ms.
        record_length = round(max(record.duration_s for record in profile.records), 2)
        problems += _derive(values, "muting", "tmin", "tmax", record_length, "s", profile)

    filtering = values["filtering"]
    if filtering["method"] == "iir" and filtering["fmax"] is None:
        fmax = IIR_FMAX_NYQUIST_FRACTION * nyquist
        problems += _derive(values, "filtering", "fmin", "fmax", fmax, "Hz", profile)

    window = values.get("correlation_window")  # passive-active only
    if window is not None and window["method"] == "mute" and window["taper"] is None:
        window["taper"] = round(CORRELATION_TAPER_S * profile.sampling_rate_hz)

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
    """The rules sigpipe checks in every window that depend on the profile, in pipeline order.

    A dispersion fmax above Nyquist is not one: sigpipe lowers it to Nyquist, with a warning.
    """
    name, nyquist = profile.name, profile.nyquist_hz
    problems: list[str] = []

    filtering = values["filtering"]
    if filtering["method"] == "iir" and filtering["fmax"] >= nyquist:
        problems.append(
            f"filtering.fmax ({filtering['fmax']:g} Hz) must be below the Nyquist frequency "
            f"of profile '{name}' ({nyquist:g} Hz)."
        )

    if (slicing := values.get("slicing")) is not None:  # passive only
        problems += _slicing_rules(slicing, profile)
        problems += _whitening_rules(values["whitening"], slicing, profile)

    if values["dispersion"]["fmin"] >= nyquist:
        problems.append(
            f"dispersion.fmin ({values['dispersion']['fmin']:g} Hz) must be below the Nyquist "
            f"frequency of profile '{name}' ({nyquist:g} Hz)."
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
