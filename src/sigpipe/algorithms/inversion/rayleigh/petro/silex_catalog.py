"""The Silex models bundled with sigpipe, and what each was trained on, read without keras (the
silex extra): a model can be chosen, and a curve checked against it, before one is loaded."""

import importlib.resources
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.interpolate import interp1d

from sigpipe.base.dispersion_curve import DispersionCurve

BUNDLED_MODELS_ROOT = Path(str(importlib.resources.files("sigpipe") / "models" / "silex"))
REQUIRED_FILES = ("silex.keras", "silex_params.json", "vocab.json")
# How far a curve's frequency or velocity range may fall short of (or overshoot) a model's
# trained range, as a fraction of that range's width: a 15-50 Hz model takes a curve starting as
# late as 22 Hz or ending as early as 43 Hz. Within the margin, the model extrapolates a little;
# past it, its prediction is unreliable, and refused.
RANGE_MARGIN_FRACTION = 0.20

# How a curve falls outside a model's trained range.
type RangeGap = Literal["starts_late", "ends_early", "too_slow", "too_fast"]


@dataclass(frozen=True, slots=True)
class SilexCard:
    """What a Silex model was trained on (its silex_params.json)."""

    name: str
    min_freq: float  # Hz, the model's frequency axis
    max_freq: float
    d_freq: float
    n_freqs: int
    min_vel: float  # m/s, the phase velocities it was trained on
    max_vel: float
    output_seq_length: int
    # The substratum every training sample sat on, in GPDC format (thickness vp vs rho per
    # line, the last line's thickness 0 for the half-space): a curve forward-modelled from a
    # prediction needs it (petro.forward.parse_under_layers).
    under_layers: str
    soils: tuple[str, ...]  # the soil types it predicts
    max_layers: int
    max_depth: float  # m, the deepest its soil columns reach
    water_table: tuple[float, float]  # m, the depths it was trained on

    @property
    def band_needed(self) -> tuple[float, float]:
        """The latest a curve may start and the earliest it may end, Hz."""
        margin = RANGE_MARGIN_FRACTION * (self.max_freq - self.min_freq)
        return self.min_freq + margin, self.max_freq - margin

    @property
    def velocities_allowed(self) -> tuple[float, float]:
        """The slowest and fastest a curve may be on the model's frequency axis, m/s."""
        margin = RANGE_MARGIN_FRACTION * (self.max_vel - self.min_vel)
        return self.min_vel - margin, self.max_vel + margin


def list_bundled_silex_models() -> list[str]:
    """Names of the Silex models bundled with sigpipe (pyproject.toml's package-data), one per
    folder of `models/silex/` holding a model's three files. Each is named
    `<site>_<min_freq>-<max_freq>hz_<min_vel>-<max_vel>mps`, after what it was trained on."""
    if not BUNDLED_MODELS_ROOT.is_dir():
        return []
    return sorted(
        path.name
        for path in BUNDLED_MODELS_ROOT.iterdir()
        if path.is_dir() and all((path / file).is_file() for file in REQUIRED_FILES)
    )


def bundled_silex_model_dir(name: str) -> Path:
    """The folder of bundled Silex model `name` (see `list_bundled_silex_models`)."""
    model_dir = BUNDLED_MODELS_ROOT / name
    if not all((model_dir / file).is_file() for file in REQUIRED_FILES):
        raise ValueError(
            f"Unknown bundled Silex model {name!r}. Available: {list_bundled_silex_models()}"
        )
    return model_dir


def load_silex_card(model_dir: Path) -> SilexCard:
    """What the Silex model in `model_dir` was trained on."""
    params: dict[str, Any] = json.loads((model_dir / "silex_params.json").read_text())
    generation: dict[str, Any] = params["generation_config"]
    return SilexCard(
        name=model_dir.name,
        min_freq=float(params["min_freq"]),
        max_freq=float(params["max_freq"]),
        d_freq=float(params["d_freq"]),
        n_freqs=int(params["n_freqs"]),
        min_vel=float(params["min_vel"]),
        max_vel=float(params["max_vel"]),
        output_seq_length=int(params["output_seq_length"]),
        under_layers=str(generation["under_layers"]),
        soils=tuple(str(soil) for soil in generation["soils"]),
        max_layers=int(generation["max_n_layers"]),
        max_depth=float(generation["max_depth"]),
        water_table=(float(generation["min_water_table"]), float(generation["max_water_table"])),
    )


def range_gaps(card: SilexCard, curve: DispersionCurve) -> tuple[RangeGap, ...]:
    """How `curve` falls outside the range the model of `card` was trained on, beyond the
    margin; none when the model covers it. Its velocities are checked once its band is: a curve
    extrapolated over a missing band has velocities of no meaning."""
    gaps: list[RangeGap] = []
    start_by, end_from = card.band_needed
    if float(np.nanmin(curve.fs)) > start_by:
        gaps.append("starts_late")
    if float(np.nanmax(curve.fs)) < end_from:
        gaps.append("ends_early")
    if gaps:
        return tuple(gaps)
    slowest, fastest = card.velocities_allowed
    velocities = _on_axis(card, curve)
    if float(np.nanmin(velocities)) < slowest:
        gaps.append("too_slow")
    if float(np.nanmax(velocities)) > fastest:
        gaps.append("too_fast")
    return tuple(gaps)


def resampled_velocities(card: SilexCard, curve: DispersionCurve) -> np.ndarray:
    """`curve`'s velocities on the model's frequency axis, extrapolated past its ends; raises
    ValueError when the curve falls outside the model's trained range beyond
    RANGE_MARGIN_FRACTION, in frequency or in velocity (range_gaps)."""
    gaps = range_gaps(card, curve)
    if "starts_late" in gaps or "ends_early" in gaps:
        raise ValueError(
            f"dispersion_curve frequency range [{float(np.nanmin(curve.fs)):g}, "
            f"{float(np.nanmax(curve.fs)):g}] Hz does not cover Silex model {card.name}'s "
            f"trained range [{card.min_freq:g}, {card.max_freq:g}] Hz within a "
            f"{RANGE_MARGIN_FRACTION:.0%} margin"
        )
    velocities = _on_axis(card, curve)
    if gaps:
        raise ValueError(
            f"dispersion_curve velocity range [{float(np.nanmin(velocities)):.0f}, "
            f"{float(np.nanmax(velocities)):.0f}] m/s (resampled onto the model's frequency "
            f"axis) falls outside Silex model {card.name}'s trained range "
            f"[{card.min_vel:.0f}, {card.max_vel:.0f}] m/s within a "
            f"{RANGE_MARGIN_FRACTION:.0%} margin"
        )
    return velocities


def _on_axis(card: SilexCard, curve: DispersionCurve) -> np.ndarray:
    """`curve`'s velocities on the model's frequency axis, extrapolated past its ends."""
    fs_grid = card.min_freq + np.arange(card.n_freqs) * card.d_freq
    resample = interp1d(curve.fs, curve.vs, fill_value="extrapolate")  # pyright: ignore[reportArgumentType]
    return np.asarray(resample(fs_grid), dtype=np.float64)
