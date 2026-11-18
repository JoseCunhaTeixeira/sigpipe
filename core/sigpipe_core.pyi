"""sigpipe's compiled core (Rust): Rayleigh- and Love-wave dispersion curves of layered models,
the seismic inversion's samplers, and the phase-shift transform's sum. sigpipe's modules call it
(data.py, forward.py, transdimensional.py, dream.py, phase_shift.py), which say what each
computes, and santiludo for its dispersion curves."""

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

type Floats = NDArray[np.float64]
# A curve: its mode (0: the fundamental), its periods (s, increasing), picked velocities and
# their uncertainties (m/s).
type Curve = tuple[int, Floats, Floats, Floats]

# A root is refined until its bracket is this share of it wide.
TOLERANCE: float

class DispersionError(Exception):
    """The fundamental mode of a layered model was not found over the periods asked."""

class TransdimensionalChain(TypedDict):
    layers: NDArray[np.int64]
    depths: Floats  # kept x (max_layers - 1), NaN beyond each model's interfaces
    vs: Floats  # kept x max_layers
    noise: Floats
    predicted: list[Floats]  # per curve: kept x points, by increasing period
    rms: Floats
    proposed: dict[str, int]  # each move's, of the first copy
    accepted: dict[str, int]
    swaps: tuple[int, int]  # the exchanges accepted, and tried
    steps: dict[str, float]
    best_depths: Floats
    best_vs: Floats
    best_misfit: float

class DreamChain(TypedDict):
    units: Floats  # kept x (parameters + 1), on the unit cube, the noise factor last
    predicted: list[Floats]  # per curve: kept x points, by increasing period
    rms: Floats
    accepted: int
    squared_moves: Floats

class DreamChains(TypedDict):
    chains: list[DreamChain]
    best_unit: Floats
    best_predicted: list[Floats]
    crossover: list[float]

def rayleigh_phase(
    periods: Floats,
    thickness: Floats,
    vp: Floats,
    vs: Floats,
    rho: Floats,
    mode: int,
    dc: float,
) -> Floats | None: ...
def phase_dispersion(
    periods: Floats,
    thickness: Floats,
    vp: Floats,
    vs: Floats,
    rho: Floats,
    mode: int = 0,
    wave: Literal["rayleigh", "love"] = "rayleigh",
    dc: float = 0.005,
) -> tuple[Floats, Floats]: ...
def period_equation(
    period: float,
    c: float,
    thickness: Floats,
    vp: Floats,
    vs: Floats,
    rho: Floats,
    wave: Literal["rayleigh", "love"] = "rayleigh",
) -> float: ...
def transdimensional_chain(
    curves: list[Curve],
    vs_bounds: tuple[float, float],
    depth_bounds: tuple[float, float],
    max_layers: int,
    least_ratio: float,
    vp_vs: float,
    noise_bounds: tuple[float, float],
    thinnest: float,
    n_iterations: int,
    n_burnin: int,
    save_every: int,
    temperatures: int,
    hottest: float,
    seed: int,
) -> TransdimensionalChain: ...
def dream_chains(
    curves: list[Curve],
    low: list[float],
    high: list[float],
    vs_template: list[float],
    thickness_template: list[float],
    vs_from: list[int],
    vs_to: list[int],
    thickness_from: list[int],
    thickness_to: list[int],
    vs_index: list[int],
    least_ratio: float,
    vp_vs: float,
    noise_bounds: tuple[float, float],
    n_iterations: int,
    n_burnin: int,
    save_every: int,
    n_chains: int,
    seed: int,
) -> DreamChains: ...
def phase_shift_on_grid(
    held: NDArray[np.complex128], fs: Floats, vs: Floats, x0: float, dx: float
) -> Floats: ...
def phase_shift_at_offsets(
    xf: NDArray[np.complex64], fs: Floats, vs: Floats, offsets: Floats
) -> Floats: ...
