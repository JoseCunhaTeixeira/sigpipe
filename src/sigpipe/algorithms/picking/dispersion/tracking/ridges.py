"""The three stages of malw-pipe's picker (../malw-pipe/docs/picking.md), adapted to Rayleigh waves.

1. `lowest_ridge` finds which ridge to follow. malw-pipe scans down from a ceiling, because
   nothing lies above the A0 Lamb mode; for Rayleigh waves the fundamental mode is the slowest and
   higher modes lie above it, so the scan goes up from a floor.
2. `corridor` fences the ridge in, so the tracking cannot wander onto another branch.
3. `track` draws the best smooth curve inside the corridor, by dynamic programming.

Images are [n_f, n_v] arrays; velocities and frequencies are regular grids.
"""

import numpy as np


def lowest_ridge(
    image: np.ndarray, start: np.ndarray, stop: np.ndarray, threshold: float
) -> np.ndarray:
    """Per frequency, the velocity index of the lowest ridge between `start` and `stop`, included.

    Scans each column up from `start` and stops at the first local maximum that reaches
    `threshold` times the column's maximum in that range. A column still rising where `stop` cuts
    it has its ridge above: `stop` is taken (malw-pipe's ceiling rule). A column still falling
    where `start` cuts it has its ridge below: `start` is taken. Either way the pick will show as
    pinned. A column with no local maximum above the threshold falls back on its brightest value.
    """
    n_f = image.shape[0]
    ridge = np.empty(n_f, dtype=int)
    for i in range(n_f):
        low, top = int(start[i]), int(stop[i])
        column = image[i, low : top + 1]
        brightest = int(np.argmax(column))
        ridge[i] = low + brightest
        if brightest in (0, column.size - 1):
            continue
        level = threshold * column[brightest]
        for j in range(1, column.size - 1):
            if column[j] >= level and column[j] >= column[j - 1] and column[j] >= column[j + 1]:
                ridge[i] = low + j
                break
    return ridge


def corridor(
    velocities: np.ndarray,
    ridge: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    half_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Index bounds, both included, of `ridge` +/- `half_width` times its velocity, kept between
    `start` and `stop`. Always at least two cells wide, so the tracking has a choice."""
    centre = velocities[ridge]
    low = np.searchsorted(velocities, centre * (1 - half_width), side="left")
    high = np.searchsorted(velocities, centre * (1 + half_width), side="right") - 1
    low = np.minimum(np.maximum(low, start), stop - 1)
    high = np.maximum(np.minimum(high, stop), low + 1)
    return low, high


def track(
    image: np.ndarray,
    velocities: np.ndarray,
    frequencies: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    smoothness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The best smooth path through the corridors, and where it sits on a corridor edge.

    Maximises the image values along the path minus, between neighbouring frequencies,
    `smoothness` times (change of log velocity per Hz)². The penalty is relative, so it means the
    same at 150 and at 800 m/s, and per Hz, so it does not depend on the frequency step. Exact
    dynamic programming (Viterbi), restricted to the corridor cells.
    """
    n_f = frequencies.size
    df = float(frequencies[1] - frequencies[0]) if n_f > 1 else 1.0
    log_v = np.log(velocities)

    total = image[0, low[0] : high[0] + 1].astype(float)
    backs: list[np.ndarray] = []
    for i in range(1, n_f):
        previous = log_v[low[i - 1] : high[i - 1] + 1]
        current = log_v[low[i] : high[i] + 1]
        penalty = smoothness * ((current[:, None] - previous[None, :]) / df) ** 2
        candidates = total[None, :] - penalty
        back = np.argmax(candidates, axis=1)
        total = image[i, low[i] : high[i] + 1] + candidates[np.arange(current.size), back]
        backs.append(back)

    path = np.empty(n_f, dtype=int)
    cell = int(np.argmax(total))
    path[-1] = low[-1] + cell
    for i in range(n_f - 1, 0, -1):
        cell = int(backs[i - 1][cell])
        path[i - 1] = low[i - 1] + cell

    on_edge = (path == low) | (path == high)
    return path, on_edge
