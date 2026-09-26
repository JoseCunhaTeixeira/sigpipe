"""What a perfect plane wave gives in a window's phase-shift image: where the window resolves
velocity at all (the picker keeps no point where it does not), and the reference G3 judges
sharpness and prominence against (the user's decisions of 2026-09-25)."""

import numpy as np

from sigpipe.base.dispersion_image import DispersionImage


def plane_wave_columns(
    image: DispersionImage, frequencies: np.ndarray, velocities: np.ndarray, floor: float
) -> np.ndarray:
    """The phase-shift columns a perfect plane wave would give at each (frequency, velocity),
    through the window's receivers and on the image's velocity grid, above the noise floor
    `floor`. The shots' offsets add a phase common to every receiver, which the magnitude
    drops, so every shot gives the same columns."""
    positions = np.array([receiver.x for receiver in image.acquisition.receivers], dtype=float)
    positions -= positions[0]
    slowness = 1 / image.vs.astype(float)
    columns = np.empty((frequencies.size, slowness.size))
    for k, (frequency, velocity) in enumerate(zip(frequencies, velocities, strict=True)):
        phases = 2j * np.pi * frequency * np.outer(positions, slowness - 1 / velocity)
        columns[k] = np.abs(np.exp(phases).mean(axis=0))
    return np.clip(columns - floor, 0, None)


def prominences(columns: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Each column's height at its peak index over its median height (both above the floor)."""
    heights = columns[np.arange(peaks.size), peaks]
    return heights / np.maximum(np.median(columns, axis=1), 1e-3)
