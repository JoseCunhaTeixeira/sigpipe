"""Measures of a dispersion image before picking (PACo's image QC, G2, judges them; PAC measures
its own runs' alike, measure_image): the noise floor, the coherent columns, peaks on the grid's
edges, competing ridges, and aliased ones."""

import math
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.signal import find_peaks

from sigpipe.algorithms.picking.dispersion.curve import (
    max_resolvable_wavelength,
    min_resolvable_wavelength,
)
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.masw.quality.measures import Measure, figure


def noise_floor(image: DispersionImage) -> float:
    """Random phases sum to about 1 / sqrt(N) on a phase-shift image (the picker's floor)."""
    return 1 / math.sqrt(len(image.acquisition.receivers))


def coherent_columns(image: DispersionImage, level: float) -> np.ndarray:
    """The frequencies (rows of fv_map) whose peak lies `level` of the way from the noise floor
    to 1."""
    floor = noise_floor(image)
    return image.fv_map.max(axis=1) > floor + level * (1 - floor)


def edge_peaks(
    image: DispersionImage, coherent: np.ndarray, edge_share: float, floor: float = 0.0
) -> tuple[int, int]:
    """How many coherent columns peak within `edge_share` of the grid's lowest velocity, and,
    looking above `floor` only (below it lie artifacts), within `edge_share` of its highest,
    among the columns where the window tells that velocity from an infinite one: a window of
    aperture L resolves slowness to about 1 / (f L), so below f = vmax / L a peak at vmax is
    energy with no moveout (noise common to every trace), not a ridge beyond the grid."""
    vs = np.asarray(image.vs, dtype=float)
    fs = np.asarray(image.fs, dtype=float)
    span = vs.max() - vs.min()
    peaks = vs[image.fv_map[coherent].argmax(axis=1)] if coherent.any() else np.array([])
    above = _above(image, floor)[coherent]
    peaks_above = vs[above.argmax(axis=1)] if coherent.any() else np.array([])
    aperture = max_resolvable_wavelength(image.acquisition) or 0.0
    resolved = (fs * aperture > vs.max())[coherent]
    low = int(np.sum(peaks <= vs.min() + edge_share * span))
    high = int(np.sum((peaks_above >= vs.max() - edge_share * span) & resolved))
    return low, high


def competing_ridges(
    image: DispersionImage,
    coherent: np.ndarray,
    ratio: float,
    separation: float,
    floor: float = 0.0,
) -> np.ndarray:
    """For each coherent column, whether a second local maximum reaches `ratio` of the highest
    while lying at least `separation` (relative) away from it in velocity; peaks below `floor`
    are artifacts, not ridges."""
    vs = np.asarray(image.vs, dtype=float)
    above = _above(image, floor)
    result = np.zeros(image.fv_map.shape[0], dtype=bool)
    for row in np.flatnonzero(coherent):
        column = above[row]
        best = int(column.argmax())
        peaks, _ = find_peaks(column, height=ratio * column[best])
        others = [peak for peak in peaks if abs(vs[peak] - vs[best]) >= separation * vs[best]]
        result[row] = bool(others)
    return result


def aliased(image: DispersionImage, coherent: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Coherent columns whose peak (above `floor`) lies below the aliasing limit v = 2 dx f (the
    shortest wavelength the receivers resolve, times f): a second ridge there is the alias of
    the first, not a mode. None of them when the receivers' positions are unknown."""
    shortest = min_resolvable_wavelength(image.acquisition)
    if shortest is None:
        return np.zeros_like(coherent)
    vs = np.asarray(image.vs, dtype=float)
    fs = np.asarray(image.fs, dtype=float)
    peaks = vs[_above(image, floor).argmax(axis=1)]
    return coherent & (peaks < shortest * fs)


def _above(image: DispersionImage, floor: float) -> np.ndarray:
    """The image with the velocities below `floor` zeroed: where a correlation's zero lag or a
    mute's edge puts energy no surface wave has."""
    vs = np.asarray(image.vs, dtype=float)
    return np.where(vs[None, :] >= floor, image.fv_map, 0.0)


class ImageLimits(BaseModel):
    """How a dispersion image is measured, and its limits (PACo's G2 judges against them, PAC
    measures its own runs' alike): provisional, measured on the demo profiles."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    coherent_level: float = Field(
        default=0.3,
        gt=0,
        lt=1,
        description="A column is coherent when its peak is this far from the noise floor\n"
        "1 / sqrt(N) towards 1, the most a plane wave gives: the same for every N.",
    )
    min_coherent_columns: float = Field(
        default=0.5, ge=0, le=1, description="Share of the image's columns that must be coherent."
    )
    edge_share: float = Field(
        default=0.02,
        gt=0,
        lt=0.5,
        description="Share of the velocity range that counts as an edge.",
    )
    max_edge_columns: float = Field(
        default=0.2, ge=0, le=1, description="Share of coherent columns peaking on an edge."
    )
    competing_ratio: float = Field(
        default=0.7, gt=0, lt=1, description="A second ridge at least this share of the first."
    )
    competing_separation: float = Field(
        default=0.15, gt=0, description="Its velocity at least this share away from the first's."
    )
    max_competing_columns: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="Share of coherent columns with a second ridge: good 24-receiver windows of "
        "the\ndemo hold one in 52 to 60 % of theirs, 5-receiver ones in 87 to 95 %.",
    )
    min_band_share: float = Field(
        default=0.5, gt=0, le=1, description="Coherent band over the usable band, at least."
    )
    vmin_floor: float = Field(
        default=30.0,
        gt=0,
        description="m/s: energy peaking at a vmin below this is an artifact, not a slower wave.",
    )


@dataclass(frozen=True)
class ImageReport:
    """A dispersion image's measures, and what its checks act on."""

    measures: tuple[Measure, ...]
    coherent: np.ndarray  # bool, by column (frequency): its peak over the level
    band: tuple[float, float] | None = None  # the coherent columns' span, Hz
    low: int = 0  # coherent columns peaking at the grid's lowest velocity
    high: int = 0  # ...at its highest (among those the window resolves)
    competing: np.ndarray | None = None  # bool, by column: a second ridge
    aliased: np.ndarray | None = None  # bool, by column: that ridge below 2 dx f
    band_share: float | None = None  # the coherent band over the usable band within the image
    usable: tuple[float, float] | None = None  # the records' usable band within the image


def measure_image(
    image: DispersionImage,
    limits: ImageLimits,
    usable_band: tuple[float, float] | None = None,
) -> ImageReport:
    """`image`'s measures against `limits`, each saying what it covers: its coherent columns,
    those peaking on the grid's edges, the coherent band against the image's ends, the columns
    with a second ridge (and those of them aliased, reported), and, with the records'
    `usable_band`, the coherent band over the part of it the image spans. One definition for
    PACo's G2 and PAC's views."""
    fs = np.asarray(image.fs, dtype=float)
    vs = np.asarray(image.vs, dtype=float)
    coherent = coherent_columns(image, limits.coherent_level)
    n_coherent = int(coherent.sum())
    share = n_coherent / max(coherent.size, 1)
    measures = [
        Measure(
            name="coherent_columns",
            value=round(share, 3),
            threshold=limits.min_coherent_columns,
            bound="min",
            passed=share >= limits.min_coherent_columns,
            of="image",
            over=f"the image's {coherent.size} columns, {figure(fs.min())}-{figure(fs.max())} "
            f"Hz: a peak {limits.coherent_level:.0%} of the way from the noise floor to 1",
        )
    ]
    if n_coherent == 0:
        return ImageReport(tuple(measures), coherent)

    band = (float(fs[coherent].min()), float(fs[coherent].max()))
    low, high = edge_peaks(image, coherent, limits.edge_share, limits.vmin_floor)
    for name, count, edge in (
        ("ridge_at_vmin", low, vs.min()),
        ("ridge_at_vmax", high, vs.max()),
    ):
        measures.append(
            Measure(
                name=name,
                value=round(count / n_coherent, 3),
                threshold=limits.max_edge_columns,
                bound="max",
                passed=count / n_coherent <= limits.max_edge_columns,
                of="image",
                over=f"the {n_coherent} coherent columns: a peak within "
                f"{limits.edge_share:.0%} of {figure(edge)} m/s",
            )
        )
    rows = np.flatnonzero(coherent)
    ends = f"the coherent band, {figure(band[0])}-{figure(band[1])} Hz, against the image's, "
    for name, touches, end in (
        ("band_at_fmin", rows[0] == 0, fs.min()),
        ("band_at_fmax", rows[-1] == fs.size - 1, fs.max()),
    ):
        measures.append(
            Measure(
                name=name,
                value=float(touches),
                threshold=0,
                bound="max",
                passed=not touches,
                of="image",
                over=f"{ends}{figure(end)} Hz",
            )
        )
    competing = competing_ridges(
        image, coherent, limits.competing_ratio, limits.competing_separation, limits.vmin_floor
    )
    competing_share = int(competing.sum()) / n_coherent
    measures.append(
        Measure(
            name="competing_ridges",
            value=round(competing_share, 3),
            threshold=limits.max_competing_columns,
            bound="max",
            passed=competing_share <= limits.max_competing_columns,
            of="image",
            over=f"the {n_coherent} coherent columns: a second peak {limits.competing_ratio:.0%} "
            f"of the first at least, {limits.competing_separation:.0%} away",
        )
    )
    # Reported: which of the second ridges lie below the aliasing limit.
    alias = aliased(image, competing, limits.vmin_floor)
    measures.append(
        Measure(
            name="aliased_ridges",
            value=round(int(alias.sum()) / max(int(competing.sum()), 1), 3),
            passed=True,
            of="image",
            over=f"the {int(competing.sum())} columns with a second ridge: under 2 dx f",
        )
    )
    band_share: float | None = None
    # The usable band where the image can use it: beyond its frequencies, nothing to compare.
    within = (
        (max(usable_band[0], float(fs.min())), min(usable_band[1], float(fs.max())))
        if usable_band is not None
        else None
    )
    if within is not None and within[1] > within[0]:
        band_share = (band[1] - band[0]) / (within[1] - within[0])
        measures.append(
            Measure(
                name="band_share_of_usable",
                value=round(band_share, 3),
                threshold=limits.min_band_share,
                bound="min",
                passed=band_share >= limits.min_band_share,
                of="image",
                over=f"the coherent band, {figure(band[0])}-{figure(band[1])} Hz, over the "
                "records' usable band within the image's, "
                f"{figure(within[0])}-{figure(within[1])} Hz",
            )
        )
    return ImageReport(
        tuple(measures),
        coherent,
        band=band,
        low=low,
        high=high,
        competing=competing,
        aliased=alias,
        band_share=band_share,
        usable=within if band_share is not None else None,
    )
