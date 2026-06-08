from typing import Callable

from src.base.beamforming import Beam

from .cross import beamform_cross

BEAMFORMING_METHODS: dict[str, Callable[..., Beam]] = {
    "cross": beamform_cross,
}
