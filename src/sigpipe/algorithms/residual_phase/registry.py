from collections.abc import Callable

from sigpipe.base.stream import Stream

from .arrival import arrival_residual_phase

RESIDUAL_PHASE_METHODS: dict[str, Callable[..., Stream]] = {
    "arrival": arrival_residual_phase,
}
