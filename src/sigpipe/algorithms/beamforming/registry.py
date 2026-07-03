from collections.abc import Callable

from sigpipe.base.beamforming import Beam


def _beamform_cross(*args: object, **kwargs: object) -> Beam:
    # Deferred import: `.cross` requires torch, an optional dependency
    # (`sigpipe[beamforming]`). Keeping it out of this module's top level
    # means importing `sigpipe.algorithms`/`sigpipe.transformers` doesn't
    # require torch unless the "cross" method is actually invoked.
    from .cross import beamform_cross

    return beamform_cross(*args, **kwargs)  # type: ignore[arg-type]


BEAMFORMING_METHODS: dict[str, Callable[..., Beam]] = {
    "cross": _beamform_cross,
}
