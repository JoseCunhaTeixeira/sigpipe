from collections.abc import Callable

from sigpipe.base.stream import Stream

from .flipping import flip

FLIPPING_METHODS: dict[str, Callable[..., Stream]] = {
    "flip": flip,
}
