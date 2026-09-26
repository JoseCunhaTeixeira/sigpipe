from collections.abc import Callable

from sigpipe.base.stream import Stream

from .shifting import shift

SHIFTING_METHODS: dict[str, Callable[..., Stream]] = {
    "shift": shift,
}
