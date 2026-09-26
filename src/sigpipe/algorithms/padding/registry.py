from collections.abc import Callable

from sigpipe.base.stream import Stream

from .padding import pad

PADDING_METHODS: dict[str, Callable[..., Stream]] = {
    "zeros": pad,
}
