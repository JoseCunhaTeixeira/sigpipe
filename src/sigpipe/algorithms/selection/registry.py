from collections.abc import Callable

from sigpipe.base.stream import Stream

from .stream.fk import selection_fk

STREAM_SELECTION_METHODS: dict[str, Callable[..., Stream | None]] = {
    "fk": selection_fk,
}
