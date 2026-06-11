from typing import Callable

from sigproc.base.stream import Stream

from .stream.fk import selection_fk

SELECTION_METHODS: dict[str, Callable[..., Stream | None]] = {
    "fk": selection_fk,
}
