from collections.abc import Callable

from sigpipe.base.stream import Stream

from .mutting import mute

MUTTING_METHODS: dict[str, Callable[..., Stream]] = {
    "mute": mute,
}
