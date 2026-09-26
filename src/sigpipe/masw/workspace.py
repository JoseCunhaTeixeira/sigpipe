"""Where the MASW layer reads profiles and writes runs, and how many windows it processes at
once: what it needs of an application's settings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Workspace(Protocol):
    """An application's settings, as the MASW layer reads them: PAC's and PACo's provide these
    three values."""

    @property
    def input_dir(self) -> Path:
        """One folder per profile."""
        ...

    @property
    def output_dir(self) -> Path:
        """One folder per profile, holding one folder per run."""
        ...

    @property
    def workers(self) -> int:
        """Records or windows processed at once, one worker process each."""
        ...


@dataclass(frozen=True, slots=True)
class Folders:
    """A Workspace from its values."""

    input_dir: Path
    output_dir: Path
    workers: int = 1
