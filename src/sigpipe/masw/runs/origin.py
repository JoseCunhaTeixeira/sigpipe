"""Who made a window's curves: a picker, automatically (the assistant, from its QC log, or PAC's
own automatic picking), or a person in PAC, by hand.

PAC records each change a person makes to a mode's curve (picks_edited.json in the window's
folder, the time of each mode's last change) and each of its own automatic picks of M0
(picks_auto.json); the assistant's picks are its QC log's picking attempts, whose times its
callers read. A curve changed by hand after its last automatic pick is the person's, verified
by them: no check judges it, and nothing automatic changes it. A higher mode is always a
person's: only a person picks one. The assistant's checks of an M0 curve are its own as long as
they came after the curve's last change in PAC."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from sigpipe.base.dispersion_curve import Mode

EDITS_FILE = "picks_edited.json"
AUTO_FILE = "picks_auto.json"
# The trigger of an assistant's attempt that judged a curve it did not pick (PAC's automatic
# pick): a check, not a pick.
JUDGED = "judge"
M0 = Mode("M", 0)
# A record of changes without modes (older ones): its one time stands for every mode.
EVERY_MODE = "*"

type Origin = Literal["auto", "hand"]


def mark_edited(window: Path, mode: Mode) -> None:
    """Record that a person changed `mode`'s curve in window folder `window`, now."""
    now = datetime.now(UTC).isoformat()
    modes = {**_edits(window), mode.label: now}
    (window / EDITS_FILE).write_text(json.dumps({"edited_at": now, "modes": modes}))


def mark_auto(window: Path) -> None:
    """Record that PAC's automatic picking picked window folder `window`'s M0, now."""
    (window / AUTO_FILE).write_text(json.dumps({"picked_at": datetime.now(UTC).isoformat()}))


def edited_at(window: Path, mode: Mode | None = None) -> datetime | None:
    """When a person last changed `mode`'s curve in window folder `window` (any mode's, None);
    None when nobody did."""
    edits = _edits(window)
    labels = list(edits) if mode is None else [mode.label, EVERY_MODE]
    known = [datetime.fromisoformat(edits[label]) for label in labels if label in edits]
    return max(known) if known else None


def auto_at(window: Path) -> datetime | None:
    """When PAC's automatic picking last picked window folder `window`'s M0; None when it never
    did."""
    path = window / AUTO_FILE
    if not path.exists():
        return None
    value = json.loads(path.read_text()).get("picked_at")
    return datetime.fromisoformat(value) if isinstance(value, str) else None


def curve_origin(window: Path, mode: Mode, assistant_at: datetime | None) -> Origin:
    """Who made `mode`'s curve in window folder `window`: automatic when its last automatic pick
    (the assistant's at `assistant_at`, or PAC's own) came after any change by hand, else a
    person's; a higher mode always a person's, and a curve no picker is known to have made (a
    run from before the records)."""
    if mode.number != 0:
        return "hand"
    picks = [at for at in (assistant_at, auto_at(window)) if at is not None]
    if not picks:
        return "hand"
    edited = edited_at(window, mode)
    return "hand" if edited is not None and edited > max(picks) else "auto"


def checks_current(window: Path, checked_at: datetime | None) -> bool:
    """Whether the assistant's latest check of window folder `window`'s M0 (its QC log's latest
    picking attempt, a pick or a judgement, ended at `checked_at`) is of the curve it holds now:
    it came after the curve's last change in PAC, by hand or by PAC's automatic picking."""
    if checked_at is None:
        return False
    changes = [at for at in (auto_at(window), edited_at(window, M0)) if at is not None]
    return not changes or checked_at >= max(changes)


def _edits(window: Path) -> dict[str, str]:
    """The window's record of changes by hand: each mode's time, or an older record's one time
    for every mode."""
    path = window / EDITS_FILE
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    modes = data.get("modes")
    if isinstance(modes, dict):
        return {str(label): str(at) for label, at in modes.items()}
    at = data.get("edited_at")
    return {EVERY_MODE: at} if isinstance(at, str) else {}
