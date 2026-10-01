"""Who made a window's curves: an automatic picker, or a person in PAC; and whether the
assistant's checks of its M0 are of the curve it holds now."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sigpipe.base.dispersion_curve import Mode
from sigpipe.masw.runs.origin import (
    EDITS_FILE,
    checks_current,
    curve_origin,
    edited_at,
    mark_auto,
    mark_edited,
)

M0, M1 = Mode("M", 0), Mode("M", 1)


def test_a_mode_changed_by_hand_after_the_assistants_pick_is_the_persons(tmp_path: Path) -> None:
    picked = datetime.now(UTC) - timedelta(hours=1)

    assert curve_origin(tmp_path, M0, picked) == "auto"
    mark_edited(tmp_path, M1)
    # M1 is always a person's; M0, left as the assistant picked it, stays its own.
    assert curve_origin(tmp_path, M1, picked) == "hand"
    assert curve_origin(tmp_path, M0, picked) == "auto"
    mark_edited(tmp_path, M0)
    assert curve_origin(tmp_path, M0, picked) == "hand"
    # Picked again by the assistant afterwards: its own again.
    assert curve_origin(tmp_path, M0, datetime.now(UTC) + timedelta(seconds=1)) == "auto"


def test_a_curve_no_picker_is_known_to_have_made_is_a_persons(tmp_path: Path) -> None:
    assert curve_origin(tmp_path, M0, None) == "hand"
    mark_auto(tmp_path)
    assert curve_origin(tmp_path, M0, None) == "auto"


def test_an_older_record_without_modes_stands_for_every_mode(tmp_path: Path) -> None:
    before = datetime.now(UTC) - timedelta(minutes=5)
    (tmp_path / EDITS_FILE).write_text(json.dumps({"edited_at": before.isoformat()}))

    mark_edited(tmp_path, M1)

    # M0's change by hand, from the older record, is kept beside M1's.
    assert edited_at(tmp_path, M0) == before
    assert edited_at(tmp_path, M1) is not None and edited_at(tmp_path, M1) > before
    assert edited_at(tmp_path) == edited_at(tmp_path, M1)


def test_the_assistants_checks_hold_until_the_curve_changes_in_pac(tmp_path: Path) -> None:
    checked = datetime.now(UTC) - timedelta(seconds=1)

    assert checks_current(tmp_path, checked)
    assert not checks_current(tmp_path, None)
    mark_edited(tmp_path, M1)  # another mode: M0's checks stand
    assert checks_current(tmp_path, checked)
    mark_auto(tmp_path)  # M0 picked again in PAC, automatically: another curve
    assert not checks_current(tmp_path, checked)
    assert checks_current(tmp_path, datetime.now(UTC) + timedelta(seconds=1))
    mark_edited(tmp_path, M0)
    assert not checks_current(tmp_path, checked)
