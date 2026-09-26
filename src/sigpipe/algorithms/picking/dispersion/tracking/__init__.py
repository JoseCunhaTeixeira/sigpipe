"""Automatic picking of dispersion curves: malw-pipe's picker, adapted to Rayleigh waves (the
registry's "tracking" method).

The fundamental mode is the lowest ridge above the aliasing floor; a corridor keeps the tracking
on its branch, and dynamic programming draws the best smooth curve inside it. Each higher mode is
searched above the one below. Every point carries its diagnostics (coherence, pinning), and the
saved curve keeps only the points the data decided.
"""

from .models import PickedMode, PickingParameters
from .modes import pick_modes, pick_tracking

__all__ = ["PickedMode", "PickingParameters", "pick_modes", "pick_tracking"]
