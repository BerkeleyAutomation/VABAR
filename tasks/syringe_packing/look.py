# SPDX-License-Identifier: Apache-2.0
"""The look every reported number for this suite was measured under: none.

A bare :class:`RenderSpec`, deliberately. It asks each backend for its own
defaults -- for the tiled tracer one key light from (-1, 1, -1), shadows, no
fill -- which is the light the numbers were taken under. Stating nothing does
not mean that any more: ``look_for(None)`` is the house room, whose 0.35 fill
light lifts the overhead frame 13-30 per cent (most in the shadows), and under
it SAM-3's wrist-view syringe mask bled two rows onto the black finger (10 of
12,248 pixels, 43 mm nearer the camera), the convex hull the graph fits its
attached collider to grew a 47 mm spur toward the wrist, and cuRobo refused
every carry heading: 0/3 on tray_pair_t00, t01, t03 against 4/4 on the tree
the numbers come from, same documents, same flags. Measured 2026-09-13; those
three were trials of the single-syringe sweep, dropped on 2026-09-15, so the
reading stands but the documents it was taken on do not.
"""

from __future__ import annotations

from robosimstudio.specs import RenderSpec

__all__ = ["BARE"]

BARE = RenderSpec()
"""What ``(:render "assets.look:BARE")`` names."""
