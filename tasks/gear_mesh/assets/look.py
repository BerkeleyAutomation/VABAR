# SPDX-License-Identifier: Apache-2.0
"""How the gear bench is lit under the path tracer.

**The house room, three times the key.** ``bench_075`` stands in the default
room (:data:`~robosimstudio.core.render.room.HANGAR_CLOSED`) and its bench and
mat are the palette's dark fixture greys -- a display value of 72 on the bench,
103 on the mat -- so under a renderer calibrated to put those values where the
tiled tracer puts them (``solvers/render/rtx.py``, ``PALETTE_IS_DISPLAY``) the
frame is dark: overhead mean 91 against 147 for the tool-sort bench, which is
brushed steel under its own shop. The tiled tracer does not read
``key_intensity`` at all (measured: bench 72 at key 1.0 and at 3.0), so the
panel can be turned up for the pictures without moving a pixel the vision graph
was tuned on. Measured on ``gearmesh-easy-single-01``'s overhead camera at the
key below: bench 97, mat 112, the pinion (252,181,70) still orange to the graph
((r-g)/(r-b) 0.39 against 0.40 on tiled), brass 0.25 against 0.26.
"""

from __future__ import annotations

from dataclasses import replace

from robosimstudio.core.render.room import HANGAR_CLOSED
from robosimstudio.core.render.spec import RenderSpec

__all__ = ["GEAR_MESH_LOOK"]

GEAR_MESH_LOOK: RenderSpec = replace(HANGAR_CLOSED, key_intensity=3.0, ambient=(0.9, 0.9, 0.9), quality="preview")
"""Every ``gear_mesh`` variant renders under this: the house room with the lab panel at three times the house key."""
