# SPDX-License-Identifier: Apache-2.0
"""The VAB cell, as a frame.

Declares no geometry, and that is the whole of it: the arena is imported whole
from the vendored export, so what a scene contributes here is the **work-surface
frame** the export never named. That frame is a measurement of one particular
cell, which is why it is not a library bench -- and it is shared rather than
task-owned because two tasks stand in it.
"""

from __future__ import annotations

from robosimstudio.scene.registry import register_cell
from robosimstudio.specs import CellSpec

ROBOSUITE_BENCH = register_cell(
    CellSpec(
        name="robosuite_bench",
        ground_z=0.0,
        # The workbench top the bins stand on. The bins' own tops are 20-40 mm
        # above it and are the task's, not the room's.
        surface_z=0.78,
        center=(0.1, 0.015, 0.0),
        half=(0.28, 0.58, 0.02),
        # No statics: the arena is imported whole by the task that owns it. What
        # this scene contributes is the *frame* -- a surface for another robot's
        # mount to be derived against, which is the thing the export never named.
        provenance="robosuite's tool-sort arena, as a work surface: assets/vab_tool_sort/scene.xml",
    )
)
