# SPDX-License-Identifier: Apache-2.0
"""The cell this task is measured in.

A scene rather than a constant, and a *task-owned* scene rather than a library
one: this is a particular pegboard workcell somebody measured, so what it
declares is only correct for the board this suite hangs things on. The generic
benches the library ships are tables at a stated height and belong to nobody.
"""

from __future__ import annotations

from robosimstudio.core.render.palette import GROUND, SORT_TARGET
from robosimstudio.scene.registry import register_cell
from robosimstudio.specs import CellSpec, StaticSpec


def _table(top_z: float, half_xy, *, solid: bool, color=GROUND) -> StaticSpec:
    """The cell's table, as a static box whose top is the work surface.

    ``solid`` is the only real choice here. The three suites that shared this
    cell each drew their own slab and disagreed about it -- 30 mm, a full-height
    block, and 20 mm of non-colliding decor 5 mm low -- which is what a table
    being nobody's responsibility looks like. One table, and a solid one: a
    30 mm slab floating at knee height reads as an error, and nothing in any of
    the three touches below the top.

    ``color`` because a cell may repaint the one table it declares -- the
    tool-hanging bench matches its table to its holders -- without that being a
    second table.
    """
    half_z = top_z / 2.0 if solid else 0.015
    return StaticSpec(
        name="table",
        half_extents=(float(half_xy[0]), float(half_xy[1]), half_z),
        xform=(0.6, 0.0, top_z - half_z, 0.0, 0.0, 0.0, 1.0),
        color=color,
    )


TOOL_HANGING_BENCH = register_cell(
    CellSpec(
        name="tool_hanging_bench",
        ground_z=0.0,
        surface_z=0.728,
        center=(0.6, 0.0, 0.0),
        half=(0.30, 0.40, 0.364),
        statics=(_table(0.728, (0.30, 0.40), solid=True, color=SORT_TARGET),),
        provenance="the YAM arena with a task-specific #978F66 tool-hanging table",
    )
)
