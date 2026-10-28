# SPDX-License-Identifier: Apache-2.0
"""Viewer decor: non-colliding beads that draw the landmarks the routines steer by.

Two families, both hand-fitted to the two-hook layouts and to nothing else:

* **The wrench hooks.** A cyan dot at each hook tip -- the exact point
  ``HangWrenchRoutine`` passes a ring over -- and, around each hook's
  centreline, a cage of beads at the distance ``encloses`` accepts: the moving
  red dot on a wrench is its ring centre, and its assigned hook passes the
  geometric clause while that dot lies inside this finite tube.
* **The screwdriver rack.** The centre and the outline of each evaluated
  hole, at the declared radius, in the colour the routine's marked target
  carries (yellow left, purple right). :func:`.records._check_marked_holes`
  asserts at import that the rack's declared holes stand at these points.

**None of this stands in a built scene.** These are ``StaticSpec``s a task
loader used to append to a variant's scenery undeclared -- 148 on ``wrenches``,
50 on ``screwdrivers`` -- and a document has no clause for scenery it does not
declare, nor a ``StaticSpec`` any place to hang body-local markers the way an
``ObjectSpec`` has ``visual_markers``. Until one of those exists, the geometry
lives here, reachable by name, and a viewer that wants it restages a spec::

    restage(spec, statics=(*spec.statics, *wrench_markers()))

The numbers are the records' own: the hook tip from ``HOOK_TIP_WORLD_*``, the
cage radius from each wrench's ``ring`` feature, the hole radii from the rack's
``hole_*`` features. A bead that disagreed with a feature would be a bead
drawn in the wrong place, and ``tests/test_tool_hanging.py`` says so.
"""

from __future__ import annotations

import numpy as np

from robosimstudio.scene.frames import point_from_surface
from robosimstudio.specs import StaticSpec

from .records import (
    BOARD_X,
    CELL,
    HOOK_STATIONS,
    HOOK_TIP_WORLD_X,
    HOOK_TIP_WORLD_Z,
    SCREWDRIVER_HOLDER_TOP_WORLD_Z,
    SCREWDRIVER_HOLE_CENTER_WORLD_X,
    SCREWDRIVER_LEFT_HOLE_WORLD_Y,
    SCREWDRIVER_RACK,
    SCREWDRIVER_RIGHT_HOLE_WORLD_Y,
    WRENCH_ADJUSTABLE_150,
    WRENCH_ADJUSTABLE_200,
)

__all__ = [
    "screwdriver_alignment_markers",
    "wrench_enclosure_markers",
    "wrench_hook_tip_markers",
    "wrench_markers",
]

WRENCH_HOOK_YS = (HOOK_STATIONS["wrench_right"][0], HOOK_STATIONS["wrench_left"][0])
"""Hook 1 and hook 2 of the wrenches document, in that order: right, then left."""

_IDENTITY = (0.0, 0.0, 0.0, 1.0)


def screwdriver_alignment_markers() -> tuple[StaticSpec, ...]:
    """Non-colliding centres and outlines of the two evaluated rack holes."""
    holes = (
        (
            "left",
            SCREWDRIVER_LEFT_HOLE_WORLD_Y,
            float(SCREWDRIVER_RACK.feature("hole_left").payload["inner_r"]),
            (1.0, 0.9, 0.0),
        ),
        (
            "right",
            SCREWDRIVER_RIGHT_HOLE_WORLD_Y,
            float(SCREWDRIVER_RACK.feature("hole_right").payload["inner_r"]),
            (0.65, 0.15, 0.90),
        ),
    )
    centers = tuple(
        StaticSpec(
            name=f"screwdriver_{side}_hole_center_marker",
            half_extents=(0.0025, 0.0025, 0.0025),
            xform=(
                *point_from_surface(CELL, (SCREWDRIVER_HOLE_CENTER_WORLD_X, hole_y, SCREWDRIVER_HOLDER_TOP_WORLD_Z)),
                *_IDENTITY,
            ),
            collide=False,
            color=color,
        )
        for side, hole_y, _, color in holes
    )
    outlines = tuple(
        StaticSpec(
            name=f"screwdriver_{side}_hole_outline_{index:02d}",
            half_extents=(0.00075, 0.00075, 0.00075),
            xform=(
                *point_from_surface(
                    CELL,
                    (
                        SCREWDRIVER_HOLE_CENTER_WORLD_X + radius * float(np.cos(angle)),
                        hole_y + radius * float(np.sin(angle)),
                        SCREWDRIVER_HOLDER_TOP_WORLD_Z + 0.0015,
                    ),
                ),
                *_IDENTITY,
            ),
            collide=False,
            color=color,
        )
        for side, hole_y, radius, color in holes
        for index, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False))
    )
    return centers + outlines


def wrench_hook_tip_markers() -> tuple[StaticSpec, ...]:
    """Cyan, non-colliding dots at the exact hook-tip points used by the policy."""
    return tuple(
        StaticSpec(
            name=f"wrench_hook_{index}_tip_marker",
            half_extents=(0.003, 0.003, 0.003),
            xform=(*point_from_surface(CELL, (HOOK_TIP_WORLD_X, hook_y, HOOK_TIP_WORLD_Z)), *_IDENTITY),
            collide=False,
            color=(0.0, 1.0, 1.0),
        )
        for index, hook_y in enumerate(WRENCH_HOOK_YS, start=1)
    )


def wrench_enclosure_markers() -> tuple[StaticSpec, ...]:
    """Visual-only cages around the hook centreline accepted by ``encloses``.

    The moving red dot is each wrench's ring centre.  Its assigned hook passes
    the geometric clause while that dot lies inside this finite tube (including
    the rounded tolerance at either end).
    """
    assignments = (
        # hook 1 is the right hook assigned to the larger 17 mm wrench.
        (1, WRENCH_HOOK_YS[0], WRENCH_ADJUSTABLE_200, (0.65, 0.15, 0.90)),
        # hook 2 is the left hook assigned to the smaller 19 mm wrench.
        (2, WRENCH_HOOK_YS[1], WRENCH_ADJUSTABLE_150, (1.0, 0.9, 0.0)),
    )
    markers: list[StaticSpec] = []
    for hook_index, hook_y, wrench, color in assignments:
        allowed = float(wrench.feature("ring").payload["inner_r"])
        for station, axis_x in enumerate(np.linspace(HOOK_TIP_WORLD_X, BOARD_X, 4)):
            for bead, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)):
                markers.append(
                    StaticSpec(
                        name=f"wrench_hook_{hook_index}_enclosure_{station}_{bead:02d}",
                        half_extents=(0.00065, 0.00065, 0.00065),
                        xform=(
                            *point_from_surface(
                                CELL,
                                (
                                    float(axis_x),
                                    hook_y + allowed * float(np.cos(angle)),
                                    HOOK_TIP_WORLD_Z + allowed * float(np.sin(angle)),
                                ),
                            ),
                            *_IDENTITY,
                        ),
                        collide=False,
                        color=color,
                    )
                )
        for bead, axis_x in enumerate(np.linspace(HOOK_TIP_WORLD_X, BOARD_X, 9)):
            markers.append(
                StaticSpec(
                    name=f"wrench_hook_{hook_index}_centerline_{bead:02d}",
                    half_extents=(0.0008, 0.0008, 0.0008),
                    xform=(*point_from_surface(CELL, (float(axis_x), hook_y, HOOK_TIP_WORLD_Z)), *_IDENTITY),
                    collide=False,
                    color=(1.0, 1.0, 1.0),
                )
            )
    return tuple(markers)


def wrench_markers() -> tuple[StaticSpec, ...]:
    """Everything the wrenches scene used to draw: both tips, both cages."""
    return wrench_hook_tip_markers() + wrench_enclosure_markers()
