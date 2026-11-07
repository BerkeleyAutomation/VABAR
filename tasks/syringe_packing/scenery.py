# SPDX-License-Identifier: Apache-2.0
"""Visual-only room furniture, so the record-time video reads as a room.

The tiled recording is the honest artefact -- it is what the policy's cameras
saw -- and without these it showed a floating arm over a near-black void: the
export strips robosuite's room shell, the ground plane draws in Newton's dark
default, and the robotiq host's ``scene_norobot.xml`` takes the export's
pedestal away with its Panda. These pieces put a floor, two back walls and a
pedestal into the *scene itself*, so every renderer -- the record-time tiled
tracer included -- shows the cell standing somewhere, with no re-render pass.

All of it is ``collide=False``: the validated physics must not move by a
millimetre for a coat of paint. The names carry the ``dressing_`` prefix so a
look that builds its own room can hide these and substitute its textured
ones. The suite's ``.suite`` stands all four in every variant.

The spare syringes' fields and the tray-floor arithmetic the sweep generators
need are here too (:func:`pose_entries`), so the recorded ``poses/`` files are
written from the suite's own numbers rather than a tool's copy of them.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from robosimstudio.assetlib import AssetRecord, register_asset
from robosimstudio.assetlib.sim import SimRequirements, contacts_with
from robosimstudio.core.math import quat_to_mat
from robosimstudio.specs import StaticSpec

from .records import (
    SYRINGE_EXPORT_MASS_KG,
    TRAY_FLOOR_COLLIDER_HALF_THICKNESS,
    TRAY_OUTER_HALF,
    TRAY_SETTLE_MARGIN,
    TRAY_WALL_COLLIDER_HALF_THICKNESS,
    TRAY_WALL_THICKNESS,
    cavity_floor_z,
)

__all__ = [
    "DRESSING",
    "PANDA_BASE",
    "SYRINGE_HALF_HEIGHT",
    "SYRINGE_HALF_LENGTH",
    "SYRINGE_HALF_WIDTH",
    "SYRINGE_PACK_COLOR",
    "pose_entries",
    "tray_collider_offsets",
]

SYRINGE_HALF_LENGTH = 0.075
"""Half the syringe's 150 mm barrel-to-flange span, along its own local y."""
SYRINGE_HALF_WIDTH = 0.011783
"""Half the 23.6 mm flange width, along the syringe's local x."""
SYRINGE_HALF_HEIGHT = 0.0068
"""Half the 13.6 mm flange height -- how far the origin sits above the surface
it rests on. See ``routine.py``'s ``GRASP_HANDLE_FRAC`` for the mesh survey
these three numbers come from."""

SYRINGE_PACK_COLOR = (0.90, 0.91, 0.93)
"""The reference splash's syringe white, so the row reads as product rather
than as nine palette-coloured boxes."""

DRESSING_FLOOR_COLOR = (0.42, 0.42, 0.41)
"""Concrete grey, the same value the RTX factory floor's texture averages to."""
DRESSING_WALL_COLOR = (0.58, 0.58, 0.56)
"""Painted plaster, robosuite's own BinsArena wall tone."""
DRESSING_PEDESTAL_COLOR = (0.35, 0.35, 0.37)
"""The export pedestal's steel grey."""
PANDA_BASE = (-0.50, -0.10, 0.912)
"""Where every binding stands the Panda's link0 [m] -- the export's own base
pose, which the standalone import is welded back to. Read off the compiled
robotiq binding's ``robot0_base`` body."""

_SCENERY_SIM = SimRequirements(domain="static", contacts=contacts_with())
"""Drawn, never touched."""


def dressing_floor() -> StaticSpec:
    """A thin slab just proud of the compiled ground plane: concrete rather than the plane's near-black."""
    return StaticSpec(
        name="dressing_floor",
        half_extents=(4.0, 4.0, 0.0005),
        xform=(0.0, 0.0, 0.0005, 0.0, 0.0, 0.0, 1.0),
        world_xform=True,
        collide=False,
        color=DRESSING_FLOOR_COLOR,
    )


def dressing_wall_west() -> StaticSpec:
    """One of the two walls the agentview camera faces (it looks from (+1.3, -0.8) towards the bench)."""
    return StaticSpec(
        name="dressing_wall_west",
        half_extents=(0.02, 4.0, 1.75),
        xform=(-2.4, 0.0, 1.75, 0.0, 0.0, 0.0, 1.0),
        world_xform=True,
        collide=False,
        color=DRESSING_WALL_COLOR,
    )


def dressing_wall_north() -> StaticSpec:
    """The other wall in the frame's background: the -x and +y quadrant."""
    return StaticSpec(
        name="dressing_wall_north",
        half_extents=(4.0, 0.02, 1.75),
        xform=(0.0, 2.8, 1.75, 0.0, 0.0, 0.0, 1.0),
        world_xform=True,
        collide=False,
        color=DRESSING_WALL_COLOR,
    )


def dressing_pedestal() -> StaticSpec:
    """A slim column under :data:`PANDA_BASE`; the resident-Panda binding's own pedestal hides it inside."""
    bx, by, bz = PANDA_BASE
    return StaticSpec(
        name="dressing_pedestal",
        half_extents=(0.10, 0.10, bz / 2.0),
        xform=(bx, by, bz / 2.0, 0.0, 0.0, 0.0, 1.0),
        world_xform=True,
        collide=False,
        color=DRESSING_PEDESTAL_COLOR,
    )


DRESSING = tuple(
    register_asset(
        AssetRecord(
            name=f"sharps_{factory.__name__}",
            summary=summary,
            tags=("dressing", "scenery"),
            features=(),
            sim=_SCENERY_SIM,
            provenance="Authored here for the record-time video; visual-only, see the module docstring.",
            build=factory,
        )
    )
    for factory, summary in (
        (dressing_floor, "A concrete-grey floor slab just proud of the ground plane."),
        (dressing_wall_west, "The plaster wall behind the bench on the -x side."),
        (dressing_wall_north, "The plaster wall behind the bench on the +y side."),
        (dressing_pedestal, "A steel-grey column under the Panda's base."),
    )
)
"""The four dressing records, in the order the suite stands them."""


def pose_entries(
    syringes: dict[str, dict[str, Any]],
    tray_pose: tuple[float, float, float],
    tray_quat: tuple[float, float, float, float],
    *,
    goal: str = "syringe_0",
) -> dict[str, dict[str, Any]]:
    """Every handle's recorded entry for one layout: what ``poses/<variant>.json`` holds under ``objects``.

    ``syringes`` maps each syringe's name to its world ``pose``/``quat``; the
    tray stands at ``tray_pose``/``tray_quat`` (its origin at mid-height).

    What each syringe is made of -- the drawn mesh, the export's diffuse map and
    the CoACD hulls it collides on -- is ``records.syringe``'s, so an entry
    states only what differs. The goal one gets the grip a parallel-jaw hand can
    hold a 150 mm lever by (``friction`` 2.0 -- see the history in
    ``routine.py``); every spare drops the map for the splash's white and takes
    a firmer grip. No entry states a ``collider``: a body colliding on hull
    proxies cannot also be told to wear a primitive, and the old
    ``bounding_box`` was the 23.6 mm flange width applied down the whole barrel.

    **"Spare" names where a syringe came from, not whether it is grasped.** Every
    document of this suite goals *every* syringe it places -- ``tray_clutter_t00``
    asks for ``syringe_0`` and ``syringe_pack_0..5`` in the box, and the pair
    documents are named ``tray_pair_both`` -- so a spare is picked up exactly like
    the goal one. This entry used to carry ``obb_grasp=False``, which reads as a
    reasonable thing to say about decoration and is fatal for a target: the
    compiler then derives no grasp box, ``sim.get_object_obb`` refuses with "has
    no grasp box", and ``target_sg.target_obb`` fails on the first spare the graph
    reaches. Only the appearance and the friction differ.

    The tray's five invisible colliders -- the floor slab and the four walls --
    stand under the tray's own pose and yaw, each one a rigid offset in the
    tray's frame rotated into the world here, because the recorded entry is a
    world placement.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, where in syringes.items():
        entry: dict[str, Any] = {
            "pose": [float(v) for v in where["pose"]],
            "quat": [float(v) for v in where["quat"]],
        }
        if name == goal:
            entry.update(
                body_name=goal,
                joint_name="",
                world_xform=True,
                mass=SYRINGE_EXPORT_MASS_KG,
                friction=2.0,
            )
        else:
            entry.update(
                body_name=name,
                joint_name="",
                asset={"texture": ""},
                world_xform=True,
                mass=SYRINGE_EXPORT_MASS_KG,
                friction=3.0,
                color=list(SYRINGE_PACK_COLOR),
            )
        out[name] = entry
    tx, ty, tz = (float(v) for v in tray_pose)
    q = [float(v) for v in tray_quat]
    out["syringe_tray"] = {"xform": [tx, ty, tz, *q]}
    for name, local in tray_collider_offsets(tz).items():
        dx, dy, dz = quat_to_mat(q) @ np.asarray(local, dtype=float)
        out[name] = {"xform": [tx + float(dx), ty + float(dy), tz + float(dz), *q]}
    return out


def tray_collider_offsets(tray_z: float) -> dict[str, tuple[float, float, float]]:
    """Each invisible tray collider's offset from the tray's origin, in the tray's frame.

    The floor slab hangs its whole thickness below the drawn cavity floor, so it
    grows downward into the bench where nothing can reach it. Each wall spans
    that slab's underside up to the rim and sits its inner face on the drawn
    wall's, thickened outward -- the pair of choices that keeps every added
    surface either inside the bench or inside the drawn tray, and nothing proud
    of what a viewer sees.
    """
    inner_x = TRAY_OUTER_HALF[0] - TRAY_WALL_THICKNESS
    inner_y = TRAY_OUTER_HALF[1] - TRAY_WALL_THICKNESS
    t = TRAY_WALL_COLLIDER_HALF_THICKNESS
    floor_top = cavity_floor_z(tray_z) - tray_z
    floor_bottom = floor_top - 2.0 * TRAY_FLOOR_COLLIDER_HALF_THICKNESS
    rim = TRAY_OUTER_HALF[2]
    wall_z = 0.5 * (rim + floor_bottom)
    return {
        "syringe_tray_floor": (0.0, 0.0, floor_top - TRAY_FLOOR_COLLIDER_HALF_THICKNESS),
        "syringe_tray_wall_xpos": (inner_x + t, 0.0, wall_z),
        "syringe_tray_wall_xneg": (-(inner_x + t), 0.0, wall_z),
        "syringe_tray_wall_ypos": (0.0, inner_y + t, wall_z),
        "syringe_tray_wall_yneg": (0.0, -(inner_y + t), wall_z),
    }


def tray_under(syringe_pose: tuple[float, float, float], syringe_quat: tuple[float, float, float, float]):
    """The tray stood under one syringe on the bench, and that syringe lifted onto its floor.

    What the ``syringe`` variant records: the tray centred on the export's
    settled syringe, yawed to its heading (the z-component of its quaternion
    alone -- the residual roll/pitch is noise a flat tray floor should not
    inherit), and the syringe raised by the wall thickness twice over so it
    lands inside the cavity rather than on the rim.
    """
    ox, oy, oz = (float(v) for v in syringe_pose)
    _, _, qz, qw = (float(v) for v in syringe_quat)
    yaw = 2.0 * math.atan2(qz, qw)
    tray_quat = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    bench_top = oz - SYRINGE_HALF_HEIGHT
    tray_pose = (ox, oy, bench_top + TRAY_OUTER_HALF[2])
    lifted = (ox, oy, cavity_floor_z(tray_pose[2]) + SYRINGE_HALF_HEIGHT + TRAY_SETTLE_MARGIN)
    return tray_pose, tray_quat, lifted
