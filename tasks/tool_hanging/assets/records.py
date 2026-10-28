# SPDX-License-Identifier: Apache-2.0
"""The tool-hanging family: four tool classes, their holders, and the wall.

Every hanging goal in the suite is geometric, and these records carry the
geometry: a wrench declares the ``loop`` its ring is, a scissor pair the loops
its two handle bows are, a screwdriver the ``rod`` its shaft is, the shared pegboard
hook the rod its arm is, and the rack the loops its holes are. ``threaded``
over those pairs is what "hung" and "inserted" mean here -- the atom's own
provenance says it was designed for these fixtures -- and the same measured
constants steer the scripted routines, so the evaluator and the baseline
cannot disagree about where a landmark is.

**Where the numbers come from.** Ring and bow centres are the branch suite's
measured CAD landmarks (the routines' job tables carry the same values); radii
and the hook's arm section were measured off the vendored meshes with trimesh
at import (vertex distance to the declared centre for a loop, tip-band extent
for the hook). All payloads are in the mesh's own frame after its import
scale, in metres, which is the frame the feature reader transforms.

**The hook's rod is a flat bar, and the radius is its half-thickness.** The
arm's section is 8.4 mm tall by 4.98 mm thick; what must pass *through* a ring
is bounded by the thin dimension, so ``radius`` declares 2.49 mm and a ring
resting on the arm's top edge reads offset + radius comfortably inside its
inner radius, while one leaning against the hook face reads no crossing at
all -- which is the discrimination the atom exists to make.

**Each record places itself.** The measured mounting transforms, spawn poses,
import scales and the CoACD hull sets the tools collide through used to be the
loader's; they are the *factories* here, and a record whose thing stands in
one place names its factory (``AssetRecord.build``), so a document typing
``ring_spanner_19 - wrench_adjustable_150`` lowers to the placed ``ObjectSpec``
with no Python beside it. A type the documents stand in more than one place --
the hook, twice per board and once centred; the hammer holder; the plier
support -- is several things: the geometry record carries the mesh and the
features and names no factory, and each *station* it stands at is a record of
its own (``pegboard_hook_wrench_left``, ``plier_support_right``) whose factory
stands that geometry there -- see :data:`HOOK_STATIONS` and its siblings, and
:func:`_stationed`. A document names the station; the station names the
placement. The factories come before the records, because a record names the
function that places it.

**Where the hook's tip is** -- :data:`HOOK_TIP_WORLD_X`, :data:`HOOK_TIP_WORLD_Z`
-- is stated here, beside the hook, because the routines steer by it and the
viewer decor in :mod:`.decor` is drawn at it: one fact, one place.

**The A/B knob.** :data:`SCISSOR_1_COLLISION_METHOD` selects scissor 1's
collider set: the automatic CoACD decomposition (default) or the hand-authored
30-piece compound proxy, in an otherwise identical scene. Change only this
value to compare them.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.assetlib.lower import declared_asset
from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.registry import register_asset
from robosimstudio.assetlib.semantics.features import FeatureSpec
from robosimstudio.assetlib.sim import SimRequirements, contacts_with
from robosimstudio.core.render.palette import SORT_SOURCE, SORT_TRAY
from robosimstudio.scene.frames import point_from_surface
from robosimstudio.specs import ObjectSpec, StaticSpec

from ..scenes import TOOL_HANGING_BENCH

__all__ = [
    "HAMMER_HOLDER",
    "HAMMER_HOLDER_STATIONS",
    "HANGING_HAMMER_1",
    "HANGING_HAMMER_2",
    "HOOK_STATIONS",
    "HOOK_TIP_WORLD_X",
    "HOOK_TIP_WORLD_Z",
    "PEGBOARD_HOOK",
    "PLIER_COMBINATION",
    "PLIER_CUTTER",
    "PLIER_SUPPORT",
    "PLIER_SUPPORT_STATIONS",
    "SCISSOR_PAIR_1",
    "SCISSOR_PAIR_2",
    "SCREWDRIVER_BOX",
    "SCREWDRIVER_GRIP",
    "SCREWDRIVER_RACK",
    "SCREWDRIVER_SLIM",
    "SKADIS_PEGBOARD",
    "STATIONED",
    "WRENCH_ADJUSTABLE_150",
    "WRENCH_ADJUSTABLE_200",
    "WRENCH_ADJUSTABLE_260",
    "hammer_holder",
    "hanging_hammer_1",
    "hanging_hammer_2",
    "pegboard_hook",
    "plier_combination",
    "plier_cutter",
    "plier_support",
    "scissor_pair_1",
    "scissor_pair_2",
    "screwdriver_box",
    "screwdriver_grip",
    "screwdriver_rack",
    "screwdriver_slim",
    "skadis_pegboard",
    "wrench_adjustable_150",
    "wrench_adjustable_200",
    "wrench_adjustable_260",
]

_TOOL_SIM = SimRequirements(domain="rigid", contacts=contacts_with("surface", "robot", "fixtures"))
_FIXTURE_SIM = SimRequirements(domain="static", contacts=contacts_with("robot", "fixtures"))

HOOK_ARM_HALF_THICKNESS = 0.00249
"""Half the hook's 4.98 mm extrusion -- the dimension a ring must clear [m]."""

HOOK_ARM_CENTER_Y = 0.0042
"""The arm section's centre height in the hook's own frame [m]: the tip band
spans local y 0 to 8.4 mm, measured off hook_2.stl."""

# --------------------------------------------------------------- the layout
ASSETS_DIR = Path(__file__).resolve().parent
"""This task's own geometry.

The collision hulls below are decomposed from the meshes beside them, so they
are found here rather than through a global asset root -- the same rule the
records themselves follow, and what makes the directory movable."""

CELL = TOOL_HANGING_BENCH
"""The repainted 0.728 m cell; the hammers stand in plain bench_0728, whose
frame is identical, so one set of surface-frame numbers serves both."""

LEVEL1_Y = 0.0
"""Board-centred y coordinate shared by the one-object variants."""

# ------------------------------------------------------------------- the wall
BOARD_X = 0.880
"""Far table edge: the robot is on the low-x side of the work surface."""
BOARD_FRONT_LOCAL_Y = 0.02509444
BOARD_CENTER_Z = 1.00800769
BOARD_COLOR = SORT_SOURCE
"""The pegboard. The reference branch's mid-brown, which *is* the palette entry.

Was written out as ``(0.6, 0.372549020, 0.184313725)`` -- exactly ``#995F2F``,
exactly :data:`~robosimstudio.core.render.palette.SORT_SOURCE`, transcribed to
nine decimals by hand. That is the failure the palette's own test docstring
names: a hex code turned into floats cannot be diffed against the palette it came
from, so the two drifted apart in principle and only happened to agree."""

HANGER_COLOR = SORT_TRAY
"""The reference branch's shared light-beige color for every tool holder.

Likewise ``#E4D6A9`` by hand. Same colour, now spelled as the thing it is."""

# ------------------------------------------------------------------ the tools
# Temporary grasp-reach experiment: place both wrench rings 50 mm closer to
# the robot (the robot stands on the low-world-X side of the work surface).
WRENCH_RING_WORLD_X = 0.610
WRENCH_LEFT_START_Y = 0.140
WRENCH_RIGHT_START_Y = -0.140
WRENCH_150_MASS = 0.18
WRENCH_200_MASS = 0.10
"""Rounded steel-equivalent masses for the 23.44 and 48.61 cm^3 meshes."""
WRENCH_150_COLLISION_ROOT = str(ASSETS_DIR) + "/wrenches/wrench_1/collision_coacd"
WRENCH_150_COLLISION_PATHS = tuple(f"{WRENCH_150_COLLISION_ROOT}/hull_{index:02d}.obj" for index in range(39))
WRENCH_200_COLLISION_ROOT = str(ASSETS_DIR) + "/wrenches/wrench_2/collision_coacd"
WRENCH_200_COLLISION_PATHS = tuple(f"{WRENCH_200_COLLISION_ROOT}/hull_{index:02d}.obj" for index in range(30))
"""The re-authored wrenches, at 39 and 30 hulls where the replaced pair had 13
and 14. The effective-density constants that stood here are gone with the meshes
they corrected: `wrench_1`/`wrench_2` state their masses outright."""

SCREWDRIVER_1_COLLISION_ROOT = str(ASSETS_DIR) + "/screwdrivers/screwdriver_1/collision_coacd"
SCREWDRIVER_1_COLLISION_PATHS = tuple(f"{SCREWDRIVER_1_COLLISION_ROOT}/hull_{i:02d}.obj" for i in range(2))
SCREWDRIVER_2_COLLISION_ROOT = str(ASSETS_DIR) + "/screwdrivers/screwdriver_2/collision_coacd"
SCREWDRIVER_2_COLLISION_PATHS = tuple(f"{SCREWDRIVER_2_COLLISION_ROOT}/hull_{i:02d}.obj" for i in range(4))
SCREWDRIVER_2_TEXTURE = str(ASSETS_DIR) + "/screwdrivers/screwdriver_2/textures/diffuse.png"
"""Beside the declaration, like every other path in this file.

It was the one that kept naming the vendored tree, from before the task-centric
move carried this geometry into the task directory. The meshes survived that
move because ``AssetRecord.files`` is normalised at registration
(``assetlib.registry._resolve_beside``); a texture is set on the ``AssetSpec``
and never passes through it, so the string stayed true of a directory that no
longer exists and ``screwdrivers`` stopped building at all -- both policies,
``FileNotFoundError`` five seconds in, before a single step."""

SCISSOR_1_COLLISION_ROOT = str(ASSETS_DIR) + "/scissors/scissor_1/collision_compound"
SCISSOR_1_COLLISION_PATHS = tuple(
    f"{SCISSOR_1_COLLISION_ROOT}/handle_{side}_{index:02d}.obj" for side in ("left", "right") for index in range(12)
) + tuple(
    f"{SCISSOR_1_COLLISION_ROOT}/{name}.obj"
    for name in ("neck_left", "neck_right", "blade_left", "blade_right", "pivot", "bridge")
)
SCISSOR_1_COACD_ROOT = str(ASSETS_DIR) + "/scissors/scissor_1/collision_coacd"
SCISSOR_1_COACD_PATHS = tuple(f"{SCISSOR_1_COACD_ROOT}/hull_{index:02d}.obj" for index in range(12))
# Change only this value to compare the hand-authored compound proxy against
# the automatic decomposition in the identical scene.
SCISSOR_1_COLLISION_METHOD = "coacd"
SCISSOR_1_COLLIDERS = {
    "compound": SCISSOR_1_COLLISION_PATHS,
    "coacd": SCISSOR_1_COACD_PATHS,
}
SCISSOR_2_COACD_ROOT = str(ASSETS_DIR) + "/scissors/scissor_2/collision_coacd"
SCISSOR_2_COACD_PATHS = tuple(f"{SCISSOR_2_COACD_ROOT}/hull_{index:02d}.obj" for index in range(10))

PLIER_1_COLLISION_ROOT = str(ASSETS_DIR) + "/pliers/plier_1/collision_coacd"
PLIER_2_COLLISION_ROOT = str(ASSETS_DIR) + "/pliers/plier_2/collision_coacd"
PLIER_1_COLLISION_PATHS = tuple(f"{PLIER_1_COLLISION_ROOT}/hull_{index:02d}.obj" for index in range(24))
PLIER_2_COLLISION_PATHS = tuple(f"{PLIER_2_COLLISION_ROOT}/hull_{index:02d}.obj" for index in range(24))

HAMMER_SCALE = 0.01
"""The supplied hammer meshes are authored in centimetres."""

# --------------------------------------------------------------- the holders
HOLDER_SCALE = 0.001
"""The supplied holders' authored millimetre scale, converted to metres."""
SCREWDRIVER_HOLDER_SCALE = 0.00120
SCREWDRIVER_HOLE_CENTER_WORLD_X = 0.856
SCREWDRIVER_LEFT_HOLE_WORLD_Y = 0.0198
SCREWDRIVER_RIGHT_HOLE_WORLD_Y = -0.0198
SCREWDRIVER_HOLDER_TOP_WORLD_Z = 0.8836
"""The 120%-scale rack's measured opening centres."""
SCREWDRIVER_BOX_SCALE = 1.0
"""The box is authored in metres, so it imports at 1:1 -- unlike the supplied
rack above, whose millimetre CAD is scaled to 120%. The two fixtures share a
mount and a hole row and nothing else, which is why they are separate
constants rather than one retuned pair."""
RACK_MOUNT = (BOARD_X, 0.093, 0.850)
"""Rack origin: y=93 mm centres its 120%-scale 155 mm row while keeping
the x=0 mounting back against the board."""
SCISSOR_HOOK_BOUNDS_CENTER_MM = (30.546192, 26.484238, 2.491525)
"""hook_2.stl's rotated CAD-bounds centre, removed so each hook stays centred
on its pin site."""
HOOK_MOUNT_WORLD_Z = 1.155
"""A hook's pin site on the board [m], before the CAD-bounds centre is removed."""
# The pegboard wall's front plane and the shared hook mesh's measured geometry:
# the scissors and the wrenches hang on the same 516-triangle hook, mounted by
# the same recipe, so its tip is one fact. The routines' job tables steer by
# it and the viewer decor draws it; both read it from here.
HOOK_REACH_TOWARD_ROBOT = 0.0611
HOOK_TIP_WORLD_X = BOARD_X - HOOK_REACH_TOWARD_ROBOT
# Exact centre of the straight 0..8.4 mm hook arm after mounting: the hook
# origin is at 1.155 - 26.484238 mm and its arm centre is local y=4.2 mm.
HOOK_TIP_WORLD_Z = 1.132715762
PLIER_SUPPORT_SCALE = 0.0012
PLIER_SUPPORT_CENTER_X_MM = 136.166073
PLIER_SUPPORT_BACK_Y_MM = 108.493484
PLIER_SUPPORT_CENTER_Z_MM = 13.819462
PLIER_SUPPORT_CENTER_WORLD_Z = 0.920
PLIER_SUPPORT_WORLD_YS = (0.075, -0.075)
PLIER_SUPPORT_TIP_WORLD_X = 0.85229
HAMMER_HOLDER_SCALE = 0.001

# ---------------------------------------------------------------- the stations
# The y each multiply-stood type stands at, by the station's name. A station is
# what a document names -- ``wrench_hook_2 - pegboard_hook_wrench_left`` -- and
# what its record's factory reads; the world x and z are the type's own.
HOOK_STATIONS: dict[str, tuple[float, str]] = {
    # The wrenches' tighter row: hook 1 is the right hook, assigned to the
    # larger 17 mm wrench; hook 2 the left, assigned to the 19 mm.
    "wrench_right": (-0.060, "the wrenches' right hook, 60 mm right of the board centre"),
    "wrench_left": (0.060, "the wrenches' left hook, 60 mm left of the board centre"),
    # The scissors' wider row, on the same mesh by the same recipe.
    "scissor_right": (-0.100, "the scissors' right hook, 100 mm right of the board centre"),
    "scissor_left": (0.100, "the scissors' left hook, 100 mm left of the board centre"),
    # Level 2 uses the same symmetric 100 mm target spacing for every family.
    "level2_right": (-0.100, "the Level-2 right hook, 100 mm right of the board centre"),
    "level2_left": (0.100, "the Level-2 left hook, 100 mm left of the board centre"),
    # The one-object reductions: one hook, centred on the board.
    "centre": (LEVEL1_Y, "the one-object reductions' hook, centred on the board"),
}
HAMMER_HOLDER_STATIONS: dict[str, tuple[float, str]] = {
    # Both on the robot's left, side by side; the hammers document hangs
    # either hammer on either one.
    "inner": (0.105, "the inner of the two holders, 105 mm left of the board centre"),
    "outer": (0.305, "the outer of the two holders, 305 mm left of the board centre"),
}
PLIER_SUPPORT_STATIONS: dict[str, tuple[float, str]] = {
    "left": (PLIER_SUPPORT_WORLD_YS[0], "the pliers' left support, 75 mm left of the board centre"),
    "right": (PLIER_SUPPORT_WORLD_YS[1], "the pliers' right support, 75 mm right of the board centre"),
    "centre": (LEVEL1_Y, "the one-object reduction's support, centred on the board"),
    "level2_left": (0.100, "the Level-2 left support, 100 mm left of the board centre"),
    "level2_right": (-0.100, "the Level-2 right support, 100 mm right of the board centre"),
}


# --------------------------------------------------- the fixtures, placed
def skadis_pegboard() -> StaticSpec:
    """The reference branch's single-piece SKADIS wall in flat BOARD_COLOR."""
    s = float(np.sqrt(0.5))
    return StaticSpec(
        name="pegboard",
        asset=declared_asset(
            SKADIS_PEGBOARD,
            approximate="bounding_box",
            keep_visual=True,
        ),
        xform=(
            *point_from_surface(CELL, (BOARD_X + BOARD_FRONT_LOCAL_Y, 0.0, BOARD_CENTER_Z)),
            0.0,
            0.0,
            s,
            s,
        ),
        color=BOARD_COLOR,
    )


def pegboard_hook(name: str, y: float) -> StaticSpec:
    """One hook_2 hook flush to the board front, under the document's name.

    The hook is an extruded side profile: local x is its 61 mm forward reach,
    local y its 53 mm height, local z its 4.98 mm thickness. Rotate local x to
    world -x (toward the robot), local y to world +z, local z to world +y;
    local x=0 stays exactly on BOARD_X, and removing the local yz bounds
    centre keeps each hook centred on its former pin site.

    Takes its placement because the documents stand this one type at five
    different y's -- two per board for the wrenches, two wider for the
    scissors, and one centred for either level-1 reduction -- which is the
    fact no zero-argument factory can carry; each of :data:`HOOK_STATIONS` is
    a record of its own that calls this with its y.
    """
    s = float(np.sqrt(0.5))
    _, center_y_mm, center_z_mm = SCISSOR_HOOK_BOUNDS_CENTER_MM
    return StaticSpec(
        name=name,
        asset=declared_asset(
            PEGBOARD_HOOK,
            scale=HOLDER_SCALE,
            approximate=None,
            keep_visual=True,
        ),
        xform=(
            *point_from_surface(
                CELL,
                (
                    BOARD_X,
                    y - center_z_mm * HOLDER_SCALE,
                    HOOK_MOUNT_WORLD_Z - center_y_mm * HOLDER_SCALE,
                ),
            ),
            0.0,
            s,
            s,
            0.0,
        ),
        color=HANGER_COLOR,
    )


def screwdriver_rack() -> StaticSpec:
    """The 155 mm graduated rack, centred on the board with its holes up.

    A 180-degree world-z turn puts the x=0 mounting back at the board and
    extends the rack toward the robot. The level-1 document stands a different
    fixture in the same place -- see :func:`screwdriver_box`.
    """
    return StaticSpec(
        name="rack",
        asset=declared_asset(
            SCREWDRIVER_RACK,
            scale=SCREWDRIVER_HOLDER_SCALE,
            approximate=None,
            keep_visual=True,
        ),
        xform=(*point_from_surface(CELL, RACK_MOUNT), 0.0, 0.0, 1.0, 0.0),
        color=HANGER_COLOR,
    )


def screwdriver_box() -> StaticSpec:
    """The screwdriver box shifted so its nominal target is at world y=0.

    A different asset from :func:`screwdriver_rack`, under the same static
    name. The full-size document threads a screwdriver's shank through the
    rack's 8 mm hole; this one drops it into a 110 mm open cup and scores the
    cup's interior as a region. Sharing one record would mean retuning
    ``InsertScrewdriverRoutine`` against the cup, which four rollouts say does
    not hang at all.
    """
    mount = (RACK_MOUNT[0], RACK_MOUNT[1] - SCREWDRIVER_LEFT_HOLE_WORLD_Y, RACK_MOUNT[2])
    return StaticSpec(
        name="rack",
        asset=declared_asset(
            SCREWDRIVER_BOX,
            scale=SCREWDRIVER_BOX_SCALE,
            approximate=None,
            keep_visual=True,
        ),
        xform=(*point_from_surface(CELL, mount), 0.0, 0.0, 1.0, 0.0),
        color=HANGER_COLOR,
    )


def hammer_holder(name: str, y: float) -> StaticSpec:
    """One hammer holder, level on the pegboard, under the document's name.

    Holder CAD bounds are x=[155, 255], y=[-57.6, 6], z=[0, 60] mm. A -90
    degree z rotation maps its long x direction across the board and its
    negative-y open side toward the robot; its y=+6 mm mounting back sits on
    BOARD_X. Takes its placement: the hammers document stands two side by side,
    each a record of its own -- :data:`HAMMER_HOLDER_STATIONS`.
    """
    s = float(np.sqrt(0.5))
    return StaticSpec(
        name=name,
        asset=declared_asset(
            HAMMER_HOLDER,
            scale=HAMMER_HOLDER_SCALE,
            approximate=None,
            keep_visual=True,
        ),
        xform=(*point_from_surface(CELL, (BOARD_X - 0.006, y, 1.120)), 0.0, 0.0, -s, s),
        color=HANGER_COLOR,
    )


def plier_support(name: str, y: float) -> StaticSpec:
    """One triangular support below the board midpoint, under the document's name.

    The STL is upright along local z; its local +y edge is the mounting back
    and the functional triangular projection extends along local -y. A single
    -90-degree yaw points that projection toward the robot. Takes its
    placement: the pliers document stands one at each of
    :data:`PLIER_SUPPORT_WORLD_YS`, the level-1 reduction one at
    :data:`LEVEL1_Y`, exactly midway -- :data:`PLIER_SUPPORT_STATIONS`, each a
    record of its own.
    """
    s = float(np.sqrt(0.5))
    return StaticSpec(
        name=name,
        asset=declared_asset(
            PLIER_SUPPORT,
            scale=PLIER_SUPPORT_SCALE,
            approximate=None,
            keep_visual=True,
        ),
        xform=(
            *point_from_surface(
                CELL,
                (
                    BOARD_X - PLIER_SUPPORT_BACK_Y_MM * PLIER_SUPPORT_SCALE,
                    y + PLIER_SUPPORT_CENTER_X_MM * PLIER_SUPPORT_SCALE,
                    PLIER_SUPPORT_CENTER_WORLD_Z - PLIER_SUPPORT_CENTER_Z_MM * PLIER_SUPPORT_SCALE,
                ),
            ),
            0.0,
            0.0,
            -s,
            s,
        ),
        color=HANGER_COLOR,
    )


# ------------------------------------------------------ the tools, placed
def _wrench(
    name: str,
    record: AssetRecord,
    collision_paths: tuple[str, ...],
    start_y: float,
    color: tuple[float, float, float],
    mass: float,
    half_thickness: float,
) -> ObjectSpec:
    """A wrench flat on the table with its closed ring boardward.

    Red marks the measured centre; green beads trace the declared inner
    radius. The overlay is lifted just above the wrench face so neither colour
    is hidden inside the rendered steel.
    """
    ring_center = tuple(record.feature("ring").payload["center"])
    inner_r = float(record.feature("ring").payload["inner_r"])
    return ObjectSpec(
        name=name,
        category=record.name,
        asset=declared_asset(
            record,
            collision_paths=collision_paths,
            approximate=None,
            keep_visual=True,
        ),
        pose=point_from_surface(
            CELL,
            (
                WRENCH_RING_WORLD_X - ring_center[0],
                start_y,
                CELL.surface_z + half_thickness,
            ),
        ),
        # Local +x runs toward the pegboard and local +y (thickness) is up.
        quat=(float(np.sqrt(0.5)), 0.0, 0.0, float(np.sqrt(0.5))),
        mass=mass,
        friction=1.5,
        color=color,
        visual_markers=(
            (
                (ring_center[0], half_thickness + 0.0015, ring_center[2]),
                0.005,
                (1.0, 0.0, 0.0),
            ),
            *(
                (
                    (
                        ring_center[0] + inner_r * float(np.cos(angle)),
                        half_thickness + 0.0015,
                        ring_center[2] + inner_r * float(np.sin(angle)),
                    ),
                    0.0011,
                    (0.0, 1.0, 0.0),
                )
                for angle in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
            ),
        ),
    )


def wrench_adjustable_150() -> ObjectSpec:
    """The 150 mm wrench -- the document's ``ring_spanner_19`` -- on the left."""
    return _wrench(
        "ring_spanner_19",
        WRENCH_ADJUSTABLE_150,
        WRENCH_150_COLLISION_PATHS,
        WRENCH_LEFT_START_Y,
        (0.24, 0.27, 0.30),
        WRENCH_150_MASS,
        0.006,
    )


def wrench_adjustable_200() -> ObjectSpec:
    """The 200 mm wrench -- the document's ``ring_spanner_17`` -- on the right."""
    return _wrench(
        "ring_spanner_17",
        WRENCH_ADJUSTABLE_200,
        WRENCH_200_COLLISION_PATHS,
        WRENCH_RIGHT_START_Y,
        (0.48, 0.51, 0.54),
        WRENCH_200_MASS,
        0.007,
    )


def wrench_adjustable_260() -> ObjectSpec:
    """A 30%-enlarged wrench at the table-centre task pose."""
    record = WRENCH_ADJUSTABLE_260
    half_thickness = 0.007 * 1.30
    return ObjectSpec(
        name="wrench_1",
        category=record.name,
        asset=declared_asset(
            record,
            collision_paths=WRENCH_200_COLLISION_PATHS,
            scale=1.30,
            approximate=None,
            keep_visual=True,
        ),
        pose=point_from_surface(CELL, (0.600, LEVEL1_Y, CELL.surface_z + half_thickness)),
        quat=(float(np.sqrt(0.5)), 0.0, 0.0, float(np.sqrt(0.5))),
        mass=WRENCH_200_MASS,
        friction=1.5,
        color=(0.48, 0.51, 0.54),
    )


def screwdriver_slim() -> ObjectSpec:
    """The slim screwdriver, horizontal with its handle toward the pegboard.

    The level-1 document stands the same tool at the table centre: its
    ``(:start-poses poses/level1_start.json)`` restates the pose over this
    spec and leaves the rest -- orientation, hulls, the tip bead -- as built.
    """
    return ObjectSpec(
        name="screwdriver_1",
        category=SCREWDRIVER_SLIM.name,
        asset=declared_asset(
            SCREWDRIVER_SLIM,
            collision_paths=SCREWDRIVER_1_COLLISION_PATHS,
            approximate=None,
            keep_visual=True,
            material_colors=True,
        ),
        pose=point_from_surface(CELL, (0.710, 0.140, CELL.surface_z + 0.0105)),
        quat=(0.0, 0.0, 1.0, 0.0),
        density=1800.0,
        friction=0.8,
        visual_markers=(((0.175, 0.0, 0.0), 0.003, (1.0, 0.0, 0.0)),),
    )


def screwdriver_grip() -> ObjectSpec:
    """The fat-handled screwdriver, horizontal with its handle toward the pegboard."""
    return ObjectSpec(
        name="screwdriver_2",
        category=SCREWDRIVER_GRIP.name,
        asset=declared_asset(
            SCREWDRIVER_GRIP,
            collision_paths=SCREWDRIVER_2_COLLISION_PATHS,
            approximate=None,
            keep_visual=True,
            # The authored diffuse map, stated rather than discovered off
            # the .mtl -- AssetSpec.texture's whole contract.
            texture=SCREWDRIVER_2_TEXTURE,
        ),
        pose=point_from_surface(CELL, (0.615, -0.140, CELL.surface_z + 0.0180)),
        quat=(0.0, 0.0, 0.0, 1.0),
        density=1600.0,
        friction=0.8,
        visual_markers=(((-0.100, -0.005697, 0.001303), 0.003, (1.0, 0.0, 0.0)),),
    )


def _hammer(name: str, record: AssetRecord, y: float, color: tuple[float, float, float]) -> ObjectSpec:
    """A supplied, life-sized hand hammer in the tabletop row.

    Rotate +90 degrees about the long local-x axis to rest on the hammer's
    broad side, then turn it end-for-end so the handle points toward the robot.
    """
    return ObjectSpec(
        name=name,
        category=record.name,
        asset=declared_asset(record, scale=HAMMER_SCALE, approximate="bounding_box", keep_visual=True),
        pose=point_from_surface(CELL, (0.700, y, CELL.surface_z + 0.031137 + 0.004)),
        quat=(0.0, float(np.sqrt(0.5)), float(np.sqrt(0.5)), 0.0),
        density=4500.0,
        friction=0.7,
        color=color,
    )


def hanging_hammer_1() -> ObjectSpec:
    """The first hammer, on the left of the row."""
    return _hammer("hammer_1", HANGING_HAMMER_1, 0.18, (1.00, 0.55, 0.05))


def hanging_hammer_2() -> ObjectSpec:
    """The second hammer, on the right of the row."""
    return _hammer("hammer_2", HANGING_HAMMER_2, -0.18, (0.05, 0.70, 1.00))


def _scissor_ring_markers(
    record: AssetRecord, face_z: float
) -> tuple[tuple[tuple[float, float, float], float, tuple[float, float, float]], ...]:
    """Red centres and green evaluated-radius circles for both handle rings."""
    markers = []
    for feature_name in ("bow", "bow_other"):
        loop = record.feature(feature_name).payload
        center = np.asarray(loop["center"], dtype=np.float64)
        radius = float(loop["inner_r"])
        markers.append(((float(center[0]), float(center[1]), face_z), 0.004, (1.0, 0.0, 0.0)))
        markers.extend(
            (
                (
                    float(center[0] + radius * np.cos(angle)),
                    float(center[1] + radius * np.sin(angle)),
                    face_z,
                ),
                0.0011,
                (0.0, 1.0, 0.0),
            )
            for angle in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        )
    return tuple(markers)


def scissor_pair_1() -> ObjectSpec:
    """Scissors 1, flat below the left arm with its handle openings boardward.

    The OBJ is flat and in metres; a +90-degree yaw points the openings toward
    the high-x pegboard. The open mesh is not watertight, so the density is a
    calibrated effective value for its measured ~82-83 g. Collides through
    whichever hull set :data:`SCISSOR_1_COLLISION_METHOD` selects. The level-1
    document stands the same pair at the table centre: its
    ``(:start-poses poses/level1_start.json)`` restates the pose over this spec.
    """
    s = float(np.sqrt(0.5))
    return ObjectSpec(
        name="scissors_1",
        category=SCISSOR_PAIR_1.name,
        asset=declared_asset(
            SCISSOR_PAIR_1,
            collision_paths=SCISSOR_1_COLLIDERS[SCISSOR_1_COLLISION_METHOD],
            approximate=None,
            keep_visual=True,
            material_colors=True,
        ),
        pose=point_from_surface(CELL, (0.540, 0.140, CELL.surface_z + 0.00550505 + 0.004)),
        quat=(0.0, 0.0, s, s),
        density=2600.0,
        friction=0.7,
        visual_markers=_scissor_ring_markers(SCISSOR_PAIR_1, 0.0070),
    )


def scissor_pair_2() -> ObjectSpec:
    """Scissors 2, flat below the right arm; its long axis is local y, so a
    -90-degree yaw points the openings boardward. Same effective-density rule
    as :func:`scissor_pair_1`."""
    s = float(np.sqrt(0.5))
    return ObjectSpec(
        name="scissors_2",
        category=SCISSOR_PAIR_2.name,
        asset=declared_asset(
            SCISSOR_PAIR_2,
            collision_paths=SCISSOR_2_COACD_PATHS,
            approximate=None,
            keep_visual=True,
            material_colors=True,
        ),
        pose=point_from_surface(CELL, (0.540, -0.140, CELL.surface_z + 0.00425 + 0.004)),
        quat=(0.0, 0.0, -s, s),
        density=5600.0,
        friction=0.7,
        visual_markers=_scissor_ring_markers(SCISSOR_PAIR_2, 0.0053),
    )


def plier_combination() -> ObjectSpec:
    """The combination pliers lying broad-side down beneath the left arm.

    Local x is the long axis: +90 degrees about x lays the broad xz profile
    flat. The level-1 document stands the same pliers at the table centre: its
    ``(:start-poses poses/level1_start.json)`` restates the pose over this spec.
    """
    s = float(np.sqrt(0.5))
    return ObjectSpec(
        name="plier_1",
        category=PLIER_COMBINATION.name,
        asset=declared_asset(
            PLIER_COMBINATION,
            collision_paths=PLIER_1_COLLISION_PATHS,
            approximate=None,
            keep_visual=True,
            material_colors=True,
        ),
        pose=point_from_surface(CELL, (0.436, 0.150, CELL.surface_z + 0.007)),
        quat=(s, 0.0, 0.0, s),
        density=2500.0,
        friction=0.8,
    )


def plier_cutter() -> ObjectSpec:
    """The cutters beneath the right arm, jaws toward the board.

    They use local y with jaws at +y: a -90-degree yaw points the jaws toward
    the board, and the CAD-origin offset puts both plier jaw tips on one
    world-x row (0.5224661 m).
    """
    s = float(np.sqrt(0.5))
    return ObjectSpec(
        name="plier_2",
        category=PLIER_CUTTER.name,
        asset=declared_asset(
            PLIER_CUTTER,
            collision_paths=PLIER_2_COLLISION_PATHS,
            approximate=None,
            keep_visual=True,
            material_colors=True,
        ),
        pose=point_from_surface(CELL, (0.5022669, -0.150, CELL.surface_z + 0.0169)),
        quat=(0.0, 0.0, -s, s),
        density=2000.0,
        friction=0.8,
    )


# ----------------------------------------------------------------- the records
# Last, because each names the factory that places it -- and that factory is
# what lets the document alone put the thing in a scene. The hook, the hammer
# holder and the plier support name none: theirs take a placement, and the
# stations that supply one are the records after them.
WRENCH_ADJUSTABLE_150 = register_asset(
    AssetRecord(
        name="wrench_adjustable_150",
        summary="A 150 x 26.1 x 12 mm wrench with a closed six-sided hanging ring.",
        tags=("wrench", "spanner", "tool"),
        files={"mesh": "wrenches/wrench_1/wrench_1.obj"},
        features=(
            FeatureSpec(
                name="ring",
                kind="loop",
                payload={"center": [0.11011492, 0.0, 0.0], "normal": [0.0, 1.0, 0.0], "inner_r": 0.00816},
                provenance="Centre of the mesh's six-sided closed opening after the authored 150 mm scaling.",
            ),
        ),
        sim=_TOOL_SIM,
        provenance="The task's 150 mm wrench_1 asset; collision hulls rebuilt by tools/build_coacd_collisions.py.",
        build=wrench_adjustable_150,
    )
)

WRENCH_ADJUSTABLE_200 = register_asset(
    AssetRecord(
        name="wrench_adjustable_200",
        summary="A 200 x 34.8 x 14 mm wrench with a closed six-sided hanging ring.",
        tags=("wrench", "spanner", "tool"),
        files={"mesh": "wrenches/wrench_2/wrench_2.obj"},
        features=(
            FeatureSpec(
                name="ring",
                kind="loop",
                payload={"center": [0.13076606, 0.0, 0.0], "normal": [0.0, 1.0, 0.0], "inner_r": 0.01088},
                provenance="Centre of the mesh's six-sided closed opening after the authored 200 mm scaling.",
            ),
        ),
        sim=_TOOL_SIM,
        provenance="The task's 200 mm wrench_2 asset; collision hulls rebuilt by tools/build_coacd_collisions.py.",
        build=wrench_adjustable_200,
    )
)


# A uniformly enlarged Level-1 variant. Its semantic feature is stated in the
# scaled mesh frame so rendering, collision, and goal evaluation agree.
WRENCH_ADJUSTABLE_260 = register_asset(
    AssetRecord(
        name="wrench_adjustable_260",
        summary="A uniformly enlarged 260 mm adjustable wrench with a larger closed hanging ring.",
        tags=("wrench", "spanner", "tool"),
        files={"mesh": "wrenches/wrench_2/wrench_2.obj"},
        features=(
            FeatureSpec(
                name="ring",
                kind="loop",
                payload={
                    "center": [0.169995878, 0.0, 0.0],
                    "normal": [0.0, 1.0, 0.0],
                    "inner_r": 0.014144,
                    "enclosure_tolerance": 0.008,
                },
                provenance="The 200 mm wrench ring geometry uniformly scaled by 1.30.",
            ),
        ),
        sim=_TOOL_SIM,
        provenance="The 200 mm wrench visual and CoACD collision geometry uniformly scaled by 1.30 for Level 1.",
        build=wrench_adjustable_260,
    )
)


SCISSOR_PAIR_1 = register_asset(
    AssetRecord(
        name="scissor_pair_1",
        summary="Household scissors, 15.5 cm, with both a CoACD and a hand-authored compound hull set.",
        tags=("scissors", "tool"),
        files={"mesh": "scissors/scissor_1/scissor_1.obj"},
        features=(
            FeatureSpec(
                name="bow",
                kind="loop",
                payload={
                    "center": [0.0217, -0.0385, 0.0],
                    "normal": [0.0, 0.0, 1.0],
                    "inner_r": 0.024,
                    "enclosure_tolerance": 0.005,
                },
                provenance=(
                    "Centroid of the bow opening, ray-mapped off the mesh (the routine's "
                    "HANDLE_CENTER steering point sits 6 mm inside it). The bow is a 30 mm-reach "
                    "oval, not a circle; the radius is sized to the measured resting hang -- the "
                    "shank crosses 18.7 mm out with the rod's 2.5 mm on top -- and stays well "
                    "inside the opening's circumscribed extent."
                ),
            ),
            FeatureSpec(
                name="bow_other",
                kind="loop",
                payload={
                    "center": [-0.02183, -0.03853, 0.0],
                    "normal": [0.0, 0.0, 1.0],
                    "inner_r": 0.024,
                    "enclosure_tolerance": 0.005,
                },
                provenance=(
                    "Centroid of the second handle opening, measured from the planar mesh silhouette; "
                    "the symmetric bows use the same effective hanging radius."
                ),
            ),
        ),
        sim=_TOOL_SIM,
        provenance="Supplied scissors; hull sets under scissor_1/collision_{coacd,compound}/ with manifests.",
        build=scissor_pair_1,
    )
)

SCISSOR_PAIR_2 = register_asset(
    AssetRecord(
        name="scissor_pair_2",
        summary="Household scissors, 16 cm, CoACD hull set.",
        tags=("scissors", "tool"),
        files={"mesh": "scissors/scissor_2/scissor_2.obj"},
        features=(
            FeatureSpec(
                name="bow",
                kind="loop",
                payload={
                    "center": [-0.0090, 0.0578, -0.00025],
                    "normal": [0.0, 0.0, 1.0],
                    "inner_r": 0.019,
                    "enclosure_tolerance": 0.005,
                },
                provenance=(
                    "As scissor_pair_1: the opening's ray-mapped centroid, radius sized to the "
                    "measured resting hang (13.8 mm out plus the rod) within the oval's 20.9 mm "
                    "circumscribed extent."
                ),
            ),
            FeatureSpec(
                name="bow_other",
                kind="loop",
                payload={
                    "center": [0.01883, 0.05487, -0.00025],
                    "normal": [0.0, 0.0, 1.0],
                    "inner_r": 0.019,
                    "enclosure_tolerance": 0.005,
                },
                provenance=(
                    "Centroid of the second handle opening, measured from the planar mesh silhouette; "
                    "it uses the same conservative effective hanging radius as the declared bow."
                ),
            ),
        ),
        sim=_TOOL_SIM,
        provenance="Supplied scissors; CoACD hulls under scissor_2/collision_coacd/ with manifest.",
        build=scissor_pair_2,
    )
)

SCREWDRIVER_SLIM = register_asset(
    AssetRecord(
        name="screwdriver_slim",
        summary="A slim screwdriver; the shaft runs along +x from the handle at the origin.",
        tags=("screwdriver", "tool"),
        files={"mesh": "screwdrivers/screwdriver_1/screwdriver_1.obj"},
        features=(
            FeatureSpec(
                name="tip",
                kind="landmark",
                payload={"position": [0.175, 0.0, 0.0]},
                provenance="The screwdriver shaft's authored +x endpoint, shared with the insertion routine.",
            ),
            FeatureSpec(
                name="shaft",
                kind="rod",
                payload={
                    "axis": [[0.175, 0.0, 0.0], [0.060, 0.0, 0.0]],
                    "radius": 0.003,
                },
                provenance=(
                    "Tip and handle-edge are the insert routine's own landmarks; the segment stops "
                    "at the handle so a rack hole threaded on it is threaded on metal."
                ),
            ),
        ),
        sim=_TOOL_SIM,
        provenance="Supplied screwdriver pair; CoACD hulls under screwdriver_1/collision_coacd/.",
        build=screwdriver_slim,
    )
)

SCREWDRIVER_GRIP = register_asset(
    AssetRecord(
        name="screwdriver_grip",
        summary="A fat-handled screwdriver with an authored diffuse texture; tip along -x.",
        tags=("screwdriver", "tool"),
        files={"mesh": "screwdrivers/screwdriver_2/screwdriver_2.obj"},
        features=(
            FeatureSpec(
                name="shaft",
                kind="rod",
                payload={
                    "axis": [[-0.100, -0.005697, 0.001303], [0.020, -0.0014, 0.0003]],
                    "radius": 0.004,
                },
                provenance="The insert routine's tip landmark, run back three quarters of the way to its handle landmark.",
            ),
        ),
        sim=_TOOL_SIM,
        provenance="Supplied screwdriver pair; CoACD hulls under screwdriver_2/collision_coacd/; textures/diffuse.png is the authored map.",
        build=screwdriver_grip,
    )
)

PLIER_COMBINATION = register_asset(
    AssetRecord(
        name="plier_combination",
        summary="Combination pliers, long axis along local x, 24-hull CoACD set.",
        tags=("pliers", "tool"),
        files={"mesh": "pliers/plier_1/plier_1.obj"},
        sim=_TOOL_SIM,
        provenance="Supplied pliers; hulls by tools/build_coacd_collisions.py.",
        build=plier_combination,
    )
)

PLIER_CUTTER = register_asset(
    AssetRecord(
        name="plier_cutter",
        summary="Diagonal cutters, long axis along local y, 24-hull CoACD set.",
        tags=("pliers", "cutters", "tool"),
        files={"mesh": "pliers/plier_2/plier_2.obj"},
        sim=_TOOL_SIM,
        provenance="Supplied Wiha-style cutters; hulls by tools/build_coacd_collisions.py.",
        build=plier_cutter,
    )
)

HANGING_HAMMER_1 = register_asset(
    AssetRecord(
        name="hanging_hammer_1",
        summary="A life-sized hand hammer, authored in centimetres.",
        tags=("hammer", "tool"),
        files={"mesh": "hammers/hammer.stl"},
        sim=_TOOL_SIM,
        provenance="Supplied hammer meshes; boxified at import (approximate=bounding_box).",
        build=hanging_hammer_1,
    )
)

HANGING_HAMMER_2 = register_asset(
    AssetRecord(
        name="hanging_hammer_2",
        summary="The second supplied hand hammer, authored in centimetres.",
        tags=("hammer", "tool"),
        files={"mesh": "hammers/hammer_2.stl"},
        sim=_TOOL_SIM,
        provenance="Supplied hammer meshes; boxified at import (approximate=bounding_box).",
        build=hanging_hammer_2,
    )
)

PEGBOARD_HOOK = register_asset(
    AssetRecord(
        name="pegboard_hook",
        summary="A 61 x 53 mm extruded J-hook, 4.98 mm thick; wrenches and scissors hang on its arm.",
        tags=("hook", "fixture"),
        files={"mesh": "scissor_holders/hook_2.stl"},
        features=(
            FeatureSpec(
                name="shank",
                kind="rod",
                payload={
                    # Tip to mount along the 61.1 mm reach, at the arm section's
                    # centre. A finite segment on purpose: a ring pushed past
                    # the mount plate is against the board, not on the hook.
                    "axis": [
                        [0.0611, HOOK_ARM_CENTER_Y, HOOK_ARM_HALF_THICKNESS],
                        [0.0, HOOK_ARM_CENTER_Y, HOOK_ARM_HALF_THICKNESS],
                    ],
                    "radius": HOOK_ARM_HALF_THICKNESS,
                },
                provenance=(
                    "Measured off hook_2.stl: reach 61.09 mm, tip band spans local y 0 to 8.4 mm, "
                    "extrusion 4.98 mm. The radius is the half-thickness -- the dimension a ring "
                    "must clear -- so a ring on the arm's top edge still reads inside its own "
                    "inner radius. See the module docstring."
                ),
            ),
        ),
        sim=_FIXTURE_SIM,
        provenance="Supplied low-resolution hook; clean 516-triangle watertight STL, collided as authored.",
    )
)

SCREWDRIVER_RACK = register_asset(
    AssetRecord(
        name="screwdriver_rack",
        summary="A 155 mm screwdriver rack; two of its graduated holes are this suite's targets.",
        tags=("rack", "holder", "fixture"),
        files={"mesh": "screwdriver_holders/screwdriver_holder.stl"},
        features=(
            FeatureSpec(
                name="hole_left",
                kind="loop",
                payload={
                    "center": [0.024, 0.0732, 0.0336],
                    "normal": [0.0, 0.0, 1.0],
                    "inner_r": 0.008,
                },
                provenance=(
                    "The suite's yellow-marked insertion target, inverse-mapped from its measured "
                    "world coordinates through the rack's mounting transform; the evaluated radius "
                    "adds 2 mm alignment tolerance to the original mesh-derived declaration."
                ),
            ),
            FeatureSpec(
                name="hole_right",
                kind="loop",
                payload={
                    "center": [0.024, 0.1128, 0.0336],
                    "normal": [0.0, 0.0, 1.0],
                    "inner_r": 0.0116,
                },
                provenance=(
                    "The purple-marked target, same derivation; a larger hole for the fatter handle's "
                    "shaft, with 2 mm alignment tolerance added to the original declaration."
                ),
            ),
        ),
        sim=_FIXTURE_SIM,
        provenance="Supplied rack STL, mounted at 120% scale; holes verified against the alignment markers by this module's own assertion.",
        build=screwdriver_rack,
    )
)

# A separate record rather than a swap of the one above, and the reason is
# measured: `InsertScrewdriverRoutine`'s approach, descend and insertion
# stations were all taken against the rack's 8 mm hole 34 mm up, and on the
# box's 28 mm opening 75 mm up the routine hangs nothing at all -- four
# rollouts, reward 0.00. The full-size `screwdrivers` document threads a
# *feature*, so it needs those numbers; `screwdrivers_level1` scores a region
# stated in the box's own local frame and reads no feature here, so the two
# fixtures can coexist without either one being retuned.
SCREWDRIVER_BOX = register_asset(
    AssetRecord(
        name="screwdriver_box",
        summary="A 100 mm wide, 105 mm deep open screwdriver box with two integral pegboard hooks.",
        tags=("box", "container", "holder", "fixture", "pegboard-mounted"),
        files={"mesh": "screwdriver_holders/screwdriver_box.stl"},
        features=(
            FeatureSpec(
                name="hole_left",
                kind="loop",
                payload={
                    "center": [0.0325, 0.0732, 0.105],
                    "normal": [0.0, 0.0, 1.0],
                    "inner_r": 0.028,
                },
                provenance=(
                    "The cup's opening at the left screwdriver's nominal station, stated for the "
                    "same reason the rack's holes are: so a plan and a score can read one geometry. "
                    "Declared rather than scored -- the level-1 document states a region instead, "
                    "because a 28 mm opening admits a leaning handle that no loop measure would."
                ),
            ),
            FeatureSpec(
                name="hole_right",
                kind="loop",
                payload={
                    "center": [0.0325, 0.1092, 0.105],
                    "normal": [0.0, 0.0, 1.0],
                    "inner_r": 0.028,
                },
                provenance="The right station, same derivation. One cup, so the two radii are equal.",
            ),
        ),
        sim=_FIXTURE_SIM,
        provenance=(
            "Authored box, built in metres by screwdriver_holders/generate_screwdriver_box.py: a "
            "57 x 102 x 61 mm open cup with two rearward mounting straps."
        ),
        build=screwdriver_box,
    )
)

PLIER_SUPPORT = register_asset(
    AssetRecord(
        name="plier_support",
        summary="A triangular support; a plier hangs with its handle gap over the tip.",
        tags=("support", "holder", "fixture"),
        files={"mesh": "plier_holders/triangular_support.stl"},
        sim=_FIXTURE_SIM,
        provenance="Supplied triangular support, mounted at 120% scale; the hung pose is scored by a region, not a feature.",
    )
)

HAMMER_HOLDER = register_asset(
    AssetRecord(
        name="hammer_holder",
        summary="An open-fronted hammer holder; the head rests across it, the handle drops through.",
        tags=("holder", "fixture"),
        files={"mesh": "hammer_holders/enforce_1250_screws.stl"},
        sim=_FIXTURE_SIM,
        provenance="Supplied holder CAD, boxified at import; licence under confirmation, see PROVENANCE.md.",
    )
)

SKADIS_PEGBOARD = register_asset(
    AssetRecord(
        name="skadis_pegboard",
        summary="A single-piece IKEA SKADIS-derived pegboard wall; the task applies its authored flat color.",
        tags=("pegboard", "wall", "fixture"),
        files={
            "mesh": "pegboards/skadis_whole/skadis_pegboard.obj",
            "texture": "pegboards/skadis_whole/material_0.png",
        },
        sim=_FIXTURE_SIM,
        provenance=(
            "Single-piece SKADIS-derived OBJ from the tool-hanging branch; its supplied texture is retained "
            "beside the mesh, while the task deliberately renders BOARD_COLOR and boxifies collision."
        ),
        build=skadis_pegboard,
    )
)


# ----------------------------------------------------- the stations, as records
def _stationed(base: AssetRecord, station: str, y: float, where: str, factory) -> AssetRecord:
    """``base``'s geometry standing at one named station, as a record of its own.

    A record names one zero-argument factory and the documents stand the hook
    at five y's, the hammer holder at two, the plier support at three. So each
    station is a record: the geometry record carries the mesh and the features,
    the station carries the placement, and ``wrench_hook_2 -
    pegboard_hook_wrench_left`` lowers to the hook on the wrenches' left
    station under the document's own name. ``replace`` keeps the features, the
    sim requirements and the resolved files, so a goal reading ``.shank`` off a
    stationed hook reads the geometry record's one measurement -- and so do
    ``tools.py``'s plans, which look the feature up by the document's type.

    The geometry record itself keeps no factory on purpose: a bare
    ``pegboard_hook`` is a hook nowhere in particular, and a document has to
    say which station it means.
    """
    name = f"{base.name}_{station}"
    return register_asset(
        replace(
            base,
            name=name,
            summary=f"{base.summary} This one is {where}.",
            provenance=f"{base.provenance} Stationed at y={y:+.3f} m by {base.name}'s placing factory.",
            build=lambda: factory(name, y),
        )
    )


STATIONED: dict[str, AssetRecord] = {
    record.name: record
    for base, stations, factory in (
        (PEGBOARD_HOOK, HOOK_STATIONS, pegboard_hook),
        (HAMMER_HOLDER, HAMMER_HOLDER_STATIONS, hammer_holder),
        (PLIER_SUPPORT, PLIER_SUPPORT_STATIONS, plier_support),
    )
    for record in (_stationed(base, station, y, where, factory) for station, (y, where) in stations.items())
}
"""Every station record, by name: ``pegboard_hook_wrench_left`` and the rest."""


def _check_marked_holes() -> None:
    """The rack's declared holes and the routine's marked targets must agree.

    Two spellings of one measurement: the record states each hole in the
    rack's frame, the markers state the same points in world. The rack mounts
    by a 180-degree z turn, so local (x, y, z) lands at
    (mount_x - x, mount_y - y, mount_z + z).
    """
    mount = np.array(RACK_MOUNT)
    for feature, world_y in (
        ("hole_left", SCREWDRIVER_LEFT_HOLE_WORLD_Y),
        ("hole_right", SCREWDRIVER_RIGHT_HOLE_WORLD_Y),
    ):
        local = np.asarray(SCREWDRIVER_RACK.feature(feature).payload["center"])
        world = np.array([mount[0] - local[0], mount[1] - local[1], mount[2] + local[2]])
        marked = np.array([SCREWDRIVER_HOLE_CENTER_WORLD_X, world_y, SCREWDRIVER_HOLDER_TOP_WORLD_Z])
        if np.linalg.norm(world - marked) > 0.005:
            raise AssertionError(
                f"rack {feature} resolves to {world.round(4).tolist()} but the marker says "
                f"{marked.tolist()}; the record and the markers have drifted apart"
            )


_check_marked_holes()
