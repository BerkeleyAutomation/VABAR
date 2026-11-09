# SPDX-License-Identifier: Apache-2.0
"""What this task's document names: a syringe, and the two bins it stands between.

Three types, and each is here rather than borrowed because **a task directory has
to be something you can hand to somebody**. This suite was carved out of
`tool_sort`, and the carving is what exposed how much it had been getting for free
from whichever sibling happened to load first: `syringe` from `tool_sort`'s own
records and `bin` from a third suite's. Neither is an import, so neither is
something a grep would have found -- the asset registry is global and every root
loads eagerly, so a borrowed declaration looks exactly like an owned one until
the directory travels somewhere its lender does not.

**Declared, not copied wholesale.** `tool_sort`'s records module declares twenty
types; this document names two of them. Bringing the other eighteen would say
this task is about hammers and tape measures, which it is not.

**The bins are `robosuite_bin`, not `bin`, and that is deliberate.** A name in the
asset registry says what a thing *is*, and these bins come out of the robosuite
export -- so the name says so, here and in every `tool_sort` document. Two
different things may not share a name; the registry refusing that is the
registry working.
"""

from __future__ import annotations

from pathlib import Path

from robosimstudio.assetlib.lower import declared_file
from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.registry import register_asset
from robosimstudio.assetlib.sim import SimRequirements, contacts_with
from robosimstudio.specs import AssetSpec, ObjectSpec, StaticSpec

__all__ = [
    "ASSETS_DIR",
    "BIN",
    "INSTRUMENT_TRAY",
    "INSTRUMENT_TRAY_FLOOR",
    "INSTRUMENT_TRAY_WALL_XNEG",
    "INSTRUMENT_TRAY_WALL_XPOS",
    "INSTRUMENT_TRAY_WALL_YNEG",
    "INSTRUMENT_TRAY_WALL_YPOS",
    "SYRINGE",
    "SYRINGE_COLLISION_PATHS",
    "SYRINGE_EXPORT_MASS_KG",
    "SYRINGE_MESH",
    "SYRINGE_TEXTURE",
    "cavity_floor_z",
    "syringe",
    "syringe_tray",
    "syringe_tray_floor",
    "syringe_tray_wall_xneg",
    "syringe_tray_wall_xpos",
    "syringe_tray_wall_yneg",
    "syringe_tray_wall_ypos",
    "wall_half_height",
]

_SIM = SimRequirements(
    domain="rigid",
    contacts=contacts_with("surface", "fixtures", "robot"),
    # Word for word what `tool_sort` states, because `AssetRecord` equality
    # compares `sim` exactly and `syringe` is a name both suites declare. A
    # rephrasing here would read as a second opinion about the physics and be
    # refused as one.
    notes="Rigid mesh in a MuJoCo cell; nothing exotic. Collision geometry comes from the export.",
)

_EXPORT_PROVENANCE = (
    "AUTOLab-GaP-NVIDIA-Collaboration/VAB-industrial ToolSortPickPlace, robosuite 1.4.0, "
    "MuJoCo 3.3.3, seed 0, dumped by vabar/scripts/dump_vab_scene.py. The export owns the geometry "
    "and the mass; this declares what the thing IS and leaves where it starts to the document."
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
"""This suite's own asset directory. Its copy of the syringe mesh and the CoACD
hull set are here rather than in the collection's ``vab_tool_sort`` export,
which ``tool_sort`` also stages: a collider is physics, and this suite's syringe
had to stop colliding as a flange-width box without moving ``tool_sort``'s."""

SYRINGE_EXPORT_MASS_KG = 0.030
"""``syringe_0_g0``'s own authored mass, read off the export's ``scene.xml`` --
every syringe gets the same figure so a policy cannot tell the goal one from its
neighbours by weight."""
SYRINGE_MESH = str(ASSETS_DIR / "syringe" / "syringe.obj")
"""The bare mesh every syringe in this suite draws -- the file the export's own
``syringe_0`` body draws, copied in beside the hulls built from it."""
SYRINGE_TEXTURE = str(ASSETS_DIR / "syringe" / "syringe_diffuse.png")
"""The export's own diffuse map (the red needle cap), so a syringe imported from
the bare mesh still draws the way perception identifies it. Built from this
module's own location rather than written down, because an ``AssetSpec.texture``
is not rewritten beside its declaration the way ``AssetRecord.files`` are -- and
a path in a recorded layout would be a path that only resolves on the machine
that wrote it."""
SYRINGE_COLLISION_PATHS = tuple(
    str(ASSETS_DIR / "syringe" / "collision_coacd" / f"hull_{index:02d}.obj") for index in range(61)
)
"""The syringe's collider: a CoACD convex decomposition of the mesh it draws.

Built by ``scripts/build_coacd_collisions.py`` from ``syringe.obj``, manifest
beside the hulls. It follows the drawn shape -- needle, an ~11.5 mm barrel, the
flange and its 23.6 mm finger grips -- where the export's authored box is
**23.6 mm across its whole 150 mm**, the mesh's own bound. That box left the
Robotiq pads some 6 mm clear of the barrel on each side while still reporting a
hold, which is the defect this replaces. Measured against the mesh: 99.9 % of
the drawn surface lies inside the hulls (the rest within 0.08 mm), and the hull
surface stays within 1.2 mm of it at the 95th percentile.

Carried from the run they were measured on rather than rebuilt here: CoACD's
MCTS is not deterministic across runs even at ``seed 0``, and the same mesh and
parameters give 58 hulls on this machine against the 61 the reported sweep
used."""


def syringe() -> ObjectSpec:
    """One syringe, imported from the mesh it draws and colliding on its hulls.

    Every syringe in this suite arrives through here -- the goal and each spare
    alike -- so there is one answer to what a syringe is made of: the drawn
    mesh, the export's diffuse map that gives it its red needle cap, and the
    CoACD hulls it collides on. A document's recorded entry says only what
    differs -- where it stands, its grip, and, a spare's alone, that it drops
    the map for the splash's white -- rather than restating the hull list
    sixty-one paths at a time.

    The export's own ``syringe_0`` body is dropped for this
    (``(:export ... (:strip-free-bodies true))``), because that body carries the
    authored box and a body cannot hold two opinions about its collider.
    """
    return ObjectSpec(
        name="syringe_0",
        category="syringe",
        asset=AssetSpec(path=SYRINGE_MESH, collision_paths=SYRINGE_COLLISION_PATHS, texture=SYRINGE_TEXTURE),
        body_name="syringe_0",
        joint_name="",
        world_xform=True,
        mass=SYRINGE_EXPORT_MASS_KG,
    )


SYRINGE = register_asset(
    AssetRecord(
        name="syringe",
        summary="A single-use syringe, which this task disposes of rather than sorts.",
        # Tagged `consumable` rather than `tool`, and `sharps` is the word that
        # grounds the instruction: the box is a *sharps* box, and what belongs in
        # one is a property of the object rather than of the receptacle.
        tags=("syringe", "sharps", "consumable"),
        # No features. The grasp box is derived from the mesh by the compiler, and
        # declaring one here would be a copy of a computation rather than a fact
        # about the object -- which matters more here than anywhere: this task
        # fails on *orientation*, and a declared grasp frame would look like an
        # answer to that.
        sim=_SIM,
        files={"export": "vab_tool_sort"},
        provenance=_EXPORT_PROVENANCE,
        # How this suite puts one in a scene -- the mesh and the hulls it
        # collides on. Excluded from record equality, which is what lets it
        # differ from `tool_sort`'s declaration of the same name.
        build=syringe,
    )
)

BIN = register_asset(
    AssetRecord(
        name="robosuite_bin",
        summary="A rectangular plastic bin from the robosuite tool-sort bench, open at the top.",
        tags=("bin", "tote", "container"),
        # No features, and for a specific reason: what is worth declaring
        # about a bin is its cavity, and a cavity is the *task's* -- this
        # document states its own quadrants over its own bins.
        sim=_SIM,
        files={"export": "vab_tool_sort"},
        provenance=_EXPORT_PROVENANCE,
    )
)

TRAY_OUTER_HALF = (0.100, 0.200, 0.006)
"""The asset's own outer half-extents [m] (``make_tray.py``'s ``HALF``)."""
TRAY_WALL_THICKNESS = 0.003
"""The asset's own wall/floor thickness [m] (``make_tray.py``'s ``WALL``)."""
TRAY_FLOOR_COLLIDER_HALF_THICKNESS = 0.010
"""Half the invisible floor slab's thickness [m]. Its top stays on the drawn
cavity floor; the thickness grows downward only, into the bench where nothing
can reach it. The old 2 mm half -- a 4 mm slab -- let a Robotiq pad pressing a
syringe drive it straight through the floor: ``tray_clutter_t03`` under
``gap_perception_v2`` sank ``syringe_pack_0``'s origin 11 mm below the slab top
during the close."""
TRAY_WALL_COLLIDER_HALF_THICKNESS = TRAY_WALL_THICKNESS / 2.0
"""Half the thickness [m] of the four invisible wall boxes: the drawn wall's own
3 mm, so both faces sit on the drawn walls (inner +-97/+-197 mm, outer
+-100/+-200 mm in the tray frame). A 20 mm version stood 17 mm proud of the
drawn outer edge -- an invisible ledge on the bench. **Before these the drawn
walls did not collide at all**, and a syringe pushed sideways left the pan."""
TRAY_SETTLE_MARGIN = 0.001
"""Extra lift so a syringe starts a hair above the tray floor rather than
inside it -- physics settles the last millimetre."""


def syringe_tray() -> StaticSpec:
    """The tray as drawn: VAB-industrial's own MJCF, decorative, stood by the recorded pose.

    Its own MJCF collides as the drawn mesh's convex hull, which -- per the
    generator's own comment -- "would fill the tray in solid"; so the mesh
    itself never collides (``collide=False``) and :func:`syringe_tray_floor`
    is the stand-in for the box-primitive collision VAB-industrial's own env
    builds against the mesh's ``cavity_min``/``cavity_max`` sites. Where it
    stands is the document's, through ``(:start-poses ...)``: the entry's
    ``xform`` is applied to what this returns.
    """
    return StaticSpec(
        name="syringe_tray",
        asset=AssetSpec(path=declared_file(INSTRUMENT_TRAY, "mjcf")),
        world_xform=True,
        collide=False,
    )


def syringe_tray_floor() -> StaticSpec:
    """The invisible slab a syringe rests on inside the tray: the cavity floor as a collider."""
    return StaticSpec(
        name="syringe_tray_floor",
        half_extents=(
            TRAY_OUTER_HALF[0] - TRAY_WALL_THICKNESS,
            TRAY_OUTER_HALF[1] - TRAY_WALL_THICKNESS,
            TRAY_FLOOR_COLLIDER_HALF_THICKNESS,
        ),
        world_xform=True,
        visible=False,
    )


def _tray_wall(name: str, half: tuple[float, float, float]) -> StaticSpec:
    """One of the four invisible boxes the drawn walls collide as."""
    return StaticSpec(name=name, half_extents=half, world_xform=True, visible=False)


def wall_half_height() -> float:
    """Half the height [m] of a wall box: the floor slab's underside up to the rim.

    Stated once here because the document records only where each wall stands;
    the shape is the tray's, not the layout's. It covers the slab as well as the
    cavity, so a syringe cannot be squeezed in between wall and floor.
    """
    span = TRAY_OUTER_HALF[2] + TRAY_WALL_THICKNESS + 2.0 * TRAY_FLOOR_COLLIDER_HALF_THICKNESS
    return 0.5 * span


def syringe_tray_wall_xpos() -> StaticSpec:
    """The +x wall, spanning the cavity's y and thickened outward."""
    hz = wall_half_height()
    t = TRAY_WALL_COLLIDER_HALF_THICKNESS
    return _tray_wall("syringe_tray_wall_xpos", (t, TRAY_OUTER_HALF[1] - TRAY_WALL_THICKNESS + 2.0 * t, hz))


def syringe_tray_wall_xneg() -> StaticSpec:
    """The -x wall; the mirror of :func:`syringe_tray_wall_xpos`."""
    hz = wall_half_height()
    t = TRAY_WALL_COLLIDER_HALF_THICKNESS
    return _tray_wall("syringe_tray_wall_xneg", (t, TRAY_OUTER_HALF[1] - TRAY_WALL_THICKNESS + 2.0 * t, hz))


def syringe_tray_wall_ypos() -> StaticSpec:
    """The +y wall, meeting the side walls rather than overlapping them."""
    hz = wall_half_height()
    t = TRAY_WALL_COLLIDER_HALF_THICKNESS
    return _tray_wall("syringe_tray_wall_ypos", (TRAY_OUTER_HALF[0] - TRAY_WALL_THICKNESS, t, hz))


def syringe_tray_wall_yneg() -> StaticSpec:
    """The -y wall; the mirror of :func:`syringe_tray_wall_ypos`."""
    hz = wall_half_height()
    t = TRAY_WALL_COLLIDER_HALF_THICKNESS
    return _tray_wall("syringe_tray_wall_yneg", (TRAY_OUTER_HALF[0] - TRAY_WALL_THICKNESS, t, hz))


def cavity_floor_z(tray_z: float) -> float:
    """Where the tray's inner floor is [m], off its origin at mid-height.

    One wall thickness above the tray's *bottom*, where the drawn mesh's cavity
    floor and its ``cavity_min`` site both sit -- z = -3 mm on a 12 mm tray.
    This used to measure down from the top instead, which put the floor 6 mm
    above the drawn one and left the walls 3 mm tall rather than 9 mm, so every
    syringe floated just under the rim of a pan it was supposed to lie in.
    """
    return tray_z - TRAY_OUTER_HALF[2] + TRAY_WALL_THICKNESS


_WALL_SIM = SimRequirements(domain="static", contacts=contacts_with("objects", "robot"))
_WALL_PROVENANCE = (
    "Computed off make_tray.py's HALF and WALL; stands where syringe_tray's recorded pose puts that "
    "cavity wall. Four separate records rather than one, because each stands at its own place and the "
    "document records where each one is."
)

INSTRUMENT_TRAY = register_asset(
    AssetRecord(
        name="instrument_tray",
        summary="A shallow 200 x 400 x 12 mm stainless instrument tray, open at the top, that syringes are laid out in.",
        tags=("tray", "instrument_tray", "container"),
        # No features, for the reason the bins give: what is worth declaring
        # about a tray is its cavity, and this task's documents score nothing
        # against it. The MJCF itself carries `cavity_min`/`cavity_max` sites;
        # `syringe_tray_floor` stands a floor collider inside it off the
        # generator's constants, since the mesh's convex hull would fill the
        # pan solid.
        sim=SimRequirements(
            domain="rigid",
            contacts=contacts_with("surface"),
            notes="Decorative mesh; the floor it is stood on is a primitive of its own (instrument_tray_floor).",
        ),
        files={"mjcf": "assets/instrument_tray/instrument_tray.xml"},
        provenance=(
            "AUTOLab-GaP-NVIDIA-Collaboration/VAB-industrial, add-all-syringes-sharps-box branch, "
            "assets/objects/instrument_tray as generated by assets/objects/_utils/make_tray.py; "
            "vendored beside this declaration. See PROVENANCE.md."
        ),
        build=syringe_tray,
    )
)

INSTRUMENT_TRAY_FLOOR = register_asset(
    AssetRecord(
        name="instrument_tray_floor",
        summary="The instrument tray's cavity floor as an invisible collider slab, 3 mm inside its walls.",
        tags=("tray", "floor", "fixture"),
        features=(),
        sim=SimRequirements(domain="static", contacts=contacts_with("objects", "robot")),
        provenance="Computed off make_tray.py's HALF and WALL; stands where syringe_tray's recorded pose puts the cavity floor.",
        build=syringe_tray_floor,
    )
)

INSTRUMENT_TRAY_WALL_XPOS = register_asset(
    AssetRecord(
        name="instrument_tray_wall_xpos",
        summary="The instrument tray's +x cavity wall as an invisible collider box, on the drawn wall's own 3 mm.",
        tags=("tray", "wall", "fixture"),
        features=(),
        sim=_WALL_SIM,
        provenance=_WALL_PROVENANCE,
        build=syringe_tray_wall_xpos,
    )
)

INSTRUMENT_TRAY_WALL_XNEG = register_asset(
    AssetRecord(
        name="instrument_tray_wall_xneg",
        summary="The instrument tray's -x cavity wall as an invisible collider box, on the drawn wall's own 3 mm.",
        tags=("tray", "wall", "fixture"),
        features=(),
        sim=_WALL_SIM,
        provenance=_WALL_PROVENANCE,
        build=syringe_tray_wall_xneg,
    )
)

INSTRUMENT_TRAY_WALL_YPOS = register_asset(
    AssetRecord(
        name="instrument_tray_wall_ypos",
        summary="The instrument tray's +y cavity wall as an invisible collider box, on the drawn wall's own 3 mm.",
        tags=("tray", "wall", "fixture"),
        features=(),
        sim=_WALL_SIM,
        provenance=_WALL_PROVENANCE,
        build=syringe_tray_wall_ypos,
    )
)

INSTRUMENT_TRAY_WALL_YNEG = register_asset(
    AssetRecord(
        name="instrument_tray_wall_yneg",
        summary="The instrument tray's -y cavity wall as an invisible collider box, on the drawn wall's own 3 mm.",
        tags=("tray", "wall", "fixture"),
        features=(),
        sim=_WALL_SIM,
        provenance=_WALL_PROVENANCE,
        build=syringe_tray_wall_yneg,
    )
)

# `sharps_box` is deliberately **not** declared, and that is not an oversight.
#
# The document names `syringe_box - sharps_box` so its region has a frame to hang
# off and so the typechecker can see the reference resolve. There is no body: the
# box is welded into bin2 as geometry by the export, and the lowering drops
# container fixtures from the object list for exactly that reason. A record
# would claim there is a thing to import, place and grasp, and the one fact worth
# having about it -- where its aperture sits inside bin2 -- is in the export's
# sidecar, measured, where the loader reads it.

# Registered on import; the loader executes records.py and nothing else of the task's build code.
from . import scenery  # noqa: E402,F401
