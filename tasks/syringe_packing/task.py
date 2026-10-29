# SPDX-License-Identifier: Apache-2.0
"""``tool_sort``, as ten documents and a loader.

The first port. Its Python definition was 571 lines of ``TaskSpec`` construction
whose semantics leaked through ``metadata`` as untyped tuples; what is left is
ten ``.bddl`` files, an import profile, and the code below that supplies the two
things BDDL genuinely cannot: **where the objects are**, and **which importer
built them**.

Six of the ten are that port. The other four came afterwards and were authored
as documents from the start -- ``sort_electrical`` and ``sort_wiring`` over two
new exports, and two restatements of the first through USD. The split is
recorded in each document's own header rather than as a constant here: the
frozen dumps that needed the list in code went with the port scaffolding.

**Poses come from the export, not the document.** BDDL has no coordinates. The
VAB environment places its pile by writing free-joint ``qpos`` at reset, and the
dump recorded what it settled into at seed 0 -- so a spec that omitted the poses
would build a scene whose whole pile sits at the world origin, under the table.
That arrangement is the export's, it is the one arrangement known to be
physically consistent, and re-sampling it would discard exactly that.

**The container conversion happens in one place.** ``syringe_box`` is welded into
``bin2`` as geometry rather than being a body, so it has a frame and no handle. A
region stated in the box's frame is restated here in ``bin2``'s, using the offset
the export recorded -- the document keeps the file's own numbers, and the
arithmetic is here and nowhere else.

**Several variants over one export.** ``hammer`` is ``sort_all``'s scene with a
goal concerning one tool, and the two USD variants are ``sort_electrical``'s
scene described a second and third way -- which is why the export is named in
each document rather than derived from the variant.

**And one variant's objects need not come from its export.**
``(:staging (:objects-from usd))`` strips the export's own free bodies and
builds each object from the stage its asset record declares instead. The three
``sort_electrical`` variants are then the same task with the geometry and the
physics arriving through two different importers, which is a comparison you can
only make when everything else is held still -- see :func:`_restage_objects`.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any

from robosimstudio_bench.goals.evaluate import evaluator_factory
from robosimstudio.assetlib.lower import declared_asset, declared_file
from robosimstudio.assetlib.registry import get_asset
from robosimstudio.paths import resolve_path
from robosimstudio.specs import (
    AssetSpec,
    Binding,
    ObjectSpec,
    RegionSpec,
    StaticSpec,
    TaskRequirements,
    TaskSpec,
    TaskSuite,
    restage,
)
from robosimstudio_bench.taskbuild import lower
from robosimstudio_bench.taskbuild.artifacts import task_resolver
from robosimstudio_bench.taskbuild.lower import Lowered, rebase_regions
from robosimstudio_bench.taskbuild.registry import register_suite
from robosimstudio_bench.taskdoc import VAB_INDUSTRIAL_V1, apply_bindings, read_problem
from robosimstudio_bench.taskdoc.sidecar import wxyz_to_xyzw

from . import assets as _own_assets  # noqa: F401 -- registers the types this document names
from . import scenes as _cell  # noqa: F401 -- registers the cell these documents stand in
from . import staging as _staging  # noqa: F401 -- registers the profile and the arrangements
from .assets.look import SYRINGE_PACKING_LOOK
from .assets.records import INSTRUMENT_TRAY

__all__ = ["DOCUMENTS", "SYRINGE_PACKING", "VARIANTS", "build", "build_all"]

HERE = Path(__file__).resolve().parent
POSES = HERE / "start_poses"

VARIANTS: dict[str, str] = {
    "syringe": "put_the_syringe_in_the_sharps_box",
    "tray_pair": "dispose_the_syringe_from_the_tray",
    "tray_pair_both": "dispose_both_syringes_from_the_tray",
}
"""Variant name to the problem file it is.

Named rather than derived: the problem files are named for what they ask and a
variant is named for what it is, and ``hammer`` is not
``pick_up_the_hammer_and_place_it_in_bin2``.

``tray_pair`` is ``syringe``'s goal over a different scene -- the goal syringe
in VAB-industrial's instrument tray beside one spare -- and its document says
so; the layout it has no coordinates for is ``start_poses/tray_pair.json``,
which :func:`_layout` reads. It was the loader-only ``ten_syringes`` for a
while, an alias of ``syringe``'s document with the tray and the spare built in
Python here, which is what a document is for.
"""

VARIANTS.update(
    {
        path.stem.replace("dispose_the_syringe_from_the_tray_", "tray_pair_"): path.stem
        for path in sorted(HERE.glob("dispose_the_syringe_from_the_tray_t[0-9][0-9].task"))
    }
)
VARIANTS.update(
    {
        path.stem.replace("dispose_both_syringes_from_the_tray_", "tray_pair_both_"): path.stem
        for path in sorted(HERE.glob("dispose_both_syringes_from_the_tray_t[0-9][0-9].task"))
    }
)
"""The ``tray_pair`` layout sweep, when it has been written: one document per
syringe location (``tools/make_tray_pair_trials.py``), each ``tray_pair_tNN``
with its own ``start_poses`` layout and nothing else of its own."""

VARIANTS.update(
    {
        path.stem.replace("dispose_the_syringe_from_the_cluttered_tray_", "tray_clutter_"): path.stem
        for path in sorted(HERE.glob("dispose_the_syringe_from_the_cluttered_tray_t[0-9][0-9].task"))
    }
)
"""Level 2's sweep (``tools/make_tray_clutter_trials.py``): one document per
5-8-syringe packing, each ``tray_clutter_tNN`` with its own ``start_poses``
layout. No bare ``tray_clutter`` variant -- unlike ``tray_pair``, there is no
single canonical packing this family transforms, only the 30 trials."""

DOCUMENTS = {name: HERE / f"{stem}.task" for name, stem in VARIANTS.items()}

ASSETS = HERE / "assets"


def _export_path(export: str, name: str) -> str:
    """One file of the staged export, from this task's own copy when it has one.

    The documents name the export the way the vendored tree does
    (``vab_tool_sort/syringe``), and this task carries its own copy of the
    robot-free scene, its sidecar and the six meshes and textures they
    reference under ``assets/`` -- so every pairing that stands a foreign robot
    in the cell resolves inside the directory, and the directory travels.
    ``scene.xml`` is deliberately not copied: it is the same cell with the
    export's Panda welded in, and the robot is the library's, not this
    task's -- the scene-resident binding reads the vendored one. The rule is
    ``register_asset``'s: beside the declaration when the file is actually
    there, the vendored tree otherwise.
    """
    beside = ASSETS / export / name
    return str(beside) if beside.exists() else f"{export}/{name}"


BINDINGS = "vab_industrial@v1"
"""Which predicate semantics these files are read under.

Reproduction tier: ``In`` binds to the weak box test the source benchmark uses,
not to the stronger contact-and-at-rest predicate this stack also has. That is a
deliberate choice for comparability and is recorded in every built spec's
provenance.
"""


def _binding_table(name: str):
    """The binding table a versioned name refers to.

    RoboSimStudio exports the one table this suite reads under as a constant
    rather than through a lookup; asking for anything else is still an error.
    """
    if name != VAB_INDUSTRIAL_V1.name:
        raise KeyError(f"unknown binding table {name!r}; available: {VAB_INDUSTRIAL_V1.name!r}")
    return VAB_INDUSTRIAL_V1


COLLIDER = "drawn_hull"
"""What a free object in these cells collides on, and why it is not the box.

The export fits each tool one oriented box and marks the mesh beside it
``contype="0" conaffinity="0"`` -- drawn, never touched. That box is VAB's own
approximation, not this repository's, and it is a big one. Measured as the
largest distance, over every heading a top-down jaw could close along, that the
collider stands proud of the mesh a viewer sees::

    object                 authored box     drawn hull
    drill                     27.1 mm         0.00 mm
    clamp                     26.2            0.00
    adjustable_wrench         19.5            0.00
    pliers                    16.8            0.00
    utility_knife             15.9            0.00
    phillips_screwdriver       8.9            0.00

Two centimetres of invisible shell is a gripper that closes on nothing and a
Robotiq linkage that deflects against it, which is what this suite looked like.

**This is a departure from the export and it is narrow.** The hammer and the
round nut come through untouched, because upstream authored *them* as real
primitives -- a cylinder handle, a box head, nine boxes for the nut -- and those
already measure 0.00 mm proud. What ``drawn_hull`` replaces is only the case
where the file itself says "this shape is for looking at": see
:meth:`~robosimstudio.scene.entities.rigid.EntityRigid._add_drawn_hull_collider`.

The old note here said boxes were chosen because "a raw convex hull leaves a thin
tool rocking in a limit cycle that creeps 15-25 mm while nominally settled". That
was true and the cause was elsewhere: mujoco_warp gives a convex pair one contact
point unless ``enable_multiccd`` is on, and one point is what a body rocks about.
``vab_500hz@v1`` states the flag; without it the same creep comes straight back.
"""

COMPARISON = ("sort_electrical", "sort_electrical_usd", "sort_electrical_usd_geom")
"""The three variants that are one scene described three ways, held still.

``sort_electrical`` and its two USD restatements exist to attribute a difference
to the *importer* and to nothing else, which only works while everything else is
identical -- including the collider. The USD stage authors a bounding box, so all
three fit one, and this triple keeps :data:`COLLIDER` off. Moving them costs a
re-export of the stage with hull colliders, which is a tools step over vendored
geometry rather than a line here.
"""

COMPARISON_COLLIDER = "bounding_box"
"""What the comparison triple fits, matching what the USD stage itself authors."""


SYRINGE_COLLIDER = ""
"""The goal syringe keeps the export's own authored box, not the drawn hull.

The one departure from :data:`COLLIDER`, and it is the object the task is
about. The hull's cross-section between the barrel and the flange is an
interpolated wedge (no vertices from -15..+45 mm -- ``routine.py``'s
``GRASP_HANDLE_FRAC`` survey), so a centre-of-mass pinch closes on two sloped
knife-edges: measured on the centre-grasp run, the syringe slid 10 mm down the
jaws during the lift and fell out at the carry's first reorientation, at
combined mu ~4.5 -- contact geometry, not friction. The export's box is
flange-width (23.6 mm) along its entire 150 mm, so flat pads meet flat faces
at the centroid with a real manifold (``enable_multiccd``), which is the
configuration the pre-hull suite measured lifting 3/3 from any station. The
box stands ~6 mm proud of the barrel per side laterally -- the exact tradeoff
the hull was brought in to remove -- accepted back *for this one object*
because a task about carrying a thing upright needs the thing holdable at its
centre more than it needs the fingers to visibly kiss the barrel."""


def collider_for(variant: str) -> str:
    """What one variant's free objects collide on.

    :data:`COLLIDER` for every variant that is a task, :data:`COMPARISON_COLLIDER`
    for the three that are an instrument. The syringe itself is the third case:
    see :data:`SYRINGE_COLLIDER`.
    """
    return COMPARISON_COLLIDER if variant in COMPARISON else COLLIDER


def _poses(meta: dict[str, Any], variant: str) -> dict[str, dict[str, Any]]:
    """Where every free object starts, keyed by the name the document uses."""
    out: dict[str, dict[str, Any]] = {}
    for name, entry in meta["objects"].items():
        out[name] = {
            "pose": tuple(float(v) for v in entry["pos"]),
            "quat": wxyz_to_xyzw(entry["quat_wxyz"]),
            "body_name": entry.get("root_body") or f"{name}_main",
            "joint_name": entry.get("joint") or f"{name}_joint0",
            # The hull of the mesh the export draws, replacing the single
            # oriented box the export fitted beside it -- except the syringe,
            # which keeps that box. See COLLIDER and SYRINGE_COLLIDER.
            "collider": SYRINGE_COLLIDER if name == "syringe_0" else collider_for(variant),
        }
    return out


OBJECT_SOURCES = ("usd", "usd_geom")
"""What ``(:staging (:objects-from ...))`` may name here, besides nothing.

Both are keys on an :class:`~robosimstudio.assetlib.record.AssetRecord`'s
``files``, which is what makes this list short and the lookup uniform: a
document names a *kind* of file and the asset says where its copy is. Empty --
the default -- means the export's own bodies, which is what every other variant
uses and what the poses were settled in.
"""


def _export_masses(export: str) -> dict[str, float]:
    """Each object's mass [kg], off its colliding geom in the merged export.

    Read from the export rather than from the per-asset MJCF because the export
    is the scene being reproduced. The two agree today -- checked for all six
    industrial tools -- but only one of them is what the MJCF variant actually
    builds, and if they ever diverge this should follow the one that ships.

    Not in the sidecar, which records each geom's frame, size and ``contype``
    and not its mass. Adding it there would mean a re-dump in the robosuite
    venv to regenerate five files for one field, so this parses the XML instead
    -- lazily, since only the two USD variants ask.
    """
    root = ET.parse(resolve_path(_export_path(export, "scene.xml"))).getroot()
    masses: dict[str, float] = {}
    for body in root.find("worldbody").findall("body"):
        colliders = [g for g in body.findall("geom") if g.get("contype") != "0" and g.get("mass")]
        if len(colliders) == 1:
            masses[(body.get("name") or "").removesuffix("_main")] = float(colliders[0].get("mass"))
    return masses


def _restage_objects(objects: tuple[ObjectSpec, ...], source: str, export: str) -> tuple[ObjectSpec, ...]:
    """The same objects, imported from their own stages instead of the export.

    Same names, same types, same settled poses -- so the goals, the evaluator
    and the regions are the MJCF variant's unchanged, and the only thing that
    differs is where the geometry and the physics come from. That is the whole
    point of these variants: a difference you then measure is the importer's,
    because nothing else moved.

    **The two flavours are told apart by the file, not by this argument.**
    ``source`` picks which file; the stage then declares whether it describes a
    rigid body, and :class:`~robosimstudio.scene.entities.rigid.EntityRigid`
    reads that. So ``approximate`` and ``mass`` are stated only for the
    geometry-only flavour -- the other carries ``PhysicsMassAPI`` and
    ``physics:approximation`` itself, and restating them here would be this
    file having an opinion about the thing the variant exists to measure.

    **The bins are not objects in this sense.** They arrive in ``spec.objects``
    because a fixture is lowered to one, but they are welded scenery the export
    owns and the strip below leaves them alone -- only free bodies are stripped,
    so restaging them would import a second pair on top of the ones still there.
    """
    if source not in OBJECT_SOURCES:
        raise ValueError(f"tool_sort: (:objects-from {source}) is not one of {', '.join(OBJECT_SOURCES)}")
    geometry_only = source == "usd_geom"
    masses = _export_masses(export) if geometry_only else {}
    restaged = []
    for spec in objects:
        if spec.static:
            restaged.append(spec)
            continue
        record = get_asset(spec.kind)
        restaged.append(
            replace(
                spec,
                asset=declared_asset(
                    record,
                    source,
                    # The MJCF variant's collider, restated for the flavour that
                    # has no opinion of its own -- and it has to be restated,
                    # because the point of these two is that everything except
                    # the importer is held still. It moved from ``bounding_box``
                    # to the hull with :data:`COLLIDER`; a variant left on the
                    # box would be comparing a collider against a different
                    # collider and calling the difference USD's.
                    # ``keep_visual`` holds the drawn mesh so the swap is
                    # invisible on camera, which is what makes the three
                    # comparable in a render as well as in a number.
                    approximate=COMPARISON_COLLIDER if geometry_only else None,
                    keep_visual=True,
                ),
                # The export's own world coordinates -- the same numbers the
                # MJCF variant's objects arrive at, and not in the surface frame
                # the scene axis would otherwise compose on.
                world_xform=True,
                # Only stated on the geometry-only flavour; the other takes its
                # mass from the stage's PhysicsMassAPI and agrees with this by
                # construction, both being the export's figure.
                mass=masses.get(spec.name) if geometry_only else None,
                # Cleared for tidiness, not for effect: the compiler already
                # ignores `collider` once a spec carries an asset (it is the
                # scene-declared path's approximation, applied to shapes the
                # arena import made). What acts here is the asset's
                # `approximate` above, and leaving both set would have the spec
                # state its collider twice through two channels, one of them
                # dead.
                collider="",
            )
        )
    if geometry_only:
        missing = [s.name for s in restaged if not s.static and s.mass is None]
        if missing:
            raise ValueError(f"tool_sort: {export}/scene.xml states no collider mass for {', '.join(missing)}")
    return tuple(restaged)


def _resolve_regions(lowered: Lowered, meta: dict[str, Any]) -> tuple[RegionSpec, ...]:
    """Regions, with any container's frame folded into its ranges.

    A container is not a body -- it is welded into ``bin2`` as geometry -- so
    there is no frame here to hang a region off. The export records the offset
    from ``bin2`` to the cavity floor, and this names it; the folding itself is
    :func:`~robosimstudio.taskbuild.lower.rebase_regions`, since the
    arithmetic is the same wherever a target turns out not to be a body and only
    the two facts feeding it are this export's.
    """
    containers = meta.get("containers") or {}
    hosts = {name: ("bin2", tuple(entry["frame_in_bin2"])) for name, entry in containers.items()}
    return rebase_regions(lowered.spec.regions, hosts)


def _statics(export: str, *, strip_objects: bool = False) -> tuple[StaticSpec, ...]:
    """The whole workcell, as one imported asset with its metadata sidecar.

    Args:
        export: Which dumped scene.
        strip_objects: Drop the export's own free-jointed bodies on import,
            leaving the bins, the bench and the robot. What the USD variants
            need: their ten objects arrive from their own stages, and importing
            the MJCF ones as well would build the pile twice -- twenty bodies
            sitting inside each other, which still compiles and then explodes
            the contact solver.

            Note what this does *not* ask for. Stripping alone no longer selects
            the collapsing arena entity: collapsing folds ``bin1`` and ``bin2``
            into the world, and the regions that hang off those two frames would
            then have nothing to hang off. See ``compile.compiler._make_static``.
    """
    return (
        StaticSpec(
            name="workcell",
            asset=AssetSpec(path=_export_path(export, "scene.xml")),
            meta="scene_meta.json",
            strip_free_bodies=strip_objects,
            # This asset *is* the room robosuite_bench names, imported here
            # because the suite predates the scene axis. A room is not placed
            # relative to itself, so its xform stays world.
            world_xform=True,
            # What is left of robosuite's arena after the export strips its room
            # shell is the workbench, the bins and their labelled compartments --
            # the scene the task is about.
            visible=True,
        ),
    )


def _hosted_bindings(export: str, stripped_file: str = "") -> tuple[Binding, ...]:
    """The pairings that stand a foreign robot in this workcell.

    Not in the document, and the reason is worth stating: each of these carries a
    ``static_overrides`` naming a *different file* -- the same arena with the
    Panda cut out of it. That is configuration, not a claim about the task, and
    which file it is depends on the export, which is why it is built here where
    the export is known rather than written out six times.

    What a document must never do is assert that a foreign pairing *works*. That
    is a fact about the matrix, discovered by running it, and a file asserting it
    would go stale with nothing to notice. These rows say only "here is how to
    stand this robot here", and every one carries the note it has always had.
    """
    stripped = (("workcell", stripped_file or _export_path(export, "scene_norobot.xml")),)
    return (
        # The same arena with the Panda cut out of it, hosting a YAM. This is
        # what static_overrides is for: the export welds its robot into the room,
        # so a foreign one needs the *stripped* file -- and which file that is
        # depends on the variant, since four of the six carry their own object
        # set. scene_resident is off because the point is that there is no longer
        # a robot in there to index.
        Binding(
            scene="robosuite_bench",
            embodiment="yam_i2rt",
            work_arm="right",
            support_arm="left",
            scene_resident=False,
            static_overrides=stripped,
            notes="compiles; no policy",
        ),
        # The same stripped arena hosting the six-jointed arm. Its hand is welded
        # on rather than shipped with it, so this row is the whole fourth axis in
        # one binding: a task's scene, another robot, and a hand that came from a
        # third file.
        Binding(
            scene="robosuite_bench",
            embodiment="ur5",
            work_arm="right",
            scene_resident=False,
            static_overrides=stripped,
            notes="compiles; no policy",
        ),
        # The same robot wearing a different hand -- the fourth axis, taking a
        # value. The i2rt gripper was measured on a YAM and its closing axis
        # holds to 0.3 degrees here, which is what makes it one hand rather than
        # two that happen to share a name.
        Binding(
            scene="robosuite_bench",
            embodiment="ur5",
            gripper="linear_4310_attached",
            work_arm="right",
            scene_resident=False,
            static_overrides=stripped,
            notes="compiles; no policy",
        ),
        # The export's own Panda, wearing a Robotiq 2F-85 instead of the
        # export's franka_hand_robosuite. Unlike the ad hoc gripper-override
        # this row used to carry, the pairing is its own registered
        # embodiment (``panda_robotiq``, robosimstudio/robot/embodiments/
        # panda.py): a purpose-built hand-free asset
        # (``panda_nohand_standalone.xml``, the standalone Panda with its
        # ``gripper0_right_gripper`` subtree physically removed) rather than
        # importing the hand-carrying standalone and disabling it at build
        # time. The Robotiq mounts on ``robot0_right_hand`` at the frame the
        # export's own native hand was welded at (zero offset, a quarter turn
        # about -z) -- robosimstudio/robot/grippers/robotiq_2f85_panda.py --
        # so the arm's IK and home pose, both authored against that frame,
        # need no re-derivation.
        Binding(
            scene="robosuite_bench",
            embodiment="panda_robotiq",
            work_arm="right",
            scene_resident=False,
            static_overrides=stripped,
            notes="compiles; no policy",
        ),
    )


def _shift_env(name: str) -> tuple[float, float]:
    """An experiment knob: ``"dx,dy"`` metres out of the environment, or zero.

    ``SHARPS_TRAY_SHIFT`` moves the goal syringe -- and with it the spare and
    the tray, since :func:`_layout` shifts every entry of a variant's layout
    together. ``SHARPS_BOX_SHIFT`` moves the sharps box (see
    :func:`_shifted_workcell`). Environment variables rather than spec fields
    because they exist for a randomization *sweep* -- twenty builds of the same
    variant at twenty layouts, each in its own process -- and a knob for an
    experiment should not become a field every document has to not-state.
    """
    import os

    raw = os.environ.get(name, "")
    if not raw:
        return (0.0, 0.0)
    dx, dy = (float(v) for v in raw.split(","))
    return (dx, dy)


def _shifted_workcell(export: str, shift: tuple[float, float]) -> str:
    """A copy of the robot-stripped workcell with the sharps box moved [m].

    The box is welded into ``bin2`` as the ``syringe_box_c*`` geom family, so
    moving it is a rewrite of those geoms' ``pos`` -- nothing else in the file
    changes. Written beside the original (same directory, so its ``../assets``
    references still resolve), named by the shift, and reused when it already
    exists. The caller shifts the region to match via the sidecar's
    ``frame_in_bin2``.
    """
    dx, dy = shift
    stem = f"scene_norobot_box{dx:+.3f}{dy:+.3f}".replace(".", "p").replace("-", "m").replace("+", "")
    src = resolve_path(_export_path(export, "scene_norobot.xml"))
    dst = src.parent / f"{stem}.xml"
    if not dst.exists():
        tree = ET.parse(src)
        moved = 0
        for geom in tree.getroot().iter("geom"):
            name = geom.get("name") or ""
            if name.startswith("syringe_box_"):
                px, py, pz = (float(v) for v in geom.get("pos", "0 0 0").split())
                geom.set("pos", f"{px + dx:.6f} {py + dy:.6f} {pz:.6f}")
                moved += 1
        if not moved:
            raise ValueError(f"{src}: no syringe_box_* geoms to shift")
        tree.write(dst, encoding="unicode")
    return _export_path(export, dst.name)


def build(variant: str) -> TaskSpec:
    """One variant, built from its document."""
    doc = apply_bindings(read_problem(DOCUMENTS[variant]), _binding_table(BINDINGS))
    staging = dict(doc.staging)
    export = str(staging.get("export", ""))
    # Read off the raw section rather than a parsed Staging: lower() does the
    # parsing and the profile merge, and this is needed a step earlier to decide
    # what to hand it. The two cannot disagree -- a profile states no
    # objects-from, so the document is the only source.
    source = str(staging.get("objects-from", "") or "")
    meta = json.loads(resolve_path(_export_path(export, "scene_meta.json")).read_text())

    tray_shift = _shift_env("SHARPS_TRAY_SHIFT")
    layout = _layout(variant, tray_shift)
    # A trial's layout may state where its box stands; the sweep knob in the
    # environment is for a run that says nothing.
    box_shift = (
        tuple(float(v) for v in layout["box_shift"]) if layout.get("box_shift") else _shift_env("SHARPS_BOX_SHIFT")
    )
    if tray_shift != (0.0, 0.0) or box_shift != (0.0, 0.0):
        import copy

        meta = copy.deepcopy(meta)
        if tray_shift != (0.0, 0.0):
            entry = meta["objects"]["syringe_0"]
            entry["pos"] = [entry["pos"][0] + tray_shift[0], entry["pos"][1] + tray_shift[1], entry["pos"][2]]
        if box_shift != (0.0, 0.0):
            box = meta["containers"]["syringe_box"]
            box["frame_in_bin2"] = [
                box["frame_in_bin2"][0] + box_shift[0],
                box["frame_in_bin2"][1] + box_shift[1],
                box["frame_in_bin2"][2],
            ]

    poses = _poses(meta, variant)
    for name, entry in layout.get("objects", {}).items():
        poses[name] = {**poses.get(name, {}), "pose": tuple(entry["pose"]), "quat": tuple(entry["quat"])}

    lowered = lower(
        doc,
        suite="syringe_packing",
        name=variant,
        make_evaluator=lambda d: evaluator_factory(d, task=f"syringe_packing/{variant}"),
        resolve_artifact=task_resolver(HERE, __package__),
        poses=poses,
        # `stage()` lets this argument beat the document's own `(:profile ...)`,
        # so it has to name the same thing the 94 documents do or they are
        # lowered against a profile none of them states -- silently, and on the
        # one field that decides what this suite's perception policy sees.
        profile="vab_industrial@v2",
        statics=_statics(export, strip_objects=bool(source)),
    )
    spec = lowered.spec
    if source:
        spec = restage(spec, objects=_restage_objects(spec.objects, source, export))
    spec = restage(spec, objects=_gripped_objects(spec.objects))
    stage = _tray_pair_staging(layout) if layout else _tray_staging
    objects, statics = stage(spec.objects, spec.statics)
    spec = restage(spec, objects=objects, statics=statics + _studio_dressing())
    stripped_file = _shifted_workcell(export, box_shift) if box_shift != (0.0, 0.0) else ""
    return _finish(spec, meta, _resolve_regions(lowered, meta), export, stripped_file)


SYRINGE_HALF_LENGTH = 0.075
"""Half the syringe's 150 mm barrel-to-flange span, along its own local y."""
SYRINGE_HALF_WIDTH = 0.011783
"""Half the 23.6 mm flange width, along the syringe's local x."""
SYRINGE_HALF_HEIGHT = 0.0068
"""Half the 13.6 mm flange height -- how far the origin sits above the surface
it rests on. See ``routine.py``'s ``GRASP_HANDLE_FRAC`` for the mesh survey
these three numbers come from.
"""

TRAY_ASSET = declared_file(INSTRUMENT_TRAY, "mjcf")
"""VAB-industrial's own ``instrument_tray`` (``add-all-syringes-sharps-box``
branch, ``assets/objects/_utils/make_tray.py``), declared in ``assets/records.py``
and shipped beside it rather than reinvented as boxes. Its own MJCF collides
as the drawn mesh's convex hull, which -- per that generator's own comment --
"would fill the tray in solid"; :func:`_tray_floor`'s collider below is
this suite's stand-in for the box-primitive collision VAB-industrial's own env
builds against the mesh's ``cavity_min``/``cavity_max`` sites, with
``StaticSpec.collide=False`` keeping the imported mesh itself decorative."""
TRAY_OUTER_HALF = (0.100, 0.200, 0.006)
"""The asset's own outer half-extents [m] (``make_tray.py``'s ``HALF``)."""
TRAY_WALL_THICKNESS = 0.003
"""The asset's own wall/floor thickness [m] (``make_tray.py``'s ``WALL``)."""
TRAY_FLOOR_COLLIDER_HALF_THICKNESS = 0.002
TRAY_SETTLE_MARGIN = 0.001
"""Extra lift so the syringe starts a hair above the new floor rather than
inside it -- physics settles the last millimetre, as the packing tray does."""


def _tray_staging(
    objects: tuple[ObjectSpec, ...], statics: tuple[StaticSpec, ...]
) -> tuple[tuple[ObjectSpec, ...], tuple[StaticSpec, ...]]:
    """Stand the syringe on a real tray instead of the bare bench.

    The export lays the syringe straight on the workbench; this stands
    :data:`TRAY_ASSET` under it, centred on wherever the export settled it,
    and lifts the syringe by the wall thickness twice over -- once for the
    tray's own floor sitting on the bench, once for the cavity floor sitting
    above *that* -- so it lands inside the cavity rather than on the rim or
    through the (decorative-only) floor.

    The tray is yawed to the syringe's own heading rather than left
    world-aligned, built from the z-component of its quaternion alone --
    the export's residual roll/pitch is noise a flat tray floor should not
    inherit. At :data:`TRAY_OUTER_HALF`'s 200 x 400 mm footprint against a
    23.6 x 150 mm syringe, there is no repeat of this suite's earlier
    box-tray mistake (walls narrower than the Robotiq's open span): the
    nearest wall is tens of millimetres further off than that ever was.
    """
    goal = next((o for o in objects if o.name == "syringe_0"), None)
    if goal is None:
        return objects, statics
    ox, oy, oz = goal.pose
    _, _, qz, qw = goal.quat
    yaw = 2.0 * math.atan2(qz, qw)
    tray_quat = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

    bench_top = oz - SYRINGE_HALF_HEIGHT
    tray, floor = _tray_statics((ox, oy, bench_top + TRAY_OUTER_HALF[2]), tray_quat)
    lifted = replace(goal, pose=(ox, oy, _cavity_floor_z(tray) + SYRINGE_HALF_HEIGHT + TRAY_SETTLE_MARGIN))
    objects = tuple(lifted if o.name == "syringe_0" else o for o in objects)
    return objects, statics + (tray, floor)


def _cavity_floor_z(tray: StaticSpec) -> float:
    """Where the tray's inner floor is [m], off its origin at mid-height."""
    return tray.xform[2] + TRAY_OUTER_HALF[2] - TRAY_WALL_THICKNESS


def _tray_statics(
    origin: tuple[float, float, float], quat: tuple[float, float, float, float]
) -> tuple[StaticSpec, StaticSpec]:
    """The tray as drawn, and the invisible floor slab it is stood on.

    See :data:`TRAY_ASSET` for why the mesh does not collide as itself.
    """
    tray = StaticSpec(
        name="syringe_tray",
        asset=AssetSpec(path=TRAY_ASSET),
        xform=(*origin, *quat),
        world_xform=True,
        collide=False,
    )
    floor = StaticSpec(
        name="syringe_tray_floor",
        half_extents=(
            TRAY_OUTER_HALF[0] - TRAY_WALL_THICKNESS,
            TRAY_OUTER_HALF[1] - TRAY_WALL_THICKNESS,
            TRAY_FLOOR_COLLIDER_HALF_THICKNESS,
        ),
        xform=(origin[0], origin[1], _cavity_floor_z(tray) - TRAY_FLOOR_COLLIDER_HALF_THICKNESS, *quat),
        world_xform=True,
        visible=False,
    )
    return tray, floor


SYRINGE_MESH_ASSET = _export_path("vab_tool_sort", "assets/syringe.obj")
"""The bare mesh from this task's copy of the export's tool pool -- the
same file the ``syringe`` export's own ``syringe_0`` body draws, imported
directly per :func:`_tray_pair_staging` rather than found by name in that export,
which carries only the one instance.
"""
SYRINGE_EXPORT_MASS_KG = 0.030
"""``syringe_0_g0``'s own authored mass, read off ``vab_tool_sort/syringe/
scene.xml`` -- the pack's extra instances get the same figure so a policy
cannot tell the goal syringe from its neighbours by weight."""
SYRINGE_PACK_COLOR = (0.90, 0.91, 0.93)
"""The reference splash's syringe white, so the row reads as product rather
than as nine palette-coloured boxes."""


def _layout(variant: str, shift: tuple[float, float] = (0.0, 0.0)) -> dict[str, Any]:
    """The settled layout a variant's document has no coordinates for, or ``{}``.

    ``start_poses/<variant>.json``: ``objects`` and ``fixtures``, each name to
    a world ``pose``/``quat``. It is the arrangement this loader used to compute
    at build -- the goal syringe lifted onto the tray floor, the spare one
    spacing along the row, the tray centred under the pair -- written out once
    so the scene a document stages is a file beside it rather than arithmetic
    here, and the same file the validated episode ran on.

    ``shift`` moves every entry together: the tray, the goal and the spare are
    one arrangement, and ``SHARPS_TRAY_SHIFT`` sweeps the arrangement rather
    than sliding a syringe out of its tray.
    """
    path = POSES / f"{variant}.json"
    if not path.is_file():
        return {}
    layout = json.loads(path.read_text())
    dx, dy = shift
    for group in ("objects", "fixtures"):
        for entry in layout.get(group, {}).values():
            x, y, z = entry["pose"]
            entry["pose"] = [x + dx, y + dy, z]
    return layout


def _tray_pair_staging(layout: dict[str, Any]):
    """The tray scene, from its recorded layout: the drawn tray, its floor, the spare.

    The document declares the tray as a fixture and the spare as a second
    ``syringe`` so the scene says what is in it; this supplies what a
    declaration cannot. The tray fixture's handle is dropped the way
    :func:`_finish` drops the container's -- a static supplied under the same
    name is what the compiler places, and a free-body lookup for it would find
    nothing in the export. The spare is imported straight from the bare mesh
    rather than found in the export (see
    :class:`~robosimstudio.scene.entities.rigid.EntityRigid`'s two arrivals),
    since the export carries only the one instance; ``approximate="convex_hull"``
    matches :data:`COLLIDER`'s reasoning for ``syringe_0`` itself, and the mass
    is the export's own rather than density times a hollow mesh's volume.
    """

    def stage(
        objects: tuple[ObjectSpec, ...], statics: tuple[StaticSpec, ...]
    ) -> tuple[tuple[ObjectSpec, ...], tuple[StaticSpec, ...]]:
        tray_at = layout["fixtures"]["syringe_tray"]
        tray, floor = _tray_statics(tuple(tray_at["pose"]), tuple(tray_at["quat"]))
        staged = []
        for spec in objects:
            if spec.name == "syringe_tray":
                continue
            if spec.name.startswith("syringe_pack"):
                spec = replace(
                    spec,
                    asset=AssetSpec(path=SYRINGE_MESH_ASSET),
                    # The same flat-faced box the goal syringe collides on
                    # (SYRINGE_COLLIDER keeps the export's authored one): the
                    # two-syringe task grasps this one at its centre of mass
                    # too, and the hull's wedge between barrel and flange is
                    # what the goal syringe was moved off for. Every pack
                    # instance beyond ``_0`` (some trials' 30-layout sweep
                    # widens the row to 3-4 syringes) is scenery the same way.
                    collider="bounding_box",
                    world_xform=True,
                    obb_grasp=False,
                    mass=SYRINGE_EXPORT_MASS_KG,
                    friction=3.0,
                    color=SYRINGE_PACK_COLOR,
                )
            staged.append(spec)
        return tuple(staged), statics + (tray, floor)

    return stage


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


def _studio_dressing() -> tuple[StaticSpec, ...]:
    """Visual-only room furniture, so the *record-time* video reads as a room.

    The tiled recording is the honest artefact -- it is what the policy's
    cameras saw -- and until now it showed a floating arm over a near-black
    void: the export strips robosuite's room shell, the ground plane draws in
    Newton's dark default, and the robotiq binding's ``scene_norobot.xml``
    takes the export's pedestal away with its Panda. These pieces put a floor,
    two back walls and a pedestal into the *scene itself*, so every renderer --
    the record-time tiled tracer included -- shows the cell standing somewhere,
    with no re-render pass needed.

    All of it is ``collide=False``: the validated physics
    (``outputs/ten_syringes-box-v5``) must not move by a millimetre for a
    coat of paint. The names carry the ``dressing_`` prefix so a look that
    builds its own room (``tools/render_offline_factory.py``) can hide these
    and substitute its textured ones.

    The pedestal stands under :data:`PANDA_BASE` for every binding. The
    resident-Panda binding already has the export's own pedestal there; this
    slimmer column hides inside it rather than beside it.
    """
    bx, by, bz = PANDA_BASE
    return (
        # Thin slab just proud of the compiled ground plane, so the camera sees
        # concrete rather than the plane's near-black default.
        StaticSpec(
            name="dressing_floor",
            half_extents=(4.0, 4.0, 0.0005),
            xform=(0.0, 0.0, 0.0005, 0.0, 0.0, 0.0, 1.0),
            world_xform=True,
            collide=False,
            color=DRESSING_FLOOR_COLOR,
        ),
        # The two walls the agentview camera actually faces (it looks from
        # (+1.3, -0.8) towards the bench, so the frame's background is the -x
        # and +y quadrant).
        StaticSpec(
            name="dressing_wall_west",
            half_extents=(0.02, 4.0, 1.75),
            xform=(-2.4, 0.0, 1.75, 0.0, 0.0, 0.0, 1.0),
            world_xform=True,
            collide=False,
            color=DRESSING_WALL_COLOR,
        ),
        StaticSpec(
            name="dressing_wall_north",
            half_extents=(4.0, 0.02, 1.75),
            xform=(0.0, 2.8, 1.75, 0.0, 0.0, 0.0, 1.0),
            world_xform=True,
            collide=False,
            color=DRESSING_WALL_COLOR,
        ),
        StaticSpec(
            name="dressing_pedestal",
            half_extents=(0.10, 0.10, bz / 2.0),
            xform=(bx, by, bz / 2.0, 0.0, 0.0, 0.0, 1.0),
            world_xform=True,
            collide=False,
            color=DRESSING_PEDESTAL_COLOR,
        ),
    )


def _gripped_objects(objects: tuple[ObjectSpec, ...]) -> tuple[ObjectSpec, ...]:
    """The syringe, with a grip a parallel-jaw hand can actually hold.

    The export's own collider states mu=1.0 -- unremarkable for a tool lying
    flat in a bin, but this suite's grasp is a short reach on a 150 mm lever
    arm (see ``routine.py``'s ``GRASP_HANDLE_FRAC``). ``ObjectSpec.friction``
    is a resident object's only path to a value other than the importer's
    own, per ``EntityRigid.find_in_scene``.

    **Not the pad-friction workaround this suite carried for a while.**
    Grasping at 4.5x realistic combined mu (pads at 4.0, this at 5.0) still
    dropped the syringe during reorientation, because the failure was never
    friction: the drawn-hull collider's cross-section between barrel and
    flange is an interpolated wedge with no real geometry under a
    centre-of-mass pinch. Fixed at the source -- :data:`SYRINGE_COLLIDER`
    keeps the export's own flat-faced box for this object -- which is what
    makes a plain 2.0 here (a fact about a plastic barrel in a rubber pad,
    not a compensation for missing contact) hold through the carry.
    """
    return tuple(replace(o, friction=2.0) if o.name == "syringe_0" else o for o in objects)


def _finish(
    spec: TaskSpec,
    meta: dict[str, Any],
    regions: tuple[RegionSpec, ...],
    export: str,
    stripped_file: str = "",
) -> TaskSpec:
    """The spec with container fixtures removed, regions resolved, hosts added, look set.

    A container is declared in the document so its region has something to hang
    off and so the typechecker can see the reference resolve -- but it has no
    body in the import, so emitting an object handle for it would send the
    compiler looking for a body that is not there. It contributes a frame and
    nothing else.
    """
    containers = set(meta.get("containers") or {})
    return restage(
        spec,
        objects=tuple(o for o in spec.objects if o.name not in containers),
        regions=regions,
        bindings=spec.bindings + _hosted_bindings(export, stripped_file),
        # scene_resident: the arena imported above already contains this Panda,
        # pedestal and all. Without it the compiler would import the extracted
        # standalone asset as well and the scene would hold two arms.
        # monolithic: this variant's arena and its robot arrive in one import, so
        # neither axis can be swapped until a binding names the robot-stripped
        # variant.
        requires=TaskRequirements(arms=1, monolithic=True),
        # The house look with the backdrop on the walls behind the bench; the
        # path tracer alone draws scenery, so perception is unchanged.
        render=SYRINGE_PACKING_LOOK,
    )


def build_all() -> tuple[TaskSpec, ...]:
    """Every variant, in the order the suite has always listed them."""
    return tuple(build(name) for name in VARIANTS)


SYRINGE_PACKING = register_suite(
    TaskSuite(
        name="syringe_packing",
        variants=build_all(),
        description="Post a single-use syringe through the aperture of a sealed sharps box (Franka Panda).",
        provenance=(
            "VAB-industrial ToolSortPickPlace, robosuite 1.4.0, "
            "MuJoCo 3.3.3, seed 0, `put_the_syringe_in_the_sharps_box` problem. Regions and goals "
            "transcribed into the document beside this file; see its header for what 'reconstructed' "
            "means and why a round trip through it proves less than it looks."
        ),
    )
)
"""The suite, built at import.

Eagerly, not lazily. Building reads the document, resolves its sidecar and binds
its goal tree -- a millisecond, once -- and doing it here means a document that
does not typecheck fails when the roots load rather than when somebody asks for
the variant it broke. A registry that can hand out a task which then fails to
build is a registry whose listing cannot be trusted.
"""
