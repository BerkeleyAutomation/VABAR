# SPDX-License-Identifier: Apache-2.0
"""Import profiles: how a body of borrowed tasks was staged.

A profile fills in what a dialect leaves to harness convention. LIBERO and VAB
problem files say what objects exist and what should become true of them; which
cell they sit in, which robot reaches for them, at what rate the physics runs and
how many episodes the task defines are all supplied by whoever runs the file.
That is fine until there are two runners, and it means a task cannot be read --
you can hold a whole problem file and still not know what it ran on.

So the convention is written down, versioned, and recorded per field. A document
naming ``vab_industrial@v1`` gets a Panda in a robosuite cell, and the lowering
step records that those two came from *here* rather than from the document. That
distinction is the point: a task that named its own Panda has decided, where one
that inherited it has not, and if the convention turns out wrong only the second
is worth re-reading.

**These live in the corpus because a profile is a claim about content.** "VAB
industrial tasks run on a Panda at 50 Hz over 500 Hz physics" is a fact about
that benchmark, not about this library. The registry that holds profiles is
machinery and lives in the toolkit; what goes in it is this file.
"""

from __future__ import annotations

import vabar  # noqa: F401 -- the scripted-routine module RoboSimStudio leaves out; see vabar/__init__.py
import vabar.ext.tool_sort  # noqa: F401 -- the robots this suite's host rows name

import numpy as np

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.core.math import look_at_matrix, xform_from_matrix
from robosimstudio.robots.registry import get_gripper
from robosimstudio.specs import ArrangementSpec, CollisionSpec, backend
from robosimstudio_bench.taskbuild import register_arrangement, register_camera_rig
from robosimstudio_bench.taskbuild.staging import Profile, Staging, register_profile
from robosimstudio_bench.taskdoc import VAB_INDUSTRIAL_V1 as VAB_DIALECT

# The profile below stages ``scene="robosuite_bench"``, so the cell it names is
# registered from here: a staging module is what the generic loader executes
# for a suite with no ``task.py``, and a profile naming a cell nobody had
# registered would fail three layers away from this file.
from . import scenes as _cell  # noqa: F401

__all__ = [
    "PANDA_WRIST_MOUNT",
    "VAB_CAMERAS",
    "VAB_INDUSTRIAL_ROBOTIQ_V1",
    "VAB_INDUSTRIAL_V1",
    "VAB_PHYSICS",
    "VAB_TOOL_SORT_V2",
]


def _panda_wrist_mount() -> tuple[float, float, float, float, float, float, float]:
    """YAM-style eye-in-hand view derived in Panda's actual TCP frame.

    ``parent_frame="work"`` resolves to ``gripper0_eef`` for the scene-resident
    Panda, and that body is already the hand's TCP.  The previous numbers were
    copied from a Robotiq *mount-body* frame and put the eye forward/inside this
    hand.  Here the eye is behind the TCP, laterally clear of the fingers, and
    looks through the jaws to a point beyond them, exactly as the YAM Level-1
    mount is constructed from its own hand geometry.
    """
    hand = get_gripper("franka_hand_robosuite")
    # Match tool_hanging's YAM construction: lateral displacement is along
    # local Y (perpendicular to Franka's local-X closing axis), while the eye
    # sits behind/above the TCP on -approach and looks beyond the fingertips on
    # +approach.  The extra height keeps the optical centre above the hand
    # instead of inside the fingers.
    eye = np.array([0.0, 0.100, -0.070], dtype=float)
    target = np.asarray(hand.tcp_offset, dtype=float) + np.array([0.0, 0.0, 0.050])
    camera_to_tcp = look_at_matrix(eye, target, hand.close_axis)
    landscape_roll = np.eye(4)
    landscape_roll[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return tuple(float(v) for v in xform_from_matrix(camera_to_tcp @ landscape_roll))


PANDA_WRIST_MOUNT = _panda_wrist_mount()
"""The wrist camera's pose in the Panda's TCP frame, shared by both rigs below.

Derived rather than written out: it evaluates to
``(0.0, 0.1, -0.07, 0.0, -0.3404252637530182, 0.9402715776831118, 0.0)``, which
is the tuple ``tool_sort_agentview@v2`` used to carry as a literal, to the digit.
"""


VAB_TOOL_SORT_V2 = register_profile(
    Profile(
        name="vab_tool_sort@v2",
        bindings=VAB_DIALECT,
        summary=(
            "VAB-industrial, robosuite 1.4.0, MuJoCo 3.3.3. "
            "One Panda in the merged workcell, 50 Hz control over 500 Hz physics, one fixed episode."
        ),
        staging=Staging(
            scene="robosuite_bench",
            embodiment="panda",
            # Every VAB export welds its Panda into the same MJCF as the
            # workbench, so the compiler must not import the standalone one as
            # well. Omitting this does not fail politely: the scene builds with
            # two arms and dies on a duplicate joint label, three layers from
            # anything that mentions robots.
            scene_resident=True,
            physics="vab_500hz@v1",
            cameras="tool_sort_agentview@v2",
            episodes=(("count", 1), ("seed", 0), ("sampler", "fixed")),
            control_mode="joint_position",
        ),
    )
)
"""The convention every VAB-industrial problem was written under.

``max_steps`` is deliberately absent. It is not a convention -- each problem's
budget is sized for that problem's goal count and destination kinds, and a
profile-wide default would silently truncate an eight-goal scene after its first
tool and report the result as the policy's.

The export is absent for the same reason and a stronger one: two variants share
one export (``hammer`` is a second goal over ``sort_all``'s scene), so it cannot
be derived from the variant and must be stated per document.
"""


VAB_INDUSTRIAL_V1 = register_profile(
    Profile(
        name="vab_industrial@v1",
        bindings=VAB_DIALECT,
        summary=(
            "VAB-industrial, robosuite 1.4.0, MuJoCo 3.3.3. "
            "One Panda in the merged workcell, 50 Hz control over 500 Hz physics, one fixed episode."
        ),
        staging=Staging(
            scene="robosuite_bench",
            embodiment="panda",
            # Every VAB export welds its Panda into the same MJCF as the
            # workbench, so the compiler must not import the standalone one as
            # well. Omitting this does not fail politely: the scene builds with
            # two arms and dies on a duplicate joint label, three layers from
            # anything that mentions robots.
            scene_resident=True,
            physics="vab_500hz@v1",
            cameras="agentview@v1",
            episodes=(("count", 1), ("seed", 0), ("sampler", "fixed")),
            control_mode="joint_position",
        ),
    )
)
"""The convention ``syringe`` reads under: ``vab_tool_sort@v2`` with the other rig.

The same cell, robot, rates, episode and dialect table as the profile above;
the one field that differs is ``cameras``, and the one camera that differs in
that rig is the overhead. It came here with ``sharps_disposal`` when that suite
merged back into this one, and it stays a profile of its own rather than a
clause on the document because the perception graph beside that document was
fitted against this overhead -- the rig is part of what the graph measured.
"""


VAB_INDUSTRIAL_ROBOTIQ_V1 = register_profile(
    Profile(
        name="vab_industrial_robotiq@v1",
        # The same upstream dialect as the two profiles above, and for the same
        # reason: these documents are transcriptions of the source benchmark's
        # problems and read `In` the way it means it -- a box test a carried
        # object satisfies -- so that a number scored here is comparable with a
        # number published there. Changing the hand does not change the dialect
        # the documents were written in.
        bindings=VAB_DIALECT,
        summary=(
            "The vab_industrial@v1 cell with a Robotiq 2F-85 in place of the export's Franka hand. "
            "Same arena, same arm, same 50 Hz over 500 Hz physics; the hand is welded on at build "
            "time rather than indexed out of the merged MJCF."
        ),
        staging=Staging(
            scene="robosuite_bench",
            embodiment="panda_robotiq",
            # The opposite of the profile above, and the whole difference between
            # them. ``vab_industrial@v1`` indexes the Panda already welded into
            # the export; this one imports a hand-free arm and bolts a 2F-85 to
            # its flange, so the export must arrive *without* a robot in it --
            # which is what ``scene_norobot.xml`` is, and what a document staged
            # under this profile has to name.
            scene_resident=False,
            physics="vab_500hz@v1",
            cameras="tool_sort_agentview@v3",
            episodes=(("count", 1), ("seed", 0), ("sampler", "fixed")),
            control_mode="joint_position",
        ),
    )
)
"""What the Level-1 documents read under: the industrial cell wearing a 2F-85.

Its thirty documents arrived (September 2026) without this registration -- they
name the profile and the delivery carried the documents, the exports and the
graph alone -- so it is transcribed here from the branch they were authored on,
RoboSimStudio's ``integrate/perception-cable`` at ``92343458``, unedited apart
from the wrist mount below being this file's constant rather than its literal.

``tool_sort_agentview@v3`` rather than ``@v2`` is that branch's choice and the
one thing here a reader should question: it is ``@v2``'s geometry with the
overhead doubled to 1280x960, and the overhead is the only 3D measurement the
sorting graph gets. See the rig's own note for the mask areas that forced it.
"""


VAB_PHYSICS = {
    "vab_500hz@v1": register_arrangement(
        "vab_500hz@v1",
        ArrangementSpec(
            control_hz=50.0,
            substeps=10,
            # One island: MuJoCo over every body, detecting its own contacts.
            # ``*`` rather than ``articulated`` because the tools are free rigid
            # bodies and belong to the same solver as the arm that sorts them.
            backends=(
                (
                    "*",
                    backend(
                        "mujoco",
                        nconmax=1024,
                        njmax=1024,
                        # Load bearing for astro, which collides on the hull of
                        # the mesh it draws: without it a hull resting on a box
                        # gets one contact point and rocks about it -- see below.
                        enable_multiccd=True,
                    ),
                ),
            ),
            collision=CollisionSpec(
                # The stiffest contact a 2 ms substep admits, critically damped
                # -- see below. Newton's default is four hundred times softer
                # and is what a bin of tools resting on each other was buzzing
                # against.
                contact_ke=6.0e4,
                contact_kd=500.0,
                rigid_gap=2.0e-3,
            ),
        ),
    ),
}
"""50 Hz control over 500 Hz physics -- the benchmark's own rate.

Ten substeps rather than the four a sparse rigid scene needs, because robosuite
runs this cell at MuJoCo's 500 Hz and the difference shows up in exactly the
scenes the benchmark cares about. Measured over 4 s of idle on the settled piles,
peak speed in the bin::

    variant      200 Hz    500 Hz
    sort_all      2.12      0.02
    sort_bench    6.90      1.28
    sort_power   18.64      0.31

At 200 Hz a bin of ten thin tools never comes to rest -- it creeps a few
centimetres a second, which moves the objects a policy planned against and scores
as motion the evaluator did not cause. The cost is real (about 1.3x realtime
becomes 0.7x), and it buys a scene that is where it says it is.

``nconmax`` is well above the settled scene's ~50 contacts: a gripper reaching
into a cluttered bin briefly needs far more, and an overflow escalates into an
illegal memory access rather than degrading.

**The rate was not the whole of it, and the rest is the contact model.** The
table above was measured over four seconds; over thirty the same piles are still
moving, and what moves them is compliance rather than momentum. Every surface in
these cells that does not state a contact model of its own -- the bench, both
bins, every geom of the export that carries no ``solref`` -- was taking Newton's
``ShapeConfig`` default of ``ke = 2.5e3, kd = 100``, which is MuJoCo's own
``solref = "0.02 1"``: soft enough that a 1.5 kg drill rests six millimetres into
the bench. ``contact_ke = 6e4, contact_kd = 500`` is that pair rewritten as the
stiffest this substep admits -- ``solref`` timeconst ``2/(kd*1e-3) = 4 ms``, which
is exactly MuJoCo's ``2*timestep`` floor at 500 Hz, at a damping ratio of 1.0.
Stiffer is not available: below the floor MuJoCo clamps, so the number would stop
meaning what it says. Measured by ``tools/stack_bench.py --task <variant>
--steps 1500 --warm 200 --repeat 2``, which is 30 s of idle with no action::

    scene                 scene mean   worst bob   peak speed
    astro/heap_sort
      before               1.36 mm/s     0.71 mm    28.7 mm/s
      this                 0.52 mm/s     0.24 mm    10.4 mm/s
    tool_sort/sort_all
      before               4.14 mm/s     5.66 mm   128.5 mm/s
      this                 1.47 mm/s     2.69 mm    82.9 mm/s

``rigid_gap`` is 2 mm rather than Newton's 0.1 m for a different reason, and it
is not a tuning: it is the distance MuJoCo is told to *look* for contact, and a
tenth of a metre is not a claim any of these files make. At the default,
``astro/heap_sort`` carries 178 contacts of which 39 can ever exert force -- the
other 139 are pairs up to 10 cm apart, occupying the buffer ``nconmax`` above is
sized against, and inflating every geom's broad-phase bound by 10 cm. At 2 mm
the same scene carries 39 and still resolves 38, so nothing real was being
found out there. On its own it changes the jitter by nothing measurable
(3.77 against 3.82 mm/s); it is here so the ceiling means what it says.

``enable_multiccd`` is the newest of the four and the only one a suite *needs*
rather than benefits from. mujoco_warp gives a convex pair a single contact point
unless it is on, and a single point cannot stop a body rocking about it -- which
is why the ``mujoco`` backend's schema calls it "not a quality knob". ``astro``
collides on the hull of the mesh it draws (see ``astro.COLLIDER``), and a
hull on one point walks: 44.8 mm of creep over 30 s of idle against 0.20 mm with
the flag on. It is not free -- about 1.3x the step, 135 to 99 steps/s on
``heap_sort`` -- and ``tool_sort``, whose objects are boxes and whose box-box
pairs get a manifold either way, is paid back anyway through the hand: its worst
bob falls 2.99 -> 2.18 mm and its peak 88.6 -> 68.4 mm/s, because the Robotiq's
and the Panda's fingers are convex meshes and every grasp contact was a point.

**What is left, and where it lives.** The bin scenes are quiet now; the one
object that still hops is ``tool_sort``'s hammer, and it is not a settings
problem. Its handle is a ``CYLINDER`` collider lying on the clamp's box, and
mujoco_warp's multi-point contact path accepts only box-and-mesh pairs
(``collision_convex.py``, the ``multicontact`` gate) -- so a 200 mm handle is
held by exactly one contact point and is free to rock about it, at any stiffness.
Raising the rate to 1000 Hz takes a further third off both scenes (``sort_all``
worst bob 2.69 -> 1.90 mm and peak 82.9 -> 50.6 mm/s; ``heap_sort`` 0.24 ->
0.12 mm and 10.4 -> 4.5 mm/s) and costs about 1.6x the step. That is a trade this
profile does not take, because its name is the rate -- a document naming
``vab_500hz@v1`` and getting 1000 would be the drift these arrangements are
versioned to prevent. Giving the handle a box or a convex-mesh collider would fix
the hammer at the geometry, which is where the fault is.
"""


VAB_CAMERAS = {
    "tool_sort_agentview@v2": register_camera_rig(
        "tool_sort_agentview@v2",
        (
            CameraSpec(
                name="agentview",
                width=640,
                height=480,
                fov_y=45.0,
                eye=(1.30, -0.80, 1.60),
                look_at=(0.10, 0.02, 0.82),
                near=0.01,
                far=5.0,
                channels=("rgb", "depth", "seg"),
            ),
            CameraSpec(
                name="overhead",
                width=640,
                height=480,
                fov_y=52.0,
                # Higher and rolled 90 degrees so the two long bins occupy the
                # landscape frame left-to-right instead of being cropped into
                # a portrait composition.
                eye=(0.10, 0.02, 2.05),
                look_at=(0.10, 0.02, 0.80),
                up=(-1.0, 0.0, 0.0),
                near=0.01,
                far=3.0,
                channels=("rgb", "depth"),
            ),
            CameraSpec(
                name="eye_in_hand",
                width=320,
                height=240,
                # Same calibrated Panda/YAM-style mount used by syringe
                # packing: behind the TCP and looking through the jaws.
                fov_y=60.0,
                parent_frame="work",
                xform=PANDA_WRIST_MOUNT,
                near=0.005,
                far=1.5,
                channels=("rgb", "depth"),
            ),
        ),
    ),
    "tool_sort_agentview@v3": register_camera_rig(
        "tool_sort_agentview@v3",
        (
            CameraSpec(
                name="agentview",
                width=640,
                height=480,
                fov_y=45.0,
                eye=(1.30, -0.80, 1.60),
                look_at=(0.10, 0.02, 0.82),
                near=0.01,
                far=5.0,
                channels=("rgb", "depth", "seg"),
            ),
            CameraSpec(
                name="overhead",
                # 1280x960, doubled from the 640x480 this rig carried as @v2.
                # The overhead frame is the only 3D measurement the sorting
                # graphs get, and at 640x480 -- 1.25 m up over a 52 deg vertical
                # fov -- one pixel was 2.54 mm, so a mask had to span about
                # 25 x 10 mm to clear the 100 px^2 an OBB fit needs from depth.
                # That is a limit on the EYE, not the hand: the tier guarantees
                # every object fits the jaws and never checked it could be seen.
                # Doubling each dimension puts a pixel at 1.27 mm and quadruples
                # every mask area, which carries aa_battery (109 -> 434 px^2) and
                # socket_head_screw_m6x60 (102 -> 409) off the floor they sat on.
                # It does NOT recover nylon_lock_nut (22 -> 89) or
                # countersunk_screw_m3x6 (3 -> 11); see the note below.
                # Do not answer a small object by lowering the 100 instead --
                # those pixels are the only 3D samples there are, and an
                # orientation fitted to thirty noisy points is a confident
                # wrong grasp.
                width=1280,
                height=960,
                fov_y=52.0,
                # Higher and rolled 90 degrees so the two long bins occupy the
                # landscape frame left-to-right instead of being cropped into
                # a portrait composition.
                eye=(0.10, 0.02, 2.05),
                look_at=(0.10, 0.02, 0.80),
                up=(-1.0, 0.0, 0.0),
                near=0.01,
                far=3.0,
                channels=("rgb", "depth"),
            ),
            CameraSpec(
                name="eye_in_hand",
                width=320,
                height=240,
                # Same calibrated Panda/YAM-style mount used by syringe
                # packing: behind the TCP and looking through the jaws.
                fov_y=60.0,
                parent_frame="work",
                xform=PANDA_WRIST_MOUNT,
                near=0.005,
                far=1.5,
                channels=("rgb", "depth"),
            ),
        ),
    ),
    "agentview@v1": register_camera_rig(
        "agentview@v1",
        (
            CameraSpec(
                name="agentview",
                width=640,
                height=480,
                fov_y=45.0,
                eye=(1.30, -0.80, 1.60),
                look_at=(0.10, 0.02, 0.82),
                near=0.01,
                far=5.0,
                channels=("rgb", "depth", "seg"),
            ),
            CameraSpec(
                name="overhead",
                width=640,
                height=480,
                fov_y=52.0,
                # Near-vertical is the reliable task view: it maximizes the
                # cap/hole pixels and removes depth parallax. The 640x480 frame
                # remains landscape, with world +Y running horizontally.
                eye=(0.10, 0.02, 1.75),
                look_at=(0.10, 0.02, 0.80),
                up=(1.0, 0.0, 0.0),
                near=0.01,
                far=3.0,
                channels=("rgb", "depth"),
            ),
            CameraSpec(
                name="eye_in_hand",
                width=320,
                height=240,
                # Match the YAM wrist camera's perspective.
                fov_y=60.0,
                parent_frame="work",
                xform=PANDA_WRIST_MOUNT,
                near=0.005,
                far=1.5,
                channels=("rgb", "depth"),
            ),
        ),
    ),
}
"""Oblique context, overhead clutter geometry, and a wrist close-up -- twice.

The latter two are the minimum perception hardware needed by the non-privileged
sorting/packing graphs; GT graphs remain compatible and simply ignore them.

``agentview@v1`` is the syringe rig, named by ``vab_industrial@v1``. Same
``agentview`` and wrist camera; its overhead sits 0.30 m lower, near-vertical
and rolled the other way, because what it has to resolve is a hole in a lid
rather than two bins side by side. The sorting rig's overhead is higher and
rolled so the two long bins fill the landscape frame.
"""
