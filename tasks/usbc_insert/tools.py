# SPDX-License-Identifier: Apache-2.0
"""The USB-C suite's tools, and the boundary that runs through the middle of them.

**Two prefixes, and the prefix *is* the boundary.** The rule is the one
:class:`~robosimstudio_gap.connector.NonPrivilegedRegistry` states:
a ``sim.`` name is withheld under ``--no-privileged``, anything else is not. So
what goes on which side is decided by one question -- *could a real cell answer
this?* -- and not by which file the code happens to live in.

``sim.*`` -- reads the simulator's own state
    **``sim.get_insertion_state``** is the task's own measurement: depth past
    the mouth, lateral error off the bore axis, tilt, the mate gap still to go,
    plus the assembly constants a correction needs and the gates the *document*
    states. It is what a verify reads and what a correction computes its offset
    from -- the reference's open-loop script seats 3.6 mm of 13 because the
    grasp shifts the plug about a millimetre in the jaws and nothing
    re-measures; a graph that reads the true pose after grasping can.

    **``sim.perceive_mate``** is the oracle body of a perception slot: where the
    plug is, where the bore's mouth and axis are, where a free receptacle's
    barrel is, and whether the receptacle is free at all. Every number in it is
    something a camera could earn, which is the point -- ``graphs/mate`` carries
    a vision body returning the same keys, and the pair is what makes swapping
    them a measurement rather than a rewrite.

    **``sim.plan_insert``** and **``sim.plan_brace``** are the schedule and the
    support approach solved against the *authored assembly*: they know where the
    parts are without looking. They are the frame-free forms of the two planners
    below, and they exist because ``graphs/gt`` calls them by these names.

``usbc.*`` -- commands the hand, or solves IK over geometry the caller supplies
    **``usbc.plan_mate``** and **``usbc.plan_hold``** are the same two planners
    with the frame passed *in*: a grasp point, a mouth, an axis, a yaw. Nothing
    in them is ground truth -- they are questions about the robot and about a
    world the caller describes -- so they survive ``--no-privileged``, which is
    what lets a perception graph plan from what it saw. Blanket-tagging these
    would make ``--no-privileged`` measure "can this graph run without a
    planner", which is a different and much less interesting question, answered
    *no* by every graph in the corpus.

    **``usbc.set_gripper``** is ``sim.set_gripper`` **under a second name**. A
    grip here is a derived partial command -- pinch to 0.4 mm of clearance,
    close to 2 mm of interference, ease to 0.6 mm for the stroke -- because
    ``robot.close_gripper`` closes *fully* and driving the pads to zero on a
    10 mm part ejects it; the widths come from the hand's own measured jaw fit,
    which is calibration and not ground truth. The ``sim.`` spelling stays
    because ``graphs/gt`` calls it, and renaming it there would change what
    ``--no-privileged`` withholds from a published result in order to make a
    point the alias already makes.

    There is deliberately **no** ``usbc.wait_steps``. The branch this came from
    carried one, and ``robot.wait_steps`` is that tool: same contract, same
    ``{"waited": n}``, already unprivileged, and it says why in its own
    docstring -- waiting on the clock names nothing. ``graphs/mate`` calls it
    with an explicit ``timeout_s`` because the derived budget is half a second
    per step and this scene renders five cameras.

    **``usbc.grip_state``** is the grasp signal a real linear gripper has:
    commanded finger position against measured, and the pad gap the measured one
    implies. It names no object, which is exactly the difference between it and
    ``sim.get_insertion_state``'s ``held`` -- that one asks "are the jaws stalled
    at *the plug's* width", and only a scene can answer it.


Joint space throughout, for the tape suite's reason: the runtime's own
``robot.go_to_pose`` solves position plus an approach axis with the wrist roll
*free*, and a keyed connector cannot afford a free roll -- the jaws would close
across the wrong dimension of the part. Every waypoint here is solved with the
arm's full-pose LM solver, warm-started along the path exactly as the live
tracking would be.
"""

from __future__ import annotations

from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.arms import arm_id_of
from robosimstudio.core.math import look_at_matrix, mat_to_quat, quat_slerp, quat_to_mat, transform_matrix
from robosimstudio.robots.embodiments import top_down_grasp_site_quat
from robosimstudio.solvers import grasp_quat
from robosimstudio.solvers.motion.path import stream_hz

from .assets.parts import DEPTH_GATE, LATERAL_GUARD, measure_insertion
from .assets.stage import BULK, INSERT_AXIS, SCALE, bulkhead_barrel_grasp, reference_assembly

__all__ = ["ToolsUsbc", "register_usbc_tools"]

# The stroke's three shape constants. They lived in `routine.py` and moved here
# when the scripted routine it drove was dropped: they describe the
# approach a *tool* plans, not a policy that no longer ships, and a module kept
# alive to hold three floats for its importer is a module with no owner.
GRASP_YAW_DEG = 180.0
"""Hand yaw over the plug [deg]. The reference's own: the jaws close across the
plug's 10 mm axis, and the other 90 degrees closes them across its 22 mm one."""

FULL_FINGER_GRASP_OFFSET = -0.032
"""Grasp the rear of the plug body, 32 mm behind its shoulder [m].

The rendered fingers retain their brackets even on the blade-only contact rig.
Both rigs therefore need the same clearance, rather than choosing the grasp
from the collision approximation.
"""

GRASP_TILT_DEG = 20.0
"""Lean the wrist away from the socket while keeping the plug horizontal."""

FINGER_PORT_CLEARANCE = 0.003
"""Minimum planned separation of the full hand from the socket mouth plane [m]."""

FINGER_BENCH_CLEARANCE = 0.001
"""Keep the tilted fingers above the measured pickup surface [m]."""

APPROACH_H = 0.06
"""How far above the part the stroke stages before descending [m]."""

TRANSPORT_TIME_SCALE = 1.0
"""Duration multiplier for held-plug transport stages."""

RELEASE_LIFT = 0.10
"""How far the hand rises off a released part before it retreats [m].

Sized for the goal's release clause, ``(tcp-distance plug) > 0.05``, against
two facts: every stroke here settles short of its command by up to a
centimetre, and the achieved height varies by more episode to episode. 0.02
(as shipped) measured 27.8 mm of distance -- a clause failure no depth could
buy back; 0.06 measured 42-57 mm, straddling the gate, and the tape shows
success latching the very step the rising hand crosses the 50 mm sphere;
0.075 still left two of five episodes under it. The mate is latched by the
detent before the hand ever opens, so height is free margin and it is taken
in full: 100 mm commands the worst measured shortfall to ~40 mm past the
clause."""

SEAT_PRELOAD = 0.010
"""How far past the nominal mated station the push carries [m].

**A position servo does not arrive.** ``move_stage`` records the same fact from
the other end -- every stroke here settles short of its command, and the tighter
settle tolerance it ships exists because 0.01 rad of joint error is ~13 mm at
the tool centre on the loaded arm. The preload is what pays for that: the push
is commanded past the seat so the arm's shortfall lands on it.

Raised from 3 mm, and the 3 was measured on the *bulkhead* for a different
mechanism -- the support brace deflects and the released mate springs 2.7 mm
back along the stroke. The bolted variant has no brace to deflect and was still
landing short, which only became visible once the layout was drawn per episode
and the wrist stopped over-reading the mouth by 3.66 mm. That bias had been
pushing the plug deeper and quietly covering this.

Swept on three drawn layouts that were failing at 0.90 / 0.00 / 0.86:

============  ==================
``PRELOAD``   reward
============  ==================
3 mm          0.82 / 0.00 / 0.79
6 mm          0.89 / 0.00 / 0.87
**10 mm**     0.87 / 0.00 / **1.00**
14 mm         0.88 / 0.28 / 0.70
============  ==================

**And why not more**, which is the more useful half: it turns over. At 14 mm the
one layout that had seated falls back to 0.70. Over-travel does not push a plug
deeper into a bore it has bottomed in -- it cocks the plug in the slot and
stores recoil, which is the same saturation the receptacle friction shows.

Three episodes is a thin sweep and this number should be re-taken if the arm
gains or loses stiffness. What it is not is a fit to the answer: the direction
is predicted by the servo shortfall, and the turnover at 14 is the mechanism
disagreeing with more.
"""

INSERT_SPEED = 0.0112
"""How fast the insert stage is driven [m/s].

The reference's own rate, restated as a rate. It shipped as a *duration* --
3.4 s for the whole stroke, scaled by the fraction of it being driven -- which
is the same thing for as long as every push starts at the mate line. It is not
the same thing once a servo push starts BEHIND the hand: a 39 mm withdraw-and-
re-enter against a 9 mm seat asked for 14.7 s, four times the speed it was
measured at. 38 mm over 3.4 s is 11.2 mm/s; a stage now takes as long as its own
length says it should.
"""

HELD_PRELOAD = 0.003
"""The same overshoot, for a receptacle the other arm is HOLDING [m].

:data:`SEAT_PRELOAD`'s sweep was taken on the bolted variant, and its own note
says what it is paying for: an arm that settles short of its command. That is a
property of the work arm and it applies here too. What does not apply is the
assumption underneath the *size* -- that whatever the push over-travels into is
rigid.

It is not. Measured from `obj_bulkhead` over a recorded episode, the braced
receptacle **turns 3.7-5.4 degrees and its mouth travels 2.9-4.4 mm under the
push**, because a pinch on a smooth barrel reacts an insertion as a moment about
the grip. Ten millimetres of commanded over-travel against that does not seat a
plug deeper; it walks the pair across the bench and stores recoil, which the
release then pays out -- and `SEAT_PRELOAD`'s own sweep saw the same turnover on
the bolted fit at 14 mm.

The evidence that more is worse here is the servo sweep. Two servo iterations
against one, same three layouts, same brace: **3.22 / 9.67 / 11.06** against
**6.96 / 10.17 / 9.94**. A second measured push made two of three shallower,
and between the two looks of the one that lost most the remaining travel went
6.0 mm to 5.8 -- a full extra push that moved the receptacle and not the mate.

3 mm is the number the bulkhead was measured at before the bolted variant's
sweep overwrote it for both.
"""

INSPECT_PITCH_DEG = 20.0
UNLOOK_BACK = 0.040
UNLOOK_LIFT = 0.030
INSPECT_RISE = 0.100
INSPECT_BACK = 0.100
"""Clearance geometry for the empty-hand receptacle inspection path [m]."""

SIDE_APPROACH = (0.0, -0.7071067811865476, -0.7071067811865476)
"""Where the support hand points to take the bulkhead: in along the barrel from
behind, pitched 45 degrees down. The reference's own, and its reasoning is worth
keeping because both alternatives were tried and measured.

The jaws close **across world x** either way -- flat on the barrel's sides,
which is what a side grasp of a cylinder means. What the 45 degrees buys is
everything else. Straight down parks the support wrist at (0.44, 0.045, 0.92)
and the work wrist at (0.44, 0.00, 0.92) during the insert: 45 mm apart, and
they foul. Straight along the axis clears the work arm but drives the hand
through the rear cable, which lies along that corridor at bore height -- an 8 mm
z-miss on the grasp. The diagonal puts the wrist 126 mm from the work wrist,
above the cable, and lands the support TCP on the grasp point to 0.0 mm.
"""

BULKHEAD_ASSET = "usb_c_bulkhead"
"""The receptacle `usbc.receptacle_shape` carries dimensions for.

Named rather than assumed: that tool states one part's barrel and flange radii,
and a suite that grew a third receptacle would need its own entry rather than
these numbers under a different name.
"""

BRACE_STANDOFF = 0.10
"""How far back along the approach the hand waits before closing in [m]."""

BRACE_LIFT = 0.030
"""Lift the held socket clear of the bench before aligning a pitched plug [m]."""

BRACE_GAP = 0.030
"""Commanded support-jaw gap [m]: how hard the support hand holds the barrel.

The command is the grip *force*, indirectly: the fingers stall on a rigid
42 mm part and the servo lag against this number is what presses. It shipped at
0.036 -- the reference's own 6 mm of interference -- and 6 mm is not enough to
react an insertion.

**Measured, on a recorded episode, from `obj_bulkhead` itself.** Through the
push the braced receptacle **turns 3.3-3.8 degrees and its mouth travels
3.2-4.2 mm**, springing back to 1.3 deg / 0.9 mm once everything lets go. That
is a friction pinch on a smooth cylinder reacting a moment of F x 34.5 mm
(`BARREL_BACK`, and there is nowhere better to grip: in front of that band sit
the 60 mm flange and the nose). The consequence is not a small error, it is a
moving target -- the re-look measures the bore where it is *deflected to*, plans
along that axis, and the pair springs back through the mate on release.

12 mm of interference doubles the normal force the pads stall at, and on three
drawn layouts it is worth 6.96 / 10.17 / 9.94 mm of depth against 3.38 / 5.39 /
11.12 at six -- the best row of the sweep in the `mate` README.

**And it does not do it by holding the part still**, which is worth recording
because it refutes the reason it was tried. Splitting the squeeze from the push
on a recorded episode, twelve millimetres of interference deflects MORE under
load than six: 5.37 deg / 4.44 mm against 3.70 / 2.91, and leaves 3.49 deg at
rest against 0.74. Both halves are measured and they do not fit one story, so
the number is kept on the outcome rather than on the explanation."""

LOOK_BACK = 0.035
LOOK_RISE = 0.120
"""Where the work hand parks to take a look at the receptacle [m].

How far short of the mate the hand stands, and how far above the bore line it
rises, for the ``look`` stage. Both are chosen against the eye-in-hand camera's
own geometry rather than against the arm's: on this hand the lens sits 70 mm to
the side of the mount and aims at a point 70 mm *below* the tool centre, so the
receptacle has to be put under the camera rather than in front of the hand.

**Measured on the shipped rig**, by projecting both targets into the resolved
wrist camera over a recorded ``port`` episode (frame 321 of 550, the look
station):

===============  ==============  =========  ==========
Target           pixel (640x480) range      resolution
===============  ==============  =========  ==========
plug, in jaws    (331, 240)      172 mm     0.33 mm/px
bore mouth       (439, 236)      277 mm     0.54 mm/px
===============  ==============  =========  ==========

The held plug lands within eleven pixels of frame centre and the mouth a fifth
of the frame to its side, which leaves room for a receptacle the brace has
moved. Standing further back walks the mouth toward the frame edge; standing
lower puts the gripper body between the lens and it.

**And the plug reading is the finest in the rig**, which was not the expectation
going in. The placed ``topdown`` resolves the bench at 0.39 mm/px and this view
resolves the *receptacle* at 0.54 -- coarser, as a hand camera standing off a
part it is holding has to be. But the plug is 172 mm away rather than 400, and
0.33 mm/px on the part in the jaws is exactly where the resolution is worth
having: the in-jaw slip is the correction that decides a 0.15 mm-a-side fit.

The second thing it buys is a viewpoint. After the brace the support arm stands
over the receptacle and directly in the fixed top-down's line of sight; the
wrist comes in from the work side, where nothing is.
"""

BARREL_BACK = float((0.5 * (BULK.band_x0 + BULK.step_x1) - BULK.mouth_x) * SCALE)
"""How far behind the mouth a side grasp takes the barrel, along the bore axis [m].

The same band midpoint :func:`~tasks/usbc_insert/assets/stage.py::bulkhead_barrel_grasp`
computes, stated as a **distance from the mouth** rather than as a world point.
That is what lets ``perceive_mate`` report a grasp station for a receptacle that
has been picked up and is no longer where it spawned: the mouth is measured, the
axis is measured, and this is a fact about the part's shape that travels with it.

34.5 mm on this mesh, which is the middle of the smooth ``band_x0..step_x1``
cylinder -- so the pads land on the barrel rather than across the flange rim at
one end or the boot's step at the other.
"""


def register_usbc_tools(registry, env, bus) -> None:
    """Register the suite's tools, privileged and otherwise.

    **Tagged per tool, not per registrar.** Every entry here used to be tagged
    ``privileged`` wholesale, which was true while every entry was a ``sim.``
    read and became false the moment the suite grew tools that only command the
    hand or solve IK over geometry the caller supplies. A blanket tag makes
    ``--no-privileged`` withhold those too, and what that measures is not "can
    this graph run without ground truth" but "can it run without a planner" --
    a different and much less interesting question, answered ``no`` by every
    graph in the corpus.

    The rule is the one :class:`~robosimstudio_gap.connector.NonPrivilegedRegistry`
    already states: the ``sim.`` prefix *is* the boundary. Tagging matches it
    rather than overriding it.

    Written as one conditional **inside** the call rather than as a ``tags``
    local above it, so the tag a reader sees is the tag that is passed.
    ``tests/test_gap_privilege.py`` reads exactly that: it scans the 400
    characters after every ``register_callable(`` in a suite and wants either a
    literal ``"sim.`` name or a literal ``tags=("privileged"`` there, on the
    grounds that a privileged tool registered under some other name and without
    the tag would slip through both of the filter's own conditions.
    """
    tools = ToolsUsbc(env, bus)
    for name, fn, summary in tools.entries():
        registry.register_callable(
            name,
            fn,
            summary=summary,
            tags=("privileged",) if str(name).startswith("sim.") else ("usbc",),
        )


class ToolsUsbc:
    """The suite's privileged tool implementations."""

    def __init__(self, env, bus) -> None:
        self.env = env
        self.bus = bus
        self.asm = reference_assembly()
        self._pair = 0

    def entries(self) -> list[tuple[str, Any, str]]:
        return [
            # -------------------------------------------- reads the simulator
            (
                "sim.get_insertion_state",
                self.get_insertion_state,
                "Plug depth/lat/tilt/mate_gap against the bore, plus the assembly frame a correction needs.",
            ),
            (
                "sim.perceive_mate",
                self.perceive_mate,
                "Oracle perception: the plug's grasp station and heading, the bore's mouth and "
                "axis, a free receptacle's barrel, and whether it is free.",
            ),
            (
                "sim.plan_insert",
                self.plan_insert,
                "The scripted schedule as joint-space stage trajectories against the AUTHORED "
                "assembly, optionally laterally corrected.",
            ),
            (
                "sim.plan_brace",
                self.plan_brace,
                "The support arm's approach onto the AUTHORED barrel position, as joint-space stages.",
            ),
            # ------------------------- commands the hand, or solves given IK
            (
                "usbc.plan_mate",
                self.plan_mate,
                "The same schedule over a frame the CALLER supplies: plug point, bore mouth, bore axis, hand yaw.",
            ),
            (
                "usbc.plan_hold",
                self.plan_hold,
                "The same support approach onto a grasp point the CALLER supplies.",
            ),
            ("sim.set_gripper", self.set_gripper, "Command a partial jaw width, ramped."),
            (
                # THE SAME CALLABLE UNDER AN UNPRIVILEGED NAME, and the pair is
                # the point -- `cable_route` carries the identical pair for the
                # identical reason. `--no-privileged` drops by the `sim.`
                # prefix, which is the right rule for a tool that reads the
                # world and the wrong one for a tool that only *commands the
                # hand*: a jaw width is a thing every real gripper is told, and
                # the generic `robot.close_gripper` cannot carry it -- it
                # commands a full close, which on a 10 mm plug is how the part
                # gets squirted out of the jaws.
                #
                # `sim.set_gripper` stays because `graphs/gt` calls it by that
                # name and that graph is the measured baseline. Renaming it
                # there would change what `--no-privileged` withholds from a
                # published result, to make a point the alias already makes.
                "usbc.set_gripper",
                self.set_gripper,
                "Command a partial jaw width: 'open' | 'pinch' | 'close' | 'insert', or metres. "
                "Unprivileged: a commanded width is an instruction to the hand, not a reading "
                "of the scene.",
            ),
            (
                "usbc.grip_state",
                self.grip_state,
                "One hand's commanded and measured finger position, the pad gap, and whether "
                "the fingers are stalled short of their command.",
            ),
            ("sim.wait_steps", self.wait_steps, "Dwell for N control steps while the policy loop holds."),
            (
                "sim.taught_receptacle",
                self.taught_receptacle,
                "Privileged: the authored pose of a bolted receptacle. Refuses free bodies.",
            ),
            (
                "usbc.plug_shape",
                self.plug_shape,
                "The plug's known local surface mesh, with its origin at the shoulder. Contains no scene pose.",
            ),
            (
                "usbc.receptacle_shape",
                self.receptacle_shape,
                "The receptacle's CATALOGUE dimensions -- barrel and flange radii, where the "
                "flange band sits behind the mouth, the bore's depth and local surface mesh. Unprivileged: a datasheet "
                "is not a reading of the scene.",
            ),
            (
                "usbc.next_pair",
                self.next_pair,
                "Which declared connector to mate next, or that the mating loop is complete.",
            ),
            (
                "usbc.instruction",
                self.instruction,
                "The command this episode was given, as language. Unprivileged: the task "
                "statement is what a cell is told, not what it can see.",
            ),
        ]

    # ------------------------------------------------------------ calibration
    def taught_receptacle(self) -> dict[str, Any]:
        """Return the authored fixture pose for oracle diagnostics only.

        This reads a simulator handle, including episode placement. It is
        registered under sim.* so camera policies cannot access it.
        """

        def read() -> dict[str, Any]:
            from robosimstudio.core.math import quat_to_mat  # noqa: PLC0415

            ctx = self.env.make_context()
            name, payload = self._receptacle()
            if payload is None:
                raise RuntimeError(f"receptacle {name!r} declares no cavity to teach")
            handle = ctx.handles.entity(name)
            if handle.body >= 0:
                raise RuntimeError(
                    f"receptacle {name!r} is a free body -- its pose is this episode's, not a "
                    "taught constant. Perceive it."
                )
            pose = np.asarray(handle.xform, dtype=np.float64)
            rot = quat_to_mat(pose[3:7])
            mouth = pose[:3] + rot @ np.asarray(payload["mouth"], dtype=np.float64)
            axis = rot @ np.asarray(payload["axis"], dtype=np.float64)
            axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
            return {
                "receptacle": str(name),
                "mouth_point": [float(v) for v in mouth],
                "bore_axis": [float(v) for v in axis],
                "taught": True,
            }

        return self.bus.call(read)

    def plug_shape(self) -> dict[str, Any]:
        """Local catalogue geometry for an RGB-D shoulder/attitude fit."""
        from .assets.stage import _load_stl  # noqa: PLC0415

        vertices, triangles = _load_stl("USB_C_Plug.stl")
        return {"surface_vertices": vertices * SCALE, "surface_triangles": triangles}

    def receptacle_shape(self) -> dict[str, Any]:
        """How big the receptacle is, and nothing about where it is.

        **A datasheet, not a measurement.** Every number here is a dimension of
        the connector the task names -- the barrel's radius, the mounting
        flange's, the band the flange occupies behind the mouth face, the bore's
        depth. A cell mating a USB-C bulkhead knows which bulkhead it is mating,
        the same way it knows the plug is a USB-C plug; what it does not know,
        and what this deliberately will not say, is where the part is.

        Why a perception body needs it. Looking DOWN at a horizontal bore, a
        camera sees the barrel's upper surface and never its axis -- and the
        mouth is on the axis. The height is then ``top - radius``, and the
        radius has to come from somewhere. Measuring it off the same silhouette
        is the self-calibrating version and it is worse: a cylinder's silhouette
        is its tangent lines seen in perspective, and on this rig that reads
        19.0 mm for a 21.0 mm barrel -- a 2 mm bias straight into the height. So
        the *shape* is looked up and the *pose* is measured, which is the split
        that survives the part moving.

        The same asset the goal is scored against, so a receptacle of another
        size needs no new number anywhere.
        """

        def read() -> dict[str, Any]:
            from robosimstudio.assetlib.registry import ASSETS  # noqa: PLC0415

            name, payload = self._receptacle()
            if payload is None:
                raise RuntimeError(f"receptacle {name!r} declares no cavity; there is no shape to report")
            entry = next((e for e in getattr(self.env.doc, "objects", ()) or () if str(e.name) == name), None)
            kind = str(getattr(entry, "type", "")) if entry is not None else ""
            if kind != BULKHEAD_ASSET:
                # REFUSES rather than substituting. The numbers below are one
                # part's -- the panel-mount barrel's -- and handing them back for
                # a receptacle of another shape would be a datasheet for the
                # wrong component, which a body would use without noticing. The
                # block port is bolted down and taught, so nothing asks.
                raise RuntimeError(
                    f"receptacle {name!r} is a {kind!r}; usbc.receptacle_shape carries the "
                    f"{BULKHEAD_ASSET!r} barrel's dimensions and has none for it"
                )
            record = ASSETS.get(kind)
            # Catalogue geometry only: these vertices contain no scene pose.
            # A wrist can fit this surface even when the braced barrel pitches
            # and its aperture is hidden behind the held plug.
            from .assets.stage import _load_stl  # noqa: PLC0415

            vertices, triangles = _load_stl("Bulkhead_Fitted.stl")
            return {
                "receptacle": str(name),
                "summary": str(getattr(record, "summary", "")),
                # The mesh's own reference constants, scaled to the world. Stated
                # from BULK rather than re-sectioned off the STL: `load_bulkhead`
                # already asserts the mesh against these, so a part that did not
                # match them would have refused to build.
                "barrel_radius": float(BULK.barrel_r * SCALE),
                "flange_radius": float(BULK.flange_r * SCALE),
                # Where the flange disc sits BEHIND the mouth face, which is what
                # a body that has just found the face needs in order to know
                # which slice of its own cloud is the flange.
                "flange_from_mouth": (
                    float((BULK.flange_x0 - BULK.mouth_x) * SCALE),
                    float((BULK.flange_x1 - BULK.mouth_x) * SCALE),
                ),
                "bore_depth": float(payload["depth"]),
                "surface_vertices": vertices * SCALE,
                "surface_triangles": triangles,
                "mouth_local": [float(BULK.mouth_x * SCALE), 0.0, 0.0],
            }

        return self.bus.call(read)

    # -------------------------------------------------------------- language
    def instruction(self) -> dict[str, Any]:
        """The episode's own command, as language.

        **A workaround, and it should not survive.** The facade publishes the
        instruction beside the camera blocks -- ``contract.ALLOWED_KEYS`` lists
        it by name as one of the four things that may cross the privilege
        boundary, "the task and the clock" -- and the pinned runtime's
        ``Connector._build_observation`` then assembles a typed ``Observation``
        out of ``cameras`` and ``arms`` and **drops it**. So a body that reads
        ``robot.get_observation()["instruction"]`` gets the empty string, always,
        and silently.

        That is not this suite's bug and it is not only this suite's problem:
        ``cable_route``'s weave body reads the same key the same way for its own
        language-derived flag, and gets the same empty string.

        The right home is ``robot.*``, and it is there now:
        :meth:`~robosimstudio_gap.robot_tools.RobotTools.instruction` registers
        ``robot.instruction`` with the same one-line body, for every suite. This
        entry stays only because the ``mate`` graph's two grounding bodies name
        ``usbc.instruction``; a new body should call the shared tool, and this
        one goes when those two are repointed.

        Not privileged. A cell is *told* what to do; being told is not seeing.
        """
        return {"instruction": str(self.env.episode.instruction or "")}

    # ------------------------------------------------------------------- wait
    def wait_steps(self, steps: int = 30) -> dict[str, Any]:
        """Block the graph until ``steps`` control steps have elapsed.

        The main loop keeps stepping hold actions while a graph node runs, so
        waiting is watching the published step counter rather than sleeping
        wall-clock -- the sim does not run at wall speed. What a grip needs
        after its ramp for the part to settle in the pads.
        """
        return {"waited": self._wait_for_step(int(steps), timeout_s=60.0)}

    # ------------------------------------------------------------------ state

    def _receptacles(self) -> list[tuple[str, dict]]:
        """Declared receptacles in document order, including fixed fixtures."""
        from robosimstudio.assetlib.registry import ASSETS  # noqa: PLC0415

        doc = self.env.doc
        entries = (*getattr(doc, "objects", ()), *getattr(doc, "fixtures", ())) if doc is not None else ()
        out: list[tuple[str, dict]] = []
        for entry in entries:
            try:
                record = ASSETS.get(entry.type)
            except KeyError:
                continue
            cavities = record.features_of_kind("cavity")
            if cavities:
                out.append((str(entry.name), dict(cavities[0].payload)))
        return out

    def next_pair(self, advance: bool = True) -> dict[str, Any]:
        """Advance the graph's per-episode connector loop without scene reads."""
        pairs = self._receptacles()
        index = int(self._pair)
        total = len(pairs)
        if total <= 0 or index >= total:
            return {
                "route": "finished",
                "index": index,
                "remaining": 0,
                "total": total,
                "receptacle": "",
                "done": [name for name, _ in pairs[:index]],
            }
        if advance:
            self._pair = index + 1
        return {
            "route": "found",
            "index": index,
            "remaining": max(0, total - index - 1),
            "total": total,
            "receptacle": pairs[index][0],
            "done": [name for name, _ in pairs[:index]],
        }

    def _receptacle(self) -> tuple[str, dict | None]:
        """Which declared thing this task mates into, and its own bore payload.

        Not a ``"port"`` literal, which is what it was. That name was true of
        the bolted-down variant and false the moment the receptacle became a
        carried bulkhead -- and it failed as a ``KeyError`` inside a graph node,
        two subgraphs deep, on a scene whose entities were perfectly fine.

        The document is the authority: every thing it declares carries a type,
        and the one whose asset declares a ``bore`` cavity is the receptacle. A
        third variant with a fourth name needs nothing here.

        The payload rides along because mouth and axis are stated in the
        receptacle's **own** frame: measuring the bulkhead with the port's
        payload composed the bore 90 degrees off its slot, and the state this
        tool fed the graph read +13 mm of depth with the plug still 46 mm out.
        """
        pairs = self._receptacles()
        if pairs:
            # next_pair advances before the pass begins.
            return pairs[min(max(0, self._pair - 1), len(pairs) - 1)]
        return "port", None

    def _gates(self) -> tuple[float, float, float]:
        """Depth, lateral and tilt gates, preferring the ones the *task* states.

        The asset's are the honest default -- 80% of the bore and the cavity's
        own half-diagonal are facts about this connector, not about the job -- and
        for as long as a document restates them, either answer is right. But a
        document is entitled to a stricter line, and a graph reading the asset's
        constant would go on reporting ``inserted: true`` while the goal said
        false, with nothing to show which of the two was being believed.

        That stopped being hypothetical when the bulkhead document added a tilt
        clause: the graph went on routing ``seated`` on depth and lat alone while
        the evaluator scored the episode 0.00, which is the exact split this
        method exists to prevent. **No tilt default**, though -- an unstated tilt
        gate is unbounded rather than the asset's, because a document that says
        nothing about tilt is not asking for one, and inventing a threshold here
        would fail the bolted ``port`` variant against a line it never stated.

        Not hypothetical even now, only harmless: the document writes its lateral
        guard to ten significant figures where the asset computes it to full
        precision, so the two have differed by 2.5e-11 all along.
        """
        from robosimstudio_bench.goals.goal import stated_gate  # noqa: PLC0415

        doc = self.env.doc
        depth = stated_gate(doc.goal, "insertion") if doc is not None else None
        lateral = stated_gate(doc.goal, "insertion:lat") if doc is not None else None
        tilt = stated_gate(doc.goal, "insertion:tilt") if doc is not None else None
        return (
            DEPTH_GATE if depth is None else float(depth),
            LATERAL_GUARD if lateral is None else float(lateral),
            float("inf") if tilt is None else float(tilt),
        )

    def get_insertion_state(self) -> dict[str, Any]:
        """Where the plug is, against where the task wants it."""
        depth_gate, lateral_guard, tilt_gate = self._gates()

        port, bore_payload = self._receptacle()

        def read() -> dict[str, Any]:
            from robosimstudio.domains.rigid.measures.jaws import jaw_state  # noqa: PLC0415

            ctx = self.env.make_context()
            m = measure_insertion(ctx, port=port, payload=bore_payload)
            held, jaw_gap = jaw_state(ctx, "plug")
            pose = ctx.pose("plug")
            return {
                # The insertion atom's four readings, spelled out rather than
                # spread. The keys are the atom's declared ``returns`` and the
                # atom always carries all four; a ``**{...}`` here made the
                # tool's output contract statically unreadable, so the frozen
                # surface lock shipped ``returns: []`` for the one tool this
                # suite's verify and correct nodes live on.
                "depth": float(m["depth"]),
                "lat": float(m["lat"]),
                "mate_gap": float(m["mate_gap"]),
                "tilt": float(m["tilt"]),
                "plug_position": [float(v) for v in pose[:3]],
                "plug_quat_xyzw": [float(v) for v in pose[3:7]],
                "mated_position": [float(v) for v in self.asm.plug_mated],
                "mouth_y": float(self.asm.mouth_y),
                "insert_axis": [float(v) for v in INSERT_AXIS],
                # Where the plug rides relative to the tool centre when the jaws
                # close at the nominal station. A node correcting an in-jaw slip
                # needs it, and used to restate it as a module constant because a
                # node script has no package to import from -- which made a fact
                # about *this suite's* grasp station into a number that would be
                # silently wrong the day the station or the hand moved.
                "grasp_offset": float(FULL_FINGER_GRASP_OFFSET),
                "bore_depth": float(self.asm.port.depth),
                "depth_gate": depth_gate,
                "lateral_guard": lateral_guard,
                "tilt_gate": tilt_gate,
                "held": bool(held),
                "jaw_gap": float(jaw_gap),
                # Tilt too, when the document states one. Depth is an axial
                # PROJECTION: a plug pivoting in an oversized bore reads as a
                # plug that stayed in, so a graph routing `seated` on depth and
                # lat alone calls a mate that hangs off the mouth a success --
                # which is what it did while the evaluator scored 0.00.
                "inserted": bool(
                    m["lat"] <= lateral_guard and m["depth"] >= depth_gate and float(m.get("tilt", 0.0)) <= tilt_gate
                ),
            }

        return self.bus.call(read)

    # ------------------------------------------------------------------ brace
    def plan_hold(self, grasp_point: Any = None) -> dict[str, Any]:
        """The support arm's approach onto a grasp point the caller supplies.

        The non-privileged half of :meth:`plan_brace`, and the split is the
        whole point: everything about *how* to take a 42 mm barrel -- the side
        approach, the standoff, the two legs, the commanded gap -- is a fact
        about this hand and this part shape, and none of it requires knowing
        where the part is. That last thing is the argument.

        ``grasp_point`` is a world ``[x, y, z]``. Left ``None`` this falls back
        to the authored barrel position, which is what makes :meth:`plan_brace`
        one line -- but a caller reaching this name is expected to pass one,
        because reaching it under ``--no-privileged`` and getting the authored
        answer would be the boundary defeated by a default.
        """
        return self._hold(grasp_point)

    def plan_brace(self) -> dict[str, Any]:
        """The support arm's approach onto the bulkhead barrel, as a trajectory.

        **Why this exists rather than a generic grasp.** The cradle the
        receptacle rests in has no axial stop -- deliberately, because reacting
        the insertion push is the support arm's job and that is what makes this
        variant bimanual rather than scenery. So an insertion run without a
        brace does not fail by jamming; it fails by shoving a 90 g receptacle
        along its own axis until the stroke ends with the plug nowhere near
        seated, which reads exactly like a stroke that was too short.

        A **side** grasp across the barrel, not a top-down one. The barrel's
        axis is the mate line, so a top-down pinch closes across the axis the
        push runs along and the part rolls out of it; closing across x puts the
        pads either side of a 42 mm cylinder with the reaction in line with the
        jaws.

        Returns the two stages in execution order with dense joint
        trajectories, the grip command that pinches the barrel, and which
        ``arm_id`` the ``robot.*`` tools should drive.

        **Privileged only in where it aims.** The barrel's position comes from
        :func:`~tasks/usbc_insert/assets/stage.py::bulkhead_barrel_grasp`, which is the
        authored assembly -- so this tool knows where the receptacle is without
        looking. Everything else it does is :meth:`plan_hold`'s, and a graph
        that has *seen* the barrel calls that instead.
        """
        return self._hold(None)

    def _hold(self, grasp_point: Any) -> dict[str, Any]:
        """The support approach, aimed at ``grasp_point`` or at the authored one."""

        def build() -> dict[str, Any]:
            env = self.env
            name = self._arm_name(env.handles.support_arm)
            ik = env.arm(name)
            # A SIDE approach, tilted 45 degrees -- not a yawed top-down one.
            #
            # This was `top_down_grasp_site_quat(hand, yaw + 90)`, which is the
            # brace the reference tried first and replaced: straight down puts
            # the two wrists 45 mm apart during the insert and they foul. See
            # SIDE_APPROACH for the measurements on all three options.
            #
            # The top-down family was measured again on main and does not
            # recover (097f2cb): the helper's ``yaw_deg`` is the world heading
            # the jaws CLOSE along, so the shipped +90 drove the blades into
            # the flange disc and the boot cone, and yaw 180 -- the work arm's
            # own heading -- is a pose the support arm's LM solve parks 11.9 mm
            # out of while FD reaches it only by giving up 24.8 deg of wrist.
            # Yaw 0 solves to 0.00 mm / 0.00 deg and is the best top-down there
            # is; it still leaves the two wrists 45 mm apart in the insert,
            # which is the thing the diagonal exists to answer.
            #
            # Off the HANDLE, not the spec: these are the hand's *measured*
            # axes, resolved at bind, and the spec carries no such thing.
            arm = env.handles.arms[name]
            quat = grasp_quat(
                arm.approach_axis,
                arm.close_axis,
                arm.ee_offset_rot,
                SIDE_APPROACH,
                (1.0, 0.0, 0.0),
            )
            # The barrel's own centre, one flange back from the mouth so the
            # pads land on the smooth band rather than across the flange rim.
            # Supplied by the caller when it has measured one, and read off the
            # authored assembly when it has not.
            centre = (
                np.asarray(bulkhead_barrel_grasp(), dtype=np.float64)
                if grasp_point is None
                else np.asarray(_vec3(grasp_point), dtype=np.float64)
            )

            # (name, start height above the barrel, end height, duration [s]).
            # Two legs for the reason the insertion plan has two: a single solve
            # from home to a point beside a 42 mm cylinder is free to route the
            # wrist through it, and staging above first keeps the descent in one
            # branch.
            # Standoff measured BACK ALONG THE APPROACH, not straight up: a
            # hand that waits above a diagonal approach and then descends is
            # tracing two sides of a triangle through the space the work arm is
            # in.
            approach_vec = np.asarray(SIDE_APPROACH, dtype=np.float64)
            stages = [
                ("stage", -BRACE_STANDOFF * approach_vec, -BRACE_STANDOFF * approach_vec, 0.6),
                ("seat", -BRACE_STANDOFF * approach_vec, np.zeros(3), 0.8),
                ("lift", np.zeros(3), np.array([0.0, 0.0, BRACE_LIFT]), 1.2),
            ]
            out: list[dict[str, Any]] = []
            # Warm-started from the live posture, exactly as plan_insert is:
            # the same solver and iteration budget the live tracking uses, run
            # kinematically.
            q_ik = ik.joint_q_ik
            wp.copy(q_ik.reshape((ik.model.joint_coord_count,)), env.state.joint_q)
            stream_hz = self._stream_hz()
            for stage, offset0, offset1, duration in stages:
                waypoints = []
                steps = max(2, int(stream_hz * duration))
                for i in range(steps):
                    t = i / (steps - 1)
                    ik.set_target(centre + offset0 + t * (offset1 - offset0), quat)
                    ik.solver.step(q_ik, q_ik, iterations=ik.iterations)
                    q = q_ik.numpy()[0]
                    waypoints.append({"positions": [float(q[c]) for c in ik.arm_coords]})
                out.append({"stage": stage, "trajectory": {"waypoints": waypoints}, "duration": float(duration)})

            # The reference's own commanded gap: 6 mm under the 42 mm barrel, so
            # the fingers stall on the part and the servo lag is the grip force.
            width = BRACE_GAP
            return {
                "arm_id": arm_id_of(env, name),
                "arm_name": name,
                "stages": out,
                "grip": {"command_m": float(max(0.0, width / 2.0))},
                "grasp_point": [float(v) for v in centre],
            }

        return self.bus.call(build)

    # ------------------------------------------------------------------ roles
    def _arm_name(self, arm: str = "") -> str:
        """The named arm, or whichever one this pairing works with.

        Never a ``"right"`` literal: the two YAM cells list their arms in
        opposite orders, so a literal names a different hand on each robot and
        the binding is the only thing that knows which is meant.
        """
        return arm or self.env.handles.work_arm

    def _hand(self, arm: str = ""):
        """The hand on that arm, for the measured grasp axes.

        The axes are the gripper's, not the arm's -- so this is what a grasp
        frame is composed from, and swapping the hand moves it.
        """
        return self.env.scene.embodiment.hands.get(self._arm_name(arm))

    def _grip_open(self, arm: str = "") -> float:
        """This arm's finger command at full open [m], read off the model."""
        return self.env.handles.arms[self._arm_name(arm)].gripper_open

    def _hand_vertices(self, ik) -> np.ndarray:
        """Full finger/housing meshes and fitted blade boxes in the live TCP frame.

        Use the robot's geometry and joint state, including visual-only finger
        brackets. No receptacle pose or other scene state is read here.
        """
        model = self.env.model
        hand = self._hand()
        names = {hand.tcp_body, *(name for side in hand.finger_bodies for name in side)}
        bodies = {i for i, name in enumerate(model.body_label) if name.split("/")[-1] in names}
        body_q = self.env.state.body_q.numpy()
        transforms = model.shape_transform.numpy()
        scales = model.shape_scale.numpy()
        types = model.shape_type.numpy()
        shape_bodies = model.shape_body.numpy()
        tcp, quat = ik.ee_pose(self.env.state)
        tcp_rotation = quat_to_mat(quat)
        points = []
        for shape, body in enumerate(shape_bodies):
            mesh = model.shape_source[shape]
            if body not in bodies:
                continue
            if mesh is not None and hasattr(mesh, "vertices"):
                local = np.asarray(mesh.vertices) * scales[shape]
            elif int(types[shape]) == int(newton.GeoType.BOX):
                # The blade-only rig adds fitted boxes. Their corners can
                # extend beyond the mesh when the hand is tilted.
                local = (
                    np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]) * scales[shape]
                )
            else:
                continue
            xf = transforms[shape]
            local = local @ quat_to_mat(xf[3:7]).T + xf[:3]
            pose = body_q[body]
            world = local @ quat_to_mat(pose[3:7]).T + pose[:3]
            points.append((world - tcp) @ tcp_rotation)
        if not points:
            raise RuntimeError("USB-C insertion needs the full hand meshes to check socket clearance")
        return np.concatenate(points)

    # ------------------------------------------------------------------- rate
    def _stream_hz(self) -> float:
        """The rate a planned stage is actually executed at [Hz].

        **The cell's control rate, not** ``PLAN_HZ``. Both planners here
        return a ``duration`` alongside their waypoints, and nothing reads it:
        ``scripts/move_stage.py`` hands the stage to ``robot.execute_trajectory``,
        whose streaming path feeds
        :meth:`~robosimstudio_gap.facade.WorldFacade.stream_joint_trajectory`
        **one waypoint per control step**. So the rate a stage is densified at
        *is* the speed it runs at.

        Densified at ``PLAN_HZ``'s 30 and streamed at this cell's 60, every stage ran
        in half its stated time: the descent onto the plug at 120 mm/s where it
        is written for 60, the retract at 85.7 where it is written for 42.9, and
        the 83 mm insert stroke into a 0.15 mm-class fit at 48.8 mm/s where the
        3.4 s it declares means 24.4. The suite it was ported alongside had the
        same defect in the same shape (``cable_route.routine.plan_station_stages``,
        fixed to read this same field); the comment left there records the belief
        that spared this copy -- that the graph path "converges on each waypoint
        in turn rather than clocking them out" -- which is what
        ``_execute_trajectory`` does only on its *fallback* path, when the facade
        exposes no ``stream_joint_trajectory``. This one does.
        """
        return stream_hz(self.env.task.physics)

    # ------------------------------------------------------------------- grip
    def set_gripper(self, width: Any = "open", arm: str = "", ramp_steps: int = 30) -> dict[str, Any]:
        """Command the jaw pair to a named or metric width, ramped.

        ``width`` is one of the derived presets, or a finger command in metres
        (0 = shut, 0.0475 = open). The command is walked from wherever it is to
        the target over ``ramp_steps`` control steps -- a step input on a
        kp=20000 finger servo slams the pads, and a slammed pinch is how a
        10 mm part gets squirted out of the jaws (the reference ramps every
        grip over its keyframe interval for the same reason). The final
        command then persists through the policy loop's hold actions.
        """
        arm = self._arm_name(arm)

        def resolve() -> tuple[float, float]:
            rig = self.env.scene.by_name["rig"]
            if isinstance(width, str):
                commands = rig.grip_commands(2.0 * self.asm.plug.half_y)
                if width not in commands:
                    raise ValueError(f"unknown gripper preset {width!r}; have {sorted(commands)} or metres")
                cmd = commands[width]
            else:
                cmd = float(np.clip(float(width), 0.0, self._grip_open(arm)))
            gripper = self.env.gripper(arm)
            return float(cmd), float(gripper.command)

        def write_fraction(fraction: float) -> None:
            gripper = self.env.gripper(arm)
            gripper.command = float(fraction)
            gripper.apply(self.env.pipeline.control.joint_target_q)

        cmd, start_fraction = self.bus.call(resolve)
        target_fraction = 1.0 - cmd / self._grip_open(arm)
        steps = max(1, int(ramp_steps))
        for i in range(1, steps + 1):
            fraction = start_fraction + (target_fraction - start_fraction) * (i / steps)
            self.bus.call(write_fraction, fraction)
            self._wait_for_step(1)
        return {"arm": arm, "command_m": float(cmd), "fraction_closed": float(target_fraction)}

    def grip_state(self, arm: str = "") -> dict[str, Any]:
        """One hand's fingers: commanded, measured, the pad gap, and the stall.

        **The grasp signal that names nothing.** ``sim.get_insertion_state``
        reports ``held``, which asks whether the jaws are stalled *at the plug's
        own width* -- two facts, one of them about an object, and only a scene
        can supply the second. This reports the first alone: the fingers were
        commanded past whatever they are holding, so a measured position that
        trails the command means they are stopped on something.

        That is weaker and it is honest, and the weakness is bounded in the one
        direction that matters here. What it cannot tell you is *what* it is
        holding; what it can tell you is that an empty hand is not about to fly a
        3.4 s insertion stroke, because empty jaws reach their command exactly.
        The measured band separates them cleanly on this rig: the pads reach
        their target within servo noise on air and stall 2 mm short of it on a
        10 mm plug.

        ``STALL_MARGIN`` is imported rather than restated, from the same module
        the ``gripped`` atom reads it from -- servo tracking noise is a property
        of the instrument, and two copies of it would let the tool and the goal
        disagree about whether a hand was holding anything.
        """
        from robosimstudio.domains.rigid.measures.jaws import STALL_MARGIN  # noqa: PLC0415

        name = self._arm_name(arm)

        def read() -> dict[str, Any]:
            env = self.env
            handle = env.handles.arms[name]
            coords = list(handle.gripper_coords)
            # Through the evaluation context, which is where the commanded
            # targets are already a host array -- `pipeline.control.joint_target_q`
            # is a warp array and refuses item indexing, and reaching for it
            # directly is the one line of this tool that has ever been wrong.
            target = np.asarray(env.make_context().joint_target_q, dtype=np.float64)
            if not coords or not target.size:
                # Fails closed, as `_jaw_stall` does: a scene with no finger
                # coordinates is a scene this cannot measure, and raising here
                # would make an unrelated pairing uncompilable.
                return {
                    "arm": name,
                    "command_m": float("nan"),
                    "measured_m": float("nan"),
                    "gap_m": float("nan"),
                    "lag_m": float("nan"),
                    "stalled": False,
                }
            joint_q = env.state.joint_q.numpy()
            measured = float(np.mean([joint_q[c] for c in coords]))
            commanded = float(np.mean([target[c] for c in coords]))
            lag = measured - commanded
            return {
                "arm": name,
                "command_m": commanded,
                "measured_m": measured,
                # Pad separation, through the hand's own span-and-travel
                # derivation. A finger coordinate is not a gap and the mapping is
                # the gripper's, not this task's.
                "gap_m": float(handle.jaw_gap(measured)),
                "lag_m": lag,
                "stalled": bool(lag > STALL_MARGIN),
            }

        return self.bus.call(read)

    def _wait_for_step(self, steps: int, timeout_s: float = 10.0) -> int:
        """Watch the published step counter advance, off the owner thread."""
        import time  # noqa: PLC0415

        start = int(self.bus.snapshot.step)
        deadline = time.monotonic() + timeout_s
        while int(self.bus.snapshot.step) < start + steps and time.monotonic() < deadline:
            time.sleep(0.002)
        return int(self.bus.snapshot.step) - start

    # ------------------------------------------------------------------- plan
    def plan_mate(
        self,
        plug_point: Any = None,
        mouth_point: Any = None,
        bore_axis: Any = None,
        tcp_point: Any = None,
        yaw_deg: float = GRASP_YAW_DEG,
        correct_x: float = 0.0,
        correct_y: float = 0.0,
        correct_z: float = 0.0,
        retry_from: float = 0.0,
        axial_only: bool = False,
        receptacle_held: bool = False,
        with_inspect: bool = False,
        mate_quat: Any = None,
    ) -> dict[str, Any]:
        """The schedule, solved to joint space, over a frame the caller supplies.

        **The non-privileged planner.** Everything about the stroke's *shape* is
        a fact about this connector and this hand -- stage above, descend, close,
        rise to the bore line, push along the axis, retract -- and none of it
        requires knowing where anything is. The four arguments are what does, and
        a graph that measured them from a camera gets the same plan the oracle
        gets from the scene.

        Args:
            plug_point: The plug body's origin in the world [m]. The grasp
                station is this displaced ``FULL_FINGER_GRASP_OFFSET`` along the plug axis, which
                is where the jaws close behind the CoM and clear of the tab.
            mouth_point: The bore mouth's centre in the world [m].
            bore_axis: Unit direction *into* the bore, in the world.
            yaw_deg: The world heading the jaws close along over the plug. The
                reference's 180 closes them across the plug's 10 mm axis; 90
                degrees off closes them across its long one. A perception body
                derives this from the plug's measured heading.
            correct_x: Lateral correction applied to the lift/insert/retract
                stages [m] -- what a graph computes from the measured
                plug-in-jaw offset after grasping.
            correct_y: Axial correction [m]: a plug that slipped back in the
                jaws needs the stroke to run correspondingly further.
            correct_z: The lateral correction, vertically.
            retry_from: When positive, the ``insert`` stage becomes a back-off
                and re-push: from that far short of the mate to the mate [m],
                over a proportionally shorter duration. What a guarded retry
                wants after a partial seat -- withdrawing the whole 80 mm
                stroke to try again drags a wedged plug back through the very
                misalignment being corrected.

        Any of the first three left ``None`` falls back to the authored
        assembly's value for it, which is what makes :meth:`plan_insert` one
        line. A caller reaching *this* name is expected to pass all three: the
        fallback is a convenience for the privileged twin, not a default a
        perception graph should be taking.

        Returns the stages in execution order with dense joint trajectories,
        the grip presets as both metres and closed-fractions, and which
        ``arm_id`` the ``robot.*`` tools should drive.
        """
        return self._mate(
            plug_point=plug_point,
            mouth_point=mouth_point,
            bore_axis=bore_axis,
            tcp_point=tcp_point,
            yaw_deg=float(yaw_deg),
            correction=(float(correct_x), float(correct_y), float(correct_z)),
            retry_from=float(retry_from),
            axial_only=bool(axial_only),
            # A HELD receptacle gets a smaller overshoot: what the push
            # over-travels into is a friction pinch, not a bench. See
            # HELD_PRELOAD for what ten millimetres of it costs.
            preload=HELD_PRELOAD if receptacle_held else SEAT_PRELOAD,
            with_inspect=bool(with_inspect),
            mate_quat=mate_quat,
            clear_entry=True,
        )

    def plan_insert(
        self,
        correct_x: float = 0.0,
        correct_y: float = 0.0,
        correct_z: float = 0.0,
        retry_from: float = 0.0,
        axial_only: bool = False,
    ) -> dict[str, Any]:
        """The schedule, solved to joint space, against the AUTHORED assembly.

        :meth:`plan_mate` with the frame left unstated, which is exactly what
        makes it privileged: it knows where the plug and the bore are without
        looking. Kept under this name and this signature because ``graphs/gt``
        calls it, and that graph is the measured baseline.

        Its overshoot is :data:`SEAT_PRELOAD`'s 10 mm on both variants, which is
        the bolted sweep's number. :data:`HELD_PRELOAD`'s 3 mm exists and is
        reachable **only** through ``usbc.plan_mate(receptacle_held=True)`` --
        no shipped graph calls this method with a held receptacle, and adding a
        flag here to pick between them would change what ``graphs/gt`` runs on
        the bulkhead, which is the 4/4 baseline.

        See :meth:`plan_mate` for what the four corrections mean.
        """
        return self._mate(
            plug_point=None,
            mouth_point=None,
            bore_axis=None,
            tcp_point=None,
            yaw_deg=GRASP_YAW_DEG,
            correction=(float(correct_x), float(correct_y), float(correct_z)),
            retry_from=float(retry_from),
            axial_only=bool(axial_only),
        )

    def _mate(
        self,
        *,
        plug_point: Any,
        mouth_point: Any,
        bore_axis: Any,
        tcp_point: Any,
        yaw_deg: float,
        correction: tuple[float, float, float],
        retry_from: float,
        axial_only: bool = False,
        preload: float = SEAT_PRELOAD,
        with_inspect: bool = False,
        mate_quat: Any = None,
        clear_entry: bool = False,
    ) -> dict[str, Any]:
        """The stroke, in one place, whoever supplied the frame.

        **Written as vectors rather than as ``(dz, dy)`` pairs**, and that is the
        whole generalisation. The reference schedule rises along world ``+z`` and
        pushes along world ``+y`` because that is where the reference put its
        bench; a receptacle a camera found is on neither axis exactly, and a
        planner written in those two scalars can only aim at the one layout it
        was measured in.

        So the stroke is stated as three directions instead:

        ``axis``
            Into the bore. Both the push and the depth are along it.
        ``rise``
            From the grasp station to the bore line -- whatever is left of the
            grasp-to-mate travel once the axial part is taken out. On the
            authored assembly this comes out as ``(0, 0, 0.0124)``, the
            reference's own ``LIFT``, to the last digit.
        ``UP``
            World vertical, and the one direction that stays a constant. The
            approach stages above the part and the retract rises off it, and both
            are *top-down grasp* geometry rather than mate geometry: the hand
            comes down onto a bench and leaves upward off it however the bore is
            pointing.
        """
        UP = np.array([0.0, 0.0, 1.0])  # portable: world vertical, not a robot fact

        def build() -> dict[str, Any]:
            env = self.env
            asm = self.asm
            rig = env.scene.by_name["rig"]
            ik = env.arm(self._arm_name())
            # TWO WRIST HEADINGS, and the difference between them is the whole
            # reason a drawn layout is harder than a fixed one.
            #
            # `yaw_deg` is the heading the jaws close ACROSS THE PLUG, which
            # follows wherever the part happens to be lying -- that is what a
            # grasp needs. The mate needs the opposite thing: the plug's own
            # insertion axis ON the bore axis. On the fixed layout those are the
            # same number, because the plug spawns on the mate line, and the
            # stroke was written with one quaternion for every stage.
            #
            # Under `staging:USBC_BENCH` they are not. Measured on the shipped
            # sampler, the plug's axis sits **8.4 degrees off the bore at spawn
            # and never comes within 4.9 of it all episode** -- because nothing
            # between the grasp and the push ever turns the wrist. Eight degrees
            # of yaw into a slot with 0.85 mm a side is a plug that jams, and it
            # is why the goal's own 5-degree tilt clause and the detent's
            # 3-degree latch gate were never once satisfied.
            #
            # So the hand grasps at the plug's heading and arrives at the bore's.
            # The turn happens across `lift`, which is the one stage between
            # closing on the part and needing it aligned.
            axis = np.asarray(INSERT_AXIS, dtype=np.float64) if bore_axis is None else _unit(bore_axis)
            hand = self._hand()

            def _quat(yaw: float, approach: tuple[float, float, float] | None = None) -> Any:
                if approach is None:
                    return top_down_grasp_site_quat(hand, float(yaw))
                turn = np.radians(float(yaw) - GRASP_YAW_DEG)
                close = np.array([np.cos(turn), np.sin(turn), 0.0])
                app = np.asarray(approach, dtype=np.float64)
                app = app - close * float(app @ close)
                return grasp_quat(
                    hand.approach_axis,
                    hand.close_axis,
                    hand.tcp_offset_rot,
                    tuple(float(v) for v in _unit(app)),
                    tuple(float(v) for v in _unit(close)),
                )

            mate_yaw = _yaw_across(axis, axis)
            grasp_axis = _plug_grasp_point(np.zeros(3), yaw_deg, axis, 1.0)
            quat_grasp = _tilted_grasp_quat(hand, float(yaw_deg), grasp_axis)
            quat_mate = _tilted_grasp_quat(hand, mate_yaw, axis)
            if mate_quat is not None:
                quat_mate = np.asarray(mate_quat, dtype=np.float64).reshape(4)
                if not np.all(np.isfinite(quat_mate)) or np.linalg.norm(quat_mate) < 1.0e-9:
                    raise ValueError("the measured mating wrist quaternion must be finite and nonzero")
                quat_mate = quat_mate / np.linalg.norm(quat_mate)
            back_quat = ik.ee_pose(env.state)[1] if mate_quat is not None and tcp_point is not None else quat_mate
            theta = np.radians(INSPECT_PITCH_DEG)
            inspect_approach = _unit(axis * np.cos(theta) - UP * np.sin(theta))
            quat_inspect = _quat(mate_yaw, tuple(float(v) for v in inspect_approach))
            plug_p = (
                np.asarray(asm.plug_spawn, dtype=np.float64)
                if plug_point is None
                else np.asarray(_vec3(plug_point), dtype=np.float64)
            )
            if mouth_point is None:
                mated = np.asarray(asm.plug_mated, dtype=np.float64)
            else:
                # Mated = the tab tip on the cavity floor, which is the mouth
                # plus the bore's depth. The plug's own origin then sits
                # `tab_tip_z` back from the tip along its insertion axis. Same
                # arithmetic `make_assembly` runs, aimed at a measured mouth.
                floor = np.asarray(_vec3(mouth_point), dtype=np.float64) + float(asm.port.mouth_z) * axis
                mated = floor - float(asm.plug.tab_tip_z) * axis

            # WHERE THE PLAN IS ANCHORED, and it is the difference between a
            # stroke that corrects a jittered grasp and one that carries it.
            #
            # With no `tcp_point` the anchor is a *reconstructed* grasp station:
            # the plug's position displaced by the nominal in-jaw offset. That is
            # right before anything has been picked up and wrong the moment
            # something has, because the offset is exactly what a grasp disturbs.
            # Measured on a drawn layout with the plug 5 mm off the mate line --
            # perception found it to 1.2 mm, the jaws centred it as they closed,
            # and the re-plan still put the mate 17.5 mm off axis.
            #
            # Given the tool centre, the plan is a DISPLACEMENT instead: the hand
            # must move by exactly (mated - plug_now), and a rigidly held plug
            # moves with it, so where it ended up in the jaws stops mattering.
            # Everything below is unchanged -- only what `base` means is.
            grasp_offset = FULL_FINGER_GRASP_OFFSET
            hand_vertices = self._hand_vertices(ik)
            base = (
                _plug_grasp_point(plug_p, yaw_deg, axis, grasp_offset)
                if tcp_point is None
                else np.asarray(_vec3(tcp_point), dtype=np.float64)
            )
            if tcp_point is None:
                # Tilting lowers one corner of the pads. Derive the pickup
                # height from the whole hand and the plug's known half-height
                # rather than driving that corner into the bench while closing.
                lowest = float(np.min((hand_vertices @ quat_to_mat(quat_grasp).T)[:, 2]))
                base += max(0.0, FINGER_BENCH_CLEARANCE - float(asm.plug.half_x) - lowest) * UP
            travel = mated - plug_p
            ins = float(travel @ axis)
            # WHAT IS LEFT OF THE TRAVEL ONCE THE AXIAL PART IS TAKEN OUT --
            # unless the tab is already in the bore, in which case there is no
            # such thing and commanding it is a lever.
            #
            # Measured on three drawn layouts: a plug seated 5-7 mm deep rests
            # 1.3-1.5 mm BELOW the bore axis, because this slot is 8.00 mm tall
            # against a 4 mm tab and the part lies on the floor of it. `rise`
            # then reads 1.5-2.1 mm and means "lift the plug to the centreline"
            # -- which, through jaws that have just closed to the carry width,
            # is the whole vertical slack applied to a tab the socket is already
            # holding. It cocks the mate instead of driving it: the second push
            # came back at 5.43 mm where the first alone reached 6.4-6.7.
            #
            # So a servo push has ONE degree of freedom, and it is the axis.
            # Lateral and vertical alignment is spent getting into the hole; a
            # correction applied after that is fighting the bore for it.
            rise = np.zeros(3) if axial_only else travel - ins * axis
            lift_len = float(np.linalg.norm(rise))
            correct = np.asarray(correction, dtype=np.float64)

            def target(offset: np.ndarray, corrected: bool) -> np.ndarray:
                return base + offset + correct if corrected else base + offset

            seat = ins + float(preload)
            if axial_only:
                # A second push is a probe, not another entry stroke. More than
                # the measured 3 mm bound cocks a short plug in the fixed port
                # (10 mm took lateral error from 2 mm to 24 mm).
                seat = float(np.clip(seat, 0.0, 0.003))
            # Bound every push, including an anchored retry after slip, by the
            # actual hand envelope. A perceived short seat is never permission
            # to advance the fingers through the receptacle. The plane is a
            # conservative bound for both the fixed port and the free barrel.
            mouth = mated + (float(asm.plug.tab_tip_z) - float(asm.port.mouth_z)) * axis
            hand_points = hand_vertices @ quat_to_mat(quat_mate).T
            clearance_limit = _clearance_travel(base + rise + correct, mouth, axis, hand_points)
            if clearance_limit < -1.0e-6:
                raise RuntimeError("USB-C grasp leaves no socket clearance; withdraw and regrasp farther back")
            if seat > clearance_limit:
                print(
                    f"[usbc] limiting push from {seat * 1e3:.1f} to {max(0.0, clearance_limit) * 1e3:.1f} mm for finger clearance",
                    flush=True,
                )
            seat = min(seat, max(0.0, clearance_limit))
            # WHERE THE PUSH STARTS, and the two cases are not the same move.
            #
            # Un-anchored, `base` is the reconstructed grasp station and the
            # stage offsets are absolute, so starting at `seat - retry_from`
            # means the hand WITHDRAWS from wherever the last push left it and
            # drives in again. That is `graphs/gt`'s back-off, and it is what
            # gets a jammed plug out of the mouth to try a second entry.
            #
            # ANCHORED on the hand, `base` is where the hand already stands, so
            # the same arithmetic clips to `insert_start = 0` -- "start here" --
            # and the withdrawal silently disappears. The first servo push was
            # written that way and could only ever shove harder on a plug that
            # had stopped: measured, a whole extra iteration moved the remaining
            # travel 6.0 mm to 5.8. So anchored, `retry_from` is stated the way
            # it reads: pull back this far from where you are, then push.
            if axial_only:
                insert_start = 0.0
            elif tcp_point is not None and retry_from > 0.0:
                insert_start = -float(retry_from)
            elif tcp_point is not None and clear_entry:
                # Align with the tab still clear of the mouth. From a raised
                # look station, projection onto a pitched bore can put the
                # perpendicular alignment target inside the socket. Keep the
                # shoulder at least 20 mm short of its mated position before
                # starting the axial stroke (the tab projects 13.3 mm).
                insert_start = min(0.0, ins - 0.020)
            else:
                back = float(np.clip(retry_from, 0.0, seat))
                insert_start = seat - back if back > 0.0 else 0.0
            span = max(seat - insert_start, 1.0e-6)
            insert_duration = max(0.6, span / INSERT_SPEED)

            # Where the hand stands to look at the receptacle, and it is stated
            # as an offset from `base` like every other station so the whole
            # schedule moves together when the frame does.
            # Anchored on the hand, the hand is already AT the look station, so
            # the round trip starts from where it stands.
            look = np.zeros(3) if tcp_point is not None else rise + (ins - LOOK_BACK) * axis + LOOK_RISE * UP
            inspect = rise + (ins - INSPECT_BACK) * axis + INSPECT_RISE * UP
            if with_inspect:
                camera = next((c for c in getattr(env, "cameras", ()) if c.name == "wrist"), None)
                if camera is not None:
                    # Aim the calibrated optical frame at the aperture. A TCP
                    # offset tuned for another camera mount can miss the bore
                    # entirely or bring the housing into the fixture.
                    tcp_camera = (
                        np.linalg.inv(transform_matrix((*hand.tcp_offset, *hand.tcp_offset_rot))) @ camera.world_matrix
                    )
                    eye = mouth - 0.30 * axis + 0.12 * UP
                    camera_up = -(quat_to_mat(quat_inspect) @ tcp_camera[:3, 1])
                    world_camera = look_at_matrix(eye, mouth, camera_up)
                    world_tcp = world_camera @ np.linalg.inv(tcp_camera)
                    inspect = world_tcp[:3, 3] - base
                    quat_inspect = mat_to_quat(world_tcp[:3, :3])
            clear = inspect - UNLOOK_BACK * axis + UNLOOK_LIFT * UP

            # (name, start offset from base, end offset, duration [s], corrected,
            #  wrist heading at the start, wrist heading at the end)
            #
            # `look` and `back` are a round trip out of the lift station and are
            # NOT part of the reference schedule -- `graphs/gt` never executes
            # them, because it reads the receptacle's pose out of the simulator
            # and has nothing to go and see. They are emitted unconditionally
            # anyway: a stage nobody drives costs the solve that built it and
            # keeps one planner serving both graphs, where a flag would make the
            # two plans differ in more than the stage list.
            stages = (
                [
                    # Inspect with an empty hand in the mate heading. Blending
                    # directly from/to the arbitrary pickup heading can send
                    # the wrist through the opposite IK branch and its limits.
                    # The approach stage turns to the grasp above the plug.
                    ("goto", APPROACH_H * UP, inspect, 1.6, False, quat_mate, quat_inspect),
                    ("inspect", inspect, inspect, 1.0, False, quat_inspect, quat_inspect),
                    ("unlook", inspect, clear, 1.2, False, quat_inspect, quat_inspect),
                    ("cross", clear, APPROACH_H * UP, 1.8, False, quat_inspect, quat_mate),
                ]
                if with_inspect
                else []
            ) + [
                ("approach", APPROACH_H * UP, APPROACH_H * UP, 0.7, False, quat_grasp, quat_grasp),
                ("descend", APPROACH_H * UP, np.zeros(3), 1.0, False, quat_grasp, quat_grasp),
                # The turn rides the lift: the plug is off the bench, held, and
                # not yet near the bore, which is the only window where turning
                # it costs nothing.
                ("lift", np.zeros(3), rise, 1.4 * TRANSPORT_TIME_SCALE, True, quat_grasp, quat_mate),
                ("look", rise, look, 1.2 * TRANSPORT_TIME_SCALE, True, quat_mate, quat_mate),
                ("back", look, rise + insert_start * axis, 1.2 * TRANSPORT_TIME_SCALE, True, back_quat, quat_mate),
                ("insert", rise + insert_start * axis, rise + seat * axis, insert_duration, True, quat_mate, quat_mate),
                (
                    "retract",
                    rise + seat * axis,
                    # Rise outside the mouth plane even when the braced bore
                    # points upward. A world-vertical lift then advances toward
                    # the flange; remove that axial component before lifting.
                    rise + seat * axis + RELEASE_LIFT * (UP - max(0.0, float(UP @ axis)) * axis),
                    1.4,
                    True,
                    quat_mate,
                    quat_mate,
                ),
            ]

            # Warm-started full-pose LM along the path: the exact solver and
            # iteration budget the live tracking uses, run kinematically.
            q_ik = ik.joint_q_ik
            wp.copy(q_ik.reshape((ik.model.joint_coord_count,)), env.state.joint_q)
            out = []
            stream_hz = self._stream_hz()
            for name, start, end, duration, corrected, q0, q1 in stages:
                n = max(2, int(duration * stream_hz))
                waypoints = []
                turning = float(np.dot(np.asarray(q0), np.asarray(q1))) < 1.0 - 1.0e-9
                for i in range(1, n + 1):
                    t = i / n
                    q = quat_slerp(q0, q1, t) if turning else q0
                    ik.set_target(target(start + t * (end - start), corrected), q)
                    ik.solver.step(q_ik, q_ik, iterations=ik.iterations)
                    q = q_ik.numpy()[0]
                    waypoints.append({"positions": [float(q[c]) for c in ik.arm_coords]})
                out.append(
                    {
                        "stage": name,
                        "trajectory": {"waypoints": waypoints},
                        "duration": float(duration),
                        "corrected": corrected,
                    }
                )

            commands = rig.grip_commands(2.0 * asm.plug.half_y)
            return {
                "arm_id": arm_id_of(env, self._arm_name()),
                "stages": out,
                "grips": {
                    k: {"command_m": float(v), "fraction_closed": float(1.0 - v / self._grip_open())}
                    for k, v in commands.items()
                },
                "grasp_point": [float(v) for v in base],
                "correction": [float(v) for v in correct],
                # What the frame came out as, so a trace can say what the plan
                # was aimed at without re-deriving it from five other keys.
                "insert_dist": seat if axial_only else ins,
                "lift_m": lift_len,
                "axis": [float(v) for v in axis],
                "yaw_deg": float(yaw_deg),
                "mate_yaw_deg": float(_yaw_across(axis, axis)),
            }

        return self.bus.call(build)

    # --------------------------------------------------------------- percieve
    def perceive_mate(self) -> dict[str, Any]:
        """Where the two halves of the mate are, read out of the scene.

        The **oracle body** of ``graphs/mate``'s perception slot, and every key
        it returns was chosen because a camera can earn it. That constraint is
        the whole design: a tool returning "the plug's exact quaternion" would be
        honest and useless, because the vision body across from it has to declare
        the same keys and cannot produce that one.

        What it does *not* return is as deliberate. There is no depth, no lateral
        error and no verdict -- those are :meth:`get_insertion_state`'s, they are
        the evaluator's own quantities, and a perception slot that handed them
        over would let a graph route on the score.

        ``held`` is the question the loader's ``HELD`` set answers at build time
        and this answers at run time: a receptacle with a body index is a free
        thing somebody has to hold, and one welded to the world has no body row
        at all. The graph branches on this, not on a variant name.
        """

        def read() -> dict[str, Any]:
            from robosimstudio.core.math import quat_to_mat  # noqa: PLC0415

            ctx = self.env.make_context()
            name, payload = self._receptacle()
            if payload is None:
                raise RuntimeError(
                    f"receptacle {name!r} declares no cavity feature; perceive_mate has no bore to report"
                )
            handle = ctx.handles.entity(name)
            free = bool(handle.body >= 0)
            pose = np.asarray(handle.xform if not free else ctx.pose(name), dtype=np.float64)
            rot = quat_to_mat(pose[3:7])
            mouth = pose[:3] + rot @ np.asarray(payload["mouth"], dtype=np.float64)
            axis = rot @ np.asarray(payload["axis"], dtype=np.float64)
            axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)

            plug = np.asarray(ctx.pose("plug"), dtype=np.float64)
            plug_rot = quat_to_mat(plug[3:7])
            # The plug's own insertion direction is its local +z -- the axis
            # `PLUG_QUAT` carries onto the world mate line.
            plug_axis = plug_rot @ np.array([0.0, 0.0, 1.0])

            return {
                "route": "found",
                "body": "oracle",
                "plug_point": [float(v) for v in plug[:3]],
                "plug_axis": [float(v) for v in plug_axis],
                "plug_yaw_deg": _yaw_across(plug_axis, axis),
                # Half the plug's height, which is what a top-down read of a
                # part lying on a bench can measure and a read of the same part
                # held in the air cannot: there is no surface under it to
                # bisect against. Carried from the bench read to the re-look.
                "plug_half_h": float(self.asm.plug.half_x),
                "mouth_point": [float(v) for v in mouth],
                "bore_axis": [float(v) for v in axis],
                "bore_depth": float(payload["depth"]),
                # Where a side grasp takes a free receptacle: on the bore axis,
                # one band back from the mouth. Derived from the mouth rather
                # than from the spawn, so it follows a receptacle that has been
                # picked up -- which is the only state this ever matters in.
                "barrel_point": [float(v) for v in (mouth + BARREL_BACK * axis)] if free else [],
                "receptacle": str(name),
                "held": free,
            }

        return self.bus.call(read)


# --------------------------------------------------------------------- helpers
def _vec3(v: Any) -> tuple[float, float, float]:
    """An ``[x, y, z]`` out of a sequence or a ``{"x": .., "y": .., "z": ..}``.

    Both spellings are in circulation across this corpus's node scripts -- gap's
    own ``Vec3`` serialises as the mapping, and a body doing its own numpy
    arithmetic hands back a list -- and a planner that raised ``TypeError`` on
    one of them would teach nothing about which it wanted.
    """
    if isinstance(v, dict):
        return (float(v["x"]), float(v["y"]), float(v["z"]))
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"expected a 3-vector, got {v!r}")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _unit(v: Any) -> np.ndarray:
    """``_vec3`` normalised, refusing a zero direction rather than dividing by it."""
    arr = np.asarray(_vec3(v), dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1.0e-9:
        raise ValueError(f"a direction cannot be the zero vector, got {v!r}")
    return arr / norm


def _clearance_travel(tcp: Any, mouth: Any, axis: Any, hand_points: np.ndarray) -> float:
    """Available axial TCP travel before any part of the hand reaches the mouth.

    ``hand_points`` are full mesh vertices relative to the TCP, rotated into
    world axes. The support of a triangle mesh along a plane normal occurs at
    a vertex, so this also bounds every face, including a hollow finger yoke.
    """
    axis = _unit(axis)
    front = float(np.max(np.asarray(hand_points, dtype=np.float64) @ axis))
    return float((np.asarray(mouth) - np.asarray(tcp)) @ axis) - front - FINGER_PORT_CLEARANCE


def _tilted_grasp_quat(hand: Any, yaw_deg: float, axis: Any) -> np.ndarray:
    """Lean away from the port without flipping the camera to the other jaw.

    Preserve the signed closing direction of ``top_down_grasp_site_quat``.
    The empty-hand inspection pose uses its opposite, which is the same pinch
    but puts the wrist camera behind the gripper housing during a held re-look.
    """
    yaw, tilt = np.radians([yaw_deg, GRASP_TILT_DEG])
    approach = _unit(axis) * np.sin(tilt) - np.array([0.0, 0.0, 1.0]) * np.cos(tilt)
    return grasp_quat(
        hand.approach_axis,
        hand.close_axis,
        hand.tcp_offset_rot,
        approach,
        (np.cos(yaw), np.sin(yaw), 0.0),
    )


def _plug_grasp_point(plug_point: Any, yaw_deg: float, bore_axis: Any, offset: float) -> np.ndarray:
    """Place the initial grasp along the perceived plug's long axis."""
    angle = np.radians(float(yaw_deg) - 90.0)
    direction = np.array([np.cos(angle), np.sin(angle), 0.0])
    if float(direction @ np.asarray(bore_axis, dtype=np.float64)) < 0.0:
        direction = -direction
    return np.asarray(plug_point, dtype=np.float64) + float(offset) * direction


def _yaw_across(long_axis: Any, reference: Any) -> float:
    """The heading the jaws close along, given a part's own long axis [deg].

    A quarter turn off the part's heading, which is what "close across it"
    means, resolved against ``reference`` -- the bore axis -- for the half-turn
    ambiguity below. Written to one rule because both bodies of the perception
    slot have to apply it: the oracle takes ``long_axis`` off the plug's
    quaternion and the vision body takes it off a mask's principal axis, and two
    conventions here would look like a perception error rather than a
    convention.

    A parallel jaw is symmetric under a half turn, so ``yaw`` and ``yaw + 180``
    are the **same grasp** -- and they are not the same *wrist*. This hand
    carries its camera 70 mm off to one side, so the two differ by swinging the
    lens to the other side of the gripper: measured, the eye-in-hand at the look
    station showed the receptacle under one and showed the gripper body and an
    empty bench under the other, and the re-look reported "no receptacle" on a
    scene where nothing was wrong.

    The ambiguity is real and it is unavoidable in the *measurement*: a
    principal axis has no sign, and an SVD returns whichever direction it likes.
    So it is resolved against the one direction in this scene that does have a
    sign -- the bore axis, which points from the plug into the receptacle. The
    representative kept is the one within a quarter turn of "across the mate
    line", which is where the reference schedule's own 180 degrees sits.
    """
    axis = np.asarray(_vec3(long_axis), dtype=np.float64)
    ref = np.asarray(_vec3(reference), dtype=np.float64)
    yaw = float(np.degrees(np.arctan2(float(axis[1]), float(axis[0]))) + 90.0)
    across = float(np.degrees(np.arctan2(float(ref[1]), float(ref[0]))) + 90.0)
    return yaw if abs((yaw - across + 180.0) % 360.0 - 180.0) <= 90.0 else yaw + 180.0
