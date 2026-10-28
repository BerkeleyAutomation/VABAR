# SPDX-License-Identifier: Apache-2.0
"""The scripted cable routines: pick-and-place, and the peg wrap.

Plan-then-stream, the tape suite's architecture: every stage endpoint of the
reference's keyframe table is solved **once** with the full-pose FD solver
(:meth:`~robosimstudio.solvers.ArmIK.solve_q_full` -- yaw authority, no
LM module), then the routine smoothsteps between the endpoint *joint*
configurations at the control rate. Not the reference's per-frame Cartesian
tracking, and deliberately so twice over:

* the per-frame LM tracker's kernels specialize over the model's joint types,
  and a model carrying cable joints pays a pathological first-ever compile;
* joint-space interpolation between converged endpoints is the motion the
  shipped graph drives (``robot.execute_trajectory`` over the same plan), and
  a routine and a graph that move identically is what makes their numbers
  comparable -- the same argument the README makes for ``--motion``.

The wrap's grasp is planned against the **live** cable -- the segment nearest
the nominal bend, jaws across its local tangent -- which is the reference's
closed-loop element collapsed to plan time. Identical until first contact: the
cable is at rest all through the approach, so tracking it per frame and
reading it once see the same rod.

**The keyframe tables live here.** They used to sit in a ``stagecraft`` module
beside a ``suite.py``, shared by this routine and by the plan tool so the two
move identically. The suite is a document now, and a table of scripted
waypoints was never a statement about the task -- it is *this solution's* data,
so it moved in here with it and the plan tool imports it from the routine it
belongs to.

**Where the parameters come from, now that ``metadata`` is gone.** Almost all of
them were never parameters: the work surface, the peg and the goal lane are
things the *scene* knows, and the routine asks it -- ``handles.surface_z``, the
peg's own static handle, the goal region's centre. What is left is the two
numbers below, which are genuinely inputs to this script and are recorded in
each document's ``(:policy-params ...)`` beside the variant they belong to.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robosimstudio.core.math import quat_to_mat
from robosimstudio.core.render.visual import shape_geometry
from robosimstudio.robots.embodiments import top_down_grasp_site_quat, work_gripper
from robosimstudio.solvers.motion.path import stream_hz as _stream_hz
from robosimstudio_bench.solutions.base import solution_param as policy_param

from .assets.cable import WEAVE_CABLE_RADIUS
from .assets.fixtures import CABLE_PEG, ROUND_FLANGE_RADIUS, ROUND_HEIGHT
from .grasp import MAX_STATION_OFFSET, choose_grasp

__all__ = [
    "GRASP_Y",
    "MOTION_SCALE",
    "REFERENCE_GRIP_OPEN",
    "RING_POINTS",
    "SEAT_BIAS",
    "WRAP_TURNS",
    "RoutineCableRoute",
    "RoutineWeave",
    "WrapPlan",
    "grasp_site_quat_legacy",
    "pickplace_rows",
    "plan_stages",
    "plan_station_stages",
    "plan_support_feed",
    "plan_support_stages",
    "station_xy",
    "wrap_rows",
]

SETTLE_STEPS = 60

GRASP_Y = {"pickplace": 0.0, "wrap": -0.18}
"""Where each variant's script reaches for the rod, in world y [m].

The reference's own two numbers, and the only pair that did not resolve into
something the scene already knows. They are ``(:policy-params (grasp-y ...))``
in the documents -- an input to *this artifact*, not a statement about the world
-- and a test pins the two spellings together, because a routine aiming
somewhere its document does not say is a policy solving a different task with
nothing to show for it.

The rod runs along +y at x = 0.58, so ``pickplace``'s 0.0 is its midpoint and
``wrap``'s -0.18 is the bend the script folds around the peg. Both are
re-detected against the live cable before the jaws close; these aim the search.
"""

GRIP_LOCAL_CLOSE_LEGACY = np.array([-0.048, 0.090, 0.0]) / np.linalg.norm(np.array([-0.048, 0.090, 0.0]))
"""The closing axis the *cable* reference grips with: ``yam_arm.py``'s
pre-measurement axis, read off the tip body origins. It is ~85 degrees off the
measured pad axis the connector sims corrected to -- and the cable sim never
adopted the correction, so its keyframes were tuned against *this* roll.
Grasping with the measured axis at the same yaw closes the jaws along the rod
instead of across it: measured as a blade clipping the cable on the way down
and a pinch that shuts on air while the rod lies beside it."""


def grasp_site_quat_legacy(hand, yaw_deg: float) -> np.ndarray:
    """The cable reference's ``grasp_quat``, in the grasp-site frame, xyzw.

    The shared composition with this suite's own closing axis substituted. The
    approach axis and the site rotation come off ``hand``, so the only thing
    pinned here is the one number this suite deliberately disagrees with the
    connector suites about.
    """
    return top_down_grasp_site_quat(hand, yaw_deg, close_axis=GRIP_LOCAL_CLOSE_LEGACY)


def cable_tangent(rod, point_xy):
    """Local cable direction from a measured centreline."""
    rod = np.asarray(rod, dtype=float).reshape(-1, 3)
    index = int(np.argmin(np.linalg.norm(rod[:, :2] - np.asarray(point_xy), axis=1)))
    local = rod[max(0, index - 2) : min(len(rod), index + 3), :2]
    return np.linalg.svd(local - local.mean(axis=0), full_matrices=False)[2][0]


def cable_grasp_quats(hand, tangent):
    """Equivalent parallel-jaw grasps that close across the measured cable."""
    tangent = np.asarray(tangent, dtype=float).reshape(2)
    if not np.all(np.isfinite(tangent)) or np.linalg.norm(tangent) < 1.0e-8:
        raise ValueError("cable grasp requires a finite, nonzero measured tangent")
    yaw = np.degrees(np.arctan2(tangent[1], tangent[0])) - 90.0
    reference = grasp_site_quat_legacy(hand, 0.0)
    choices = [top_down_grasp_site_quat(hand, yaw + turn) for turn in (0.0, 180.0)]
    return sorted(choices, key=lambda q: -abs(float(np.dot(q, reference))))


def cable_grasp_quat(hand, tangent):
    """The nearer wrist roll whose measured jaw axis closes across the cable."""
    return cable_grasp_quats(hand, tangent)[0]


#: The reference's own open command, kept only as the documented default for a
#: caller that has no scene to ask. Live code passes the arm's own value, since
#: the open command is a property of whichever hand the pairing brought.
REFERENCE_GRIP_OPEN = 0.0475
GRIP_CLOSE = 0.003
"""Grip at the cable's diameter: over-squeezing squirts it out [m]."""

MOTION_SCALE = 2.6
"""Multiplier on every weave leg's duration.

**The legs were written at 0.4 to 0.7 s for moves of about 100 mm** -- roughly
200 mm/s with a gripper closed on a cable, which is faster than anything on this
bench needs to be and reads, watching it, as the arm teleporting between grasps.
It also matters physically rather than only visually: the rod is a chain of
capsules solved by VBD, and a hand that crosses a station in half a second drags
it through contacts the solver resolves as an impulse, so the cable whips instead
of following.

2.6 puts a crossing at about 75 mm/s. The number is the ratio between what the
legs were written at and the speed the reference's own scripted routine moves at,
and slowing them changes only the streaming -- the waypoints, the IK and the
grip commands are untouched.
"""

SEAT_BIAS = 0.012
"""How far PAST the seat's centre a crossing places the rod [m].

**The rod relaxes after the jaws open, and the place point has to pay for it.**
A crossing carries the cable to a target and lets go; what is scored is where it
comes to rest, and a Kirchhoff rod that has just been bent around a post keeps
some of that bend as stored energy and springs back toward it.

Measured on ``weave3`` with the initial condition varied (episode 3): station 1
was aimed at 17.6 mm past its post -- the middle of a box that runs 3 to 45 mm --
and came to rest at **+0.5 mm**, on the correct side and 2.5 mm short of the box.
The template was right (the side decision ran at 77 mm of margin) and the carry
executed; the rod simply did not stay.

12 mm is chosen against the box, not against that one episode: aiming at
17.6 + 12 = 29.6 mm leaves 15 mm before the box's far edge, so an episode that
relaxes very little is still inside, and one that relaxes as much as episode 3
did lands near the middle instead of falling out.

**This is not the SEAT_OVERSHOOT that was built and reverted.** That one was
aimed at a station whose margin was NEGATIVE -- the rod was on the far side of
the post it should have crossed, and pressing harder cannot fix a cable that is
on the wrong side. The signed margin is what separates the two cases, and
``verify_weave_vision`` reports it alongside a reason, which is how this one was
identified as a near miss rather than the same failure again.
"""

WEAVE_PEG_CLEAR = 0.030
"""Longitudinal hand offset that clears the peg flange while placing [m]."""

TRANSIT_LIFT = 0.0
"""Retired: lifting the drag traverse clears the flange and does not help [m].

ZERO, and the measurement is why. The collision it removes is real -- see below
-- but lifting the traverse 6 mm scored WORSE than not doing it: per-seat
8/20 = 40% against a 57% baseline, with press-time readings unchanged (44%
clearing against 43%). The drag template's own comment predicted it: a rod
lifted clear of the table swings, and a swinging rod lands somewhere the plan
did not choose. Trading a flange strike for a swing is not a trade worth making.

Kept as a constant rather than deleted because the collision it was aimed at is
measured and unfixed: the closed hand's yoke is 17.3-18.6 mm wide 10-13 mm ABOVE
the TCP, rest_z puts it at 20.3-23.3 mm, and the post's top flange occupies
19.7-23.5 mm -- so 8.8% of every episode has the yoke inside the rim. Whatever
fixes that has to do it without lifting the rod.

**The drag template was ploughing the yoke through the spools.** It moves the
held point from the grasp to the seat entirely at ``rest_z``, which is
``surface + 10.3 mm`` -- and the closed hand's widest part sits 10-13 mm ABOVE
the TCP, so the yoke rides at 20-23 mm, inside the post's top flange band
(``ROUND_GROOVE_Z[1]`` 19.7 mm to ``ROUND_HEIGHT`` 23.5 mm). Measured over eight
episodes of recorded ``right_ee_pose``: 8.8% of every episode is spent with the
yoke in that band AND within the 30.93 mm the hand needs, closing to 13.7-26.6 mm.

That is why the hourglass matters. The post is 4.75 mm at cable height and
12.5 mm at yoke height, so there is no single clearance radius -- the fingers sit
in the groove while the yoke sweeps the flange above them.

Six millimetres is what puts the yoke over the top rather than through it:
``rest_z + 6`` = 16.3 mm carries the yoke to 26.3-29.3 mm, clear of the 23.5 mm
rim. It is deliberately NOT ``carry_z`` (53.5 mm): the drag exists because a rod
lifted clear of the table swings and lands somewhere the plan did not choose, and
6 mm keeps the rod essentially on the bench while moving the metal that was
hitting the fixture."""

WRAP_PAD_CLEAR = 0.016
"""The closed hand's swept half-width, which is what the arc has to clear [m].

**Not a pad margin past the flange.** The TCP is a point; the hand is not. At
``weave_grip_close`` the blade yoke measures 17.3 mm of radial extent from the
approach axis at 10 mm above the TCP and flares 0.43 mm per mm above that --
and ``rest_z`` (``PAD_FLOOR_CLEAR`` is 0.0) puts the peg's flange band at
dz +9.7..+13.5, exactly where the yoke is 17.3-18.6 mm wide. Sampled against
``round_peg.stl``'s own r(z) -- 4.75 mm shaft flaring to a 12.5 mm rim over
z 19.68-23.5 mm -- the smallest collision-free TCP radius is 30.93 mm.

An arc at the longitudinal clearance alone does not clear it: that is 30 mm on
Easy and 28-30 mm on Medium, which drives the yoke 0.93 mm into the flange on
Easy and 2.93 mm on Medium, over 28% and 59% of the sweep. The colliding
bearings form two lobes 180 degrees apart, so no starting yaw escapes them and
the fixed wrist this plan solves once cannot dodge it. And the contact is live
rather than filtered: ``assets/cable.py`` sets ``contact_only`` for exactly
these two tiers, so ``filter_fingertip_scenery`` is False and the blade-vs-static
exclusion pairs are never installed.

Sixteen millimetres puts the floor at ``12.5 + 3.5 + 16`` = 32 mm, which clears
the measured worst case by 1.1 mm and also clears the arrangement's 2 mm contact
detection margin, so no contact rows are generated at all."""

WRAP_SWEEP = 1.0471975511965976
"""Half the arc's sweep about the post [rad] -- 60 degrees, so 120 in total.

A half turn ends the sweep ON the station's axis, where the hand's x offset from
the post is exactly zero -- and the scored seat starts 3 mm off the axis
(``(:ranges ((0.003 -0.035 0.045 0.035)))``). Releasing there leaves the held
point outside the very box the crossing is measured in.

Sixty degrees either side of the required face keeps BOTH ends inside it: at
32 mm the ends stand 16 mm off the axis on the required side and 27.7 mm along
the run, against a box that runs 3-45 mm and +/-35 mm. It is still 120 degrees
of post the span is drawn around."""

WRAP_LEG_S = 0.18
"""Seconds per waypoint of the wrap arc. The whole half turn is ~1.3 s, which is
the pace the carry already runs at; faster and the span whips off the post
instead of being drawn onto it."""

WRAP_ARC_POINTS = 7
"""Waypoints in the half-turn around a station. Seven is what keeps the chord
inside the cable's own radius at the radius the arc runs: a 180-degree sweep at
30 mm in six chords leaves a 3.2 mm sagitta, under the rod's 3.5 mm, so the
held point never cuts a corner the cable would have to be dragged across."""


def wrap_arc(peg_xy, side: float, radius: float, z: float, points: int = WRAP_ARC_POINTS):
    """The hand path that lays a span around ONE FACE of a station.

    A half turn about the post, from the port end of the seat's own scored
    window round to the anchor end, passing the face the crossing is required to
    use -- ``+x`` when ``side`` is positive, ``-x`` when it is negative. The
    cable is held at the TCP, so the span between the anchor and the hand sweeps
    the post's circumference and comes to rest against the shaft; nothing is
    pressed, and nothing has to reach inside the flange.

    Args:
        peg_xy: The station's ``(x, y)`` in world.
        side: Which face the route requires, ``+1`` or ``-1``.
        radius: Arc radius from the station's axis [m]. The caller is expected
            to have cleared the flange; :func:`wrap_radius` is that rule.
        z: Height to run the arc at [m] -- the rod's rest height, so the span is
            laid down rather than swung.
        points: Waypoints in the sweep.

    Returns:
        ``(points + 1, 3)`` waypoints, first at ``+y`` of the station and last
        at ``-y``, both a ``radius`` out.
    """
    sx, sy = float(peg_xy[0]), float(peg_xy[1])
    # +90 degrees is the port end of the window, where the carry already stands.
    # Sweeping NEGATIVE passes +x; sweeping positive passes -x. Either way the
    # arc finishes at -90, the anchor end, having gone round the required face.
    # Both ends INSIDE the scored box, on the required face. +x is bearing 0 and
    # -x is bearing pi; the sweep runs WRAP_SWEEP either side of whichever the
    # route asks for, and always from the port end of the window to the anchor
    # end so successive stations are worked in the direction the run travels.
    face = 0.0 if float(side) > 0.0 else np.pi
    start, stop = face + WRAP_SWEEP, face - WRAP_SWEEP
    if float(side) <= 0.0:
        start, stop = face - WRAP_SWEEP, face + WRAP_SWEEP
    out = []
    for k in range(int(points) + 1):
        angle = start + (stop - start) * (k / float(points))
        out.append([sx + radius * np.cos(angle), sy + radius * np.sin(angle), float(z)])
    return np.asarray(out, dtype=np.float64)


def wrap_radius(clearance: float) -> float:
    """How far out the wrap arc has to run [m].

    Never inside the flange plus the cable plus a pad margin, and never further
    than the longitudinal clearance the caller already chose -- which on a tight
    board is what keeps the arc's ends inside the seat's own +/-35 mm window.
    """
    return max(float(clearance), ROUND_FLANGE_RADIUS + WEAVE_CABLE_RADIUS + WRAP_PAD_CLEAR)
PEG_RADIUS = float(CABLE_PEG.feature("shaft").payload["radius"])
"""How far out the wrap has to stand off, asked of the peg [m].

The number was written here as ``0.013`` and again in the asset that declares
the peg. It sets ``ring_radius`` and so the whole wrap trajectory, which makes it
exactly the kind of copy that is fine until somebody re-scales a peg and only one
of the two moves -- and the one that would not move is the one steering the arm.
"""

LIFT = 0.08
"""Pick-and-place carry height above the cable's rest [m]. The reference
scripts 0.15, but with the tool vertical this arm's reachable envelope tops
out below that over the far table: the elevated waypoints solve 76 mm short
(the reference's soft per-frame tracking simply sagged through the same
limit without anyone measuring). 80 mm keeps the rod well clear of the table
and every waypoint inside millimetres."""
WRAP_DZ = 0.04
"""Wrap carry height above the cable's rest [m], the reference's own. Lower
was tried against the free rod's slip-over-the-tip and lost more than it
bought: at 0.025 the carried bight drags on the table the whole lap and the
cable pivots instead of winding. With the hold latch carrying the end, the
reference height lays the loop onto the peg the way its script intends."""
WRAP_RADIUS = 0.055
"""Gripper orbit clearance beyond the peg surface [m]."""
WRAP_TURNS = 3.0
"""Scripted laps. The reference scripts 2.0, which nets ~1 retained turn while
held -- and springs back to a quarter turn within two seconds of release: one
lap around a 13 mm peg has no capstan margin against this rod's bend energy.
A third lap is what buys self-locking retention, and it is the reference's own
--wrap-turns parameter turned, not new mechanics."""
RING_POINTS = 16
"""Waypoints per lap."""


def pickplace_rows(grasp_y: float = 0.0, grip_open: float = REFERENCE_GRIP_OPEN) -> np.ndarray:
    """The reference's 11 rows: approach, close, lift, out-and-back, lay, release.

    ``[duration, dz, grip, dx, dy]`` -- offsets from the grasp base
    ``(0.58, grasp_y, surface + radius)``.
    """
    del grasp_y  # the base carries it; rows are offsets
    go, gc = float(grip_open), GRIP_CLOSE
    return np.array(
        [
            [1.2, LIFT, go, 0.0, 0.0],  # approach above the cable
            [0.9, 0.0, go, 0.0, 0.0],  # descend
            [0.6, 0.0, go, 0.0, 0.0],  # settle
            [0.6, 0.0, gc, 0.0, 0.0],  # close
            [0.5, 0.0, gc, 0.0, 0.0],  # firm hold
            [1.4, LIFT, gc, 0.0, 0.0],  # lift
            [2.2, LIFT, gc, 0.05, -0.10],  # move out
            [2.2, LIFT, gc, 0.0, 0.12],  # move to the goal lane
            [1.0, 0.0, gc, 0.0, 0.12],  # lower
            [0.8, 0.0, go, 0.0, 0.12],  # release
            [0.8, LIFT, go, 0.0, 0.12],  # retract
        ],
        dtype=np.float64,
    )


@dataclass
class WrapPlan:
    """The wrap's derived geometry: where to grasp, and the ring."""

    peg_xy: tuple[float, float] = (0.44, -0.06)
    bend_y: float = -0.18
    cable_z: float = 0.761
    cable_x: float = 0.58

    ring_start_angle: float = field(init=False)
    grasp_yaw: float = field(init=False)

    def __post_init__(self) -> None:
        self.ring_start_angle = float(np.arctan2(self.bend_y - self.peg_xy[1], self.cable_x - self.peg_xy[0]))
        # After the grasp keys the wrist keeps the yaw the straight cable
        # implied: closing across a +y-running cable is yaw 0.
        self.grasp_yaw = 0.0

    @property
    def grasp_xy(self) -> tuple[float, float]:
        return (self.cable_x, self.bend_y)

    @property
    def ring_radius(self) -> float:
        return WRAP_RADIUS + PEG_RADIUS

    def ring_point(self, angle: float) -> tuple[float, float]:
        return (
            float(self.peg_xy[0] + self.ring_radius * np.cos(angle)),
            float(self.peg_xy[1] + self.ring_radius * np.sin(angle)),
        )


def wrap_rows(plan: WrapPlan, grip_open: float = REFERENCE_GRIP_OPEN) -> np.ndarray:
    """The reference's wrap: grasp at the bend, carry to the ring, two laps,
    settle, release, retract.

    ``[duration, x, y, dz, grip]`` -- absolute xy, dz above the cable's rest
    height. The first three rows' xy is overridden live by the cable detection;
    the ring is ``WRAP_TURNS`` laps of ``RING_POINTS`` waypoints about the peg
    at radius ``wrap_radius + peg_radius``.
    """
    go, gc = float(grip_open), GRIP_CLOSE
    bx, by = plan.grasp_xy
    rows = [
        [1.2, bx, by, 0.08, go],  # approach above the bend (live-detected)
        [0.9, bx, by, 0.0, go],  # descend (live)
        [0.5, bx, by, 0.0, go],  # settle: the close fires on a quiet arm
        [0.6, bx, by, 0.0, gc],  # close (live)
        [0.5, bx, by, WRAP_DZ, gc],  # lift slightly
    ]
    x0, y0 = plan.ring_point(plan.ring_start_angle)
    rows.append([1.4, x0, y0, WRAP_DZ, gc])  # carry onto the ring
    total = int(RING_POINTS * WRAP_TURNS)
    for k in range(1, total + 1):
        angle = plan.ring_start_angle + 2.0 * np.pi * WRAP_TURNS * (k / total)
        x, y = plan.ring_point(angle)
        rows.append([2.6 / RING_POINTS + 0.03, x, y, WRAP_DZ, gc])
    rows.append([0.8, x0, y0, 0.0, gc])  # settle onto the table
    rows.append([1.0, x0, y0, 0.02, go])  # release
    rows.append([0.8, bx, by, 0.15, go])  # retract
    return np.array(rows, dtype=np.float64)


def peg_xy_of(env, name: str = "peg0") -> tuple[float, float]:
    """Where a peg stands, off its own handle rather than a copied constant.

    A welded static has no ``body_q`` row, so the compiler resolves its world
    transform once and hangs it on the handle -- which is the same fact the goal
    reads when it measures a winding about ``peg0.axis``. Asking for it here is
    what keeps the script and the score aimed at one peg.
    """
    return tuple(float(v) for v in env.handles.statics[name].xform[:2])  # type: ignore[return-value]


def station_xy(env) -> list[tuple[float, float]]:
    """Every station of this weave, in order, off the compiled scene.

    **The scene, not the layout table.** ``WEAVE_LAYOUTS`` is where a variant's
    stations are *declared*, and for the fixed arrangements the two are the same
    number. They are not the same number for a board variant: its stations are
    drawn per episode, and the table holds the board the level was read off
    rather than the one this episode put down. A plan made from the table on
    such an episode aims at pegs that are 30 mm away and reports nothing wrong.

    So the privileged answer is the compiled scene's, which is also the one the
    goal scores against -- the seats hang off these same handles. The layout
    table stops being something the routine reads at all, which is the right end
    state regardless: it was only ever a copy of what the scene knows.
    """
    names = [n for n in env.handles.statics if n.startswith("peg") and n[3:].isdigit()]
    # Numeric, not lexical: `peg10` sorts before `peg2`, and a route walked in
    # that order is a different route that still builds and still scores.
    return [peg_xy_of(env, name) for name in sorted(names, key=lambda n: int(n[3:]))]


def plan_stages(env, mode: str = "") -> dict:
    """Solve the keyframe schedule into per-stage joint endpoints.

    Shared by the routine (which interpolates them live) and the plan tool
    (which returns them, densified, for a graph to stream) -- one solve, one
    motion, two drivers.

    Args:
        env: The live environment.
        mode: Which script to run. Empty takes the variant's own name, which is
            what ``metadata["mode"]`` held and duplicated: the two variants are
            called ``pickplace`` and ``wrap``, and the mode was those strings.

    Returns ``{"stages": [(name, q_start, q_end, duration, grip_cmd)],
    "grips": {...}, "grasp_yaw": float}`` with every q a full ``joint_q``.
    """
    # Asked of the document, which states it in `(:policy-params (mode ...))`.
    # The variant name is the fallback and happens to agree -- the two variants
    # are called `pickplace` and `wrap` and the modes were those strings -- but
    # agreeing by coincidence is what this stops relying on.
    mode = mode or policy_param(env, "mode", env.task.name)
    cable = env.scene.by_name[env.task.deformables[0].name]
    surface_z = float(env.handles.surface_z)
    ik = env.arm(env.handles.work_arm)

    if mode == "wrap":
        plan = WrapPlan(
            peg_xy=peg_xy_of(env),
            bend_y=float(policy_param(env, "grasp-y", GRASP_Y["wrap"])),
            cable_z=surface_z + float(env.task.deformables[0].radius),
        )
        rows = wrap_rows(plan, env.handles.arm("work").gripper_open)
        positions = cable.positions(env.state.body_q.numpy())
        nominal = np.asarray(plan.grasp_xy)
        seg = int(np.argmin(np.linalg.norm(positions[:, :2] - nominal, axis=1)))
        point = positions[seg]
        lo, hi = max(0, seg - 1), min(len(positions) - 1, seg + 1)
        tangent = positions[hi, :2] - positions[lo, :2]
        grasp_yaw = float(np.degrees(np.arctan2(tangent[1], tangent[0])) - 90.0)
        # The live grasp: the approach/descend/settle/close rows all target the
        # detected segment, jaws across its tangent.
        rows[:4, 1] = point[0]
        rows[:4, 2] = point[1]
        names = (
            ["approach", "descend", "settle", "close_hold", "hoist", "carry"]
            + [f"ring_{k:02d}" for k in range(len(rows) - 9)]
            + ["lay", "release_hold", "retract"]
        )
        # The *hand*, not the arm: the approach axis and the site rotation this
        # composition reads are the gripper's, since the "a hand is its own axis"
        # split moved them off ArmSpec. The pickplace branch below already asks
        # for the hand; this branch was missed, so every wrap under the scripted
        # routine died on AttributeError before its first step.
        hand = work_gripper(env)
        quats = [grasp_site_quat_legacy(hand, grasp_yaw)] * 5 + [grasp_site_quat_legacy(hand, plan.grasp_yaw)] * (
            len(rows) - 5
        )

        def target_of(row):
            return np.array([row[1], row[2], plan.cable_z + row[3]])

        def grip_of(row):
            return float(row[4])

    else:
        grasp_xy = (0.58, float(policy_param(env, "grasp-y", GRASP_Y["pickplace"])))
        rows = pickplace_rows(grip_open=env.handles.arm("work").gripper_open)
        base = np.array([grasp_xy[0], grasp_xy[1], surface_z + float(env.task.deformables[0].radius)])
        names = [
            "approach",
            "descend",
            "settle",
            "close_hold",
            "firm",
            "lift",
            "move_out",
            "move_goal",
            "lower",
            "release_hold",
            "retract",
        ]
        quats = [grasp_site_quat_legacy(work_gripper(env), 0.0)] * len(rows)
        grasp_yaw = 0.0

        def target_of(row):
            return base + np.array([row[3], row[4], row[1]])

        def grip_of(row):
            return float(row[2])

    # Seed from the declared home, not the live state: at episode 1+ the arm
    # still stands where the last episode left it while its targets drive it
    # home through the settle, and a chained solve seeded from that transient
    # lands in a different elbow branch -- measured as the plan gate refusing
    # episodes the fresh-scene episode passes.
    q_prev = np.asarray(env.state.joint_q.numpy(), dtype=np.float64).copy()
    # Off the handle, not an embodiment module: the strict zip below pairs the
    # posture against this arm's own coordinate count, so an imported constant
    # would be the one thing guaranteed to raise on a different robot.
    home_index = env.handles.arm("work")
    for coord, value in zip(home_index.arm_coords, home_index.home_q, strict=True):
        q_prev[coord] = float(value)
    prev_target: np.ndarray | None = None
    stages = []
    errors: dict[str, float] = {}
    for name, row, quat in zip(names, rows, quats, strict=True):
        target = target_of(row)
        # Chained seeds along a SUBDIVIDED Cartesian line. Joint-space
        # interpolation only pins the endpoints -- between an above-the-grasp
        # via and the grasp itself the lerp bows the hand sideways, and the
        # blade tips (35 mm below the TCP, swinging with the wrist) clipped the
        # rod on the way down: measured as one finger shoved from open to 5 mm
        # and the cable kicked 8 mm before the close. Waypoints every 4 cm keep
        # the bow at millimetres.
        span = 0.0 if prev_target is None else float(np.linalg.norm(target - (prev_target)))
        cuts = max(1, int(np.ceil(span / 0.04)))
        qs = [q_prev.copy()]
        for i in range(1, cuts + 1):
            sub = target if prev_target is None else prev_target + (i / cuts) * (target - prev_target)
            q_prev = np.asarray(ik.solve_q_full(q_prev, sub, quat), dtype=np.float64)
            qs.append(q_prev.copy())
        pos_err, rot_err = ik.pose_errors(q_prev, target, quat)
        # One number per stage, both halves: a plan can score perfect position
        # with the wrist half a turn out, and that must not pass a gate.
        errors[name] = float(pos_err + 0.1 * rot_err)
        stages.append((name, qs, float(row[0]), grip_of(row)))
        prev_target = target

    # The stages where a solve error becomes a physical lie: where the jaws
    # meet the cable, and the first leg that carries it.
    critical = ("descend", "close_hold", "hoist" if mode == "wrap" else "lift")
    return {
        "stages": stages,
        "grips": {
            "open": env.handles.arm("work").gripper_open,
            "close": grip_of(rows[3] if mode != "wrap" else rows[2]),
        },
        "grasp_yaw": grasp_yaw,
        "errors": errors,
        "worst_grasp_error": max(errors[k] for k in critical if k in errors),
    }


class RoutineCableRoute:
    """The scripted route, pickplace or wrap per the variant being run.

    One class for two scripts, as it always was -- what changed is where the
    branch reads from. It used to be ``metadata["mode"]``, an untyped string
    beside a variant already called ``pickplace`` or ``wrap``; a copy of a name
    can disagree with the name, and this one had nowhere to say so.
    """

    def __init__(self, env) -> None:
        self.env = env
        self.mode = env.task.name
        self.plan: dict | None = None
        self.frame = 0
        self._left_hold: np.ndarray | None = None
        self._stage_starts: np.ndarray | None = None
        self._latched = False

    @property
    def control_mode(self) -> str:
        return "joint_position"

    def reset(self) -> None:
        self.plan = None
        self.frame = 0
        self._left_hold = None
        self._latched = False

    def _ensure_plan(self) -> None:
        if self.plan is not None:
            return
        self.plan = plan_stages(self.env, self.mode)
        worst = float(self.plan["worst_grasp_error"])
        if worst > PLAN_ERROR_LIMIT:
            self.env.request_termination(f"route plan unreachable: worst grasp endpoint error {worst * 1000:.0f} mm")
        durations = np.array([s[2] for s in self.plan["stages"]])
        self._stage_starts = np.concatenate([[0.0], np.cumsum(durations)])

    def act(self) -> np.ndarray:
        env = self.env
        if self._left_hold is None:
            index = env.handles.arm("support")
            q = env.state.joint_q.numpy()
            anchor_grip = 0.0
            if env.task.deformables[0].anchor_grasp is not None:
                anchor_grip = 1.0 - float(env.task.deformables[0].anchor_grip) / env.handles.arm("work").gripper_open
            self._left_hold = np.append(np.array([q[c] for c in index.arm_coords], dtype=np.float64), anchor_grip)

        # Stand at home through the settle *before* planning. The plan reads
        # the live rod once, and the rod moves during the settle: the anchor's
        # closing pads swing it ~7 cm about the pinch in the first thirty
        # steps -- measured as the work jaws descending on where the rod used
        # to be, the latch refusing at >5 cm, and the whole wrap orbiting air
        # (0.00 turns, 0 segments engaged). Planned against the settled rod,
        # the grasp lands on it.
        if self.frame < SETTLE_STEPS:
            self.frame += 1
            index = env.handles.arm("work")
            q = env.state.joint_q.numpy()
            right = np.append(np.array([q[c] for c in index.arm_coords], dtype=np.float64), 0.0)
            return np.concatenate([right, self._left_hold]).astype(np.float32)
        self._ensure_plan()

        dt = 1.0 / env.task.physics.control_hz
        total = float(self._stage_starts[-1])
        t = np.clip((self.frame - SETTLE_STEPS) * dt, 0.0, total - 1e-6)
        stage_index = min(int(np.searchsorted(self._stage_starts, t, side="right")) - 1, len(self.plan["stages"]) - 1)
        _name, qs, duration, grip_cmd = self.plan["stages"][stage_index]

        # The wrap's hold: latch the pinched segment once the close has run,
        # drop it at the release. The drag around the peg is what a friction
        # pinch bleeds under; the winding physics stays the rod's own.
        if self.mode == "wrap":
            cable = env.scene.by_name[env.task.deformables[0].name]
            if _name in ("hoist", "carry") and not self._latched:
                if cable.hold_segment(True, env.state) >= 0:
                    self._latched = True
            if _name in ("release_hold", "retract") and self._latched:
                cable.hold_segment(False)
                self._latched = False
        local = float(np.clip((t - self._stage_starts[stage_index]) / max(duration, 1e-6), 0.0, 1.0))
        # Smoothstep the stage clock, then walk the subdivided chain piecewise
        # -- eased at the stage boundaries, straight along the Cartesian line.
        s = local * local * (3.0 - 2.0 * local)
        arc = s * (len(qs) - 1)
        lo = min(int(arc), len(qs) - 2)
        q_cmd = qs[lo] + (arc - lo) * (qs[lo + 1] - qs[lo])

        if self.frame > SETTLE_STEPS + total * env.task.physics.control_hz + 60:
            env.request_termination("route script complete")

        ik = env.arm(env.handles.work_arm)
        right = np.append(
            np.array([q_cmd[c] for c in ik.arm_coords], dtype=np.float64),
            1.0 - float(grip_cmd) / env.handles.arm("work").gripper_open,
        )
        self.frame += 1
        return np.concatenate([right, self._left_hold]).astype(np.float32)

    def get_action(self, obs) -> np.ndarray:
        del obs
        return self.act()


# ---------------------------------------------------------------- the weave
#
# A second solution in this suite's one solution module, because that is where a
# suite's solutions go: under `tasks/`, only `routine.py`, `tools.py` and
# `__init__.py` may be Python, and `tests/test_layout.py` enforces it so
# that "a task is a document" cannot erode one convenience file at a time. This
# arrived as `weave_routine.py` and the gate refused it, correctly.
#
# It shares nothing with RoutineCableRoute but the grip constants: the wrap runs
# a keyframe table, and this streams a plan that is re-solved against the live
# rod between stations.

WEAVE_GRIP_STEPS = 24
"""Steps held at a stage boundary for the jaws to travel.

A finger command is a target, not a state: stepping straight into the carry
leaves the pads still opening, and a latch taken then records the rod where it
is about to not be.
"""

WEAVE_SETTLE_STEPS = 60
"""Steps stood at home before the first plan, and again after the last leg.

The rod settles from its spawn drop in well under a second, and a plan made
against it mid-bounce records a grasp x the settled rod no longer occupies.
The same dwell closes the run: the verdict wants the rod at rest, not the
frame the hand let go.
"""

WEAVE_CLOSED = ("move", "cross", "place")
"""The legs the rod is carried through. Everything else runs with open jaws."""

PAD_FLOOR_CLEAR = 0.0
"""Air left between the pad tips and the tabletop at the weave's lowest legs [m].

**Zero, and the zero is the grasp.** A rod lying on the bench has its widest
point half a diameter up, so a pinch that only reaches the tabletop has the
whole rod between its pads and one that stops 2 mm short meets it above the
equator -- where the pad normals point *down* the rod's shoulder and the
squeeze ejects what it was meant to hold. On this 10 mm rod the 2 mm cost the
lower 40% of the section: measured with the jaws closing at
``surface + 12.8 mm`` (pads bottoming at 2.4 mm, rod centre at 5) against
``surface + 10.4 mm``, which is the pose the pinch actually carries a rod
from -- 12 contacts, dragged 46 mm and lifted 109 mm under friction alone.

The clearance was a guard against a fight that is no longer there: with the
blades' table pairs re-admitted into the main pipeline (see
``robosimstudio.scene.rod_solvers.filtered_pairs``) the tips ground on the tabletop and the
arm stalled on them, so leaving air was the only way to keep a grasp leg out
of a servo saturation. Filtered as they were always declared to be, the pad
tips pass through the surface the rod rests on and this floor is free to sit
where the rod is."""

PLAN_ERROR_LIMIT = 0.025
"""Endpoint residual above which a plan is refused rather than streamed [m,
with rotation folded in at a tenth].

The wrap plan's own gate, spelled once: it was a bare ``0.025`` in the wrap
routine and grew two more copies when the weave gained its gates, and a
threshold retuned in one copy and not another is how the next aim bug streams
anyway. The weave graph path still applies no gate at all -- ``run_weave.py``
streams whatever ``sim.plan_station`` returns -- which is recorded there."""

WEAVE_MAX_STEP_RAD = 0.05
"""Largest joint move a weave leg may command between adjacent frames [rad].

3 rad/s at the 60 Hz stream -- a brisk arm, nowhere near the 2.9-per-frame a
basin flip smuggled into the cross leg. Only a leg containing a flip pays:
below this, resampling is skipped and the leg is exactly the solve's own."""

WEAVE_PINCH_ENGAGE = 0.0005
"""How far each pad presses into the weave rod when closed [m].

:func:`weave_grip_close` derives the close command from the task's own rod
rather than reusing ``GRIP_CLOSE``, and this is the engagement it aims for.
``GRIP_CLOSE`` is 3 mm -- a 6 mm pad gap, calibrated against the wrap's 22 mm
reference rod -- and on the weave's 10 mm rod that is a 2 mm-per-side crush:
the squeeze reaction through the 20x proxies (~80 N, the anchor's own
measurement) convulsed the arm the whole time it actually held the rod --
20-35 rad/s joint spikes through the cross and place legs, measured on
station 0. The same constant-vs-spec mismatch ``hold_gains`` documents for
the latch, wearing a different symptom. Half a millimetre a side keeps firm
contact for the latch to take over from without turning the carry into a
fight."""


def weave_grip_close(env) -> float:
    """The weave's close command [m]: pads at the rod's diameter less the engagement.

    The finger command is per-finger travel, so the pad gap is twice it -- the
    anchor's calibration (a 10 mm command leaving ~1.5 mm a side on a 22 mm
    rod) is the measurement behind that factor. Floored at ``GRIP_CLOSE`` so a
    rod thinner than the reference pinch never commands past it.
    """
    radius = float(env.task.deformables[0].radius)
    return max(GRIP_CLOSE, radius - WEAVE_PINCH_ENGAGE)


def pad_tip_drop(env, arm: str = "") -> float:
    """How far the pad tips reach past the grasp site, along the approach [m].

    Measured off that hand's own finger colliders in the grasp-site frame, at
    whatever pose the arm currently holds -- the fingers of a linear jaw
    translate across the approach axis, so the projection is pose- and
    opening-invariant. 10.7 mm on the i2rt hand.

    Args:
        arm: Which arm to measure. Empty means the work arm, which is every
            caller but the anchor -- and the anchor is the reason this takes a
            name at all: both YAM hands are the same gripper, but reading one
            arm's colliders to place the *other* one's pads is the class of
            copy this suite keeps finding.

    This number is why the weave's low legs carry a floor. The grasp site sits
    at the pad nub field (``GRASP_SITE_OFFSET``'s own measurement), so a site
    commanded to ``surface + 5 mm`` demands the tips 5 mm *under* the table.
    While the arm-vs-table pair belonged to no solver that was invisible; with
    the table solid it is a fight the servo cannot win and does not stop
    fighting -- measured on weave3 station 0 as 28 rad/s joint spikes, 90 mm
    single-step TCP jumps, the site pressed 2.5 mm below the tabletop, and the
    rod batted at 8 m/s by pads grinding past it.
    """
    name = arm or env.handles.work_arm
    ik = env.arm(name)
    hand = env.scene.embodiment.hands.get(name) or work_gripper(env)
    site_pos, site_quat = ik.ee_pose(env.state)
    # The direction the tips extend, expressed in the site's own frame: in the
    # planner's top-down grasp orientation the tips point along world down, so
    # world down rotated into *that* orientation's site frame is a constant of
    # the hand -- readable at whatever pose the arm holds right now.
    down_in_site = quat_to_mat(np.asarray(grasp_site_quat_legacy(hand, 0.0), dtype=np.float64)).T @ np.array(
        [0.0, 0.0, -1.0]
    )
    live_rot = quat_to_mat(np.asarray(site_quat, dtype=np.float64))

    model = env.model
    body_q = env.state.body_q.numpy()
    shape_body = model.shape_body.numpy()
    shape_type = model.shape_type.numpy()
    shape_scale = model.shape_scale.numpy()
    shape_xform = model.shape_transform.numpy()
    arm_handle = env.handles.arms[name]
    finger_bodies = {int(b) for side in arm_handle.finger_groups for b, _ in side}

    from robosimstudio.core.math import transform_matrix  # noqa: PLC0415

    drop = 0.0
    for s in range(model.shape_count):
        if int(shape_body[s]) not in finger_bodies:
            continue
        geometry = shape_geometry(model.shape_source[s], int(shape_type[s]), shape_scale[s])
        if geometry is None:
            continue
        vertices, _ = geometry
        world = transform_matrix(body_q[int(shape_body[s])]) @ transform_matrix(shape_xform[s])
        points = vertices @ world[:3, :3].T + world[:3, 3]
        in_site = (points - site_pos) @ live_rot
        drop = max(drop, float((in_site @ down_in_site).max()))
    return drop


def _leg_rows(ik, quat, q_prev: np.ndarray, start, end, duration: float, stream_hz: float):
    """One Cartesian leg for ``ik``'s arm, as streamed joint rows.

    The weave's leg machinery, shared by the station planner and the support
    hand's own legs so the two arms move as one method. Returns
    ``(rows, q_prev, error)``: an ``[n, coords]`` array to stream one row per
    control period, the full joint vector the chain ended on (which seeds the
    next leg), and the endpoint residual ``pos + 0.1 * rot`` the caller gates
    against :data:`PLAN_ERROR_LIMIT`.

    Solved at 4 cm CUT POINTS and smoothstepped as a JOINT chain to the stream
    -- ``plan_stages``' own structure, adopted after measuring what solving
    every stream frame actually did. Two things, both bad. Each per-frame solve
    paid solve_q's full budget whether or not it had converged (the early exit
    cannot fire before the axis term switches on), so a station cost 157-485 s
    of planning against 4.5 s this way. And the frame-by-frame chain had no
    recovery: one solve escaping its basin poisoned every seed after it,
    measured as the far-side legs ending 286-727 mm off target with ``place``
    aimed 15 cm below the tabletop -- while the same targets solved as cut
    points converge to 0.1 mm. Fewer, better-seeded solves are both the speed
    and the geometry fix; the interpolation between them bows the hand by
    millimetres at this cut length, which is the same trade the wrap's plan
    already makes and documents.

    The host walk is safe *here* because these solves are re-made against the
    live rod every station -- there are no tuned keyframes to hold a particular
    device-path fixed point (the wrap's plan stays outside the scope for
    exactly that reason; ``ArmIK.host_fk`` carries the measurement). The
    endpoint residual is measured inside the same scope and returned, because
    every caller gates on it and a ``pose_errors`` call outside the scope pays
    a full-model device FK and a readback per leg on a sim that is
    launch-bound.

    A chained solve has no continuity guarantee: adjacent Cartesian targets can
    land in different elbow basins, and the joint path then jumps between two
    frames 1/30 s apart -- measured on the weave3 layout as 2.9 rad inside the
    cross leg and 3.7 inside the retreat, streamed as a whip the servo chases
    at 20-41 rad/s. The flip itself is a fact of the workspace; what is
    repairable is the timing, so a leg that contains one is resampled by
    joint-space arc length at a bounded step per frame: same path, same
    endpoints, the reconfiguration spread over the frames it needs instead of
    one.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    steps = max(2, int(stream_hz * duration))
    cuts = max(1, int(np.ceil(float(np.linalg.norm(end - start)) / 0.04)))
    qs = [q_prev.copy()]
    with ik.host_fk():
        for i in range(1, cuts + 1):
            point = start + (i / cuts) * (end - start)
            q_prev = np.asarray(ik.solve_q_full(q_prev, point, quat), dtype=np.float64)
            qs.append(q_prev.copy())
        pos_err, rot_err = ik.pose_errors(q_prev, end, quat)
    error = float(pos_err + 0.1 * rot_err)
    rows = []
    for i in range(steps):
        t = i / (steps - 1)
        s = t * t * (3.0 - 2.0 * t)
        arc_pos = s * (len(qs) - 1)
        lo = min(int(arc_pos), len(qs) - 2)
        q = qs[lo] + (arc_pos - lo) * (qs[lo + 1] - qs[lo])
        rows.append([float(q[c]) for c in ik.arm_coords])
    q_rows = np.asarray(rows, dtype=np.float64)
    arc = np.abs(np.diff(q_rows, axis=0)).max(axis=1)
    if len(arc) and float(arc.max()) > WEAVE_MAX_STEP_RAD:
        s_cum = np.concatenate([[0.0], np.cumsum(arc)])
        frames = max(len(q_rows), int(np.ceil(s_cum[-1] / WEAVE_MAX_STEP_RAD)) + 1)
        grid = np.linspace(0.0, float(s_cum[-1]), frames)
        q_rows = np.stack([np.interp(grid, s_cum, q_rows[:, j]) for j in range(q_rows.shape[1])], axis=1)
    return q_rows, q_prev, error


def plan_station_stages(
    env,
    index: int = 0,
    carry_clear: float = 0.030,
    approach: float = 0.055,
    scene: dict[str, Any] | None = None,
    seat_bias: float | None = None,
    max_offset: float | None = None,
    slack_margin: float | None = None,
    final_pass: bool | None = None,
    _grasp_choice: int = 0,
) -> dict[str, Any]:
    """One weave station as joint-space stage trajectories, solved live.

    The planning core of ``sim.plan_station`` -- the physics of a crossing,
    the anchored-end ordering and the reason a station is planned only when it
    is reached are documented on ``ToolsCable.plan_station`` in ``tools.py``, which wraps this in a bus
    call. :class:`RoutineWeave` is already on the stepping thread and calls it
    directly, the same split :func:`plan_stages` has with ``sim.plan_route``.

    Args:
        env: The environment; its task must be a weave variant.
        index: Which station of the execution order to plan.
        carry_clear: How far above the spool tops the rod is carried [m].
        approach: How far above the rod the hand stages before descending [m].
        scene: Where the stations, their sides, the scored seat abscissae and
            the rod are, when something other than the simulator supplies them
            -- the keys ``stations``, ``sides``, ``seats_x`` and ``rod``. The
            default reads all four out of the scene, which is the privileged
            answer and the one every caller took before a perceptual body
            existed.

            **The split is the point, not a convenience.** Everything below this
            block is kinematics and choreography: which template a station runs,
            the pass order, the pad floor, the arc re-timing. None of it is
            privileged, and a policy that had measured the same four things from
            a camera would want every line of it. Keeping them in one function
            with a substitutable front means the vision body inherits the
            choreography rather than reimplementing it -- and that a plan made
            from perceived geometry differs from a privileged one in exactly the
            four numbers it was handed, which is what makes the two comparable
            at all.
    """
    from .assets.layout import (  # noqa: PLC0415
        WEAVE_CABLE_X,
        WEAVE_LAYOUTS,
        WEAVE_SIDES,
        seat_offset,
    )

    variant = env.task.name
    if variant not in WEAVE_LAYOUTS:
        raise RuntimeError(f"plan_station is for the weave variants; this is {variant!r}")

    # THE RATE THE WAYPOINTS ARE STREAMED AT, which is the cell's control rate
    # and not :data:`PLAN_HZ`.
    #
    # `RoutineWeave` consumes one waypoint per control period, so the rate a
    # leg is densified at *is* the speed it runs at. Densified at PLAN_HZ's 30
    # and streamed at this cell's 60, every leg ran in half its stated duration
    # -- a 55 mm descent given 0.4 s took 0.2 s, and the servo arrived 20 mm
    # short of the rod. The function already assumed the stream rate elsewhere
    # and assumed 60: WEAVE_MAX_STEP_RAD's 0.05 rad per frame is quoted as
    # "3 rad/s at the 60 Hz stream", which is only true at 60. So the two
    # halves of one function disagreed about how fast their own output moves.
    #
    # The module docstring states the intent -- "smoothsteps between the
    # endpoint joint configurations **at the control rate**" -- and the wrap
    # routine beside this one reads `physics.control_hz` for its own clock.
    # The graph path needs the same rate, and the claim that stood here -- that
    # it "converges on each waypoint in turn rather than clocking them out" -- is
    # false. `robot.execute_trajectory` takes that per-waypoint path only as a
    # FALLBACK, when the facade exposes no `stream_joint_trajectory`; this one
    # does, so the graph streams one waypoint per control step exactly as this
    # routine does. Measured on usbc_insert, which held the same PLAN_HZ on the
    # strength of that belief and ran its whole schedule at 2x.
    stream_hz = _stream_hz(env.task.physics)

    # The four substitutable numbers, resolved once. A body that supplies some
    # but not all of them is answered per key rather than all-or-nothing: the
    # spools are rigid and easy to see, the rod is neither, and a bring-up that
    # perceives the posts while still reading the rod is a real intermediate
    # state rather than a mistake.
    scene = dict(scene or {})
    stations = [tuple(float(v) for v in st) for st in scene["stations"]] if "stations" in scene else station_xy(env)
    sides = (
        [int(v) for v in scene["sides"]]
        if "sides" in scene
        else [WEAVE_SIDES[k % len(WEAVE_SIDES)] for k in range(len(stations))]
    )
    if len(sides) != len(stations):
        raise ValueError(f"{len(stations)} stations but {len(sides)} sides; a weave needs one side per station")
    ik = env.arm(env.handles.work_arm)
    hand = work_gripper(env)
    # Jaws across the rod: it runs along +y, so closing along x is yaw 0
    # -- pickplace's own grasp, on a rod laid the same way.
    quat = grasp_site_quat_legacy(hand, 0.0)

    surface_z = float(env.handles.surface_z)
    # The height that puts the PADS AROUND THE ROD, which is what a pinch is.
    # The site rides at the nub field and the tips ride `pad_drop` below it, so
    # a site at ``surface + pad_drop`` bottoms the tips on the tabletop the rod
    # rests on and the whole 10 mm section is between the blades. Commanded at
    # the rod's own centre instead the tips would be half a diameter *under*
    # the table -- which is why the blades' table pairs are filtered (see
    # ``robosimstudio.scene.rod_solvers.filtered_pairs``) rather than why the hand should stay
    # high: 2 mm of clearance here met the rod above its equator and the
    # squeeze pushed it out. See :data:`PAD_FLOOR_CLEAR`.
    pad_drop = pad_tip_drop(env)
    rest_z = max(surface_z + WEAVE_CABLE_RADIUS, surface_z + pad_drop + PAD_FLOOR_CLEAR)
    carry_z = surface_z + ROUND_HEIGHT + carry_clear
    # This board's own seat, not the spool's: the panel stands on 9.5 mm cleats
    # where the F1's groove is 4.75, and a feed detour measured against the
    # wrong one is 4.75 mm off at every vertex. See ``seat_offset``.
    seat = seat_offset(variant)

    # `solve_q_full`, not the LM solver `plan_insert` steps: a cable
    # scene's model holds rod joints, the LM kernel specialisation on it
    # is pathological, and ArmIK builds no `solver` at all. This is the
    # same full-pose call the wrap's own plan makes -- yaw authority
    # included, which a grasp across a rod cannot trade away.
    q_prev = np.asarray(env.state.joint_q.numpy(), dtype=np.float64)
    stages: list[dict[str, Any]] = []
    errors: dict[str, float] = {}

    def leg(name: str, start, end, duration: float, grip: float) -> None:
        # The machinery lives in :func:`_leg_rows` -- the cut-point solve, the
        # host FK walk, the basin-flip resample and the measurements behind all
        # three -- because the support hand's legs are solved by the same
        # method, and two copies of a motion planner is how two arms end up
        # moving differently.
        nonlocal q_prev
        # The residual per leg, the same figure plan_stages gates its grasp
        # on. A solve that cannot reach its target does not error -- it
        # returns its best basin and the chain streams it -- so the only
        # honest reading is to measure where the chain ended. The aim bug
        # this arrived with left 349 mm here and nothing looked.
        # MOTION_SCALE, applied once and here rather than at each call: the
        # durations below are the shape of the schedule and the scale is how
        # fast the arm runs it. Nothing else about a leg changes.
        q_rows, q_prev, residual = _leg_rows(ik, quat, q_prev, start, end, duration * MOTION_SCALE, stream_hz)
        # WORST, not last. ``errors`` is keyed by stage name and the wrap emits
        # nine legs all called "cross", so a plain assignment reports only the
        # ninth -- and the whole point of this dict is that
        # ``worst_grasp_error`` sees every leg that was streamed.
        errors[name] = max(residual, errors.get(name, 0.0))
        stages.append(
            {
                "stage": name,
                "trajectory": {"waypoints": [{"positions": [float(v) for v in row]} for row in q_rows]},
                "duration": float(len(q_rows) / stream_hz),
                "grip_command_m": float(grip),
            }
        )

    # ONE STATION, SOLVED AGAINST THE LIVE ROD, and that is the whole
    # reason this is per station rather than one plan up front.
    #
    # A plan made once puts every grasp at the rod's *starting* x. The
    # first gather moves the rod, so the second grasp descends where the
    # rod used to be and the latch refuses -- measured, `hold_segment`
    # returning -1 at stations two and three while the hand closed on
    # nothing and dragged nothing. The rod's position is an outcome of
    # the previous station, so it has to be read, not predicted.
    #
    # A station tidies with a DRAG along the table or crosses with a lift
    # over the spool; which one is decided against the live rod at the
    # template branch below, not by the caller and -- since weave4's middle
    # seat -- not by the side alone.
    # FAR-SIDE CROSSINGS FIRST, near-side seats last, each pass from
    # the anchored end outwards.
    #
    # The last station worked is the one that survives, and the passes
    # disturb each other in one direction only: a far-side crossing
    # drags the whole rod forward, which pushes a near-side seat done
    # earlier straight out the far edge of its own box (measured: 0.533
    # against a seat ending at 0.517). A near-side drag does not return
    # the favour -- it pulls slack along the table without lifting
    # anything over a spool.
    #
    # So the destructive pass goes first and the tidying pass goes last,
    # which is the order a person weaving by hand uses for the same
    # reason.
    by_y = sorted(range(len(stations)), key=lambda i: -stations[i][1])
    order = [i for i in by_y if sides[i] > 0]
    order += [i for i in by_y if sides[i] < 0]
    i = order[int(index) % len(order)]
    sx, sy = stations[i]
    side = sides[i]

    # Where the rod actually is at this station, right now: the nearest
    # segment's own (x, y), and a constant height.
    #
    # The aim used to mix one segment's x with the *station's* y -- an identity
    # while the rod shipped eight times too stiff and ended every episode at
    # ``straightness`` 0.99, and a real miss at the rod's own stiffness (see
    # ``assets/cable.py::weave_bend``): the first crossing genuinely
    # bends it -- straightness 0.13 -- and the two near-side stations then
    # descended 48 and 65 mm from the nearest segment and closed on air, 0.1 to
    # 5.7 contacts a step against the far-side crossing's 8 to 18.
    #
    # **Grasping the segment's own (x, y) is half of one change, and the other
    # half is the support hand (measured 2026-08-11).** Fixed alone, all three
    # stations hold the rod -- 6.9 to 21 contacts a leg, the rod 6 to 12 mm off
    # the TCP -- and weave3 went from 2/3 seats to 0/3, because a station that
    # actually takes the rod also drags the *free* rod bodily across the bench
    # (26 of 42 segments piled at one station, none left at the next) where a
    # station that misses leaves the previous crossing alone. The far end has
    # something to hold it now -- ``RoutineWeave`` opens by taking it with the
    # support hand (:func:`plan_support_stages`) and pays out slack through the
    # carries -- so a grasp that lands is a bend, not a haul. The roster half
    # is written up in ``robosimstudio.scene.rod_solvers.proxy_arms_for``.
    if "rod" in scene:
        # A perceived centreline, in the same (N, 3) world-metre layout the
        # entity publishes. Its N is the fitter's, not the chain's -- nothing
        # below counts segments, it takes the nearest point in y.
        rod = np.asarray(scene["rod"], dtype=np.float64).reshape(-1, 3)
        if len(rod) < 2:
            raise ValueError(f"a perceived rod needs at least two points, got {len(rod)}")
    else:
        rod = env.scene.by_name[env.task.deformables[0].name].positions(env.state.body_q.numpy())
    # AIM AT THE MIDDLE OF THE BOX THE GOAL SCORES, not at its edge.
    #
    # Read off the scene's own region rather than recomputed here, so
    # the plan and the evaluator cannot disagree about where a seat is.
    # Aiming at the inner edge was worth ~20 mm of margin and the rod
    # spent all of it: released against a neighbouring crossing's
    # tension it springs *forward*, and a segment placed on the edge
    # left the box the moment the next station pulled. The centre has
    # 21 mm either side of it, which is more than the largest relaxation
    # measured on this layout.
    #
    # ``center()`` is a WORLD point -- the handle resolves `(:target pegN)`'s
    # frame and adds the box's own middle -- so it is the place entire, not an
    # offset. It was read as an offset and added to the station's x, which
    # aimed every carry at ``2*sx + 24 mm``: x = 1.064 on this bench, half a
    # metre past the peg line, unreachable. The solver left residuals of 70
    # to 349 mm there, the resampler stretched the garbage chain into a whip,
    # and the whip is what actually visited the seats -- the transient
    # [6, 6, 3] mid-fling, the 0/0.33/0.67 lottery, and a rod torn out of
    # both hands. The gate below is what turns the next aim bug into a
    # refusal instead of a lottery.
    # HOW FAR OUT THE BOX TO AIM, on top of that. See :data:`SEAT_BIAS`.
    bias = SEAT_BIAS if seat_bias is None else float(seat_bias)
    if "seats_x" in scene:
        # Perceived: the seat is the post's own centre offset by post plus rod
        # radius on the named side, which is what the box was drawn around. A
        # body that measured the two radii computes the same number the
        # evaluator scores without ever reading the evaluator.
        target_x = float(scene["seats_x"][i]) + side * bias
    else:
        seat_region = env.handles.regions.get(f"seat{i}")
        target_x = (
            float(seat_region.center()[0]) + side * bias
            if seat_region is not None
            else sx + side * (seat + bias)
        )

    # A terminated route's first cable body is welded to the bench. A grasp
    # abreast of its first peg does not contain enough material to make the
    # lateral diagonal, so select the nearest *perceived* point that does.
    cables = getattr(env.task, "deformables", None) or getattr(env.task, "extras", ())
    cable = cables[0]
    fixed_anchor = (
        "start" in tuple(getattr(cable, "anchors", ()))
        or bool(getattr(cable, "pin_start", False))
    )
    if fixed_anchor and side < 0:
        # The fixed start pulls a released near-side bend back toward -x.
        # Land farther inside the scored box so relaxation does not erase it.
        target_x += 2.0 * bias
    clearance = WEAVE_PEG_CLEAR
    if fixed_anchor:
        pitch = min((abs(sy - y) for j, (_x, y) in enumerate(stations) if j != i), default=.12)
        # Keep the held point inside the seat's +/-35 mm longitudinal score
        # window while clearing this peg and staying short of the next one.
        clearance = min(WEAVE_PEG_CLEAR, .4 * pitch)
    work_y = sy + clearance
    anchor_xy = rod[int(np.argmin(rod[:, 1])), :2] if fixed_anchor else None
    effective_margin = (0.020 if slack_margin is None else float(slack_margin)) if fixed_anchor else slack_margin
    pick = choose_grasp(
        rod,
        (sx, work_y),
        (target_x, work_y),
        anchor_xy=anchor_xy,
        max_offset=(0.120 if fixed_anchor else MAX_STATION_OFFSET)
        if max_offset is None
        else float(max_offset),
        slack_margin=effective_margin,
    )
    print(f"[grasp] station={i} {'take' if pick.ok else 'REFUSE'}: {pick.reason}", flush=True)
    if fixed_anchor and not pick.ok:
        raise ValueError(f"No feasible grasp relative to the fixed anchor: {pick.reason}")
    grasp = np.array([float(pick.point[0]), float(pick.point[1]), rest_z])
    if fixed_anchor:
        quat = cable_grasp_quats(hand, cable_tangent(rod, grasp[:2]))[_grasp_choice]
    material_s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(rod, axis=0), axis=1))]
    support_index = int(np.argmax(rod[:, 1]))
    support_material = abs(float(material_s[pick.index] - material_s[support_index]))
    # The motion template is a property of the cable abreast of the post, not
    # of the inboard point selected to borrow slack.
    nearest = rod[int(np.argmin(np.abs(rod[:, 1] - sy)))]

    over = grasp + np.array([0.0, 0.0, approach])
    close = weave_grip_close(env)
    # From where the hand actually stands to the station's staging point. The
    # reach leg used to hold at `over` alone, so a station opened with a step
    # input -- the whole inter-station transit compressed into one frame,
    # measured as 72 mm single-step TCP jumps at every entry. A leg that
    # starts at the live pose makes the transit a motion instead.
    here, _ = ik.ee_pose(env.state)
    leg("enter", np.asarray(here, dtype=np.float64), over, 0.6, REFERENCE_GRIP_OPEN)
    # Which template a station runs is the ROD's question, not the side's.
    # A station whose rod already lies on its required side is a tidy -- a
    # drag at rest height, because a rod lifted clear of the table swings,
    # and a swinging rod lands somewhere the plan did not choose. A station
    # whose rod lies on the WRONG side must cross its own spool, and there
    # is no dragging through a fixture: weave4's middle near seat sits
    # between its two far crossings, the span between them runs east of the
    # peg line, and the rest-height drag jammed that span against the spool
    # flange and stripped the pinch 25 mm short of the box (contacts
    # 11 -> 6, seat2 0/42 for the rest of the episode). The lift-over is
    # the template the far side has always used; only the target x differs.
    # Keying it on the side alone was an identity on weave3 -- every
    # near-side span there starts and stays west -- and weave4's middle
    # seat is where the identity breaks.
    # THE WRAP, shared by both templates below.
    #
    # Both used to finish the same way -- put the held point down at the seat's
    # x, 30 mm along the run, open, and hope the span drooped into the box. It
    # is why a Medium seat was never MARGINAL: measured over 8 episodes, the 13
    # failed seats read 0.0000 twelve times and 0.0164 once, against a 0.0333
    # threshold, while the ones that took read 0.049-0.098. The cable was either
    # laid round the post or nowhere near it, and which of the two was not
    # something the place point decided.
    #
    # So the hand now carries the span AROUND the required face instead
    # (:func:`wrap_arc`), at a radius that keeps the pads outside the flange the
    # seat itself sits inside. ``place`` becomes the dwell at the far end of that
    # arc, which is the stage the seat skill holds and measures against.
    _arc_end: list = []

    def _wrap(entry) -> None:
        arc = wrap_arc((sx, sy), side, wrap_radius(clearance), rest_z)
        above = np.array([arc[0][0], arc[0][1], float(entry[2])])
        leg("cross", np.asarray(entry, dtype=np.float64), above, 0.4, close)
        leg("cross", above, arc[0], 0.3, close)
        for k in range(1, len(arc)):
            leg("cross", arc[k - 1], arc[k], WRAP_LEG_S, close)
        # A dwell, not a motion: the span is already round the post, and what
        # this buys is a stage ``seat_station`` can hold, measure and report on.
        leg("place", arc[-1], arc[-1], 0.35, close)
        _arc_end.append(arc[-1])

    rod_side = 1 if float(nearest[0]) > sx else -1
    if rod_side == side:
        place = np.array([target_x, work_y, rest_z])
        leg("reach", over, over, 0.4, REFERENCE_GRIP_OPEN)
        leg("take", over, grasp, 0.4, REFERENCE_GRIP_OPEN)
        leg("move", grasp, place, 0.7, close)
        _wrap(place)
        # A PLACE LEG, so the seat skill has something to run. The drag template
        # ended at `move` and went straight to `clear`, so `seat` -- which asks
        # for the "place" stage -- was a NO-OP on every dragged station and two
        # of three crossings were never seated or checked at all. Measured: the
        # node printed "nothing to place" for stations 1 and 2 of every episode.
        #
        # It is a dwell rather than a motion: the rod is already at the target,
        # and what this stage buys is a stage the seat skill can hold, measure
        # and correct against.
        leg("clear", _arc_end[0], _arc_end[0] + np.array([0.0, 0.0, approach]), 0.4, REFERENCE_GRIP_OPEN)
    else:
        place_x = target_x
        up = np.array([grasp[0], work_y, carry_z])
        across = np.array([place_x, work_y, carry_z])
        place = np.array([place_x, work_y, rest_z])
        leg("reach", over, over, 0.4, REFERENCE_GRIP_OPEN)
        leg("take", over, grasp, 0.4, REFERENCE_GRIP_OPEN)
        leg("move", grasp, up, 0.5, close)
        leg("cross", up, across, 0.6, close)
        _wrap(across)
        leg("clear", _arc_end[0], np.array([_arc_end[0][0], _arc_end[0][1], carry_z]),
            0.4, REFERENCE_GRIP_OPEN)

    # Out of the frame, and ONLY after the last station.
    #
    # The goal wants the hand 50 mm off the nearest segment, and a hand parked
    # over the weave is not out of it -- but that is a thing to do once, at the
    # end. Run on every pass it is 0.8 s of travel out and back between
    # crossings, and worse than wasted: the hand crosses the rod it just placed
    # on the way back in. It also opened from ``stations[order[-1]]`` rather
    # than from where the hand stands, so every non-final pass began the leg
    # with a jump to a station it was not at.
    is_final = final_pass if final_pass is not None else int(index) % len(order) == len(order) - 1
    if is_final:
        leg(
            "retreat",
            np.array([target_x, work_y, carry_z]),
            np.array([WEAVE_CABLE_X - 0.12, work_y, carry_z]),
            0.8,
            REFERENCE_GRIP_OPEN,
        )

    result = {
        "variant": variant,
        "stations": len(stations),
        "station": int(i),
        "side": int(side),
        "stages": stages,
        "grips": {
            "open": {"command_m": float(REFERENCE_GRIP_OPEN)},
            "close": {"command_m": float(close)},
        },
        "seat_m": float(seat),
        # Where this station takes the rod and where it lets it go, for the
        # support hand's feed arithmetic: the slack a crossing demands is a
        # fact about these two points and the pinch, not about the leg list.
        "grasp_ok": bool(pick.ok),
        "grasp_reason": str(pick.reason),
        "grasp_index": int(pick.index),
        "grasp_slack_m": support_material,
        "fixed_anchor_material_m": float(pick.slack_m) if fixed_anchor else None,
        "grasp_need_m": float(pick.need_m),
        "seat_bias_m": float(bias),
        "grasp_m": [float(v) for v in grasp],
        # WHERE THE HAND ACTUALLY ENDS, which since the wrap is the far end of
        # the arc and not the pre-wrap set-down point. ``place`` is still the
        # target the arc is built around, but the hand leaves the span at
        # ``_arc_end[0]`` -- up to a wrap radius away along the run -- and a
        # support hand that pays cable to the old point under-feeds every carry
        # by that much.
        "place_m": [float(v) for v in _arc_end[0]],
        # The far-side crossings already seated, as the detour vertices the
        # rod's path takes between the pinch and this station. The physical
        # seat -- shaft plus rod radius -- rather than the scored box's
        # middle, because the vertex is where the rod presses the spool.
        # Emitted here because only this function knows the execution order;
        # what lies between the two ends is the feed's own question.
        "feed_via": [
            [float(stations[j][0] + seat), float(stations[j][1]), rest_z]
            for j in order[: int(index) % len(order)]
            if sides[j] > 0
        ],
        "errors": errors,
        # The legs where a residual becomes a physical lie: every one flown
        # with the rod in the jaws, plus the take that puts it there. Derived
        # from WEAVE_CLOSED rather than restated, so a leg added to the
        # carried set cannot escape the gate by omission here.
        "worst_grasp_error": max(errors[k] for k in ("take", *WEAVE_CLOSED) if k in errors),
    }

    # A parallel-jaw grasp has two equivalent orientations. A small change in
    # the observed start joints can put the first one in a failed IK basin.
    # Re-plan the full carry from the current joints with the second orientation;
    # retain the lower residual, and let the caller's existing gate reject both.
    result["grasp_orientation_choice"] = _grasp_choice
    if fixed_anchor and _grasp_choice == 0 and result["worst_grasp_error"] > .010:
        alternative = plan_station_stages(
            env, index=index, carry_clear=carry_clear, approach=approach,
            scene=scene, seat_bias=seat_bias, max_offset=max_offset,
            slack_margin=slack_margin, final_pass=final_pass, _grasp_choice=1,
        )
        if alternative["worst_grasp_error"] < result["worst_grasp_error"]:
            print(f"[grasp] equivalent wrist orientation: "
                  f"{result['worst_grasp_error']*1000:.1f} -> "
                  f"{alternative['worst_grasp_error']*1000:.1f} mm residual", flush=True)
            return alternative
    return result


WEAVE_FEED_MARGIN = 0.015
"""Slack fed beyond the flat-chord deficit, per station [m].

The chord :func:`plan_support_feed` measures is the xy straight line from the
pinch to the place point; the carry actually travels higher -- the rod rides
``carry_z`` over the spools mid-cross, which lengthens the worst instantaneous
chord by ~5 mm on this layout -- and the pinch point sits a few millimetres
off the TCP the arithmetic uses. Fifteen covers both with room; slack is
cheap (a bowed rod on a table) where shortfall is the whole failure."""

WEAVE_FEED_MIN = 0.005
"""Below this owed pay-out the hand stands still [m]. A feed the size of the
solver's own placement noise is not a motion worth streaming."""

WEAVE_FEED_MAX = 0.09
"""The most one station may walk the far hand [m]. A ceiling, not a target:
weave3's first crossing computes ~38 mm. What it guards against is a bad
segment match turning the arithmetic into a lunge across the bench."""

WEAVE_FEED_STANDOFF = 0.03
"""How short of its path's first vertex the feeding hand must stop [m].

The vertex is a spool's seat point -- or the place the work hand occupies
when no vertex stands between -- and the pay-out hand rides at rod height,
below the spool tops. A feed clamped by the ceiling alone could walk the
pinch into the fixture once earlier pay-outs have brought it close; thirty
millimetres clears the 12.5 mm flange with the pad's own reach to spare."""


def _feed_target(a_pos, place, via, arc: float) -> np.ndarray | None:
    """Where the far hand should walk to pay out one station, in xy -- or
    ``None`` where the geometry owes nothing.

    The pure half of :func:`plan_support_feed`, split out because it is the
    arithmetic a regression re-breaks silently and the only part a GPU-free
    test can hold: the y-window that admits committed vias between the two
    ends, the walk order from the pinch toward the station, the path-versus-
    arc deficit against :data:`WEAVE_FEED_MARGIN` and :data:`WEAVE_FEED_MIN`,
    and the two clamps -- :data:`WEAVE_FEED_MAX` for a bad segment match,
    :data:`WEAVE_FEED_STANDOFF` so the fed pinch stops short of whatever its
    path's first vertex is.

    Args:
        a_pos: The pinch, world [3] (z ignored).
        place: The station's place point, world [3].
        via: Committed far-side seat vertices, world [3] each.
        arc: Material rod length between the pinched and grasped segments [m].
    """
    a_pos = np.asarray(a_pos, dtype=np.float64)
    place = np.asarray(place, dtype=np.float64)
    lo, hi = sorted((float(a_pos[1]), float(place[1])))
    pts = sorted(
        (np.asarray(v, dtype=np.float64) for v in via if lo < float(v[1]) < hi),
        key=lambda v: -float(v[1]),
    )
    if float(a_pos[1]) < float(place[1]):
        pts.reverse()
    points = [a_pos, *pts, place]
    path = float(sum(np.linalg.norm(b[:2] - a[:2]) for a, b in itertools.pairwise(points)))
    if path < 1e-6:
        return None
    deficit = path + WEAVE_FEED_MARGIN - float(arc)
    if deficit < WEAVE_FEED_MIN:
        return None
    first = points[1]
    span = float(np.linalg.norm(first[:2] - a_pos[:2]))
    feed = min(deficit, WEAVE_FEED_MAX, span - WEAVE_FEED_STANDOFF)
    if feed < WEAVE_FEED_MIN:
        return None
    return a_pos[:2] + feed * (first[:2] - a_pos[:2]) / max(span, 1e-6)


def plan_support_stages(
    env,
    approach: float = 0.055,
    rod: Any = None,
    grasp_xy: Any = None,
    grasp_axis_z: float | None = None,
    grasp_tangent: Any = None,
    arm: str = "",
) -> dict[str, Any]:
    """The support hand's opening move: take the rod's far end, to hold and pay out.

    The other half of the bimanual weave ``robosimstudio.scene.rod_solvers.proxy_arms_for``
    starts, and the move ``assets/cable.py::weave_cable`` promised when
    it dropped the pre-posed anchor: taking the far end is a thing the *policy*
    does, on its own first move. Without it the rod cannot be woven at all --
    a free rod answers a sideways pull by translating, not bending, and the
    first crossing hauled 26 of 42 segments to one station while the next had
    none left to grasp.

    The grasp recipe is the station planner's own, on the other arm: the
    segment nearest :func:`~cable_route.assets.layout.weave_anchor_y` at its own
    (x, y), the height that puts the pads around the rod measured off *this*
    hand's colliders (``pad_tip_drop`` takes the arm name for exactly this
    call), jaws across the rod, the same leg solver. A contact pinch at the
    same half-millimetre engagement as the work hand's -- no latch: the hold
    the score sees has to be the hold the fingers earn.

    Two legs -- enter and take -- because the close is the routine's to command
    and to dwell for, exactly as it is at a station. Returns the stages, the
    grasp, and the solved chain's end ``q_end`` with the grasp ``quat``, which
    seed every feed this hand later solves.
    """
    from .assets.layout import WEAVE_LAYOUTS, weave_anchor_y  # noqa: PLC0415

    variant = env.task.name
    if variant not in WEAVE_LAYOUTS:
        raise RuntimeError(f"plan_support_stages is for the weave variants; this is {variant!r}")
    name = str(arm) or env.handles.support_arm
    if not name:
        raise RuntimeError(f"{variant!r} is bound with no support arm; a weave's far end needs a hand")
    ik = env.arm(name)
    hand = env.scene.embodiment.hands.get(name) or work_gripper(env)
    quat = grasp_site_quat_legacy(hand, 0.0)
    quats = [quat] if grasp_tangent is None else cable_grasp_quats(hand, grasp_tangent)
    stream_hz = _stream_hz(env.task.physics)
    surface_z = float(env.handles.surface_z)
    radius = float(env.task.deformables[0].radius)
    rest_z = max(surface_z + radius, surface_z + pad_tip_drop(env, name) + PAD_FLOOR_CLEAR)
    if grasp_axis_z is not None:
        axis_z = float(grasp_axis_z)
        if not np.isfinite(axis_z):
            raise ValueError("the measured support grasp height must be finite")
        rest_z += max(0.0, axis_z - (surface_z + radius))
    if rod is None:
        rod = env.scene.by_name[env.task.deformables[0].name].positions(env.state.body_q.numpy())
        perceived = False
    else:
        rod = np.asarray(rod, dtype=np.float64).reshape(-1, 3)
        if len(rod) < 2:
            raise RuntimeError("a support grasp needs at least two measured centreline points")
        perceived = True
    # ``weave_anchor_y(variant)``, not the bare constant: on a terminated run the
    # rod is 0.82 m rather than the 0.50 the constant was written for, and taking
    # the segment nearest +0.17 puts this hand 123 mm from where the other one is
    # about to work. See the function -- it carries the measurement.
    seg = int(np.argmin(np.abs(rod[:, 1] - weave_anchor_y(variant))))
    xy = np.asarray(rod[seg, :2] if grasp_xy is None else grasp_xy, dtype=np.float64)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError("the measured support grasp must contain two finite x/y coordinates")
    grasp = np.array([float(xy[0]), float(xy[1]), rest_z])
    over = grasp + np.array([0.0, 0.0, approach])
    here, _ = ik.ee_pose(env.state)
    q_seed = np.asarray(env.state.joint_q.numpy(), dtype=np.float64)
    candidates = []
    for quat in quats:
        q_prev = q_seed.copy()
        stages: list[dict[str, Any]] = []
        errors: dict[str, float] = {}
        for leg_name, start, end, duration in (
            ("enter", np.asarray(here, dtype=np.float64), over, 1.0),
            ("take", over, grasp, 0.5),
        ):
            rows, q_prev, errors[leg_name] = _leg_rows(
                ik, quat, q_prev, start, end, duration, stream_hz
            )
            stages.append(
                {
                    "stage": leg_name,
                    "trajectory": {"waypoints": [{"positions": [float(v) for v in row]} for row in rows]},
                    "duration": float(len(rows) / stream_hz),
                    "grip_command_m": float(REFERENCE_GRIP_OPEN),
                }
            )
        candidates.append((max(errors.values()), quat, q_prev, stages, errors))
        if max(errors.values()) <= .010:
            break
    # A parallel-jaw grasp has two physically equivalent wrist rolls. The
    # support arm reaches the terminal fringe with only one of them on this
    # cell, so solve both and retain the lower residual.
    _, quat, q_prev, stages, errors = min(candidates, key=lambda item: item[0])
    return {
        "stages": stages,
        "grasp_m": [float(v) for v in grasp],
        "segment": -1 if perceived else int(seg),
        "close": float(weave_grip_close(env)),
        "quat": [float(v) for v in np.asarray(quat, dtype=np.float64)],
        "q_end": [float(v) for v in q_prev],
        "errors": errors,
        "worst_grasp_error": errors["take"],
    }


def plan_support_feed(
    env, plan: dict[str, Any], q_support: np.ndarray, quat, segment: int = -1
) -> tuple[np.ndarray | None, np.ndarray]:
    """Rows paying out rod through one station's carry, or ``None`` where the
    geometry owes nothing.

    **Why a fixed pinch fails, and why squeezing was never the fix.** The far
    pinch holds material point A; the station is about to carry its grasped
    segment G to the place P. The rod between two pinches is inextensible, so
    once the chord |P - A| exceeds the material run between A and G something
    must give -- and what gave, measured 2026-08-11 with the far tip pinned
    rigid, was the work hand: the ``cross`` leg pulled the rod straight out of
    its 0.5 mm pinch, 0/3 with one divergence. Harder was tried and is worse
    both ways (1.5 mm a side diverged; 2.0 pushed the rod out during the cross
    -- :data:`WEAVE_PINCH_ENGAGE`'s own record). The answer is the one a person
    weaving by hand uses: the far hand pays rod out as the crossing takes it
    up.

    The pay-out is measured, not scripted: the rod's path from the pinch to P,
    less the material arc between the pinched and grasped segments (their
    index gap at the rod's own pitch), plus :data:`WEAVE_FEED_MARGIN` for the
    carry's lift. **The path, not the chord.** The route already committed
    bends the rod around every seated far-side crossing between the pinch and
    the station -- the plan hands those over as ``feed_via`` -- and a chord
    that cuts through a spool undercounts what the drag will demand. It did:
    weave4's middle near seat sits between the two crossings, its drag needs
    ~20 mm the chord could not see, the feed stood still, and the work pinch
    lost the rod 25 mm short of the box (contacts 11 -> 6, TCP-to-rod 6 ->
    22 mm, seat2 0/42 segments for the rest of the episode).

    Positive walks the hand that far along the path's first leg -- the
    direction a unit of travel buys a unit of slack -- streamed through
    exactly the legs that take the slack up (:data:`WEAVE_CLOSED`, the
    routine's job). Zero or negative stands the hand still, which is every
    near-side drag the committed route does not tax: their slack comes from
    the free tail, not from the anchored side.

    Args:
        env: The live environment.
        plan: One station's plan, as :func:`plan_station_stages` returns it.
        q_support: The support chain's last solved full joint vector.
        quat: The support hand's grasp orientation, as taken.
        segment: The pinched segment's rod index, as
            :func:`plan_support_stages` recorded it at the take. Negative
            falls back to the xy-nearest segment -- a proximity match that a
            doubled-back pass can pull toward the wrong material point,
            inflating the arc and standing the feed still with nothing to say
            so, which is why the recorded index is the one to pass.

    Returns ``(rows, q_end)``: the streamed rows or ``None``, and the joint
    vector the next feed's chain seeds from.
    """
    name = env.handles.support_arm
    ik = env.arm(name)
    q_support = np.asarray(q_support, dtype=np.float64)
    a_pos, _ = ik.fk_ee(np.asarray(q_support, dtype=np.float32))
    a_pos = np.asarray(a_pos, dtype=np.float64)
    grasp = np.asarray(plan["grasp_m"], dtype=np.float64)
    place = np.asarray(plan["place_m"], dtype=np.float64)
    measured_slack = plan.get("grasp_slack_m")
    if measured_slack is not None:
        arc = float(measured_slack)
    else:
        rod = env.scene.by_name[env.task.deformables[0].name].positions(env.state.body_q.numpy())
        i_a = int(segment) if int(segment) >= 0 else int(np.argmin(np.linalg.norm(rod[:, :2] - a_pos[:2], axis=1)))
        i_g = int(np.argmin(np.linalg.norm(rod[:, :2] - grasp[:2], axis=1)))
        spec = env.task.deformables[0]
        arc = abs(i_a - i_g) * float(spec.length) / int(spec.segments)
    end_xy = _feed_target(a_pos, place, plan.get("feed_via", ()), arc)
    if end_xy is None:
        return None, q_support
    end = a_pos.copy()
    end[:2] = end_xy
    frames = sum(len(s["trajectory"]["waypoints"]) for s in plan["stages"] if s["stage"] in WEAVE_CLOSED)
    stream_hz = _stream_hz(env.task.physics)
    rows, q_end, error = _leg_rows(
        ik, np.asarray(quat, dtype=np.float64), q_support, a_pos, end, frames / stream_hz, stream_hz
    )
    if error > PLAN_ERROR_LIMIT:
        # The same refusal every carried leg gets, on the one carried motion
        # that had none: a feed chain that escaped its basin would sweep the
        # pinching hand across the bench mid-carry. Standing still risks a
        # taut cross; whipping the far pinch guarantees worse.
        return None, q_support
    return rows, q_end


class RoutineWeave:
    """Drive one weave, station by station, with both hands.

    The work hand runs the stations; the support hand opens the episode by
    taking the rod's far end (:func:`plan_support_stages`) and thereafter holds
    it, paying out slack through each carry that demands it
    (:func:`plan_support_feed`). One hand at a time is not a weave at all --
    the measured haul-aways and pinch losses are written up on those two
    functions. A pairing with no support arm still runs rather than refusing,
    but what it runs is a single-handed attempt with the corrected aim and
    nothing holding the far end -- the configuration measured at 0/3, useful
    as a debugging posture and not as a policy.

    Args:
        env: The environment to drive. Its task must be a weave variant.
    """

    MODE = "joint_position"
    """The action space this emits into. Read by ``PolicyScripted``, which is
    what lets a Cartesian routine and a joint-space one share a scene."""

    def __init__(self, env) -> None:
        from .assets.layout import WEAVE_LAYOUTS  # noqa: PLC0415

        if env.task.name not in WEAVE_LAYOUTS:
            raise RuntimeError(f"RoutineWeave drives the weave variants; this is {env.task.name!r}")
        self.env = env
        # Counted off the compiled scene rather than off the layout table: a
        # board variant's stations are drawn per episode, and while its count
        # does not vary, reading it from the one place that is always current
        # is what keeps that true when somebody makes the count vary too.
        self.stations = len(station_xy(env))
        self.reset()

    def reset(self) -> None:
        self.station = 0
        self.stages: list[dict[str, Any]] = []
        self.stage = 0
        self.frame = 0
        self.settle = WEAVE_SETTLE_STEPS
        self.dwell = 0
        self.grip = self._open
        self.grip_target = self._open
        self.done = False
        self.command: list[float] | None = None
        """The last waypoint actually streamed to the work arm.

        Re-sent through every dwell, which is what makes a dwell a *hold*. See
        :meth:`_action` for what echoing the measured joints back did instead.
        """
        self.support: list[float] | None = None
        """The support arm's standing command: home until the anchor legs
        stream, then wherever the anchor grasp and its feeds have walked it.
        The wrap routine's ``_left_hold``, generalised to a hand that moves."""
        bimanual = bool(self.env.handles.support_arm)
        self.anchored = not bimanual
        """Whether the far end is held. Starts true only when the pairing
        brings no second hand, which degrades to the single-handed weave."""
        self.anchor_stages: list[dict[str, Any]] = []
        self.anchor_stage = 0
        self.anchor_frame = 0
        self.support_grip: float | None = self._support_open if bimanual else None
        self.support_grip_target: float | None = self.support_grip
        self.feed_rows: np.ndarray | None = None
        """Waypoints streaming to the support arm right now -- the pay-out."""
        self.feed_next: np.ndarray | None = None
        """The coming station's pay-out, waiting for its first closed leg."""
        self.feed_frame = 0
        self._support_q: np.ndarray | None = None
        """The support chain's last solved full joint vector. Feeds seed here
        rather than from the live state so the streamed command is continuous
        with the command that stands, not with wherever load has sagged it."""
        self._support_quat: np.ndarray | None = None
        self._support_seg = -1
        """The rod index the anchor take pinched, handed to every feed as the
        material point A. Recorded rather than re-derived by proximity: a
        doubled-back pass can put a different segment nearer the TCP, and an
        inflated arc stands the feed still with nothing to say so."""
        self._holds: dict[str, list[float]] = {}
        """Standing commands for arms that are neither role. ``self.support``
        used to serve here too, safely, while it was a posture taken once; it
        is a streamed trajectory now, and a third arm sharing it would replay
        the anchor legs through its own workspace. No shipped cell has a
        third arm; the dict is what keeps that latent rather than silent."""

    # ------------------------------------------------------------------ grips
    @property
    def _open(self) -> float:
        return float(self.env.handles.arms[self.env.handles.work_arm].gripper_open)

    @property
    def _close(self) -> float:
        return float(weave_grip_close(self.env))

    @property
    def _support_open(self) -> float:
        return float(self.env.handles.arms[self.env.handles.support_arm].gripper_open)

    # ----------------------------------------------------------------- driving
    def act(self) -> np.ndarray:
        """One control period.

        Returns the full bimanual vector: the work arm tracks the plan, the
        support arm holds wherever it is. Held rather than commanded home each
        step -- a support arm re-driven to home every period fights whatever it
        is doing, and this task's second hand is idle rather than absent.

        **The rod is carried by the pinch, and by nothing else.** This routine
        used to engage ``EntityCable.hold_segment``
        at the end of the take dwell -- a kinematic pin between the segment and
        the hand -- on the stated grounds that a friction grasp could not hold
        a dragged rod. What the latch was actually covering for was a hand that
        never touched the rod: with it disabled the shipped routine made *zero*
        gripper-vs-rod contacts across a whole episode and moved the cable
        0.1 mm. The pin was doing all of the work and the contact none of it,
        which is exactly the arrangement that cannot be told apart from a grasp
        by looking at the score.

        With the hand actually arriving at the rod (:meth:`_action`), at the
        depth that puts the pads around it (:func:`plan_station_stages`), and
        with the contacts no longer being dropped on the floor of a 64-row
        buffer (``tasks/cable_route/staging.py``), the pinch carries it: 266 contacts at
        peak, contact on 56% of the episode's steps, 390 mm of rod moved, and
        the same 2 of 3 seats the latched version scored. The latch stays where
        it is still load-bearing -- the wrap's anchor hand and its wind, where
        the rod is dragged around a peg under tension -- and is gone from here.

        **The far end is held the same way.** The support hand's anchor grasp
        and its pay-outs are contact pinches through the same proxies, streamed
        by the same machinery; nothing in this routine touches a latch slot.
        """
        if self.settle > 0:
            self.settle -= 1
            return self._action(None)
        if self.dwell > 0:
            self.dwell -= 1
            return self._action(None)
        if not self.anchored:
            return self._anchor_step()
        if not self.stages:
            if self.station >= self.stations:
                # First arrival settles; the second asks out. The dwell is what
                # puts the verdict on the rod at rest, not the release frame.
                if self.done:
                    self.env.request_termination("weave script complete")
                else:
                    self.done = True
                    self.dwell = WEAVE_SETTLE_STEPS
                return self._action(None)
            plan = plan_station_stages(self.env, index=self.station)
            worst = float(plan["worst_grasp_error"])
            if worst > PLAN_ERROR_LIMIT:
                # The wrap routine's own refusal, at the wrap's own threshold.
                # A leg the solver could not reach streams its best basin
                # otherwise -- with the rod in the jaws -- and what that looks
                # like end to end is a whip that flings the rod, scored as a
                # lottery. Refusing is the reading a person can act on.
                self.env.request_termination(
                    f"station {plan['station']} plan unreachable: worst carried-leg error {worst * 1000:.0f} mm"
                )
            self.stages = list(plan["stages"])
            self.stage = 0
            self.frame = 0
            # The station's pay-out, planned against the same live rod its
            # legs were: how much is owed is settled here, when it is owed is
            # the closed legs' own boundary below.
            self.feed_next = None
            if self._support_q is not None:
                self.feed_next, self._support_q = plan_support_feed(
                    self.env, plan, self._support_q, self._support_quat, segment=self._support_seg
                )

        stage = self.stages[self.stage]
        waypoints = stage["trajectory"]["waypoints"]
        if self.frame == 0:
            # Boundary work, before the leg rather than after: the jaws are
            # opened ahead of the leg that lifts away, and given the dwell to
            # travel in, or the leg starts while they are still on the rod.
            name = stage["stage"]
            if name == "clear":
                self.grip_target = self._open
                self.dwell = WEAVE_GRIP_STEPS
            if name in WEAVE_CLOSED:
                self.grip_target = self._close
                if self.feed_next is not None:
                    # The pay-out runs with the carry it pays for: it starts
                    # with the first closed leg and spans exactly their frames.
                    self.feed_rows = self.feed_next
                    self.feed_next = None
                    self.feed_frame = 0

        target = waypoints[min(self.frame, len(waypoints) - 1)]["positions"]
        self.frame += 1
        if self.frame >= len(waypoints):
            name = stage["stage"]
            self.frame = 0
            self.stage += 1
            if name == "take":
                # Command the close and dwell for it: the ramp is
                # WEAVE_GRIP_STEPS and the pads need as long again to travel
                # and load against the rod. A carry begun before that is a
                # carry begun on jaws that are still open.
                self.grip_target = self._close
                self.dwell = 2 * WEAVE_GRIP_STEPS
            if self.stage >= len(self.stages):
                self.stages = []
                self.station += 1
        return self._action(target)

    def _anchor_step(self) -> np.ndarray:
        """One period of the opening move: the support hand takes the far end.

        The station loop in miniature, on the other arm: stream the planned
        legs one waypoint per period, and close at the end of ``take`` with the
        same double dwell the stations give their own grasp. The plan is made
        on first entry -- after the settle, against the rod at rest -- and the
        chain's solved end seeds every feed this hand later pays out.

        A last half-settle follows the close before the first station plans:
        the closing pads move the rod (the wrap's anchor swung it centimetres),
        and station 0 must read where the pinch left it, not where the spawn
        did.
        """
        if not self.anchor_stages:
            plan = plan_support_stages(self.env)
            worst = float(plan["worst_grasp_error"])
            if worst > PLAN_ERROR_LIMIT:
                self.env.request_termination(
                    f"support anchor plan unreachable: take endpoint error {worst * 1000:.0f} mm"
                )
            self.anchor_stages = list(plan["stages"])
            self.anchor_stage = 0
            self.anchor_frame = 0
            self._support_q = np.asarray(plan["q_end"], dtype=np.float64)
            self._support_quat = np.asarray(plan["quat"], dtype=np.float64)
            self._support_seg = int(plan["segment"])
        if self.anchor_stage >= len(self.anchor_stages):
            self.anchored = True
            self.dwell = WEAVE_SETTLE_STEPS // 2
            return self._action(None)
        stage = self.anchor_stages[self.anchor_stage]
        waypoints = stage["trajectory"]["waypoints"]
        self.support = [float(v) for v in waypoints[min(self.anchor_frame, len(waypoints) - 1)]["positions"]]
        self.anchor_frame += 1
        if self.anchor_frame >= len(waypoints):
            name = stage["stage"]
            self.anchor_frame = 0
            self.anchor_stage += 1
            if name == "take":
                self.support_grip_target = float(weave_grip_close(self.env))
                self.dwell = 2 * WEAVE_GRIP_STEPS
        return self._action(None)

    def _action(self, positions) -> np.ndarray:
        """The bimanual vector: work arm tracking, support arm holding.

        **A dwell re-sends the last waypoint; it does not re-send the
        measurement.** Both arms used to be held by reading ``state.joint_q``
        back out and returning it as the command, and a position servo told to
        go where it already is has no error to answer -- so it stops closing
        whatever gap it had, gravity and contact take another bite, and the
        next period reads *that* back as the new instruction. A hold written
        that way is a ratchet that lets go.

        It cost this routine the whole task. Every leg ends into a dwell, and
        the one after ``take`` -- 48 steps, the jaws' own travel -- is where the
        grasp happens. The 55 mm descent had 12 frames to run and the servo was
        still ~20 mm short of the rod when the leg ended; the dwell then froze
        the hand there and closed the jaws on air, 20 mm above a 10 mm rod.
        Measured across a whole weave3 episode with the latch disabled: zero
        gripper-vs-rod contacts in the proxy pipeline and 0.1 mm of rod motion.
        Everything that looked like manipulation was the latch.

        Holding the commanded waypoint instead, the servo keeps its error and
        converges during the dwell it was given for exactly that. The wrap
        routine beside this one has always done it this way -- ``_left_hold``
        is taken once and re-sent -- which is also why the wrap works.

        The support arm is streamed by the same rule: the anchor legs and the
        feeds each advance its standing command one waypoint per period, and
        between them the last waypoint stands. Its fingers walk exactly as the
        work hand's do.
        """
        # Walk the finger commands toward their targets rather than stepping
        # them: a step input on the finger servo slams the pads
        # (sim.set_gripper ramps for the same reason), and a slam against a
        # held rod is an impulse the release pays out.
        step = (self._open - self._close) / float(WEAVE_GRIP_STEPS)
        if self.grip < self.grip_target:
            self.grip = min(self.grip_target, self.grip + step)
        elif self.grip > self.grip_target:
            self.grip = max(self.grip_target, self.grip - step)
        if self.support_grip is not None:
            # That hand's own span, not the work hand's: the anchor take's
            # dwell is budgeted as WEAVE_GRIP_STEPS of travel, and a support
            # hand with a different open width walked at the work hand's step
            # would still be moving when it expires. The two YAM cells give
            # both arms one gripper, so today the spans are equal; stated per
            # arm because that is a fact about the pairing, not a law.
            step_s = (self._support_open - self._close) / float(WEAVE_GRIP_STEPS)
            if self.support_grip < self.support_grip_target:
                self.support_grip = min(self.support_grip_target, self.support_grip + step_s)
            elif self.support_grip > self.support_grip_target:
                self.support_grip = max(self.support_grip_target, self.support_grip - step_s)
        env = self.env
        q = env.state.joint_q.numpy()
        if positions is not None:
            self.command = [float(v) for v in positions]
        if self.feed_rows is not None:
            # The pay-out: one row per period onto the support arm, the leg's
            # last row left standing as the new hold.
            self.support = [float(v) for v in self.feed_rows[min(self.feed_frame, len(self.feed_rows) - 1)]]
            self.feed_frame += 1
            if self.feed_frame >= len(self.feed_rows):
                self.feed_rows = None
        out: list[float] = []
        for spec in env.scene.embodiment.spec.arms:
            handle = env.handles.arms[spec.name]
            if spec.name == env.handles.work_arm:
                # Before the first waypoint there is nothing to hold but the
                # start pose, which is what the opening settle is standing in.
                if self.command is None:
                    self.command = [float(q[c]) for c in handle.arm_coords]
                out += self.command
                # `self.grip` is a finger command in metres; the action slot is
                # the 0 (open) .. 1 (closed) fraction every arm contributes.
                out += [1.0 - float(self.grip) / self._open]
            elif spec.name == env.handles.support_arm and self.support_grip is not None:
                if self.support is None:
                    self.support = [float(q[c]) for c in handle.arm_coords]
                out += self.support
                out += [1.0 - float(self.support_grip) / self._support_open]
            else:
                # An arm that is neither role holds its OWN first posture --
                # see the `_holds` attribute for why it no longer shares
                # `self.support`.
                held = self._holds.setdefault(spec.name, [float(q[c]) for c in handle.arm_coords])
                out += held
                # The standing command, not the measured joint: echoing the
                # coordinate back as a command would read a sagging finger as
                # an instruction to sag further.
                out += [float(env.gripper(spec.name).command)]
        return np.asarray(out, dtype=np.float64)
