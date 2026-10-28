# SPDX-License-Identifier: Apache-2.0
"""The tool-hanging suite's privileged tools: the job cursor, the plans, the state.

Six tools, for the things the generic ``sim.*`` set cannot know. Four of them
carry the design; ``sim.get_phase`` hands a planned phase to the node that
drives it and ``sim.give_up`` retires a job the cursor should stop offering.

**``sim.next_job``** is the cursor the workflow loops on. These variants hang two
tools per episode and the document says which tool goes on which holder -- as a
pair of ``threaded`` clauses, in its own order -- so "what is left to do" is a
question about the *goal tree* read against the evaluator's verdict, exactly as
``tool_sort``'s ``sim.next_goal`` is. Reading the pairing off the goal is what
lets one workflow serve three variants: the wrenches' hooks, the scissors' wider
row and the screwdrivers' rack are three different fixtures and one question.

**``sim.plan_pick``** and **``sim.plan_place``** are the motion, solved to joint
space against the arm's own full-pose objective. Two tools rather than one
because the second is *closed loop*: the pick's stations are computed from where
the tool lies, and the place's from where the tool actually ended up **in the
jaws** after the lift and the turn. That difference is the whole reason a graph
can beat the scripted routine here. The routine plans its entire path from the
pre-grasp measurement and never looks again, so the millimetres the close shifts
the tool -- and the centimetre the turn-to-vertical can swing it -- are spent at
the hook.

**``sim.get_hang_state``** is the measurement both of those and the verify read:
where the tool's own landmark is, where the holder's is, and the ``threaded``
verdict between them computed from the same declared features the evaluator
scores. A graph and a score that disagree about what "hung" means is the failure
this exists to prevent.

**What is restated here and why.** Only :data:`TOOL_STATIONS` -- per tool class,
where to close the jaws on it and how high its loop must ride to clear a hook
tip. Both are measurements of the *mesh*, taken on the branch this suite was
ported from, and neither is derivable: the grasp centre is a choice about which
part of a spanner is a handle, and the clearance is set by the physical opening
of an oval bow rather than by the ``inner_r`` its record declares for hanging.
Everything else -- the loops, the rods, the hook tips, the rack holes -- is read
off the asset records through :func:`_feature_world`, so the plan aims at the
same geometry the goal is scored on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robosimstudio.solvers.motion.path import stream_hz
from robosimstudio_bench.goals.goal import about_key, leaves
from robosimstudio.domains.rigid.measures.geometry import encloses, threaded
from robosimstudio.assetlib.registry import ASSETS
from robosimstudio.assetlib.semantics.kinds import get_kind, is_registered_kind
from robosimstudio.core.math import (
    quat_from_axis_angle,
    quat_multiply,
    quat_slerp,
    quat_to_mat,
)
from robosimstudio.arms import arm_id_of
from robosimstudio.robots.embodiments import top_down_grasp_site_quat

__all__ = ["HANG_ATOMS", "ToolsToolHanging", "register_tool_hanging_tools"]

HANG_ATOMS = ("threaded", "encloses")
"""The geometry atoms a document may state "hung" with, strictest first.

Both take ``(loop, rod)`` and both are scored the same way, so a job is a job
whichever one the clause names. Reading only ``threaded`` is what made this
graph unrunnable on two of the three variants that declare it: ``wrenches`` and
``scissors`` state ``encloses`` -- deliberately, because a released tool tilts
and ``threaded`` wants the rod through the loop's exact plane -- so ``jobs()``
found nothing to do and the workflow went straight to ``verify`` and aborted,
at 318 and 320 steps with ``clauses: 0``. The measured table in
``graphs/gt/README.md`` predates that rewrite and wants re-taking.
"""


# --------------------------------------------------------------- measurements


def _vec3_gap(value) -> dict[str, float]:
    """Newton xyz sequence to GAP's named Vec3 mapping."""
    return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}


def _quat_gap(value) -> dict[str, float]:
    """Newton xyzw sequence to GAP's wxyz Quaternion mapping."""
    return {"w": float(value[3]), "x": float(value[0]), "y": float(value[1]), "z": float(value[2])}


@dataclass(frozen=True)
class ToolStation:
    """What a tool class's own mesh says about handling it.

    Attributes:
        grasp_local: Where the jaws close, in the mesh's frame [m]. A choice
            about which part of the object is a handle, so it is measured rather
            than derived -- the ring end of a spanner is not a handle and its
            centroid is not either.
        loop_above_tip: How far above a hook's rod axis the tool's loop must
            ride to pass over the tip [m], or 0 for a tool that goes into a
            holder rather than onto one. **Not derivable from the loop's
            declared ``inner_r``**: a record's radius is sized to the resting
            hang, and a scissor bow is a 30 mm oval whose physical opening is
            larger than the circle that describes where it comes to rest.
    """

    grasp_local: tuple[float, float, float]
    loop_above_tip: float = 0.0
    placement_local: tuple[float, float, float] | None = None
    min_lift: float = 0.0
    motion_scale: float = 1.0
    hook_past: float = 0.028


TOOL_STATIONS: dict[str, ToolStation] = {
    # The spanners: closed across the flat mid-shaft, well clear of both ends.
    "wrench_ring_19": ToolStation((0.14409, 0.22179, 0.00425), loop_above_tip=0.015),
    "wrench_ring_17": ToolStation((0.154142, 0.178765, 0.00380), loop_above_tip=0.015),
    "wrench_adjustable_150": ToolStation(
        (0.066, 0.0, 0.0), loop_above_tip=0.015, min_lift=0.180
    ),
    "wrench_adjustable_200": ToolStation(
        (0.050, 0.0, 0.0), loop_above_tip=0.015, min_lift=0.180
    ),
    # The scissors: closed at the pivot, which is the one place both halves are
    # held at once. 10 mm of tip clearance rather than the wrenches' 15 because
    # the bow's physical opening is far wider than the circle it hangs on.
    "scissor_pair_1": ToolStation(
        (0.000, -0.018, 0.000), loop_above_tip=0.010, min_lift=0.190, motion_scale=2.0
    ),
    "scissor_pair_2": ToolStation(
        (0.000, 0.020, -0.00025), loop_above_tip=0.010, min_lift=0.190, motion_scale=2.0
    ),
    # The screwdrivers: closed on the handle, tip-down into the rack.
    "screwdriver_slim": ToolStation((0.030, 0.0, 0.0)),
    "screwdriver_grip": ToolStation((0.060, 0.0, 0.0)),
    # A plier is seated by the point inside its handle gap, not an asset
    # feature used by the evaluator.  The evaluator scores its body against a
    # support-relative region, so this is the one additional measured station.
    "plier_combination": ToolStation(
        (0.036, 0.004, -0.0009),
        placement_local=(-0.010, 0.004, -0.0009),
        min_lift=0.180,
        motion_scale=2.0,
    ),
    "plier_cutter": ToolStation(
        (0.000, -0.015, -0.0035),
        placement_local=(0.000, -0.045, -0.0035),
        min_lift=0.180,
        motion_scale=2.0,
    ),
}
"""Per tool class, measured off the vendored meshes on the branch this ported
from. Keyed by the *asset record* rather than by the scene's instance name, so a
second wrench of the same class needs nothing here.

**The two wrench entries are stranded, and ``wrenches`` cannot run until they are
re-measured.** ``7520a52`` did not rename those records -- it swapped the meshes:
``wrench_ring_19`` was a 19 mm ring spanner
(``wrenches/8mm_to_19mm/thickened/obj_1_19mm_thickened.stl``) and
``wrench_adjustable_150`` is a different tool entirely (``wrenches/wrench_1/wrench_1.obj``),
same for 17 mm -> ``wrench_adjustable_200``. The coordinates below are measurements
of geometry no longer in the scene, so re-keying them onto the new records would
hand the planner a grasp station for the wrong mesh -- a plausible number and a
silently wrong grip. ``_station`` raises by name instead, which is what
``tool_hanging/wrenches`` does today.

Until ``7520a52`` this was hidden: ``jobs()`` read only ``threaded`` clauses and that
commit also rewrote these goals to ``encloses``, so the cursor found no work, the
graph aborted at ~318 steps, and nothing ever asked for a wrench's station.
``scissors`` was in the same shape and is solved by the atom fix alone -- its
records were not swapped. Nothing here can be derived:
``tasks/tool_hanging/rigs/record_tool_hanging_poses.py`` writes settled *object poses*, not grasp
stations, and these were measured by hand."""

GRASP_YAW_DEG = 90.0
"""World heading the jaws close along at the pick [deg].

Every tool here lies with its long axis pointing at the board (world +x), so the
fingers have to cross it -- which is what 90 degrees means to
``top_down_grasp_site_quat``, composed off the live hand's own measured axes."""

APPROACH_ABOVE_GRASP = 0.120
"""How far above the grasp the hand stages before descending [m]."""

FINGER_TABLE_CLEARANCE = 0.004
"""Gap left between the fingertips and the work surface at the grasp [m].

**The single change that turns this suite's grasp from a lottery into a grasp.**
Every tool here is a flat thing lying on a table, and the measured landmark the
plan steers by is its *mid-thickness* -- 4.25 mm above the surface on the 19 mm
spanner. Commanding the tool centre there drives the fingertips into the table
about ten millimetres before they arrive, and then two things happen at once:
the position servo keeps pushing (measured: the descend ends 8.2 mm short and
stays there, and asking for 25 mm more changes nothing but the shortfall), and
the pads jam. Traced on the shipped routine, the jaws sit at 92-95 mm through
the whole close and only run shut once the *lift* unloads them -- by which time
the hand is already above the tool. That is the coin flip the README records as
"the 17's jaws close ~1 mm above its thinner shaft".

So the descend stops where the fingers can still move: the surface, plus the
hand's own declared tip reach, plus this. Non-zero because the declared reach
understates the collider -- the arm bottoms out with the tool centre 16.5 mm
above the surface against a 12.14 mm stated reach. 4 mm rather than more because
the gripping band is only the last few millimetres of the pad. Swept on both
spanners at 1 mm steps, measuring whether the tool leaves the table:

    2 mm   jaws still jammed part-open at 26-28 mm; neither lifts
    4 mm   both lift; the tool slips 7-10 mm in the jaws
    5 mm   both lift; 6-9 mm of slip
    6 mm   the 17 lifts, the 19 does not
    7 mm+  the pads close above the shaft; neither lifts

The window is narrow, which is why the workflow does not rely on hitting it:
``carry`` checks whether the tool actually came up, and the cursor gives the job
a second pass from wherever the first one left it.
"""

LIFT_CLEARANCE = 0.015
"""Margin under a tool once it has been stood on end [m].

Added to the tool's own reach past the grasp point, which is what actually sets
the lift: a screwdriver gripped 145 mm from its tip has to rise at least that
far before it can be turned vertical without ploughing the table.

15 mm reproduces all three of the branch's hand-picked lifts to within a
few millimetres -- 150, 165 and 180 -- from each tool's own box. 30 mm does not:
it puts the screwdrivers' turn station 180 mm above a grasp the arm can only
reach 170 mm above, and the plan gate refuses it."""

JAW_STALL_MARGIN = 0.003
"""How far past its own shut width the gap has to sit to mean "on something" [m].

The jaws reach 0.7-1.2 mm on air and stall at 13-17 mm on a spanner shaft, so
the signal is not subtle -- the margin only has to clear the servo's own
overshoot."""

OFF_SURFACE = 0.020
CARRY_RADIUS = 0.30
"""What "the hand has it" looks like geometrically: the tool's lowest point this
far above the work surface [m], and its box centre within this of the tool
centre [m]. The radius is generous on purpose -- these tools are up to 300 mm
long and ride well off the pads -- and it is doing one job, telling a carried
tool from one lying on the table or dropped on the way."""

PRE_HOOK_STANDOFF = 0.045
"""How far in front of the hook tip the loop stages before it is pushed on [m]."""

PAST_HOOK_TIP = 0.028
"""How far along the hook's arm, past the tip, the loop is carried [m]."""

SEAT_PRELOAD = 0.002
"""How far past resting contact the loop is pressed onto the rod [m].

The lower stage is **derived, not tabulated**: the loop is carried down until it
sits on the rod, which is ``loop_above_tip`` plus the loop's own hanging radius
(``inner_r - radius``, both declared). This is the margin on top, so the loop is
demonstrably in contact rather than a tenth of a millimetre above it.

The shipped routine lowers a flat 12 mm instead, which on the 19 mm spanner
leaves the ring centre 9.7 mm above where it will come to rest -- so the ring
*drops* onto the arm when the jaws open, and lands cocked. That matters here
more than it sounds: ``threaded`` allows ``inner_r - rod_r`` of offset, which is
6.71 mm on the 17 mm spanner, and a ring that settles 1 mm off in y reads
6.8 mm and fails a goal it is physically on."""

RETRACT_FROM_BOARD = 0.100
RETRACT_UPWARD = 0.050
"""Where the emptied hand goes before it heads home [m]."""

TIP_CLEARANCE_ABOVE_HOLE = 0.030
"""How high above a rack hole the shaft tip is aligned before it goes in [m]."""

INSERTION_DEPTH = 0.060
"""How far past the hole's plane the shaft tip is driven [m]."""

RETRACT_ABOVE_HOLE = 0.080
"""How far the emptied hand rises off a seated screwdriver [m]."""

SEAM_HOLD_STEPS: dict[str, int] = {"carry": 20}
"""Control steps the arm holds still at a seam, per phase. Absent means none.

**Only ``carry``, and the first attempt at this is why the table is per phase.**
Held at every seam of every phase it costs about 294 steps against a document
that truncates at 2400, and measured over ten episodes that is the whole story:
9/10 and a median 2094 steps became **6/10 and a median 2388**, with truncations
going 1 to 4. The diagnosis below was right and the application was not -- a
settle is worth its steps at the seam that drops the tool and nowhere else.

``carry`` is ``[lift, orient]`` and so has exactly one seam, which is the one
the measurement names. 20 steps, once per tool.

**Where the tool is actually lost, measured per control step over five episodes
of ``wrenches``.** Tracking the spanner's offset from the tool centre through a
whole pass, the close and the rise are innocent and the *turn* is not:

===== ======  ==================  =================  ==================
ep    reward  delta close->rise   delta during rise  delta during turn
===== ======  ==================  =================  ==================
0     1.000   +0.01 mm            +0.01 mm           +0.11 mm
1     0.500   +0.27 mm            +0.02 mm           **+119.34 mm**
2     1.000   -0.06 mm            +0.01 mm           +0.12 mm
3     1.000   -0.04 mm            +0.02 mm           +0.11 mm
4     0.500   +0.13 mm            +0.22 mm           **+65.52 mm**
===== ======  ==================  =================  ==================

Both failures lose the tool during the 90-degree wrist turn, holding it through
everything before. And the turn is where :meth:`ToolsToolHanging.get_phase`
merged two stages into one trajectory: ``carry`` is a 0.60 s rise and a 1.00 s
turn concatenated, so the turn begins on the step the rise ends, from an arm
still carrying the rise's velocity.

The merge is right and stays -- it exists because every
``robot.execute_trajectory`` call ends in a settle floored at 60 control steps,
and a nine-stage plan paying that nine times does not fit a 2400-step document.
What it removed was the *settle*, not the call. Repeating the seam waypoint
buys the settle back at its real cost: 20 steps, against the 60-step floor a
separate call would charge, and only at the seams a phase actually has.

Why a hold rather than more clamp: the clamp is 1.4 N (``ke`` 200 at the 7.2 mm
standing error the 19 mm shaft leaves) against a 2.126 N tool measured in every
episode, so the turn's moment is not covered by the grip at all. Raising the
finger stiffness is the other half and is a change to the *robot*, not to this
task; this is the half that is this task's own."""

HOLD_TOLERANCE = 0.012
"""What "this arm can hold that pose" means when a station is being chosen [m].

Much tighter than the plan gate below, and separately named because it answers a
different question. The gate asks whether a solved plan is worth running at all;
this asks which of several candidate stations to *pick*, and there the answer
"42 mm out but under the limit" is the wrong one -- it passes the gate and then
drops the tool on the way. Measured: the screwdriver rack's second insertion,
grasped and lifted and gone before the turn finished, on a lift station the gate
had accepted at 42 mm."""

HOLD_LADDER_STEP = 0.010
"""How far a station steps down per rung while a ladder looks for one [m]."""

FLAT_LIFT_FLOOR = 0.120
"""How low the top-down half of the rise may be driven [m].

Only the *first* half of the rise is top-down -- the turn finishes it -- so this
does not have to reach the tool's own clearance requirement. But it may not go
much under the pregrasp stand-off either: the screwdrivers lie with their tips
already at the rack's front face, and a turn started low sweeps a 160 mm shaft
through it. Measured: dropped to 50 mm, the first screwdriver was knocked out of
reach on its own first pass (its re-plan then reported a 261 mm approach) and the
variant scored 0.00 where the same graph at 120 mm scored 0.50."""

POSITION_LIMIT = 0.045
ROTATION_LIMIT = float(np.radians(20.0))
"""What a station has to solve to before the plan is handed over.

The scripted routine's own limits, kept rather than tightened so a plan this
refuses is a plan the baseline would have refused too. Both halves are checked:
:meth:`~robosimstudio.solvers.ik.arm_ik.ArmIK.pose_errors` documents a solve
that scores 0.2 mm of position with the wrist half a turn out, and a tool held
at the wrong roll misses a hook by the width of its own ring."""

STAGE_SECONDS: dict[str, float] = {
    "approach": 0.70,
    "descend": 0.50,
    "lift": 0.60,
    "orient": 1.00,
    # The hook leg.
    "pre_hook": 0.90,
    "over_hook": 0.60,
    "onto_hook": 0.45,
    # The rack leg.
    "raise": 0.70,
    "above_hole": 1.00,
    "into_hole": 0.90,
    # Shared.
    "retract": 0.70,
    "home": 0.90,
}
"""How long each stage runs [s].

Between a half and two thirds of the scripted routine's own frame counts at this
cell's 60 Hz, on the same waypoints -- so the graph is moving the same path
somewhat faster, not a different path.

Trimmed rather than matched, because the graph spends steps the routine does
not: a settle at the end of every phase, two planning nodes the rollout keeps
stepping through, and a second pass at a job the first pass lost. The document
truncates at 2400. Measured at the routine's own pace, a clean two-job episode
finished at ~2300 and any episode that had to retry anything truncated with the
second tool still on the table -- which scores as a failed hang and is not one.
"""

CONVERGE_ROUNDS = 24
"""Solver rounds spent converging a station before a stage is densified from it.

The per-waypoint budget is sized for *tracking* -- consecutive targets a
millimetre apart -- and a station is a jump. Converging first is what keeps the
first waypoint of a stage from being a step input the arm chases."""



# ------------------------------------------------------------------ registrar


def register_tool_hanging_tools(registry, env, bus) -> None:
    """Register the suite's ``sim.*`` tools: the cursor, the plans, the state."""
    tools = ToolsToolHanging(env, bus)
    for name, fn, summary in tools.entries():
        registry.register_callable(name, fn, summary=summary, tags=("privileged",))


@dataclass(frozen=True)
class Job:
    """One tool and the holder its own goal clause names."""

    index: int
    tool: str
    """The manipulable object -- whichever operand has a body in the scene."""
    holder: str
    """The fixture it goes on or into."""
    tool_feature: str
    holder_feature: str
    tool_is_loop: bool
    """True where the tool carries the loop and the fixture the rod -- a spanner
    over a hook. False is the mirror: a screwdriver's shaft into a rack hole, and
    one atom scores both."""
    atom: str
    """Which predicate this clause states, so a node routing on the geometry and
    the evaluator scoring it test the same one.

    One of :data:`HANG_ATOMS` for a loop/rod job, or ``containment`` for a region
    job -- the level-1 documents state a support that way, because a plier over a
    triangular tip threads nothing."""
    keys: tuple[str, ...]
    """The metric key(s) the evaluator publishes this clause's verdict under.

    Usually one. A tool with two candidate loop features scored by an ``Or`` --
    scissors_1's ``bow`` and ``bow_other``, either catching the hook counts --
    publishes one leaf key per feature, and the job is satisfied when *any* of
    them reads true."""
    placement: str = "threaded"
    """``threaded`` for loop/rod jobs, ``support`` for a containment region.

    A second axis from :attr:`atom` rather than a restatement of it: the atom
    says which predicate scores the clause, this says which *geometry* the plans
    are built from -- a seat over a tip, or a feature mated into a hole."""
    region: str = ""
    """The region a ``support`` job is scored in; empty for the rest."""


class ToolsToolHanging:
    """The suite's privileged tool implementations."""

    def __init__(self, env, bus) -> None:
        self.env = env
        self.bus = bus
        self._jobs: tuple[Job, ...] | None = None
        self._arms: dict[str, str] = {}
        self.attempts: dict[str, int] = {}
        """Passes charged against each tool, for :data:`MAX_ATTEMPTS`."""
        self.given_up: set[str] = set()
        """Tools abandoned for the rest of this episode.

        Episode-scoped because the connector -- and with it this object -- is
        rebuilt per episode by ``PolicyGraph.reset``. Held here rather than
        threaded through the workflow because a subgraph re-entered by the loop
        edge starts with a fresh scope, so there is nowhere in the graph for
        "the tools I have already failed on" to live."""
        self._expected: dict[str, list[float]] = {}
        """Per tool, where :meth:`plan_pick` predicted its landmark would be once
        the hand reached the turn station. What :meth:`get_hang_state` reports
        the live landmark against, which is the in-jaw slip a place corrects."""
        self._phases: dict[str, dict[str, Any]] = {}
        """The last plan's stages, grouped by phase, held here rather than
        carried through the workflow.

        A dense stage is a few hundred joint vectors, and the alternative is
        threading it out of one subgraph's outputs and into another's inputs --
        which makes the plan a value the *graph* has to route correctly rather
        than a thing the planner owns. Grouped by phase rather than listed,
        because the four phases are the same for every variant while the stages
        inside them are not: a spanner's ``place`` legs onto a hook and a
        screwdriver's into a rack have nothing in common. One workflow can name a
        phase; it could only name one of the two stage lists."""
        self._current = ""
        """The tool the cursor last handed out -- what an unqualified call means."""
        self._motion_scale = 1.0

    MAX_ATTEMPTS = 2
    """Passes at one tool before it is abandoned.

    The livelock this prevents does not look like a failure: a wrench is carried
    to the hook, released, slides off and lands on the table. Nothing raised, so
    the cursor returns the same job and the graph does it again -- until the
    episode truncates with the *other* tool untouched. One retry is worth having
    because a slide is often not repeatable; on ``tool_sort`` a third never paid.
    """

    def entries(self) -> list[tuple[str, Any, str]]:
        return [
            ("sim.next_job", self.next_job, "The next tool this document still wants hung, or done."),
            ("sim.plan_pick", self.plan_pick, "Approach, descend, lift and turn-to-vertical, as joint stages."),
            ("sim.get_phase", self.get_phase, "One phase of the last plan: its stages, in order, and the arm."),
            (
                "sim.plan_place",
                self.plan_place,
                "The hang or insertion, re-planned around where the tool actually sits in the jaws.",
            ),
            ("sim.get_hang_state", self.get_hang_state, "Landmark, holder, threaded verdict, and whether it is held."),
            (
                "sim.get_gap_target",
                self.get_gap_target,
                "Ground-truth grasp geometry for GAP, without planning a robot trajectory.",
            ),
            (
                "sim.get_fixture_waypoints",
                self.get_fixture_waypoints,
                "Pose-only feature-mating waypoints from current object and fixture poses.",
            ),
            ("sim.give_up", self.give_up, "Abandon one tool for the rest of the episode and get on with the rest."),
        ]

    # ------------------------------------------------------------------- jobs
    def jobs(self) -> tuple[Job, ...]:
        """The document's tool/holder clauses, in its own order.

        Read from the goal tree rather than from a table beside it, for
        ``placements``' own reason: a copy can disagree with its original, and
        one that does produces a policy diligently hanging tools nobody is
        scoring. It is also what makes one workflow serve three variants -- the
        rack's clauses name the loop on the *fixture*, and this notices.

        The linking leaf is ``threaded`` on some documents and an explicit
        ``encloses`` on others -- wrenches and scissors state it directly (with
        an ``Or`` of two candidate loop features on scissors, either bow
        catching the hook counts), screwdrivers still call ``threaded``. Both
        name a tool and a holder the same way, so both build the same
        :class:`Job`; a tool/holder pair seen more than once (the ``Or`` case)
        merges into the one job its later leaves add a scoring key to, rather
        than being counted twice.
        """
        if self._jobs is not None:
            return self._jobs
        objects = self.env.handles.objects
        by_pair: dict[tuple[str, str], Job] = {}
        order: list[tuple[str, str]] = []
        # Through ``env.goal`` rather than ``env.evaluator.goal``: the same tree
        # on a single world, and the difference between running and not on a
        # fleet. A world's connector is built against a ``WorldView``, and a view
        # over one world of a wave has no evaluator of its own to reach through
        # -- so the batched form died here, on every world's first cursor call,
        # with ``'WorldView' object has no attribute 'evaluator'``. ``sim_tools``
        # states the same rule over its own goal reader.
        goal = self.env.goal
        for leaf in leaves(goal) if goal is not None else ():
            atom, operands = leaf.about
            if atom == "containment" and len(operands) >= 2:
                tool, region_name = str(operands[0]), str(operands[1])
                region = self.env.handles.regions.get(region_name)
                if tool not in objects or region is None:
                    continue
                pair = (tool, region.frame)
                if pair not in by_pair:
                    by_pair[pair] = Job(
                        index=len(order),
                        tool=tool,
                        holder=region.frame,
                        tool_feature="",
                        holder_feature="",
                        tool_is_loop=False,
                        atom=atom,
                        keys=(about_key(atom, operands),),
                        placement="support",
                        region=region_name,
                    )
                    order.append(pair)
                continue
            if atom not in HANG_ATOMS or len(operands) < 2:
                continue
            loop_ref, rod_ref = str(operands[0]), str(operands[1])
            loop_name, _, loop_feature = loop_ref.partition(".")
            rod_name, _, rod_feature = rod_ref.partition(".")
            if loop_name in objects:
                tool, tool_feature, holder, holder_feature, tool_is_loop = (
                    loop_name,
                    loop_feature,
                    rod_name,
                    rod_feature,
                    True,
                )
            elif rod_name in objects:
                tool, tool_feature, holder, holder_feature, tool_is_loop = (
                    rod_name,
                    rod_feature,
                    loop_name,
                    loop_feature,
                    False,
                )
            else:
                continue
            key = about_key(atom, operands)
            pair = (tool, holder)
            existing = by_pair.get(pair)
            if existing is not None:
                by_pair[pair] = Job(
                    index=existing.index,
                    tool=existing.tool,
                    holder=existing.holder,
                    tool_feature=existing.tool_feature,
                    holder_feature=existing.holder_feature,
                    tool_is_loop=existing.tool_is_loop,
                    atom=existing.atom,
                    keys=(*existing.keys, key),
                )
                continue
            by_pair[pair] = Job(
                index=len(order),
                tool=tool,
                holder=holder,
                tool_feature=tool_feature,
                holder_feature=holder_feature,
                tool_is_loop=tool_is_loop,
                atom=atom,
                keys=(key,),
            )
            order.append(pair)
        self._jobs = tuple(by_pair[pair] for pair in order)
        return self._jobs

    def _job(self, tool: str = "") -> Job:
        """The job a call means: the named one, or the one the cursor handed out.

        ``_current`` is what makes an unqualified call mean *this* pass rather
        than the document's first clause. Every node after the cursor asks about
        the tool it is working on without naming it -- and a release node asking
        for "the outstanding job" would get the *next* tool, because the one in
        its hand has just become satisfied. That opens the wrong hand.
        """
        jobs = self.jobs()
        if not jobs:
            raise RuntimeError(
                f"task {self.env.task.full_name!r} states no {'/'.join(HANG_ATOMS)} goal to plan against"
            )
        wanted = tool or self._current
        if wanted:
            found = next((j for j in jobs if j.tool == wanted), None)
            if found is None:
                raise ValueError(f"no job hangs {wanted!r}; this document hangs {[j.tool for j in jobs]}")
            return found
        return jobs[0]

    def next_job(self, count_attempt: bool = False) -> dict[str, Any]:
        """The first tool whose hanging clause is still unsatisfied.

        Args:
            count_attempt: Charge this call as a pass at whatever it returns,
                abandoning the job once it has had :data:`MAX_ATTEMPTS`. Only the
                loop gate passes it -- every other node asks the same question to
                learn *which* tool it is working on, and counting those would
                abandon each job on its first pass.

        Returns ``route`` ``"more"`` with the job, or ``"done"`` when the
        document has nothing left it will accept.
        """
        metrics = self.bus.snapshot.info.get("metrics") or {}
        outstanding = [j for j in self.jobs() if not any(bool(metrics.get(k, False)) for k in j.keys)]
        for job in outstanding:
            if job.tool in self.given_up:
                continue
            if count_attempt:
                self.attempts[job.tool] = self.attempts.get(job.tool, 0) + 1
                if self.attempts[job.tool] > self.MAX_ATTEMPTS:
                    self.given_up.add(job.tool)
                    continue
            self._current = job.tool
            return {
                "route": "more",
                "job_index": job.index,
                "object_name": job.tool,
                "holder_name": job.holder,
                "arm_name": self._arm_for(job),
                "arm_id": arm_id_of(self.env, self._arm_for(job)),
                "remaining": len(outstanding),
                "attempt": int(self.attempts.get(job.tool, 0) or 1),
            }
        return {
            "route": "done",
            "job_index": -1,
            "object_name": "",
            "holder_name": "",
            "arm_name": "",
            "arm_id": 0,
            "remaining": len(outstanding),
            "attempt": 0,
        }

    def give_up(self, object_name: str = "") -> dict[str, Any]:
        """Abandon one tool for the rest of the episode.

        Empty means the job the cursor is on, which is what a failure branch
        almost always is. The alternative to giving up is retrying the same tool
        until the episode truncates, and on a two-tool document that means one
        ungraspable spanner costs the other its turn.
        """
        name = object_name or self._current or str(self.next_job()["object_name"])
        if name:
            self.given_up.add(name)
        return {"object_name": name, "given_up": sorted(self.given_up)}

    # ------------------------------------------------------------------- arms
    def _arm_for(self, job: Job) -> str:
        """Which hand takes this tool: the one whose rest pose is nearest it.

        Never a ``"left"`` literal. The two YAM cells list their arms in opposite
        orders and a literal names a different physical hand on each, so the only
        portable rule is a geometric one -- and it is the right rule anyway,
        because a bimanual cell lays its tools out one under each shoulder and
        reaching across the body is what makes a plan unsolvable.
        """
        if job.tool in self._arms:
            return self._arms[job.tool]
        # The **grasp station**, not the body origin. A wrench's mesh origin is
        # 180 mm up its own shaft and the scene spawns it so the *ring* lands
        # where the layout wants it, which puts the 17's origin on the far side
        # of the centre line from the 17 -- and picked the wrong hand for it.
        here = self._grasp_world(job)[:2]
        best, best_d = "", float("inf")
        for name in self.env.handles.arms:
            rest = np.asarray(self.env.arm(name).home_pose[0][:2], dtype=np.float64)
            distance = float(np.linalg.norm(rest - here))
            if distance < best_d:
                best, best_d = name, distance
        self._arms[job.tool] = best
        return best

    def _grasp_world(self, job: Job) -> np.ndarray:
        """Where the jaws close on this tool, as it lies now [m]."""
        pose = self.env.handles.entity(job.tool).pose(self._body_q())
        return np.asarray(pose[:3], dtype=np.float64) + quat_to_mat(pose[3:7]) @ np.asarray(
            self._station(job.tool).grasp_local, dtype=np.float64
        )

    def get_gap_target(self, object_name: str = "", count_attempt: bool = False) -> dict[str, Any]:
        """Pose-only oracle output for the GAP grasp subgraph.

        This is deliberately not :meth:`plan_pick`: it reads the object's pose
        and CAD landmark but performs no IK and returns no joint trajectory.
        A perception implementation can produce the same mapping later.
        """

        def read() -> dict[str, Any]:
            job = self._job(tool=object_name)
            if count_attempt:
                self.attempts[job.tool] = self.attempts.get(job.tool, 0) + 1
                if self.attempts[job.tool] > self.MAX_ATTEMPTS:
                    raise RuntimeError(f"{job.tool} exhausted {self.MAX_ATTEMPTS} GAP grasp attempts")
            station = self._station(job.tool)
            entity = self.env.handles.entity(job.tool)
            pose = entity.pose(self._body_q())
            rotation = quat_to_mat(pose[3:7])
            grasp = np.asarray(pose[:3], dtype=np.float64) + rotation @ np.asarray(
                station.grasp_local, dtype=np.float64
            )
            arm = self._arm_for(job)
            hand = self.env.scene.embodiment.hands[arm]
            grasp[2] = max(
                float(grasp[2]),
                float(self.env.handles.surface_z) + float(hand.tip_reach) + FINGER_TABLE_CLEARANCE,
            )
            flat = np.asarray(top_down_grasp_site_quat(hand, GRASP_YAW_DEG), dtype=np.float64)
            # A small box centred on the measured handle landmark. The open
            # grasp skill consumes an OBB, while the exact pose below refines
            # the first yaw-fan candidate without adding simulator access to it.
            half = np.asarray(entity.obb.half if entity.obb is not None else (0.02, 0.02, 0.01), dtype=float)
            target_obb = {
                "center": _vec3_gap(grasp),
                "extent": _vec3_gap(np.minimum(half, np.array([0.025, 0.025, 0.012]))),
                "orientation": _quat_gap(pose[3:7]),
            }
            return {
                "object_name": job.tool,
                "holder_name": job.holder,
                "arm_id": arm_id_of(self.env, arm),
                "target_obb": target_obb,
                "grasp_pose": {"position": _vec3_gap(grasp), "rotation": _quat_gap(flat)},
            }

        return self.bus.call(read)

    def get_fixture_waypoints(self, object_name: str = "", phase: str = "place") -> dict[str, Any]:
        """Convert live object/fixture poses into portable placement poses.

        Only feature geometry is solved here. The returned poses are executed
        by the open ``executing-feature-mating`` skill through ordinary robot tools.
        """

        def read() -> dict[str, Any]:
            job = self._job(tool=object_name)
            arm = self._arm_for(job)
            ik = self.env.arm(arm)
            here, flat = ik.ee_pose(self.env.state)
            here = np.asarray(here, dtype=np.float64)
            flat = np.asarray(flat, dtype=np.float64)
            carry_only = str(phase) == "carry"
            upright = quat_multiply(quat_from_axis_angle((0.0, 1.0, 0.0), -np.pi / 2.0), flat) if carry_only else flat

            pose = self.env.handles.entity(job.tool).pose(self._body_q())
            if str(phase) == "place":
                entity = self.env.handles.entity(job.tool)
                centre = np.asarray(pose[:3], dtype=np.float64)
                if entity.obb is not None:
                    centre = centre + quat_to_mat(pose[3:7]) @ np.asarray(entity.obb.center_local, dtype=np.float64)
                if float(np.linalg.norm(centre - here)) >= CARRY_RADIUS or centre[2] <= float(self.env.handles.surface_z) + OFF_SURFACE:
                    raise RuntimeError(f"{job.tool} did not rise with the gripper; retry grasp from its live pose")
            landmark = np.asarray(pose[:3], dtype=np.float64) + quat_to_mat(pose[3:7]) @ self._landmark_local(job)
            in_hand = quat_to_mat(flat).T @ (landmark - here)

            def hand_for(target: np.ndarray) -> np.ndarray:
                return np.asarray(target, dtype=np.float64) - quat_to_mat(upright) @ in_hand

            lift = max(self._station(job.tool).min_lift, 0.180)
            waist = here + np.array([0.0, 0.0, min(0.120, lift)])
            lifted = here + np.array([0.0, 0.0, lift])
            if carry_only:
                points: list[tuple[np.ndarray, bool]] = [(waist, True), (lifted, True)]
            else:
                hand = self.env.scene.embodiment.hands[arm]
                canonical_flat = np.asarray(top_down_grasp_site_quat(hand, GRASP_YAW_DEG), dtype=np.float64)
                upright = quat_multiply(
                    quat_from_axis_angle((0.0, 1.0, 0.0), -np.pi / 2.0), canonical_flat
                )
                # Correct the residual carry wrist before translating toward
                # any fixture. This is shared by loop, rod and support mates.
                points = [(here, True)]

            if carry_only:
                pass
            elif job.placement == "support":
                origin = self.env.handles.regions[job.region].origin_now(self._body_q())
                tip = origin + np.array([0.10249, -0.16340, 0.01659])
                seated = tip + np.array([0.003, 0.0, 0.018])
                points.extend(
                    [(hand_for(seated + np.array([-0.048, 0.0, 0.0])), False),
                     (hand_for(seated), True),
                     (hand_for(seated + np.array([0.0, 0.0, -0.015])), True)]
                )
            elif job.tool_is_loop:
                rod = self._feature_world(job.holder, job.holder_feature)
                loop = self._feature_world(job.tool, job.tool_feature)
                axis = np.asarray(rod["axis"], dtype=np.float64)
                tip, inward = axis[0], axis[1] - axis[0]
                inward /= max(float(np.linalg.norm(inward)), 1.0e-12)
                station = self._station(job.tool)
                seated = tip + station.hook_past * inward + np.array([0.0, 0.0, station.loop_above_tip])
                hang = max(0.0, float(loop["inner_r"]) - float(rod["radius"]))
                points.extend(
                    [(hand_for(seated - (PRE_HOOK_STANDOFF + station.hook_past) * inward), False),
                     (hand_for(seated), True),
                     (hand_for(seated - np.array([0.0, 0.0, station.loop_above_tip + hang + SEAT_PRELOAD])), True)]
                )
            else:
                hole = self._feature_world(job.holder, job.holder_feature)
                mouth = np.asarray(hole["center"], dtype=np.float64)
                shaft = np.asarray(self._feature_world(job.tool, job.tool_feature)["axis"], dtype=np.float64)
                along = shaft[1] - shaft[0]
                along /= max(float(np.linalg.norm(along)), 1.0e-12)
                # Move to the robot's measured canonical upright frame. The
                # GT controller uses this same frame; preserving the open
                # grasp's residual wrist tilt leaves a long screwdriver free
                # to lean against the rack after release.
                along = quat_to_mat(upright) @ (quat_to_mat(flat).T @ along)
                along /= max(float(np.linalg.norm(along)), 1.0e-12)
                aligned = mouth + TIP_CLEARANCE_ABOVE_HOLE * along
                points.extend(
                    [(hand_for(aligned), False),
                     (hand_for(mouth - INSERTION_DEPTH * along), True)]
                )

            waypoints = []
            for index, (point, cartesian) in enumerate(points):
                rotation = flat if carry_only and index == 0 else upright
                waypoints.append(
                    {"pose": {"position": _vec3_gap(point), "rotation": _quat_gap(rotation)},
                     "cartesian": bool(cartesian)}
                )
            return {"object_name": job.tool, "arm_id": arm_id_of(self.env, arm), "waypoints": waypoints}

        return self.bus.call(read)

    # --------------------------------------------------------------- geometry
    def _feature_world(self, entity: str, feature: str) -> dict[str, Any]:
        """One declared feature's payload, in world coordinates.

        The same conversion ``assetlib.bind``'s feature reader makes for the
        evaluator, done here so the plan aims at the geometry the goal scores
        rather than at a second copy of it: points rotate and translate,
        directions only rotate, radii are neither and pass through. A welded
        fixture has no ``body_q`` row -- its pose is where the compiler put it.
        """
        handle = self.env.handles.entity(entity)
        record = ASSETS.get(self._category(entity))
        spec = record.feature(feature)
        body_q = self._body_q()
        if 0 <= handle.body < len(body_q):
            pose = np.asarray(body_q[handle.body], dtype=np.float64)
        else:
            pose = np.asarray(handle.xform, dtype=np.float64)
        origin, rotation = pose[:3], quat_to_mat(pose[3:7])
        kind = get_kind(spec.kind) if is_registered_kind(spec.kind) else None
        points = set(kind.points if kind else ())
        directions = set(kind.directions if kind else ())
        out: dict[str, Any] = {}
        for name, value in spec.payload.items():
            array = np.asarray(value, dtype=np.float64) if name in points or name in directions else None
            if name in points:
                out[name] = (array @ rotation.T + origin) if array.ndim > 1 else (rotation @ array + origin)
            elif name in directions:
                out[name] = (array @ rotation.T) if array.ndim > 1 else (rotation @ array)
            else:
                out[name] = value
        return out

    def _category(self, entity: str) -> str:
        """The asset record an entity was declared as."""
        doc = self.env.doc
        for spec in getattr(doc, "objects", ()) if doc is not None else ():
            if spec.name == entity:
                return str(spec.type)
        raise KeyError(f"nothing in this document declares {entity!r}")

    def _station(self, entity: str) -> ToolStation:
        category = self._category(entity)
        if category not in TOOL_STATIONS:
            raise KeyError(
                f"{entity!r} is a {category!r}, which has no measured grasp station; "
                f"TOOL_STATIONS knows {sorted(TOOL_STATIONS)}"
            )
        return TOOL_STATIONS[category]

    def _landmark_local(self, job: Job) -> np.ndarray:
        """The point on the tool the plan steers, in the tool's own frame.

        A loop's centre for something that hangs, a rod's far end for something
        that goes into a hole -- read off the record either way, so the routine's
        job table and the goal's operands cannot drift apart.
        """
        if job.placement == "support":
            station = self._station(job.tool)
            if station.placement_local is None:
                raise KeyError(f"{job.tool!r} has no measured support placement landmark")
            return np.asarray(station.placement_local, dtype=np.float64)
        spec = ASSETS.get(self._category(job.tool)).feature(job.tool_feature)
        if job.tool_is_loop:
            return np.asarray(spec.payload["center"], dtype=np.float64)
        return np.asarray(spec.payload["axis"], dtype=np.float64)[0]

    # ---------------------------------------------------------------- solving
    def _hz(self) -> float:
        """The rate a planned stage is actually executed at [Hz].

        The cell's control rate, because ``robot.execute_trajectory``'s streaming
        path feeds one waypoint per control step: the rate a stage is densified
        at *is* the speed it runs at. Densifying at a nominal 30 and streaming at
        60 would run every stage in half its stated time, which on the turn to
        vertical is the difference between a tool that rotates in the jaws and
        one that is flung out of them.
        """
        return stream_hz(self.env.task.physics)

    def _seeded(self, ik):
        """The solver's scratch, seeded from the live posture.

        Through ``.numpy()`` and ``assign`` rather than ``wp.copy``, and that is
        about the fleet rather than about taste. Batched, ``env`` is a
        ``WorldView`` and ``state.joint_q`` is one world's *slice* of the wave's
        array -- an object that answers ``.numpy()`` with this world's
        coordinates and forwards everything else, but is not itself a
        ``wp.array``. ``wp.copy`` type-checks its arguments, so the batched form
        died here on every world's first plan with "Copy source and destination
        must be arrays", one layer under the evaluator miss that used to mask
        it. The seed is a few dozen floats once per plan, so the host round trip
        it costs is not a hot path.
        """
        q_ik = ik.joint_q_ik
        live = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float32).reshape(1, -1)
        q_ik.assign(live)
        return q_ik

    def _converge(self, ik, q_ik, position: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Drive the solver onto one station and return the full joint vector."""
        if ik.solver is None:
            # The FD path, for a model whose joint types make the LM kernels
            # unbuildable. Not this suite today; kept so a scene that grows a
            # cable joint degrades rather than raising inside a graph node.
            seed = np.asarray(q_ik.numpy()[0], dtype=np.float64)
            solved = ik.solve_q_full(seed, position, quat, iterations=120)
            q_ik.assign(np.asarray(solved, dtype=np.float32).reshape(1, -1))
            return np.asarray(solved, dtype=np.float64)
        ik.set_target(position, quat)
        for _ in range(CONVERGE_ROUNDS):
            ik.solver.step(q_ik, q_ik, iterations=ik.iterations)
        return np.asarray(q_ik.numpy()[0], dtype=np.float64)

    @staticmethod
    def _smooth(t: float) -> float:
        return float(t * t * (3.0 - 2.0 * t))

    def _cartesian_stage(self, ik, q_ik, name: str, p0, q0, p1, q1) -> tuple[dict[str, Any], np.ndarray]:
        """One stage, densified in **Cartesian** space and solved per waypoint.

        Not the routine's joint-space smoothstep between two converged ends. Two
        legs make the difference visible: the turn to vertical is a rotation *in
        place*, and interpolating joints between its ends swings the tool through
        a bow of several centimetres while a per-waypoint solve holds the tool
        centre still; and pushing a ring over a hook tip is a straight line, which
        joint interpolation makes an arc that catches the tip.
        """
        seconds = STAGE_SECONDS[name] * self._motion_scale
        steps = max(2, int(round(seconds * self._hz())))
        p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
        q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
        waypoints = []
        for i in range(1, steps + 1):
            alpha = self._smooth(i / steps)
            self._solve_at(ik, q_ik, p0 + alpha * (p1 - p0), quat_slerp(q0, q1, alpha))
            row = q_ik.numpy()[0]
            waypoints.append({"positions": [float(row[c]) for c in ik.arm_coords]})
        end = np.asarray(q_ik.numpy()[0], dtype=np.float64)
        return {"stage": name, "trajectory": {"waypoints": waypoints}, "duration": seconds}, end

    def _solve_at(self, ik, q_ik, position: np.ndarray, quat: np.ndarray) -> None:
        if ik.solver is None:
            seed = np.asarray(q_ik.numpy()[0], dtype=np.float64)
            q_ik.assign(
                np.asarray(ik.solve_q_full(seed, position, quat, iterations=30), dtype=np.float32).reshape(1, -1)
            )
            return
        ik.set_target(position, quat)
        ik.solver.step(q_ik, q_ik, iterations=ik.iterations)

    def _joint_stage(self, ik, name: str, q_from: np.ndarray, q_to: np.ndarray) -> dict[str, Any]:
        """One stage interpolated in joint space, for a leg with no Cartesian shape.

        Two of them: the reach out of the home posture, where a straight line
        between a folded arm and a point over the table is not a motion anybody
        asked for, and the return to it.
        """
        seconds = STAGE_SECONDS[name] * self._motion_scale
        steps = max(2, int(round(seconds * self._hz())))
        waypoints = []
        for i in range(1, steps + 1):
            alpha = self._smooth(i / steps)
            blended = q_from + alpha * (q_to - q_from)
            waypoints.append({"positions": [float(blended[c]) for c in ik.arm_coords]})
        return {"stage": name, "trajectory": {"waypoints": waypoints}, "duration": seconds}

    def _gate(self, ik, errors: dict[str, list[float]], name: str, q, position, quat) -> None:
        pos_err, rot_err = ik.pose_errors(np.asarray(q, dtype=np.float32), position, quat)
        errors[name] = [float(pos_err), float(np.degrees(rot_err))]
        if pos_err > POSITION_LIMIT or rot_err > ROTATION_LIMIT:
            raise RuntimeError(
                f"{name} does not solve: position {pos_err * 1000:.0f} mm, "
                f"rotation {np.degrees(rot_err):.1f} deg (limits "
                f"{POSITION_LIMIT * 1000:.0f} mm, {np.degrees(ROTATION_LIMIT):.0f} deg)"
            )

    # ------------------------------------------------------------------- pick
    def plan_pick(self, object_name: str = "") -> dict[str, Any]:
        """Approach, descend, lift and turn-to-vertical, for one tool.

        Every station is measured off the tool as it lies now, so a re-plan after
        a failed grasp aims at wherever the first attempt left it.

        Returns the stages in execution order, the arm to drive them on, and
        ``expected_landmark`` -- where the tool's own hang point will be once the
        hand reaches the turn station, *if* the jaws take the tool exactly where
        they were aimed. :meth:`plan_place` re-measures rather than believing it;
        the prediction is reported so the difference can be read.
        """

        def build() -> dict[str, Any]:
            job = self._job(tool=object_name)
            self._motion_scale = self._station(job.tool).motion_scale
            arm = self._arm_for(job)
            ik = self.env.arm(arm)
            hand = self.env.scene.embodiment.hands[arm]
            flat = np.asarray(top_down_grasp_site_quat(hand, GRASP_YAW_DEG), dtype=np.float64)
            upright = quat_multiply(quat_from_axis_angle((0.0, 1.0, 0.0), -np.pi / 2.0), flat)

            pose = self.env.handles.entity(job.tool).pose(self._body_q())
            rotation = quat_to_mat(pose[3:7])
            station = self._station(job.tool)
            grasp = np.asarray(pose[:3], dtype=np.float64) + rotation @ np.asarray(
                station.grasp_local, dtype=np.float64
            )
            landmark = np.asarray(pose[:3], dtype=np.float64) + rotation @ self._landmark_local(job)

            lift = self._lift_for(job, pose, grasp)
            approach = grasp + np.array([0.0, 0.0, APPROACH_ABOVE_GRASP])
            # Stop where the fingers can still close. See FINGER_TABLE_CLEARANCE.
            floor = float(self.env.handles.surface_z) + float(hand.tip_reach) + FINGER_TABLE_CLEARANCE
            seat = grasp.copy()
            seat[2] = max(float(seat[2]), floor)

            q_ik = self._seeded(ik)
            live = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float64)
            errors: dict[str, list[float]] = {}

            q_approach = self._converge(ik, q_ik, approach, flat)
            self._gate(ik, errors, "approach", q_approach, approach, flat)
            reach = [self._joint_stage(ik, "approach", live, q_approach)]

            stage, q_grasp = self._cartesian_stage(ik, q_ik, "descend", approach, flat, seat, flat)
            self._gate(ik, errors, "descend", q_grasp, seat, flat)
            reach.append(stage)

            # Two ladders, one per wrist. The rise is done top-down as far as
            # this arm can hold that wrist, and the *turn* does the rest of the
            # rising -- see _holdable_height. The tool sweeps down as it turns,
            # so finishing the rise inside the turn rather than before it is
            # what keeps a 160 mm screwdriver off the table on an arm that
            # cannot hold a top-down wrist at 167 mm.
            wanted_lift = lift
            flat_h, _ = self._holdable_height(ik, q_ik, grasp, lift, flat, FLAT_LIFT_FLOOR)
            lift, _ = self._holdable_height(ik, q_ik, grasp, lift, upright, flat_h)
            waist = grasp + np.array([0.0, 0.0, min(flat_h, lift)])
            lifted = grasp + np.array([0.0, 0.0, lift])

            stage, q_lift = self._cartesian_stage(ik, q_ik, "lift", seat, flat, waist, flat)
            self._gate(ik, errors, "lift", q_lift, waist, flat)
            carry = [stage]

            stage, q_turn = self._cartesian_stage(ik, q_ik, "orient", waist, flat, lifted, upright)
            self._gate(ik, errors, "orient", q_turn, lifted, upright)
            carry.append(stage)

            # Where the landmark rides relative to the hand, if the jaws take the
            # tool where they were aimed. The place re-measures this; reporting
            # the prediction is what makes the slip readable.
            in_hand = quat_to_mat(flat).T @ (landmark - seat)
            expected = lifted + quat_to_mat(upright) @ in_hand
            self._expected[job.tool] = [float(v) for v in expected]
            return {
                "route": "planned",
                "object_name": job.tool,
                "holder_name": job.holder,
                "arm_name": arm,
                "arm_id": arm_id_of(self.env, arm),
                **self._hold([("reach", reach), ("carry", carry)], arm),
                "grasp_point": [float(v) for v in grasp],
                "lift_m": float(lift),
                "lift_wanted_m": float(wanted_lift),
                "expected_landmark": self._expected[job.tool],
                "errors_mm_deg": errors,
            }

        return self.bus.call(build)

    # ------------------------------------------------------------ stage store
    def _hold(self, phases: list[tuple[str, list[dict[str, Any]]]], arm: str) -> dict[str, Any]:
        """Keep the dense trajectories, hand back only what a router reads."""
        arm_id = arm_id_of(self.env, arm)
        for phase, stages in phases:
            self._phases[phase] = {"stages": stages, "arm_id": arm_id}
        return {
            "phases": {phase: [str(s["stage"]) for s in stages] for phase, stages in phases},
            "seconds": float(sum(float(s["duration"]) for _, stages in phases for s in stages)),
        }

    def get_phase(self, phase: str) -> dict[str, Any]:
        """One phase of the last plan, as a **single** joint trajectory.

        Concatenated rather than handed over stage by stage, and the reason is
        the step budget. Each ``robot.execute_trajectory`` call ends in a settle
        that is floored at 60 control steps whatever it is asked for, so a nine
        stage plan pays up to 540 steps standing still -- against a document that
        truncates at 2400 for two tools. Merging the stages of a phase costs
        nothing (they are already contiguous: each begins where the last ended)
        and buys back most of that.

        Args:
            phase: ``reach``, ``carry``, ``place`` or ``retreat``.
        """
        held = self._phases.get(str(phase))
        if held is None:
            raise KeyError(f"nothing has planned a phase called {phase!r}; have {sorted(self._phases)}")
        waypoints: list[dict[str, Any]] = []
        seam = int(SEAM_HOLD_STEPS.get(str(phase), 0))
        for index, stage in enumerate(held["stages"]):
            rows = stage["trajectory"]["waypoints"]
            # A seam is where one stage's motion becomes the next one's, and
            # merging them removed the only thing that used to stand between:
            # the settle at the end of an execute_trajectory call. Repeating the
            # last row holds the arm there for SEAM_HOLD_STEPS control steps --
            # the settle back, at its real cost rather than the 60-step floor
            # per call this merge exists to avoid.
            if index and waypoints and seam:
                waypoints.extend([dict(waypoints[-1])] * seam)
            waypoints.extend(rows)
        return {
            "phase": str(phase),
            "arm_id": int(held["arm_id"]),
            "stages": [str(s["stage"]) for s in held["stages"]],
            "trajectory": {"waypoints": waypoints},
            "duration": float(sum(float(s["duration"]) for s in held["stages"])),
        }

    def _holdable_height(
        self, ik, q_ik, grasp: np.ndarray, wanted: float, quat: np.ndarray, floor: float
    ) -> tuple[float, float]:
        """The highest station at or below ``wanted`` this arm can hold ``quat`` at.

        The tool's own box says how far it has to rise before it can be stood on
        end without ploughing the table; the arm says how far it can hold a given
        wrist out over the work surface, and the two disagree by a lot. Measured
        on the fatter screwdriver, at its 167 mm turn station: the **upright**
        pose solves to 0.001 mm and the **top-down** one parks 43 mm out. Same
        point, same arm, two wrists. So the two are laddered separately and the
        turn does the last of the rising, which is the shape
        ``cloth_fold``'s ``_holdable_roll`` argues for -- screen a leg at the
        pose it will actually be held in, and take the first that holds.

        Accepted at :data:`HOLD_TOLERANCE` rather than at the plan gate's 45 mm,
        deliberately. A station that solves 42 mm out passes the gate and then
        drops the tool on the way to it, which is what the screwdriver's second
        insertion did: grasped, lifted, and gone before the turn finished.

        The scratch is restored before returning, so the chain the caller is
        building goes on from the station it was already at.

        Returns:
            The height chosen [m] and the position error it solves to [m].
        """
        keep = np.asarray(q_ik.numpy()[0], dtype=np.float32).copy()
        best: tuple[float, float] | None = None
        height = float(wanted)
        while height >= floor - 1.0e-9:
            point = grasp + np.array([0.0, 0.0, height])
            q_ik.assign(keep.reshape(1, -1))
            q = self._converge(ik, q_ik, point, quat)
            pos_err, rot_err = ik.pose_errors(np.asarray(q, dtype=np.float32), point, quat)
            if best is None or pos_err < best[1]:
                best = (height, float(pos_err))
            if pos_err <= HOLD_TOLERANCE and rot_err <= ROTATION_LIMIT:
                q_ik.assign(keep.reshape(1, -1))
                return height, float(pos_err)
            height -= HOLD_LADDER_STEP
        q_ik.assign(keep.reshape(1, -1))
        return (float(best[0]), float(best[1])) if best else (float(wanted), float("inf"))

    def _lift_for(self, job: Job, pose: np.ndarray, grasp: np.ndarray) -> float:
        """How far the tool rises before it is turned on end [m].

        Its own reach past the grasp point plus a margin, from the body-frame
        box: a screwdriver held 145 mm up its shaft must clear 145 mm before its
        tip can swing under the hand, and a tool that does not is turned through
        the table. The three variants' hand-picked lifts -- 150, 165 and 180 mm --
        are this quantity, measured once each and then frozen.
        """
        obb = self.env.handles.entity(job.tool).obb
        rotation = quat_to_mat(pose[3:7])
        origin = np.asarray(pose[:3], dtype=np.float64)
        if obb is None:
            return APPROACH_ABOVE_GRASP + LIFT_CLEARANCE
        half = np.asarray(obb.half, dtype=np.float64)
        centre = np.asarray(obb.center_local, dtype=np.float64)
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
        corners = origin + (centre + signs * half) @ rotation.T
        reach = float(np.max(np.linalg.norm(corners - grasp, axis=1)))
        return max(APPROACH_ABOVE_GRASP, reach + LIFT_CLEARANCE, self._station(job.tool).min_lift)

    # ------------------------------------------------------------------ place
    def plan_place(self, object_name: str = "") -> dict[str, Any]:
        """The hang or the insertion, around where the tool **actually** sits.

        The closed-loop half, and the reason this graph is not the routine with
        extra steps. The hand is at the turn station holding the tool; this reads
        the tool's live pose, computes where its landmark rides in the hand *now*,
        and puts the stations where that landmark lands on the holder. The routine
        computes the same stations from the pre-grasp measurement and never looks
        again, so every millimetre the close shifted the tool -- and the swing the
        turn adds -- is spent at the hook.

        Two shapes, chosen by which side of the goal clause the tool is on:

        * a **loop** on the tool goes over a rod. Stage in front of the tip, push
          the loop past it, lower onto the arm.
        * a **rod** on the tool goes into a loop on the fixture. Rise clear of the
          holder, cross to above the hole, drive the tip down into it.
        """

        def build() -> dict[str, Any]:
            job = self._job(tool=object_name)
            arm = self._arm_for(job)
            ik = self.env.arm(arm)
            q_ik = self._seeded(ik)
            here, here_quat = ik.ee_pose(self.env.state)
            here = np.asarray(here, dtype=np.float64)
            upright = np.asarray(here_quat, dtype=np.float64)

            pose = self.env.handles.entity(job.tool).pose(self._body_q())
            landmark = np.asarray(pose[:3], dtype=np.float64) + quat_to_mat(pose[3:7]) @ self._landmark_local(job)
            # Where the landmark rides in the hand, measured rather than assumed.
            in_hand = quat_to_mat(upright).T @ (landmark - here)

            def hand_for(target: np.ndarray) -> np.ndarray:
                """The tool-centre point that puts the landmark on ``target``."""
                return np.asarray(target, dtype=np.float64) - quat_to_mat(upright) @ in_hand

            errors: dict[str, list[float]] = {}
            stages: list[dict[str, Any]] = []
            if job.placement == "support":
                region = self.env.handles.regions[job.region]
                support_origin = region.origin_now(self._body_q())
                # The triangular support's tip relative to its compiled frame,
                # measured once from the same mesh used by the scripted policy.
                support_tip = support_origin + np.array([0.10249, -0.16340, 0.01659])
                seated = support_tip + np.array([0.003, 0.0, 0.018])
                staged = seated + np.array([-0.048, 0.0, 0.0])
                lowered = seated + np.array([0.0, 0.0, -0.015])
                away = lowered + np.array([-RETRACT_FROM_BOARD, 0.0, RETRACT_UPWARD])
                legs = [
                    ("pre_hook", hand_for(staged)),
                    ("over_hook", hand_for(seated)),
                    ("onto_hook", hand_for(lowered)),
                ]
                release_at = hand_for(lowered)
                retract_to = away - quat_to_mat(upright) @ in_hand
                target = seated
            elif job.tool_is_loop:
                rod = self._feature_world(job.holder, job.holder_feature)
                tool_loop = self._feature_world(job.tool, job.tool_feature)
                axis = np.asarray(rod["axis"], dtype=np.float64)
                tip, inboard = axis[0], axis[1] - axis[0]
                inboard = inboard / max(float(np.linalg.norm(inboard)), 1.0e-12)
                station = self._station(job.tool)
                above = station.loop_above_tip
                seated = tip + station.hook_past * inboard + np.array([0.0, 0.0, above])
                staged = seated - (PRE_HOOK_STANDOFF + station.hook_past) * inboard
                # Down until the loop rests on the rod, plus a preload: the
                # loop's own hanging radius is declared, so this is measured
                # rather than picked. See SEAT_PRELOAD.
                hang_radius = max(0.0, float(tool_loop["inner_r"]) - float(rod["radius"]))
                lowered = seated - np.array([0.0, 0.0, above + hang_radius + SEAT_PRELOAD])
                away = lowered - RETRACT_FROM_BOARD * inboard + np.array([0.0, 0.0, RETRACT_UPWARD])
                legs = [
                    ("pre_hook", hand_for(staged)),
                    ("over_hook", hand_for(seated)),
                    ("onto_hook", hand_for(lowered)),
                ]
                release_at = hand_for(lowered)
                retract_to = away - quat_to_mat(upright) @ in_hand
                target = seated
            else:
                hole = self._feature_world(job.holder, job.holder_feature)
                mouth = np.asarray(hole["center"], dtype=np.float64)
                # **Along the shaft's own axis, not straight down.** ``threaded``
                # asks where the rod's *axis line* crosses the hole's disc, and
                # the plan steers the rod's *tip*: 60 mm apart, so a shaft
                # hanging six degrees off vertical puts the tip dead centre and
                # the crossing 6.6 mm out -- past a 5.2 mm budget on the fatter
                # screwdriver, which is exactly the miss this variant showed.
                # Driving along the measured axis puts the whole line through
                # the hole centre, and makes the stroke axial rather than a
                # six-degree wedge into a 9 mm bore.
                shaft = np.asarray(self._feature_world(job.tool, job.tool_feature)["axis"], dtype=np.float64)
                along = shaft[1] - shaft[0]
                along = along / max(float(np.linalg.norm(along)), 1.0e-12)
                aligned = mouth + TIP_CLEARANCE_ABOVE_HOLE * along
                seated = mouth - INSERTION_DEPTH * along
                # Clear the rack's own height before crossing to it, at this
                # hand's current stand-off from the board.
                raised = hand_for(aligned).copy()
                raised[0], raised[1] = here[0], here[1]
                legs = [
                    ("raise", raised),
                    ("above_hole", hand_for(aligned)),
                    ("into_hole", hand_for(seated)),
                ]
                release_at = hand_for(seated)
                retract_to = release_at + np.array([0.0, 0.0, RETRACT_ABOVE_HOLE])
                target = seated

            start, q_end = here, None
            for name, point in legs:
                stage, q_end = self._cartesian_stage(ik, q_ik, name, start, upright, point, upright)
                self._gate(ik, errors, name, q_end, point, upright)
                stages.append(stage)
                start = point

            stage, q_end = self._cartesian_stage(ik, q_ik, "retract", release_at, upright, retract_to, upright)
            self._gate(ik, errors, "retract", q_end, retract_to, upright)
            retreat = [stage]

            home = np.asarray(q_end, dtype=np.float64).copy()
            for coord, value in zip(ik.arm_coords, self.env.handles.arms[arm].home_q, strict=True):
                home[coord] = float(value)
            going_home = [self._joint_stage(ik, "home", np.asarray(q_end, dtype=np.float64), home)]

            expected = self._expected.get(job.tool)
            slip = (
                [float(landmark[i] - expected[i]) * 1.0e3 for i in range(3)]
                if expected is not None
                else [0.0, 0.0, 0.0]
            )
            return {
                "route": "planned",
                "object_name": job.tool,
                "arm_name": arm,
                "arm_id": arm_id_of(self.env, arm),
                **self._hold([("place", stages), ("retreat", [*retreat, *going_home])], arm),
                "target": [float(v) for v in target],
                "in_hand_mm": [float(v) * 1.0e3 for v in in_hand],
                # How far the tool moved in the jaws between the plan and now.
                # The quantity the routine cannot see and this graph corrects.
                "slip_mm": slip,
                "errors_mm_deg": errors,
            }

        return self.bus.call(build)

    # ------------------------------------------------------------------ state
    def get_hang_state(self, object_name: str = "") -> dict[str, Any]:
        """Where this tool is against where its own goal clause wants it.

        The verdict is the clause's own atom over the same declared features the
        evaluator reads, so a node routing on it and the score cannot disagree
        about what hung means. ``offset`` and ``clearance`` come with it, because
        a near miss and a wild one are different situations and a bare boolean
        cannot say which happened.
        """

        def read() -> dict[str, Any]:
            job = self._job(tool=object_name)
            arm = self._arm_for(job)
            pose = self.env.handles.entity(job.tool).pose(self._body_q())
            landmark = np.asarray(pose[:3], dtype=np.float64) + quat_to_mat(pose[3:7]) @ self._landmark_local(job)
            held, carried, jaw_gap, jaw_shut = self._grip_state(job.tool, arm)
            expected = self._expected.get(job.tool)
            metrics = self.bus.snapshot.info.get("metrics") or {}
            if job.placement == "support":
                contained = any(float(metrics.get(k, 0.0)) >= 1.0 for k in job.keys)
                verdict = {
                    "threaded": contained,
                    "crossings": contained,
                    "offset": 0.0 if contained else float("inf"),
                    "clearance": 0.0 if contained else float("-inf"),
                }
            else:
                tool = self._feature_world(job.tool, job.tool_feature)
                holder = self._feature_world(job.holder, job.holder_feature)
                loop, rod = (tool, holder) if job.tool_is_loop else (holder, tool)
                verdict = (threaded if job.atom == "threaded" else encloses)(loop, rod)
                if job.atom == "encloses":
                    verdict = {
                        "threaded": bool(verdict["encloses"]),
                        "crossings": bool(verdict["encloses"]),
                        "offset": verdict["offset"],
                        "clearance": verdict["clearance"],
                    }
            # Every branch above normalises onto the strict atom's key, so this
            # reads the same whether the clause said `threaded`, `encloses` or
            # `containment`.
            seated = bool(verdict["threaded"])
            return {
                "route": "hung" if seated else "short",
                "object_name": job.tool,
                "holder_name": job.holder,
                "arm_name": arm,
                "arm_id": arm_id_of(self.env, arm),
                "held": bool(held),
                "carried": bool(carried),
                "jaw_gap_mm": float(jaw_gap) * 1.0e3,
                "jaw_shut_mm": float(jaw_shut) * 1.0e3,
                # Something between the pads: they stalled short of where the
                # same command reaches on air. What a *pre-lift* check can read,
                # because contact needs load and a tool still on the table
                # supplies none.
                "pinched": bool(jaw_gap > jaw_shut + JAW_STALL_MARGIN),
                # Named for the strict atom because that is what the graph's
                # seat check reads; it means "satisfies this clause's own atom".
                "threaded": seated,
                # ``encloses`` reports no plane crossing -- not requiring one is
                # the whole difference between it and ``threaded``.
                "crossings": bool(verdict.get("crossings", seated)),
                "offset_mm": float(verdict["offset"]) * 1.0e3 if np.isfinite(verdict["offset"]) else -1.0,
                "clearance_mm": float(verdict["clearance"]) * 1.0e3,
                "landmark": [float(v) for v in landmark],
                "expected_landmark": list(expected) if expected is not None else [],
                "slip_mm": (
                    float(np.linalg.norm(landmark - np.asarray(expected, dtype=np.float64))) * 1.0e3
                    if expected is not None
                    else -1.0
                ),
                "scored": any(bool(metrics.get(k, False)) for k in job.keys),
            }

        return self.bus.call(read)

    def _grip_state(self, object_name: str, arm: str) -> tuple[bool, bool, float, float]:
        """Is the tool in the hand -- by contact, and by where it actually is.

        Contact alone is the test the evaluator's grasp predicate uses, and on
        *this* hand and these tools it is not enough on its own. Measured across
        a clearance sweep: the arm lifted both spanners 30 cm off the table with
        the jaws stalled on them, and the contact test read false at every
        instant it was sampled -- a pinch on a thin flat shaft does not
        necessarily push both pads outwards within the angle the predicate
        allows. Routing a carry on that alone abandons a tool the hand is
        holding.

        So contact is one of two signals, and the other is ground truth about
        the world: the tool is off the surface and travelling with the hand.
        That cannot be faked by a tool resting on scenery (it has to be near
        *this* tool centre) and it cannot be missed the way an impulse can.

        Returns ``(held, carried, jaw_gap, jaw_shut)`` -- the verdict, the
        geometric half of it on its own, and the jaw measurements a *pre-lift*
        check reads instead, since nothing on a table loads a pad.
        """
        handle = self.env.handles.arms.get(arm)
        gripper = self.env.scene.embodiment.grippers.get(arm)
        jaw_gap = float(gripper.width(self.env.state)) if gripper is not None else 0.0
        jaw_shut = float(gripper.width_for_command(1.0)) if gripper is not None else 0.0

        contact = False
        contacts = getattr(self.env, "contacts", None)
        if contacts is not None and handle is not None and handle.finger_groups:
            contact = bool(
                contacts.check_grasp(
                    handle.finger_groups,
                    self.env.handles.entity(object_name).collision_shapes,
                    np.asarray(self.env.state.body_q.numpy(), dtype=np.float64),
                    open_axis_signs=handle.open_axis_signs,
                )
            )

        entity = self.env.handles.entity(object_name)
        pose = entity.pose(self._body_q())
        carried = False
        if entity.obb is not None:
            rotation = quat_to_mat(pose[3:7])
            half = np.asarray(entity.obb.half, dtype=np.float64)
            centre = np.asarray(pose[:3], dtype=np.float64) + rotation @ np.asarray(
                entity.obb.center_local, dtype=np.float64
            )
            signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
            lowest = float(np.min((centre + (signs * half) @ rotation.T)[:, 2]))
            tcp = np.asarray(self.env.arm(arm).ee_pose(self.env.state)[0], dtype=np.float64)
            carried = bool(
                lowest > float(self.env.handles.surface_z) + OFF_SURFACE
                and float(np.linalg.norm(centre - tcp)) < CARRY_RADIUS
            )
        return bool(contact or carried), carried, jaw_gap, jaw_shut

    # ---------------------------------------------------------------- helpers
    def _body_q(self) -> np.ndarray:
        snapshot = self.bus.snapshot
        if snapshot.body_q is not None:
            return snapshot.body_q
        return np.asarray(self.env.state.body_q.numpy(), dtype=np.float64)
