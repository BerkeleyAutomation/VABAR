# SPDX-License-Identifier: Apache-2.0
"""The benchmark's own scripted sort, over this package's OSC action space.

A port of ``VAB-industrial/mujoco/run_tool_sort.py`` -- ``sort_one`` and
``post_upright`` -- which is the baseline the tool-sort variants were authored
against. The point of porting it rather than writing a routine that merely
succeeds is that the two runs are then *comparable*: same grasp heuristic, same
waypoints, same per-step Cartesian command law, so a variant this scores badly on
is a variant the benchmark's baseline scores badly on, and the difference between
the two numbers is the simulator rather than the policy.

Three things were converted rather than copied, and each is a claim worth
checking:

* **Holds are stated in seconds, not steps.** The reference holds each waypoint
  60 control steps at 20 Hz; this suite runs at 50 Hz. Copying the step count
  would give every leg 1.2 s instead of 3 s, and since the arm's speed is set by
  the impedance gains rather than by the control rate (see
  :mod:`robosimstudio.solvers.motion.osc`) a leg would simply stop 60 % short.
  So the reference's number is stated as the duration it is -- 3 s -- and the
  step count follows the environment's rate.
* **The pile is re-read every round.** Tools start stacked, so the order is
  recomputed from live z before each pick rather than fixed up front. That is the
  reference's one concession to clutter and it is kept; the descent is still
  blind to neighbours, so a buried tool can still foul the gripper.
* **Compartment geometry comes from the task's regions.** The reference derives
  the drop point from ``bin2_pos`` plus a quadrant offset; here the same point is
  the centre of the ``RegionSpec`` the ``.bddl`` file declares. They agree by
  construction -- the suite transcribes the file's own ranges -- and going
  through the region means a routine and an evaluator cannot disagree about where
  a compartment is.

**Measured against the reference's published baselines**, at its own
``--handle_frac 0.0``, episode 0::

    variant     reference   this port
    hammer          1.0       1.0      (success at step 784 of 2400)
    syringe         1.0       1.0      (success at step 907 of 11400)
    sort_all       ~0.12      0.25     (hammer sorted; drill, clamp, round_nut not)

``sort_all`` reads higher for the reason its own docstring gives: the reference's
number is over eight goals and this export carries one instance per type, so both
runs sort exactly one tool -- the hammer -- and fail the same three. Those three
are ungraspable by construction, and the reference's limits table says so to the
millimetre: against an 0.08 m span, the drill is 0.211 m across, the clamp 0.147
and the round nut 0.108. A failed pick still disturbs the pile, which is why the
hammer must be attempted after the drill has already been dropped on it.

**It names its own control mode.** Every action it emits is tagged ``osc_pose``,
so it runs on a variant whose declared mode is ``joint_position`` for the sake of
its shipped graph without either having to give way::

    env = Env("tool_sort/hammer")     # declares joint_position, for the graph
    policy = PolicyScripted(env)      # emits osc_pose, and says so

The environment carries both controllers and hands the arm over on the first
tagged action, re-seeded -- see
:meth:`~robosimstudio.core.env.EnvBase.use_control`. Which is the point: a graph
that plans joint trajectories and a routine that servos Cartesian deltas are two
baselines on *one* scene, and making them two scenes would have made them
incomparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import newton
import numpy as np

from robosimstudio.core.math import quat_from_axis_angle, quat_multiply, quat_slerp, quat_to_mat
from robosimstudio.solvers import (
    ActionOsc,
    OscWaypoints,
    Waypoint,
    grasp_from_obb,
    grasp_quat,
    osc_action,
    top_down_grasp_quat,
)
from robosimstudio_bench.goals.goal import placements

from .grasps import FIT_MARGIN, POS_TOL, Grasp, _Leg, face_grasps, support

__all__ = ["GRASP_HANDLE_FRAC", "RoutineSortFaces", "RoutineSyringe", "RoutineToolSort"]

GRASP_HANDLE_FRAC = -0.70
"""Slide the grasp 52.5 mm along the syringe towards its plunger flange.

What :class:`RoutineSyringe` changes and nothing else. **Grasping the centroid
stopped working when the collider stopped being a box, and this is the whole of
that story.** `75a953f` replaced the VAB export's collision boxes with convex
hulls of the drawn mesh -- right, because those boxes stood up to 27 mm proud of
what a viewer sees. The syringe's box was 23.6 mm wide along its **entire**
length, so closing to 19.6 mm bit into it anywhere, centroid included. The hull
is not:

    along the long axis          hull width
    -75 .. -45 mm (needle)          0.4 mm
    -45 .. -15 mm (barrel)         10.7 mm
    -15 .. +45 mm                  no vertices -- interpolated
    +45 .. +75 mm (flange)         23.6 x 13.6 mm

`_grip_command` reads its closing width off the **OBB**, and 23.6 mm is the
*flange*. At the centroid the hull is nowhere near that, so the jaws swept shut
to 19.6 mm through open air, the arm lifted nothing, and `_post_legs` then
reported the syringe "hangs 36.8 cm below the gripper" -- which was it measuring
to an object still lying on the bench. Bisected: 3/3 at `76c93e3`, 0/3 at
`75a953f`, and 0/3 every commit since.

-0.70 puts the grasp at +52.5 mm, on the flange, where the hull really is the
width the grip assumes. Measured, three episodes each, with the grip command
**unchanged**: centroid lifts 0.1 mm; +45 mm lifts 195.9; +52.5 mm lifts 197.6;
+60 mm lifts 188.6. The window is wide and this sits inside it.

**It does not make the task pass, and nothing here claims to.** The syringe is
now picked up and carried and then lost during `over_hole`, the leg that
translates and rotates it upright at once. That failure is indifferent to every
knob reachable from out here -- grip 0.755 to 1.00, friction mu 1 to 10, leg
duration 1x to 2x, offset +45 to +60 mm all lift it and all drop it. Rotating a
150 mm syringe upright while holding one end swings its mass through a long lever
against position-servoed fingers, and fixing that means changing the *sequence*
(rotate before lifting, or grasp near the centre of mass and give the collider
something to hold there) rather than tuning a number.
"""

HOLD_SECONDS = 3.0
"""Seconds per waypoint -- the reference's ``--hold 60`` at its 20 Hz."""

APPROACH_UP = 0.14
"""Height above the grasp point the approach hovers at [m]."""
LIFT_UP = 0.20
"""Height above the grasp point a grasped tool is lifted to [m]."""
CLEARANCE = 0.16
"""Height above the release point the transit passes through [m]."""
COMPARTMENT_DROP = 0.14
"""Release height above a compartment's floor [m].

The bin's walls keep a bouncing tool in, so the reference fills one from a
height rather than lowering into it.
"""
CONTAINER_DROP = 0.025
"""Release height above a *container's* rim [m].

Not 0.14: a tool let go that far above a lid either bounces off it or misses the
hole. 0.025 still clears the fingers, whose collision boxes reach about 0.012 m
below the grip site.
"""
POST_CLEARANCE = 0.01
"""How far above the rim a posted tool's lower end is brought before release [m]."""

OVER_HOLE_LEGS = 8
"""How many waypoints the carry-to-the-hole leg is flown as.

**One waypoint does not fly this leg, it fires it.** The setpoint is rate-limited
per axis and the two limits are not the same trip: at these gains the wrist turns
at 3.06 rad/s and the hand crosses at 0.306 m/s, so a single waypoint asking for
90 degrees and 470 mm at once finishes its rotation in the first ~26 steps and
spends the remaining ~80 translating, tool already upright. Standing a 150 mm
syringe on end that early is what put the whole of its weight along its own axis
while the arm was still accelerating sideways.

Interpolating the pose ties the two together -- each leg is an eighth of the turn
and an eighth of the reach -- and the arm then tracks the path instead of
overshooting it: measured on ``syringe``, one waypoint leaves the hand **267 mm**
from the aperture at release, having swung past it to y=0.59 against a target of
0.16.

**Eight rather than four, and the reason is repeatability rather than reach.**
Four gets the tool into the box, but the episode is bimodal -- six sequential
runs went 5/6, the successes landing at 1365-1415 steps and the failure at 1651,
which is the goal's one retry being spent. Eight is 6/6 at **exactly 1958 steps,
every run**, on a stack whose own note says the same configuration can produce
different outcomes. Paying 543 steps against an 11400 budget for a leg that
stops being a coin flip is the trade this variant wants; a benchmark's scripted
solution exists to be reproducible.

One is kept reachable through the label, so a single-waypoint leg still reads
``over_hole`` rather than ``over_hole1``.

Every other variant releases over an open compartment and never reaches this
branch, so nothing else in the suite changes length."""

_UPRIGHT_ROTATIONS = ((0.0, 0.0, np.pi / 2.0), (np.pi / 2.0, 0.0, 0.0))
"""World rotations off the home pose that stand a held tool on end, left to right.

The reference's own composition, and its reasoning is geometric rather than
tunable: home points the gripper down with the fingers along world y, so the tool
it holds lies along world x, and standing that up needs a rotation about y -- but
a bare 90 degrees about y also swings the gripper round to reach in from +x, away
from the arm's base, which the Panda cannot hold over the far side of the bench.
Yawing about z first puts the tool along y, so the stand-up becomes a rotation
about x and the gripper reaches in +y with the arm on the near side. Same tool
orientation, a pose the arm holds to under a millimetre.
"""


def _up(height: float) -> np.ndarray:
    """A vertical offset, as a vector -- so the additions below read as geometry."""
    return np.array([0.0, 0.0, float(height)])


class RoutineToolSort:
    """Pick each goal's tool out of the pile and place it in its region.

    Args:
        env: The environment to drive. It must offer an :data:`MODE` controller,
            which every scene with a gripper does.
        arm: Which arm sorts. Empty uses the first.
        handle_frac: How far along the long axis to slide the grasp from the
            centroid towards the light end, as a fraction of the half-extent.
            0.0 because that is what the reference's *recorded baselines* use
            (``--handle_frac 0.0``, stated under its task table), not the 0.5 its
            scripts happen to carry as an argparse default. Comparing against a
            published number means running the configuration it was published
            at.
        yaw_offset_deg: Added to the OBB's handle yaw. The reference exposes this
            for the case where the gripper closes *along* a handle instead of
            across it; 0 is its default.
        hold_seconds: Duration of one waypoint. See :data:`HOLD_SECONDS`.
        grip_squeeze: How far inside a tool's own width the fingers are told to
            close [m]. See :meth:`_grip_command` -- this only applies to a tool
            whose box the gripper actually fits.
        only: Comma-separated subset of object names to sort, for a run that
            wants one tool rather than a whole variant. Empty sorts every goal.
    """

    MODE = "osc_pose"
    """The controller this routine's actions are for.

    Named on every action rather than required of the environment. A task variant
    declares whichever mode its shipped graph speaks, and the runtime's own
    commands select the joint controller for the same reason -- so a scene can
    carry a graph and this routine without either dictating the other's action
    space. See :meth:`~robosimstudio.core.env.EnvBase.use_control`.
    """

    def __init__(
        self,
        env,
        *,
        arm: str = "",
        handle_frac: float = 0.0,
        yaw_offset_deg: float = 0.0,
        hold_seconds: float = HOLD_SECONDS,
        grip_squeeze: float = 0.004,
        only: str = "",
    ) -> None:
        space = env.controllers.get(self.MODE)
        if not isinstance(space, ActionOsc):
            raise ValueError(
                f"{type(self).__name__} emits normalized OSC deltas, and {env.task.full_name!r} "
                f"offers no {self.MODE!r} controller (it has {sorted(env.controllers)}). "
                f"An embodiment with no gripper or with arms of unequal width can be missing one."
            )
        self.env = env
        self.arm_name = arm or next(iter(env.handles.arms))
        self.handle_frac = float(handle_frac)
        self.yaw_offset = np.radians(float(yaw_offset_deg))
        self.hold = max(1, int(round(float(hold_seconds) * float(env.task.physics.control_hz))))
        self.grip_squeeze = float(grip_squeeze)

        # Asked of the goal, not of a copy beside it. This used to read
        # `task.metadata["goals"]` -- a hand-written duplicate of the pairs the
        # goal already states, which could disagree with it and produce a routine
        # diligently sorting into compartments nobody was scoring.
        goals = placements(env.evaluator.goal)
        if not goals:
            raise ValueError(f"task {env.task.full_name!r} declares no goals for a routine to sort")
        wanted = {name.strip() for name in only.split(",") if name.strip()}
        if wanted:
            unknown = wanted - {obj for obj, _ in goals}
            if unknown:
                raise ValueError(f"no goal names {sorted(unknown)}; have {sorted(obj for obj, _ in goals)}")
            goals = tuple((obj, region) for obj, region in goals if obj in wanted)
        self.goals = goals

        self.home_quat = np.asarray(env.arm(self.arm_name).home_pose[1], dtype=np.float64)
        self.containers = dict((env.handles.meta or {}).get("containers") or {})

        self.remaining = [obj for obj, _ in self.goals]
        self.region_of = dict(self.goals)
        self.track = OscWaypoints(output_max=space.controllers[0].output_max)
        self.tool = ""
        self.message = ""
        self.legs: list[str] = []
        self.finished = False
        self._posted = False
        """Whether the current tool's container post has been planned."""

    # ------------------------------------------------------------------- state
    @property
    def ik(self):
        return self.env.arm(self.arm_name)

    @property
    def active(self) -> bool:
        """Whether there is work left. What a teleop panel stops stepping on."""
        return not self.finished

    @property
    def phase(self) -> str:
        """Which leg is being flown, for a readout."""
        if self.finished:
            return "done"
        leg = self.track.current
        return leg.label if leg is not None else "idle"

    def plan(self) -> str:
        """Start the first tool and say what is being attempted.

        The shipped-routine interface a teleop panel drives: it calls this once,
        then :meth:`act` per step. Planning is per *tool* rather than up front --
        every pick disturbs the pile, so a plan made now for the fourth tool
        would be a plan against a layout that no longer exists by the time it
        runs.
        """
        self._advance()
        return self.message

    def reset(self) -> None:
        """Forget the plan. The next :meth:`act` re-plans against the new pile."""
        self.remaining = [obj for obj, _ in self.goals]
        self.track.load([])
        self.tool = ""
        self.message = ""
        self.finished = False
        # Left set, an episode interrupted mid-post would make the next one skip
        # the post entirely and drop its tool flat on the container's lid.
        self._posted = False

    # -------------------------------------------------------------- geometry
    def _pick_order(self) -> list[str]:
        """Outstanding tools, top of the pile first.

        They start stacked on one another, so lifting the buried one first drags
        whatever is on top of it. Read from live z each round rather than fixed
        up front, since every pick disturbs the pile.
        """
        body_q = self.env.state.body_q.numpy()
        return sorted(self.remaining, key=lambda name: -float(self.env.handles.objects[name].pose(body_q)[2]))

    def _grasp(self, tool: str) -> tuple[np.ndarray, np.ndarray]:
        """Where and how to take one tool, from its oriented bounding box."""
        handle = self.env.handles.objects[tool]
        if handle.obb is None:
            raise ValueError(f"object {tool!r} has no OBB; its spec sets obb_grasp=False")
        body_q = self.env.state.body_q.numpy()
        position, handle_yaw = grasp_from_obb(handle.obb, handle.pose(body_q), self.handle_frac)
        return position, top_down_grasp_quat(self.home_quat, handle_yaw + self.yaw_offset)

    def _grip_command(self, tool: str) -> float:
        """How hard to close on one tool: 0 open .. 1 fingers together.

        The reference's ``CLOSE`` is ``+1`` on a *torque*-controlled gripper, so it
        means "squeeze at a fixed force". Here the fingers are position-servoed
        (``target_ke=1500``), so 1 means "drive the pads to touching" and the grip
        force is whatever the servo makes of the standing error -- which for a
        smooth 24 mm cylinder is enough to extrude it out of the jaws. Measured on
        ``tool_sort/syringe``: grasped and lifted at a 22.7 mm opening, then
        squeezed to 1.0 mm and dropped during the leg that turns it upright.

        So a tool whose box the gripper actually fits is closed on to just inside
        its own width, which is a real grip with a bounded force. A tool whose box
        it does *not* fit has no trustworthy width to use -- the box is the whole
        object, and a claw hammer reads 132 mm across a 36 mm handle -- so those
        close fully, which is what the reference does and what works for them.
        """
        obb = self.env.handles.objects[tool].obb
        if obb is None or not obb.fits_gripper:
            return 1.0
        span = float(self.env.gripper(self.arm_name).span)
        separation = max(0.0, obb.grasp_width - self.grip_squeeze)
        return float(np.clip(1.0 - separation / span, 0.0, 1.0))

    def _container_of(self, region_name: str) -> dict | None:
        """The container a goal region belongs to, or ``None`` for a compartment.

        A compartment is open from above, so releasing a tool over it works. A
        sharps box is a sealed shell whose lid is solid everywhere but one hole,
        and a tool let go flat over it lands *on* the lid -- which reads as a
        completed script and scores zero. The two destinations need different
        endings, so they are told apart before the waypoints are built.
        """
        for name, entry in self.containers.items():
            if region_name == f"{name}_region":
                return entry
        return None

    def _release_pose(self, region_name: str) -> tuple[np.ndarray, np.ndarray, dict | None]:
        """The point and orientation to release at, plus the container if any."""
        body_q = self.env.state.body_q.numpy()
        region = self.env.handles.regions[region_name]
        centre = region.center(body_q)
        container = self._container_of(region_name)
        if container is None:
            floor = float(region.origin_now(body_q)[2] + region.z_range[0])
            return np.array([centre[0], centre[1], floor + COMPARTMENT_DROP]), self.home_quat, None
        aperture_x, aperture_y, _radius = (float(v) for v in container["aperture"])
        rim = float(container["rim_z"])
        return np.array([aperture_x, aperture_y, rim + CONTAINER_DROP]), self._upright_quat(), container

    def _upright_quat(self) -> np.ndarray:
        """The wrist pose that stands a held tool on end. See :data:`_UPRIGHT_ROTATIONS`."""
        quat = self.home_quat
        for axis_angle in _UPRIGHT_ROTATIONS:
            vector = np.asarray(axis_angle, dtype=np.float64)
            quat = quat_multiply(quat_from_axis_angle(vector, float(np.linalg.norm(vector))), quat)
        return quat / np.linalg.norm(quat)

    def _tool_lowest_z(self, tool: str) -> float:
        """World z of the lowest corner of a tool's box, as it hangs right now.

        Measured rather than predicted from the grasp: the tool settles in the
        fingers differently depending on where along it they closed, and when it
        is being posted through a hole a centimetre of that is the difference
        between going in and catching the rim.
        """
        handle = self.env.handles.objects[tool]
        pose = handle.pose(self.env.state.body_q.numpy())
        rotation = quat_to_mat(pose[3:7])
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
        corners = pose[:3] + (handle.obb.center_local + signs * handle.obb.half) @ rotation.T
        return float(corners[:, 2].min())

    # ------------------------------------------------------------------- plan
    def _plan(self, tool: str) -> list[Waypoint]:
        """The reference's eight-leg cycle for one tool."""
        grasp_pos, grasp_quat = self._grasp(tool)
        release_pos, release_quat, container = self._release_pose(self.region_of[tool])
        hold = self.hold
        grip = self._grip_command(tool)
        over = release_pos + _up(CLEARANCE)

        legs = [
            Waypoint("approach", tuple(grasp_pos + _up(APPROACH_UP)), tuple(grasp_quat), 0.0, hold),
            Waypoint("descend", tuple(grasp_pos), tuple(grasp_quat), 0.0, hold),
            Waypoint("close", tuple(grasp_pos), tuple(grasp_quat), grip, hold),
            Waypoint("lift", tuple(grasp_pos + _up(LIFT_UP)), tuple(grasp_quat), grip, hold),
        ]
        if container is None:
            # Transit and release in the home orientation: a compartment is wider
            # than any tool, and the yawed grasp pose is not reachable over bin2.
            legs += [
                Waypoint("over_cell", tuple(over), tuple(release_quat), grip, hold),
                Waypoint("descend2", tuple(release_pos), tuple(release_quat), grip, hold),
                Waypoint("release", tuple(release_pos), tuple(release_quat), 0.0, hold),
                Waypoint("retract", tuple(over), tuple(release_quat), 0.0, hold),
            ]
        else:
            # Carry it to the hole already turned upright: the arm reconfigures
            # into that pose over the length of the move, where asking for the
            # same rotation on the spot once it has arrived swings the hand a
            # quarter of a metre off the hole. The post legs are appended once the
            # hang has been measured -- see :meth:`_post_legs`.
            start_pos = np.asarray(legs[-1].position, dtype=np.float64)
            start_quat = np.asarray(legs[-1].quat, dtype=np.float64)
            for i in range(1, OVER_HOLE_LEGS + 1):
                fraction = i / OVER_HOLE_LEGS
                legs += [
                    Waypoint(
                        f"over_hole{i}" if OVER_HOLE_LEGS > 1 else "over_hole",
                        tuple(start_pos + fraction * (over - start_pos)),
                        tuple(quat_slerp(start_quat, release_quat, fraction)),
                        grip,
                        hold,
                    )
                ]
        return legs

    def _post_legs(self, tool: str, container: dict) -> list[Waypoint]:
        """Post an upright tool through a container's hole, from the measured hang.

        The release height is computed here rather than in :meth:`_plan` because
        it depends on how far the tool actually hangs below the gripper once it is
        upright, which is a centimetre or two off whatever the grasp intended.
        """
        aperture_x, aperture_y, _radius = (float(v) for v in container["aperture"])
        rim = float(container["rim_z"])
        upright = self._upright_quat()
        hang = float(self.ik.ee_pose(self.env.state)[0][2] - self._tool_lowest_z(tool))
        post_z = rim + POST_CLEARANCE + hang
        self.message = f"{tool} hangs {hang * 100:.1f} cm below the gripper; posting at z={post_z:.3f}"
        grip = self._grip_command(tool)
        return [
            Waypoint("post", (aperture_x, aperture_y, post_z), tuple(upright), grip, self.hold),
            Waypoint("release", (aperture_x, aperture_y, post_z), tuple(upright), 0.0, self.hold),
            Waypoint("retract", (aperture_x, aperture_y, rim + 0.22), tuple(upright), 0.0, self.hold),
        ]

    # -------------------------------------------------------------------- run
    def act(self) -> np.ndarray:
        """One control step's action, in the task's OSC space."""
        if self.track.done:
            self._advance()
        if self.track.done:
            return self.env.idle_action(self.MODE)

        position, quat = self.ik.ee_pose(self.env.state)
        arm_action = self.track.act(position, quat)
        return self._pack(arm_action)

    def _advance(self) -> None:
        """Finish the current tool and plan the next, or report the routine done."""
        if self.tool:
            container = self._container_of(self.region_of[self.tool])
            if container is not None and not self._posted:
                # The transport legs are done and the tool is over the hole; the
                # post is planned now that the hang can be measured.
                self._posted = True
                self.track.load(self._post_legs(self.tool, container))
                return
            self.remaining.remove(self.tool)
            self.tool = ""

        order = self._pick_order()
        if not order:
            self.finished = True
            self.message = f"routine finished: {len(self.goals)} goal(s) attempted"
            self.env.request_termination(self.message)
            return
        self.tool = order[0]
        self._posted = False
        legs = self._plan(self.tool)
        self.legs = [leg.label for leg in legs]
        self.track.load(legs)
        self.message = f"sorting {self.tool} into {self.region_of[self.tool]}"

    # ---------------------------------------------------------------- actions
    def _pack(self, arm_action: np.ndarray) -> np.ndarray:
        """Put one arm's action into the full vector, holding every other arm."""
        space = self.env.controllers[self.MODE]
        full = self.env.idle_action(self.MODE)
        index = list(self.env.scene.embodiment.iks).index(self.arm_name)
        base = index * space.per_arm
        full[base : base + space.per_arm] = np.asarray(arm_action, dtype=np.float32)
        return full

    # ----------------------------------------------------------- policy shape
    def get_action(self, obs) -> np.ndarray:
        """The :class:`~robosimstudio_bench.policies.PolicyBase` interface."""
        del obs
        return self.act()


class RoutineSyringe(RoutineToolSort):
    """The same sort, with the grasp slid onto the syringe's plunger flange.

    What ``syringe``'s ``(:routine ...)`` names. One number differs from the
    parent and it is the ``handle_frac`` default: negative slides the grasp the
    other way from the reference's convention, towards the heavy end, which is
    what this variant needs -- see :data:`GRASP_HANDLE_FRAC` for the
    measurement.

    ``RoutineToolSort`` keeps 0.0, because that is what the reference's
    *recorded baselines* use (``--handle_frac 0.0``, stated under its task
    table) rather than the 0.5 its scripts happen to carry as an argparse
    default. Comparing against a published number means running the
    configuration it was published at. This variant has no published number
    left to match: the one it had was measured against a collider it no longer
    has.

    Every other argument is the parent's, and passing ``handle_frac`` explicitly
    overrides the default here exactly as it does there.
    """

    def __init__(self, env, *, handle_frac: float = GRASP_HANDLE_FRAC, **kwargs) -> None:
        super().__init__(env, handle_frac=handle_frac, **kwargs)


# =============================================================================
# The Level-1 tier's routine, and why it stands beside the class above rather
# than under it.
#
# Upstream holds the four members below on a shared ``_ArmRoutine`` base and
# re-bases ``RoutineToolSort`` onto it. That refactor is behaviour-preserving
# and it is deliberately not taken here: ``RoutineToolSort`` is the reference
# benchmark's baseline, its numbers are in README's parity table, and the
# thirty documents this block arrived with do not need it moved. So the base
# is added for the new routine alone and the old one keeps its own copies.
# =============================================================================


class _ArmRoutine:
    """What both tool-sort routines do the same way, held once.

    Four methods that were byte-identical in ``RoutineToolSort`` and
    ``RoutineSortFaces``: the arm handle, the "is there work left" flag a teleop
    panel stops stepping on, the policy interface, and packing one arm's action
    into a vector sized for every arm. None of them is about *how* either
    routine sorts -- they are about being an arm-driving routine at all, which
    is exactly the part two of them should not each have a copy of.

    Subclasses supply ``env``, ``arm_name`` and ``finished``.
    """

    MODE = "osc_pose"
    """The controller both routines drive. A subclass that needs another says so."""

    @property
    def ik(self):
        return self.env.arm(self.arm_name)

    def active(self) -> bool:
        """Whether there is work left. What a teleop panel stops stepping on."""
        return not self.finished

    def get_action(self, obs) -> np.ndarray:
        """The :class:`~robosimstudio_bench.policies.PolicyBase` interface."""
        del obs
        return self.act()

    def _pack(self, arm_action: np.ndarray) -> np.ndarray:
        """Put one arm's action into the full vector, holding every other arm."""
        space = self.env.controllers[self.MODE]
        full = self.env.idle_action(self.MODE)
        index = list(self.env.scene.embodiment.iks).index(self.arm_name)
        base = index * space.per_arm
        full[base : base + space.per_arm] = np.asarray(arm_action, dtype=np.float32)
        return full


# =============================================================================
# A hand-crafted sort for the industrial kit: grasp a **face**, not a footprint.
#
# :class:`RoutineToolSort` above is the reference benchmark's own baseline,
# ported to the digit so the two runs are comparable. This is the other kind of
# solution -- one written to *solve* a variant rather than to reproduce somebody's
# number. ``sort_electrical`` and ``sort_mechanical`` both declare it, and it is
# named for what it does rather than for either of them -- it was
# ``RoutineSortElectrical`` while only one variant used it, which is a name that
# would have had to be re-read every time a second one did. The numbers for both
# are at the end of this block.
#
# The founding difference between it and the baseline is one geometric decision,
# stated in ``sort_electrical``'s own document::
#
#     HALF THIS SET IS STILL OUT OF REACH of a top-down parallel jaw, deliberately
#     -- easiest is not easy. `grasp_from_obb` closes across the OBB's short
#     *horizontal* axis. The multimeter's is 0.087 m against the gripper's 0.080 m,
#     missing by 7 mm, and the safety glasses' is 0.157 m, missing by twice that.
#     Neither is a thinness failure: at 0.053 m and 0.061 m thick both are *inside*
#     the span edge-on.
#
# The document goes on to say a baseline scores 4/8 here. **It does not, and the
# gap is worth knowing before reading anything below.** Measured, episode 0,
# ``examples.run --policy scripted`` at the routine above: 0/8, every containment
# false, 9601 steps. Two things are wrong rather than one. The four plates are the
# document's own point and are discussed next -- but the four tools that *do* fit
# the jaws are lost as well, and to the carry rather than to the grasp: the
# baseline aims each release at a compartment's centre, ``bin2``'s far quadrant
# centre is at the edge of this Panda's reach, and the arm that gets there has
# straightened its elbow into a corner of joint space it does not come back from.
# Whatever it was holding is dropped somewhere on the way. So this routine is
# written against two failures, and only the first of them is about grasping.
#
# What is wrong with the grasp is not the number it scores but the shape of
# the question it asks: :func:`~robosimstudio.robot.grasp_from_obb` picks
# one degree of freedom -- a yaw about world +z -- and derives everything else
# from the box's *longest* axis, so it can only ever close across the widest of
# the remaining two. Two consequences, and this variant hits both:
#
# * **A standing object proposes a meaningless grasp.** ``safety_glasses_1``
#   settles on edge, with its long axis 6 degrees off world vertical. The yaw is
#   ``arctan2`` of that axis projected into the table plane -- two components that
#   are both approximately zero -- so the wrist angle is noise, and the width the
#   jaws are asked to span is 0.168 m rather than the 0.065 m the object is
#   actually thick in the direction the fingers could have closed.
# * **A tilted object is grasped as if it were flat.** Both multimeters lean on
#   the pile -- measured on the shipped export, their thin axes sit 21.3 and 19.8
#   degrees off vertical. That thickness is 0.053 m and well inside the span, but
#   reaching it means the jaws close along an axis a fifth of a turn off
#   vertical, which a yaw-only grasp cannot express at all.
#
# This routine picks all six of a box's faces instead. A grasp here is a pair
# ``(approach, close)`` of world directions -- the hand points along the first and
# the jaws shut across the second -- built into a wrist quaternion by
# :func:`~robosimstudio.robot.grasp_quat`, which is the general form of the
# baseline's own helper and reads the hand's *measured* axes rather than assuming
# a home posture. The search enumerates every ``(closing axis, approach)`` pair
# the object's oriented box admits, throws away the ones whose width does not fit
# or whose fingers would end up inside the bin floor, another object or a wall,
# and ranks what is left. For a battery lying flat the winner is exactly the
# baseline's top-down grasp; for the standing glasses it is a top-down grasp
# closing across the thickness; for a plate leaning steeply enough it is a hand
# coming in near the tilt limit, sliding along the raised face and pinching it.
#
# **Three plates are left on this pile, and what excludes them is two degrees.**
# ``multimeter_0``, ``multimeter_1`` and ``safety_glasses_0`` lean 21.3, 19.8 and
# 17.2 degrees off vertical. The approach perpendicular to a face tilted by
# theta sits at ``90 - theta`` from vertical, so those three want the hand at
# 68.7, 70.2 and 72.8 degrees -- against a :data:`MAX_TILT_DEG` of 68. They are
# not flat and they are not out of reach of the *geometry*; they are outside a
# ceiling that exists because past it the wrist runs out of room over an 0.08 m
# bin wall and the lower finger spends the descent inside the pile.
#
# That is a genuinely different failure from a plate lying dead flat, where every
# horizontal width is wider than the jaws and the one that fits can only be
# closed across from directly above or below -- one of which is where the hand is
# and the other is where the floor is. No wrist angle solves *that*, because the
# obstruction is not the wrist. What solves it is changing the object's pose
# first, so the routine has one non-prehensile maneuver, :meth:`_tip_legs`: drive
# the open fingers into the plate's side face above its mid-plane and push it
# into the nearest bin wall, so the far edge jams and the plate rides up.
#
# **It does not work, and it is kept because it is the honest shape of what is
# missing.** Six attempts across five episodes moved both plates about the bin
# floor and stood neither of them up: the rim is 0.08 m above the floor, which is
# less than half the length of the plate being levered against it, so the plate
# slides to the wall and stops. It is fenced (:data:`TIP_ATTEMPTS`, and only ever
# aimed at an object whose *pose* refuses every grasp) so that failing costs two
# pushes rather than a goal, and ``tip_up=False`` turns it off -- which is the
# comparison worth having, since the grasping policy is what the numbers above
# are about. Standing a flat plate up wants a primitive this routine does not
# have: a wall taller than the plate, a fixture, or a second contact.
#
# **Every pick is verified and retried.** The baseline flies its eight legs open
# loop and moves on whether or not the tool came with it, which on a variant where
# half the picks are marginal turns one failure into one lost goal. Here a pick
# that did not leave the object between the fingers is *known* to have failed --
# compare the object's position against the tool centre after the lift -- and the
# pile has been disturbed by the attempt, so re-reading the geometry and trying
# again is a different attempt rather than the same one repeated. Placement is
# checked the same way, against the region the evaluator will score, and a tool
# that missed its compartment goes back on the queue.
#
# **Legs end on arrival rather than on a clock.** The reference holds every
# waypoint a fixed three seconds because it has no feedback to end on; this one
# ends a leg when the tool centre is within :data:`POS_TOL` of it and has held
# there briefly, with the fixed budget kept only as the ceiling. That is what pays
# for the retries: a pick costs about 6 s of simulated time rather than 24, so the
# document's 19200-step budget is 60-odd attempts rather than eight.
#
# **What it scores.** Each of these variants is one ``And`` over eight goals, so
# the episode is a failure until all eight hold and the number to read is the
# reward, which is the share of them that do. Measured through ``examples.run``::
#
#                              sort_electrical        sort_mechanical
#     RoutineToolSort          0.00                   0.00
#     RoutineSortFaces         0.625                  1.000
#
# ``sort_electrical`` is five of eight in about 5000 steps of the document's
# 19200, and *which* five is not stable -- which is the more useful fact about
# it. Its ninth body is a distractor, its eight targets are dropped as a pile,
# and a pick that misses disturbs what it missed, so the set that lands moves
# around underneath a score that does not. On the shipped export the two objects
# that begin with the most candidates (both batteries, eight apiece) are the two
# it fails, three attempts each, while two plates that begin with *no* grasp at
# all are placed once their neighbours have gone. Read the reward as "five of
# eight", never as "these five".
#
# The distractor costs the routine nothing and is not evidence that it can
# reject: nothing is asked of an object that appears in no goal, so it is never
# queued. What the variant asks of a *perceiving* policy is exactly what this one
# is excused from.
#
# **Both numbers were measured on ``bounding_box`` colliders, and only one of the
# two variants still fits one.** ``sort_electrical`` is in :data:`COMPARISON`, so
# it keeps the box and 0.625 is a number about the code below. ``sort_mechanical``
# and ``sort_electronics`` now build on ``drawn_hull`` -- the hull of the mesh the
# export draws, which is up to 27 mm narrower than the box VAB fitted around it --
# so 1.000 is a claim about a collider this variant no longer has. It has not been
# re-measured. A hull that is narrower than the box the search planned against
# should if anything help, since :meth:`RoutineSortFaces._read_colliders` reads
# whatever the scene holds, but "should help" is not a measurement.
#
# ``sort_mechanical`` is all eight, in 3871 steps, each on its first attempt.
# That is the easiest rung of the industrial ladder rather than a general result
# -- eight separated objects, all inside the jaws' span -- but getting there took
# three fixes that the *piled* variants had never exposed, and each of them was
# the routine believing something about geometry that was not so:
#
# * **The box the fingers meet is not always the box the handles publish.**
#   ``compute_obb`` bounds a body's shapes axis-aligned in the body frame, so an
#   object whose one box collider sits a few degrees off that frame gets bounded
#   by something wider than it is -- 60 mm for a wrench that is 47. The jaws then
#   close to 4 mm inside a width the object has not got and meet in the air.
#   :meth:`RoutineSortFaces._read_colliders`.
# * **The fingers are the small part of what comes down.** The palm above them is
#   200 mm across and enters a bin 31 mm above the fingertips, over walls 80 mm
#   tall, so a clearance test that knows only about fingertips will happily fly a
#   grasp that lands the hand on the rim -- the arm stops 33 mm short with no
#   contact anywhere near the object. :meth:`RoutineSortFaces._read_palm`.
# * **A fixed release height is a fixed impact speed.** 0.12 m over a compartment
#   is nothing to a 400 g gear and passes a 32 g wire spool straight through the
#   compartment floor. :meth:`RoutineSortFaces._place_legs` now descends to
#   the rim, which is as low as a hand wider than a compartment can go.
#
# All three were found on ``sort_mechanical`` and all three help
# ``sort_electrical``, which is what took it from 0.25-0.50 to 0.625.
#
# Run either as the variant's routine, or point a policy straight at it::
#
#     uv run -m examples.run --task tool_sort/sort_mechanical --policy scripted
# =============================================================================


# --------------------------------------------------------------------- geometry
SQUEEZE = 0.004
"""How far inside the measured width the jaws are told to close [m].

The same number and the same reason as the baseline's: these fingers are
position-servoed, so commanding them shut means "drive the pads together" and
the grip force is whatever the servo makes of the standing error -- which
extrudes a smooth object out of the jaws. Closing to just inside the width the
box actually is bounds the force instead.
"""

STANDOFF = 0.10
"""How far back along the approach the pre-grasp pose stands [m]."""

LIFT_ABOVE = 0.18
"""How far above the grasp point a taken tool is lifted [m]."""

BACK_OFF = 0.05
"""How far back along the approach the lift starts [m].

For a top-down grasp this is simply part of the lift. For a tilted one it is the
difference between extracting the tool along the way the hand came in and
levering it against whatever it was resting on.
"""

TRANSIT_CLEAR = 0.16
"""Height above the release point the carry passes through [m]."""

DROP_CLEAR = 0.03
"""How far above the compartment floor a *hanging* tool's lowest corner is let
go from [m]. See :meth:`RoutineSortFaces._place_legs`."""

AIM_INSET = 0.06
"""How far inside a compartment's near edge a tool is released [m].

Enough that a tool dropped from :data:`DROP` lands and settles inside the
quadrant rather than on its rim, and no more -- every centimetre of it is
reach. See :meth:`RoutineSortFaces._aim_point`.
"""

DROP = 0.12
"""Release height above a compartment's floor [m].

The bin's walls keep a bouncing tool in, so a compartment is filled from a
height rather than lowered into -- the baseline's finding, kept.
"""

FLOOR_CLEAR = 0.010
"""How far above the bin floor the pads must stay [m]."""

OBJECT_CLEAR = 0.008
"""How far the fingers must clear a *neighbouring* object's box [m]."""

FINGER_CLEAR = 0.012
"""How much room each finger is given beside the object it is closing on [m].

Both the width the jaws are pre-opened to and the width the clearance tests are
made at -- one number, because they are the same fact. Three times the tolerance
the stand-off is flown to, so the descent has room for the arm's own error, and
far short of the 0.040 a side that full open would sweep. See
:meth:`RoutineSortFaces._open_for`.
"""

BIN_FLOOR = 0.02
"""Height of a bin's floor above its own frame origin [m].

Read off the export this variant is built from (``bin1``'s floor geom is a box
of half-height 0.02 centred on the frame), and the routine is allowed to know
it: a solution is written against a cell, unlike a document, which is why this
is here and not in the ``.bddl`` file.
"""

BIN_WALL_TOP = 0.10
"""Height of a bin's wall rim above its own frame origin [m]. Same source."""

WALL_INSET = 0.01
"""How far inside a bin's wall a region's declared range stops [m].

Measured on the export and the same on both axes: ``bin1``'s floor slab is
0.40 x 0.50, its walls' inner faces stand at 0.19 and 0.24 from the frame, and
``bin1_region`` is 0.18 by 0.23. So the region is 10 mm short of the floor the
bin actually has -- a centimetre of reachable floor all the way round, which is
the difference between a grasp and no grasp on an object laid out near an edge.
"""

WALL_THICK = 0.02
"""A bin wall's own thickness [m]. Same source: 0.01 half, so 20 mm of timber.

With :data:`WALL_INSET` this says where the solid part of a bin is: from the
region's edge plus 10 mm out to plus 30 mm, at every height below
:data:`BIN_WALL_TOP`. Past that there is no bin. See
:meth:`RoutineSortFaces._clear_of_cell`, which used to read the region as
the whole obstacle and so had the wall in the wrong place twice over -- 10 mm
too far in, and with nothing solid outside it at all.
"""

CELL_MARGIN = 0.006
"""How far inside a bin's own floor the hand is kept [m]."""

CROSS_CLEAR = 0.15
"""How far above the tallest bin rim a long move is flown [m].

The fingers hang at the commanded point and the tools stand up to 0.10 m out of
the pile, so this is the height at which crossing the cell touches nothing.
"""

CROSS_TOL = 0.03
"""Arrival tolerance for a transit leg [m]. It is a height to be at, not a
place to be, so converging on it to the millimetre is time spent on nothing."""

# ------------------------------------------------------------------- the tip-up
TIP_ATTEMPTS = 2
"""How many times a plate with no feasible grasp is pushed at before it is left.

Low deliberately. The maneuver is worth trying and is not worth grinding on: a
plate that has not tipped in two goes has usually been pushed flat against the
wall, where the third attempt is the second one repeated against a shorter run
up.
"""

TIP_HEIGHT = 0.035
"""Height above the bin floor the pushing fingers are driven along [m].

Above the mid-plane of a plate 0.053-0.065 m thick, so the push makes a moment
about the far bottom edge once that edge is against the wall, and below its top
face, so the fingers push rather than slide over.
"""

TIP_OVERRUN = 0.04
"""How far past the wall the push is aimed [m].

The plate stops when it hits; the command has to keep asking, because a leg that
ends the moment the plate stops moving ends before it has tipped.
"""

# --------------------------------------------------------------------- the legs
AIM_TOL = 0.004
"""The tighter tolerance the pre-grasp stand-off is flown to [m].

Half of :data:`POS_TOL`, for the one leg where the number is a grasp rather than
a waypoint -- see :meth:`RoutineSortFaces._pick_legs`.
"""

ROT_TOL_DEG = 8.0
"""Orientation tolerance a leg counts as arrived at [deg]."""

SETTLE_FRAMES = 4
"""Control steps a leg has to stay inside tolerance before it ends."""

SETTLE_STEPS = 90
"""Control steps allowed on top of the travel for the servo to close in.

1.8 s, and measured rather than guessed: a leg given only its travel at cruise
ends about 20 mm short, because under this controller the setpoint arrives
before the arm does and the tail is the position servo working off a standing
error. The convergence test is what actually ends the leg -- this is only how
long the routine is willing to wait before flying the next one anyway.
"""

CRUISE = 0.30
"""Cruise speed the OSC gains imply [m/s], for sizing a leg's step ceiling.

Not a command: :class:`~robosimstudio.robot.OscSetpoint` sets the speed
and this only decides how long to wait before giving up on a leg. See that
module for why it is 0.3 and not the 1.0 that ``output_max`` over the control
period suggests.
"""

GRIP_FRAMES = 30
"""Control steps held while the fingers close or open."""

ATTEMPTS = 3
"""Picks attempted per tool before it is abandoned."""

GRASP_RADIUS = 0.12
"""How near the tool centre a lifted object has to be to count as held [m]."""

GRASP_RISE = 0.04
"""How far the object has to have risen to count as held [m]."""

DISTURBED = 0.008
"""How far a tool may shift during the descent before the leg is abandoned [m].

Past the settling noise of a pile that was stepped to rest, and well inside
the clearance the fingers descend with. See
:meth:`RoutineSortFaces._shoving`.
"""

SHOVED_DOWN = 0.010
"""How far the jaws may drive a tool *downward* while closing [m].

Separate from :data:`DISTURBED` and directional, because the direction is what
matters: a tool rolling a little as the pads seat on it is a grasp happening,
and a tool going down is a tool being pressed into the bin floor -- which this
simulator does not reliably stop. Driven, a tool here goes 21 mm into a 40 mm
slab in twelve control steps and does not come back from about 35, ending under
the bench or on the world ground plane, out of the task for good.

**This is a mitigation and not the fix, and the numbers say so.** Over runs of
this variant::

    500 Hz, no guard      through the floor in 3 of 5 runs    2-5 goals
    500 Hz, guard  5 mm   through in 0 of 3                   2-3 goals
    500 Hz, guard 10 mm   through in 1 of 2                   2 and 4 goals
    1 kHz,  no guard      through in 0 of 1                   3 goals

Tightening the guard buys safety by refusing grasps that were seating normally,
and loosening it hands the tools back to the solver. 10 mm is the loose end of
that trade, chosen because the *real* remedy is the physics rate -- at 1 kHz the
unguarded routine drove a tool 29 mm deep and it came back out. See
:meth:`RoutineSortFaces._shoving`, and ``vab_500hz@v1`` in
``tasks/tool_sort/staging.py``.
"""

ROLL_WEIGHT = 0.5
"""How much a grasp's score prefers the wrist nearer the arm's home posture.

Small: it is a tie-break between the two wrists that make the *same* grasp, and
it must not outweigh a geometric reason to prefer one grasp over another.
"""

STALLED = 0.05
"""How far short of its last waypoint the arm counts as stuck [m].

Well past what the servo's own standing error explains (about 20 mm at the end
of a converged leg, and less once it has settled), so this fires on an arm that
was *stopped* rather than on one that was merely slow.
"""

LEVEL_TILT_DEG = 30.0
"""Approach tilt above which a carry keeps the grasp's heading [deg].

Below it the hand simply carries in the home posture. See
:meth:`RoutineSortFaces._transit_quat`.
"""


@dataclass
class _Attempt:
    """What has been tried on one tool, for the readout and the row extras."""

    picks: int = 0
    tips: int = 0
    places: int = 0
    notes: list[str] = field(default_factory=list)


class RoutineSortFaces(_ArmRoutine):
    """Sort every goal's object into its compartment, grasping faces.

    Args:
        env: The environment to drive. It must offer an ``osc_pose`` controller.
        arm: Which arm sorts. Empty uses the first.
        attempts: Picks attempted per tool before it is abandoned.
        tip_up: Whether to try the non-prehensile tip-up on an object no grasp
            fits. Off gives the pure grasping policy, which is the comparison
            worth having when reading what the maneuver is worth.
        only: Comma-separated subset of object names to sort. Empty sorts every
            goal.
    """

    def __init__(
        self,
        env,
        *,
        arm: str = "",
        attempts: int = ATTEMPTS,
        tip_up: bool = True,
        only: str = "",
    ) -> None:
        space = env.controllers.get(self.MODE)
        if not isinstance(space, ActionOsc):
            raise ValueError(
                f"{type(self).__name__} emits normalized OSC deltas, and {env.task.full_name!r} "
                f"offers no {self.MODE!r} controller (it has {sorted(env.controllers)})"
            )
        self.env = env
        self.arm_name = arm or env.handles.work_arm or next(iter(env.handles.arms))
        self.hand = env.handles.arms[self.arm_name]
        self.attempts = int(attempts)
        self.tip_up = bool(tip_up)

        goals = placements(env.evaluator.goal)
        if not goals:
            raise ValueError(f"task {env.task.full_name!r} declares no goals for a routine to sort")
        wanted = {name.strip() for name in only.split(",") if name.strip()}
        if wanted:
            unknown = wanted - {obj for obj, _ in goals}
            if unknown:
                raise ValueError(f"no goal names {sorted(unknown)}; have {sorted(obj for obj, _ in goals)}")
            goals = tuple((obj, region) for obj, region in goals if obj in wanted)
        self.goals = goals
        self.region_of = dict(goals)

        containers = dict((env.handles.meta or {}).get("containers") or {})
        posted = [r for r in self.region_of.values() if any(r == f"{n}_region" for n in containers)]
        if posted:
            raise ValueError(
                f"{type(self).__name__} releases over an open compartment and {posted} are containers; "
                f"tasks.tool_sort.routine:RoutineToolSort is the one that posts through a lid"
            )

        self.output_max = np.asarray(space.controllers[0].output_max, dtype=np.float64)
        self.span = max(0.0, float(self.hand.gripper_span) - FIT_MARGIN)
        self.home_quat = np.asarray(env.arm(self.arm_name).home_pose[1], dtype=np.float64)

        rims = [float(region.origin[2]) + BIN_WALL_TOP for region in env.handles.regions.values()]
        self.transit_z = (max(rims) if rims else float(env.handles.surface_z)) + CROSS_CLEAR
        """Height every long move is flown at [m]: the tallest bin rim plus a hand."""

        home_position = np.asarray(env.arm(self.arm_name).home_pose[0], dtype=np.float64)
        self.staging = np.array([home_position[0], home_position[1], self.transit_z])
        """Where every long move starts and ends [m].

        The home tool centre, lifted to :attr:`transit_z` -- which is over the
        bench between the two bins, near the base, and above everything. Both
        plans begin there for a reason that is about the *arm* rather than the
        route: the posture that reaches into a bin is folded and the posture
        that reaches the far compartment is not, and the Cartesian servo cannot
        get from one to the other along a straight line through the middle of
        the cell. Going back through the pose the arm was born in keeps every
        long move a move it has made before.
        """

        base = env.handles.arms[self.arm_name].base_body
        self.base_xy = np.asarray(env.state.body_q.numpy()[base][:2], dtype=np.float64) if base >= 0 else np.zeros(2)
        """Where the arm is mounted, in the table plane [m] -- what "near" means."""

        self.colliders = self._read_colliders()
        """Per object, the box the *fingers* meet, or nothing. See :meth:`_box`."""

        self.palm = np.zeros((0, 3))
        """The hand's collision points in the tool frame [m]. See :meth:`_clear_palm`.

        Filled by :meth:`reset`, which is where the pose it is measured off is
        one the environment has actually built.
        """

        self.palm_lift = 0.0
        """How far behind the fingertips the hand's nearest point is [m].

        31 mm on this Panda. What a release has to keep above a compartment rim,
        since the hand is wider than a compartment -- see :meth:`_place_legs`.
        """

        self.queue: list[str] = []
        self.record: dict[str, _Attempt] = {}
        self.legs: list[_Leg] = []
        self.index = 0
        self.frame = 0
        self.held = 0
        self.tool = ""
        self.stage = ""
        self.message = ""
        self.finished = False
        self._grasp: Grasp | None = None
        self._grasp_z = 0.0
        self._object_at = np.zeros(3)
        self.reset()

    # ------------------------------------------------------------------- state
    @property
    def phase(self) -> str:
        """Which leg is being flown, for a readout."""
        if self.finished:
            return "done"
        leg = self._leg
        return f"{self.stage}/{leg.label}" if leg is not None else "idle"

    def plan(self) -> str:
        """Start the first tool and say what is being attempted."""
        self._advance()
        return self.message

    def reset(self) -> None:
        """Forget the plan. The next :meth:`act` re-plans against the new pile.

        The hand is re-measured here as well as at construction, because a
        routine is built *before* the environment's first reset -- the runner
        makes the policy and then warms the scene -- and :meth:`_read_palm` takes
        the hand's points into the tool frame off a live pose. The shape is a
        constant; the pose it was read at is not.
        """
        self.palm = self._read_palm()
        self.palm_lift = -float(np.max(self.palm[:, 2])) if len(self.palm) else 0.0
        self.queue = [obj for obj, _ in self.goals]
        self.record = {obj: _Attempt() for obj in self.queue}
        self.legs = []
        self.index = 0
        self.frame = 0
        self.held = 0
        self.tool = ""
        self.stage = ""
        self.message = ""
        self.finished = False
        self._grasp = None

    def row_extras(self) -> dict:
        """Per-tool outcomes, so a run's own row says which picks were hard."""
        return {
            "picks": {name: r.picks for name, r in self.record.items()},
            "placed": sorted(name for name, r in self.record.items() if r.places),
            "tips": {name: r.tips for name, r in self.record.items() if r.tips},
            "notes": {name: r.notes for name, r in self.record.items() if r.notes},
            "unsorted": sorted(self.queue),
        }

    # ---------------------------------------------------------------- geometry
    def _read_colliders(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Each object's lone collision box, in its body's frame, or nothing.

        **The box the fingers meet is not the box the handles publish, and on
        this suite the two can differ by 12 mm.** ``handle.obb`` comes from
        :func:`~robosimstudio.robot.compute_obb`, which bounds a body's
        shapes *axis-aligned in the body frame*. That is the right box for the
        baseline, which grasps by a yaw about world +z and wants a bound it can
        take a footprint from -- and it is the wrong one here, because a tool
        whose collision geom is a box rotated a few degrees inside its own body
        gets bounded by something bigger than the box it is::

            pipe_wrench    collision 0.152 x 0.047 x 0.020    obb 0.155 x 0.060 x 0.020
            wire_spool     collision 0.050 x 0.050 x 0.024    obb 0.055 x 0.055 x 0.024

        The wrench's geom sits about 5 degrees off its body axes, and 5 degrees
        of a 152 mm object is 13 mm of width the object does not have. A pick
        planned on the wide number closes the jaws to 4 mm inside *it* -- 56 mm
        around a 47 mm object -- and the fingers meet in the air with the wrench
        still on the floor. Measured on ``sort_mechanical``: the descent
        converged to 0.4 mm, the close ran to its commanded width, and the tool
        never moved. Eight goals, eight of those.

        So the search reads the collision shape directly when a body has exactly
        one, which is every object in this suite (the exporter writes one box
        collider per tool and a visual mesh beside it). Anything else -- several
        colliders, a convex mesh, no collider at all -- keeps the published OBB,
        which is the honest bound when there is no single box to prefer.
        """
        model = self.env.scene.pipeline.model
        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        flags = model.shape_flags.numpy()
        scale = model.shape_scale.numpy()
        transform = model.shape_transform.numpy()
        out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for name, handle in self.env.handles.objects.items():
            if handle.body < 0:
                continue
            # Every *collision* shape on the body, not every box on it: a body
            # carrying a box beside a convex hull is not described by the box,
            # and preferring it would trade a bound that is too big for one that
            # is too small. Visual meshes and sites are not collision shapes and
            # do not count.
            colliders = [
                s
                for s in range(model.shape_count)
                if int(shape_body[s]) == handle.body and int(flags[s]) & int(newton.ShapeFlags.COLLIDE_SHAPES)
            ]
            if len(colliders) != 1 or int(shape_type[colliders[0]]) != int(newton.GeoType.BOX):
                continue
            shape = transform[colliders[0]]
            out[name] = (
                np.asarray(shape[:3], dtype=np.float64),
                quat_to_mat(np.asarray(shape[3:7], dtype=np.float64)),
                np.asarray(scale[colliders[0]], dtype=np.float64),
            )
        return out

    def _read_palm(self) -> np.ndarray:
        """The hand's own collision points, in the tool frame [m].

        Everything welded to the tool centre -- the palm and whatever else rides
        with it -- as the points it is made of, measured once off the model.
        The fingers are *not* here: they move with the grip, so they are tested
        analytically at the width each candidate opens to (:meth:`_candidates`),
        and this is the part whose shape is a constant.

        **It is here because the palm is much bigger than the fingers and sits
        just above them.** On this Panda, measured in the tool frame, the palm
        reaches 100 mm each way along the closing axis and 32 mm across it,
        standing 31 mm behind the fingertips -- a slab 200 mm wide, entering the
        bin 31 mm above the pads. A bin wall here stands 80 mm above its floor,
        so a top-down grasp of anything less than about 100 mm from a wall puts
        that slab on the rim, and the arm stops there with the fingers still
        30 mm above the tool. It is not subtle when it happens and it is
        invisible to a test that only asks where the fingertips go: measured on
        ``sort_mechanical``, four of eight picks stalled that way, 33 mm short,
        with no contact anywhere near the object.

        Which wrist the search picks is what fixes it, and the geometry says why:
        the slab is long in the closing direction and thin across it, so closing
        across the axis *parallel* to the near wall presents 32 mm to that wall
        instead of 100. That candidate is already in the list -- it was being
        ranked below one that cannot be flown.
        """
        model = self.env.scene.pipeline.model
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_type = model.joint_type.numpy()
        # Every body welded to the tool centre: the end-effector body and its
        # fixed-joint ancestors. The palm is one link up from the site body the
        # tool centre is measured off, which is why this walks rather than
        # reading one body.
        welded = {int(self.ik.ee_body)}
        while True:
            up = [
                int(joint_parent[j])
                for j in range(model.joint_count)
                if int(joint_child[j]) in welded and int(joint_type[j]) == int(newton.JointType.FIXED)
            ]
            fresh = {b for b in up if b >= 0 and b not in welded}
            if not fresh:
                break
            welded |= fresh
        fingers = {int(body) for side in self.hand.finger_groups for body, _ in side}
        welded -= fingers

        body_q = self._body_q()
        tcp, tcp_quat = self.ik.ee_pose(self.env.state)
        axes = quat_to_mat(np.asarray(tcp_quat, dtype=np.float64))
        shape_body = model.shape_body.numpy()
        flags = model.shape_flags.numpy()
        scale = model.shape_scale.numpy()
        transform = model.shape_transform.numpy()
        points: list[np.ndarray] = []
        for s in range(model.shape_count):
            if int(shape_body[s]) not in welded or not int(flags[s]) & int(newton.ShapeFlags.COLLIDE_SHAPES):
                continue
            source = model.shape_source[s]
            vertices = getattr(source, "vertices", None)
            half = np.asarray(scale[s], dtype=np.float64)
            local = (
                np.asarray(vertices, dtype=np.float64) * half
                if vertices is not None
                else np.array(
                    [[i * half[0], j * half[1], k * half[2]] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
                )
            )
            shape = transform[s]
            local = local @ quat_to_mat(np.asarray(shape[3:7], dtype=np.float64)).T + np.asarray(shape[:3])
            pose = body_q[int(shape_body[s])]
            world = local @ quat_to_mat(np.asarray(pose[3:7], dtype=np.float64)).T + np.asarray(pose[:3])
            points.append((world - np.asarray(tcp, dtype=np.float64)) @ axes)
        return np.concatenate(points, axis=0) if points else np.zeros((0, 3))

    def _body_q(self) -> np.ndarray:
        return self.env.state.body_q.numpy()

    def _box(self, tool: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One object's box right now: ``(centre, axes, half)``.

        The collision box when the object has exactly one -- see
        :meth:`_read_colliders` for why that is not the same box as the
        published OBB and what it costs to grasp the wrong one -- and the OBB
        otherwise. Either way the centre is the *box's*, not the body's: they
        differ by a centimetre on the safety glasses, which is a centimetre of
        grasp.
        """
        handle = self.env.handles.objects[tool]
        pose = handle.pose(self._body_q())
        rotation = quat_to_mat(pose[3:7])
        collider = self.colliders.get(tool)
        if collider is not None:
            offset, spin, half = collider
            return pose[:3] + rotation @ offset, rotation @ spin, half
        obb = handle.obb
        if obb is None:
            raise ValueError(f"object {tool!r} has neither a box collider nor an OBB to grasp by")
        return pose[:3] + rotation @ obb.center_local, rotation, np.asarray(obb.half, dtype=np.float64)

    def _lowest_z(self, tool: str) -> float:
        """World z of the lowest corner of a tool's box as it hangs right now [m].

        Measured rather than predicted from the grasp: a tool settles in the
        fingers differently depending on where along it they closed, and a
        centimetre of that is the difference between a release and a collision.
        """
        centre, axes, half = self._box(tool)
        return float(centre[2] - support(axes, half, np.array([0.0, 0.0, 1.0])))

    def _bin_of(self, point: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
        """The bin a point is over: its frame origin and its footprint.

        Read off the task's own regions rather than off the export's metadata,
        so the floor the fingers are kept out of is the floor the evaluator
        measures against. ``bin1_region`` is the pile; each compartment region
        is a quadrant of ``bin2``, and its frame is what a release is aimed at.

        The footprint returned is the region's, and the point is matched against
        it **widened to the outside of the wall** -- because the walls stand
        outside the floor the region declares, so a point just past the region's
        edge is not beside the bin, it is over the rim. See
        :meth:`_clear_of_cell`, which is what does something with the two.
        """
        body_q = self._body_q()
        reach = WALL_INSET + WALL_THICK
        for name, region in self.env.handles.regions.items():
            if not name.startswith("bin1"):
                continue
            origin = region.origin_now(body_q)
            x0, y0, x1, y1 = region.xy_ranges
            if origin[0] + min(x0, x1) - reach <= point[0] <= origin[0] + max(x0, x1) + reach and (
                origin[1] + min(y0, y1) - reach <= point[1] <= origin[1] + max(y0, y1) + reach
            ):
                return origin, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        return None

    def _clear_of_cell(self, point: np.ndarray) -> bool:
        """Whether a point of the hand may be here without being in the bin.

        A bin is three things to a descending hand and this answers all three
        from one footprint: over the floor and below it is inside the floor;
        over the *wall band* and below the rim is inside the wall; and anything
        above the rim, or out past the wall entirely, is free air. The rim
        standing only 0.08 m up is what lets the hand reach in over an edge at
        all.

        **Where the wall is, is a fact about the export and not about the
        region.** The region stops :data:`WALL_INSET` short of the floor the bin
        really has and the wall is :data:`WALL_THICK` of timber beyond that, so
        reading the region's own edge as the obstacle is wrong twice: it gives
        away a centimetre of reachable floor, and it declares the wall itself --
        where the palm actually lands -- to be free air. Both mattered. The
        second is what let a grasp be flown that put the hand on the rim with
        every fingertip test passing; the first rejected a perfectly good grasp
        of a gear by one millimetre.
        """
        over = self._bin_of(point)
        if over is None:
            return True
        origin, (x0, y0, x1, y1) = over
        floor = np.array([origin[0] + x0, origin[1] + y0]) - WALL_INSET + CELL_MARGIN
        ceiling = np.array([origin[0] + x1, origin[1] + y1]) + WALL_INSET - CELL_MARGIN
        if np.all(point[:2] >= floor) and np.all(point[:2] <= ceiling):
            return bool(point[2] >= origin[2] + BIN_FLOOR + FLOOR_CLEAR)
        return bool(point[2] >= origin[2] + BIN_WALL_TOP)

    def _clear_of_others(self, tool: str, points: np.ndarray) -> bool:
        """Whether none of these points is inside another object's box."""
        body_q = self._body_q()
        for name, handle in self.env.handles.objects.items():
            if name == tool or handle.obb is None or handle.body < 0:
                continue
            pose = handle.pose(body_q)
            axes = quat_to_mat(pose[3:7])
            centre = pose[:3] + axes @ handle.obb.center_local
            local = np.abs((points - centre) @ axes)
            if np.any(np.all(local <= np.asarray(handle.obb.half) + OBJECT_CLEAR, axis=1)):
                return False
        return True

    def _clear_palm(self, grasp: Grasp) -> bool:
        """Whether the hand itself fits where this grasp would put it.

        The same cell test the fingertips get (:meth:`_clear_of_cell`), asked of
        the points the hand is actually made of -- see :meth:`_read_palm` for
        where they come from and why two fingertips were never enough of an
        answer. The wrist a grasp implies is what places them, so this is a
        question about the *candidate* and not about the object.
        """
        if not len(self.palm):
            return True
        points = grasp.position + self.palm @ quat_to_mat(self._quat(grasp.approach, grasp.close)).T
        return all(self._clear_of_cell(point) for point in points)

    # -------------------------------------------------------------- the search
    def _candidates(self, tool: str, *, relaxed: bool = False) -> list[Grasp]:
        """Every grasp of one object's box that fits and is not blocked here.

        :func:`face_grasps` answers the geometric half -- which faces the jaws
        fit around, from where -- and this adds the four things that need a
        scene: the fingers may not be inside the bin floor or a wall, the *hand*
        may not be either, the fingers may not be inside a *neighbour*, and the
        wrist the grasp implies should be the one nearer the posture the arm is
        already in.

        Args:
            tool: The object to grasp.
            relaxed: Drop the test that keeps the fingers out of the other
                objects. Even measured at the width the fingers actually
                descend at, that test can reject every grasp of a tool that is
                perfectly takeable once its neighbour has gone.
                :meth:`_start_next` uses the strict answer for every tool before
                it will accept a relaxed one for any, so the pile is thinned in
                the order that disturbs it least and the relaxation is what
                breaks a deadlock rather than what plans a pick.

                It does **not** relax the cell tests, and the hand's is the
                reason to say so: a neighbour is a thing another pick can take
                away, and a bin wall is not.
        """
        centre, axes, half = self._box(tool)
        out: list[Grasp] = []
        for grasp in face_grasps(centre, axes, half, span=self.span):
            reach = 0.5 * grasp.width + FINGER_CLEAR
            pads = np.array([grasp.position + reach * grasp.close, grasp.position - reach * grasp.close])
            if not all(self._clear_of_cell(pad) for pad in pads):
                continue
            if not self._clear_palm(grasp):
                continue
            # Where the fingers will be, and the path the hand flies in along.
            # The object itself is excluded by name -- the fingers are meant to
            # end up around it -- but everything else in the pile is not.
            if not relaxed:
                path = np.concatenate(
                    [pads, [grasp.position - grasp.approach * f * STANDOFF for f in (0.25, 0.5, 1.0)]]
                )
                if not self._clear_of_others(tool, path):
                    continue
            # The tie a parallel jaw always offers: the sign of a closing
            # direction is arbitrary to the *hand* and not to the arm, since the
            # two wrists differ by half a turn of the last joint and on a Panda
            # reaching across itself one of them is against its limit.
            roll = abs(float(np.dot(self._quat(grasp.approach, grasp.close), self.home_quat)))
            out.append(replace(grasp, score=grasp.score + ROLL_WEIGHT * roll))
        out.sort(key=lambda grasp: -grasp.score)
        return out

    def _gap(self, gap: float) -> float:
        """The gripper command that holds the pads this far apart: 0 open .. 1 shut."""
        return float(np.clip(1.0 - max(0.0, gap) / float(self.hand.gripper_span), 0.0, 1.0))

    def _grip(self, width: float) -> float:
        """How far to close on a measured width: 0 open .. 1 fingers together."""
        return self._gap(width - SQUEEZE)

    def _open_for(self, width: float) -> float:
        """How far to *pre-open* for an object of this width.

        Not all the way, which is the point. The jaws open to 0.080 m and the
        things in this bin are 0.030 to 0.065 m thick, so descending at full
        open sweeps a volume half again as wide as it needs and, in a pile whose
        closest pair is 68 mm apart, that volume is almost always somebody
        else's. The visible symptom is not a collision report -- it is that
        *every* grasp fails the neighbour test, so every pick falls through to
        the relaxed pass, which is the one that does not check. Four of five
        picks in a row missed that way, each of them shoving the pile about on
        the way down. Opening to the object plus :data:`FINGER_CLEAR` a side
        makes the test mean something and the descent fit.
        """
        return self._gap(width + 2.0 * FINGER_CLEAR)

    def _quat(self, approach: np.ndarray, close: np.ndarray) -> np.ndarray:
        """The wrist that points along an approach and closes across a direction."""
        return grasp_quat(self.hand.approach_axis, self.hand.close_axis, self.hand.ee_offset_rot, approach, close)

    # ------------------------------------------------------------------- plans
    def _chain(self, moves: list[tuple[str, np.ndarray, np.ndarray, float, int, float]]) -> list[_Leg]:
        """Size each leg's step ceiling from the leg before it, not from now.

        A plan is built in one go and flown over several seconds, so measuring
        every leg's length from where the arm is *at planning time* would give
        the carry the budget of the descent that precedes it. The cursor starts
        at the tool centre and walks the plan.

        The budget is the travel at cruise plus :data:`SETTLE_STEPS`, and the
        constant is the load-bearing half of that sum. A move under this
        controller does not end when the setpoint arrives: the setpoint leads
        the arm by design and the last centimetre is the servo closing a
        standing error, so a leg cut off at "distance over cruise" ends about
        two centimetres short every time. Which is survivable on a battery the
        jaws have 23 mm of slack around and fatal on the safety glasses, where
        they have 7.
        """
        hz = float(self.env.task.physics.control_hz)
        cursor = np.asarray(self.ik.ee_pose(self.env.state)[0], dtype=np.float64)
        legs = []
        for label, target, quat, grip, hold, tol in moves:
            goal = np.asarray(target, dtype=np.float64)
            travel = float(np.linalg.norm(goal - cursor)) / CRUISE * hz
            steps = hold or int(np.clip(travel + SETTLE_STEPS, 40.0, 260.0))
            legs.append(_Leg(label, goal, np.asarray(quat, dtype=np.float64), float(grip), steps, not hold, tol))
            cursor = goal
        return legs

    def _pick_legs(self, grasp: Grasp) -> list[_Leg]:
        """Approach, descend, close, lift -- around one proposed grasp.

        The stand-off is where the precision has to be spent, which is why it is
        the leg with the tight tolerance. It is flown in free space above the
        object, and the descent that follows it travels *along the approach* --
        so whatever lateral error the hand arrives with is the lateral error it
        still has when the pads are level with the object, and there is nowhere
        left to correct it from.

        The ``cross`` leg in front of it is the other half of that: it takes the
        hand to :attr:`transit_z` over the *column* of the grasp before anything
        descends. Without it a pick planned from over the far compartment is one
        straight line 0.75 m long, and this controller flies a straight line --
        so the hand loses its height at the same rate it crosses, arrives at the
        gap between the bins with the fingertips at rim height and drags them
        through both piles. The arm ends that move stalled against a shoulder
        limit and every later pick inherits it.
        """
        quat = self._quat(grasp.approach, grasp.close)
        grip = self._grip(grasp.width)
        ready = self._open_for(grasp.width)
        stand = grasp.position - grasp.approach * STANDOFF
        lift = grasp.position - grasp.approach * BACK_OFF
        lift[2] = max(lift[2] + LIFT_ABOVE, self.transit_z)
        return self._chain(
            [
                ("stage", self.staging, quat, ready, 0, CROSS_TOL),
                ("cross", np.array([stand[0], stand[1], self.transit_z]), quat, ready, 0, CROSS_TOL),
                ("approach", stand, quat, ready, 0, AIM_TOL),
                ("descend", grasp.position, quat, ready, 0, POS_TOL),
                ("close", grasp.position, quat, grip, GRIP_FRAMES, POS_TOL),
                ("lift", lift, quat, grip, 0, POS_TOL),
            ]
        )

    def _transit_quat(self) -> np.ndarray:
        """The wrist to carry in: the arm's home posture, yawed if it must be.

        The grasp orientation is the wrong thing to carry in and the reason is
        the arm rather than the grip. ``bin2``'s far compartment is 0.86 m from
        this Panda's base, which is its reach, so the carry arrives with the
        shoulder and the last wrist joint both against their limits -- and a
        wrist pinned at 2.88 rad of 2.897 has nothing left to give when the next
        pick asks it to turn the other way. Measured: three picks in a row after
        one such carry never got within 45 cm of their targets, with the arm
        wandering along its own limit surface the whole time.

        Turning the hand about the *approach* axis costs the grip nothing -- the
        pads keep their contact and the tool turns with them -- so a top-down
        grasp simply carries home. A tilted grasp has to be levelled to be
        carried at all, and that rotation is a real risk to a face pinch, so it
        keeps the heading it grasped with and gives up only the tilt. If the
        tool is lost on the way, :meth:`_placed` sees it and the tool goes back
        on the queue.
        """
        grasp = self._grasp
        down = np.array([0.0, 0.0, -1.0])
        if grasp is None:
            return self.home_quat
        if float(np.degrees(np.arccos(np.clip(-grasp.approach[2], -1.0, 1.0)))) <= LEVEL_TILT_DEG:
            return self.home_quat
        for axis in (grasp.close, np.cross(grasp.approach, grasp.close)):
            flat = np.array([axis[0], axis[1], 0.0])
            if float(np.linalg.norm(flat)) > 0.3:
                return self._quat(down, flat / float(np.linalg.norm(flat)))
        return self.home_quat

    def _recover_legs(self) -> list[_Leg]:
        """Nothing, or one leg back to the posture the arm started in.

        The escape hatch for a pick planned from a *stuck* arm. Everything else
        here reasons about where the tool is; this is the one thing that reasons
        about where the arm is, because an arm left leaning on a joint limit --
        which the far compartment can do, being at the edge of the reach -- does
        not fail the next leg so much as ignore it, and no amount of re-reading
        the pile fixes that. The home pose is the one pose known to be
        reachable, since the episode began in it, so a plan that starts from
        there starts from somewhere.
        """
        here = np.asarray(self.ik.ee_pose(self.env.state)[0], dtype=np.float64)
        leg = self.legs[self.index - 1] if 0 < self.index <= len(self.legs) else None
        if leg is None or float(np.linalg.norm(here - leg.position)) < STALLED:
            return []
        position, quat = self.ik.home_pose
        self.record.setdefault(self.tool, _Attempt()).notes.append("recovered to home")
        return self._chain([("recover", np.asarray(position, dtype=np.float64), quat, 0.0, 0, POS_TOL)])

    def _place_legs(self, tool: str) -> list[_Leg]:
        """Carry to the compartment and let go over it.

        **The release height is measured, not assumed.** A tool taken 25 mm from
        its end hangs most of its length below the hand, and lowering the *hand*
        to a fixed 0.12 m over a compartment then drives the tool through the
        compartment's floor -- the arm stalls against it, still holding on, and
        the pose it is left standing in is the one that had the next three picks
        never reach their targets. So the drop is taken from the lowest corner
        of the tool as it actually hangs right now, which is the same measurement
        the baseline makes before posting a syringe through a lid and for the
        same reason.

        **And it is as low as the hand fits, not as high as the baseline had
        to.** ``DROP``'s 0.12 m comes from a routine that had no model of the
        hand: fill a compartment from well above and let the walls keep a
        bouncing tool in. The cost of the height is the landing, and one object
        here does not survive it -- ``wire_spool`` weighs 32 g, and let go 89 mm
        over a compartment it arrives fast enough to pass straight through the
        floor, the bench under it, and out of the task. (The same contact
        failure :data:`SHOVED_DOWN` guards against under a driven squeeze, on a
        free drop instead.)

        So the release descends until the *palm* is level with the compartment
        rim and no further, which is a real limit rather than a chosen one: the
        hand is 200 mm across and a compartment is 195 mm wide, so nothing about
        this hand goes down into one. On this cell that is 49 mm over the floor
        instead of 120, which takes a third off the impact speed, and it is
        still the walls that keep the tool in.
        """
        body_q = self._body_q()
        region = self.env.handles.regions[self.region_of[tool]]
        floor = float(region.origin_now(body_q)[2] + region.z_range[0])
        hang = float(np.asarray(self.ik.ee_pose(self.env.state)[0])[2] - self._lowest_z(tool))
        aim = self._aim_point(region, body_q)
        rim = float(region.origin_now(body_q)[2]) + BIN_WALL_TOP
        lowest = max(rim - self.palm_lift, floor + hang + DROP_CLEAR)
        release = np.array([aim[0], aim[1], min(lowest, floor + max(DROP, hang + DROP_CLEAR))])
        over = np.array([aim[0], aim[1], max(release[2] + TRANSIT_CLEAR, self.transit_z)])
        grasp = self._grasp
        quat = self._transit_quat()
        grip = self._grip(grasp.width) if grasp is not None else 1.0
        return self._chain(
            [
                ("stage", self.staging, quat, grip, 0, CROSS_TOL),
                ("carry", over, quat, grip, 0, POS_TOL),
                ("lower", release, quat, grip, 0, POS_TOL),
                ("release", release, quat, 0.0, GRIP_FRAMES, POS_TOL),
                ("retract", over, quat, 0.0, 0, POS_TOL),
            ]
        )

    def _aim_point(self, region, body_q: np.ndarray) -> np.ndarray:
        """Where in a compartment to let go: its near edge, not its centre.

        A compartment is 0.195 x 0.245 m and the goal is satisfied anywhere in
        it, so which point is aimed at is free -- and it is not free to the arm.
        ``bin2``'s far quadrant centre is at the edge of this Panda's reach, and
        an arm that gets there gets there by *straightening its elbow*: the
        posture that arrives is 2.3 rad away from the one it started in, on a
        joint the Cartesian servo has no way to fold back, so the pick after it
        spends its whole budget stuck against a shoulder limit. Aiming at the
        corner of the quadrant nearest the base, one drop radius in, costs
        nothing that is scored and takes 0.1 m off the reach.
        """
        origin = region.origin_now(body_q)
        x0, y0, x1, y1 = region.xy_ranges
        lo = origin[:2] + np.array([min(x0, x1), min(y0, y1)])
        hi = origin[:2] + np.array([max(x0, x1), max(y0, y1)])
        near = np.clip(self.base_xy, lo, hi)
        inset = np.minimum(AIM_INSET, 0.5 * (hi - lo))
        return np.clip(near + np.sign(0.5 * (lo + hi) - near) * inset, lo + inset, hi - inset)

    def _tip_legs(self, tool: str) -> list[_Leg]:
        """Push a flat plate into the nearest bin wall so that it tips up.

        The one non-prehensile move here, and the one place this routine is
        *hoping* rather than reasoning. What it does is well defined: bring the
        shut fingers to the plate's near side face at :data:`TIP_HEIGHT`, above
        the mid-plane of a plate that is 0.053-0.065 m thick, and drive them at
        the wall the plate is nearest. The plate slides until its far edge meets
        the wall and then has a moment about that edge.

        What it cannot promise is the outcome. The rim stands only 0.08 m above
        the floor, so a plate that keeps going does not lean against the wall so
        much as pivot over it, and one that has already been pushed flat to the
        wall has no run-up left. Hence :data:`TIP_ATTEMPTS`, and hence
        ``tip_up=False`` being a supported way to run: the pure grasping policy
        is the honest measurement of the face search, and this is a bet placed
        on top of it.
        """
        centre, axes, half = self._box(tool)
        over = self._bin_of(centre)
        if over is None:
            return []
        origin, (x0, y0, x1, y1) = over
        walls = (
            (np.array([1.0, 0.0, 0.0]), origin + np.array([x1, 0.0, 0.0])),
            (np.array([-1.0, 0.0, 0.0]), origin + np.array([x0, 0.0, 0.0])),
            (np.array([0.0, 1.0, 0.0]), origin + np.array([0.0, y1, 0.0])),
            (np.array([0.0, -1.0, 0.0]), origin + np.array([0.0, y0, 0.0])),
        )
        # The nearest wall in the direction being pushed, measured from the
        # plate's own edge rather than its centre -- and the fingers have to fit
        # in behind the plate to push it at all, so a candidate whose run-up
        # would start inside the opposite wall is dropped rather than clipped.
        height = origin[2] + BIN_FLOOR + TIP_HEIGHT
        footprint = np.array([[origin[0] + x0, origin[1] + y0], [origin[0] + x1, origin[1] + y1]])
        best: tuple[np.ndarray, np.ndarray, float] | None = None
        for direction, point in walls:
            edge = support(axes, half, direction)
            gap = float(np.dot(direction, point - centre)) - edge
            if gap <= 0.01:
                continue
            start = centre - direction * (edge + 0.045)
            if np.any(start[:2] < footprint[0] + 0.012) or np.any(start[:2] > footprint[1] - 0.012):
                continue
            if best is None or gap < best[2]:
                best = (direction, np.array([start[0], start[1], height]), gap)
        if best is None:
            return []
        direction, start, gap = best
        edge = support(axes, half, direction)
        finish = start + direction * (gap + edge + TIP_OVERRUN)
        # Open, and across the push. Two fingertips 0.08 m apart along the face
        # push it squarely; one shut pair is a blade at a point, and a point on
        # a 0.18 m plate turns it in yaw instead of standing it up.
        quat = self._quat(np.array([0.0, 0.0, -1.0]), np.cross(np.array([0.0, 0.0, 1.0]), direction))
        above = start + np.array([0.0, 0.0, 0.14])
        return self._chain(
            [
                ("tip_over", above, quat, 0.0, 0, POS_TOL),
                ("tip_down", start, quat, 0.0, 0, POS_TOL),
                ("tip_push", finish, quat, 0.0, 90, POS_TOL),
                ("tip_clear", finish - direction * 0.03 + np.array([0.0, 0.0, 0.16]), quat, 0.0, 0, POS_TOL),
            ]
        )

    # ------------------------------------------------------------- verification
    def _held(self, tool: str) -> bool:
        """Whether the lift left the object between the fingers."""
        position = self.env.handles.objects[tool].position(self._body_q())
        tcp = np.asarray(self.ik.ee_pose(self.env.state)[0], dtype=np.float64)
        return bool(np.linalg.norm(position - tcp) < GRASP_RADIUS and position[2] > self._grasp_z + GRASP_RISE)

    def _placed(self, tool: str) -> bool:
        """Whether the object is in the region the evaluator will score it in."""
        body_q = self._body_q()
        region = self.env.handles.regions[self.region_of[tool]]
        origin = region.origin_now(body_q)
        x0, y0, x1, y1 = region.xy_ranges
        lo = origin + np.array([min(x0, x1), min(y0, y1), region.z_range[0]])
        hi = origin + np.array([max(x0, x1), max(y0, y1), region.z_range[1]])
        position = self.env.handles.objects[tool].position(body_q)
        return bool(np.all(position >= lo) and np.all(position <= hi))

    # --------------------------------------------------------------------- run
    @property
    def _leg(self) -> _Leg | None:
        return self.legs[self.index] if self.index < len(self.legs) else None

    def act(self) -> np.ndarray:
        """One control step's action, in the task's OSC space."""
        if self._leg is None:
            self._advance()
        leg = self._leg
        if leg is None:
            return self.env.idle_action(self.MODE)

        position, quat = self.ik.ee_pose(self.env.state)
        delta = osc_action(position, quat, leg.position, leg.quat, output_max=self.output_max)
        self.frame += 1
        if leg.converge and self._arrived(position, quat, leg):
            self.held += 1
        else:
            self.held = 0
        if self.frame >= leg.max_steps or self.held >= SETTLE_FRAMES or self._shoving(leg):
            self.index += 1
            self.frame = 0
            self.held = 0
        return self._pack(np.append(delta, leg.grip))

    def _shoving(self, leg: _Leg) -> bool:
        """Whether the hand is pushing the tool instead of closing around it.

        The one piece of feedback in a leg besides its own convergence, and it
        is about the *object*. A descent or a close that has started moving what
        it is reaching for has already failed as a grasp -- the fingers are on
        the outside of it -- and every step after that is a stiff servo leaning
        on a rigid box.

        **This is a guard against the simulator as much as against a bad
        grasp.** Held against the bin floor by a closing jaw, a tool here does
        not stop at the floor: measured on ``tactical_flashlight_1``, its centre
        is driven 21 mm into a 40 mm slab over twelve control steps and then
        passes through both the bin floor and the bench beneath it, ending on
        the world ground plane 0.8 m below the workcell -- from where it can
        never be picked and the goal is gone. The scene is perfectly stable when
        nobody is touching it (30 s of idle moves nothing by more than a
        millimetre), so this is a contact failure under a driven squeeze rather
        than anything wrong with where the tools are or what shape they are.
        Not being the one who drives the squeeze is the only part of that this
        routine controls.

        So the leg ends and, if the fingers were closing, they close where they
        are. Sometimes that is a grasp anyway; when it is not, the pick is
        scored a miss and the retry re-reads a pile that has genuinely changed.
        """
        if self.stage != "pick" or leg.label not in ("descend", "close") or not self.tool:
            return False
        position = self.env.handles.objects[self.tool].position(self._body_q())
        moved = position - self._object_at
        if leg.label == "descend":
            return bool(np.linalg.norm(moved) > DISTURBED)
        # During the close the jaws are meant to move the object a little as
        # they seat on it, and only *downward* travel means it is being driven
        # into whatever it is resting on.
        return bool(-moved[2] > SHOVED_DOWN)

    def _arrived(self, position, quat, leg: _Leg) -> bool:
        if float(np.linalg.norm(np.asarray(position) - leg.position)) > leg.tol:
            return False
        dot = abs(float(np.dot(np.asarray(quat, dtype=np.float64), leg.quat)))
        return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))) <= ROT_TOL_DEG

    def _advance(self) -> None:
        """Decide what happens next: verify what just ran, then load the next legs."""
        stage, self.stage = self.stage, ""
        if stage == "pick" and self.tool:
            if self._held(self.tool):
                self._load("place", self._place_legs(self.tool))
                return
            self.record[self.tool].notes.append("pick missed")
            self.message = f"{self.tool}: nothing in the fingers, re-reading the pile"
        elif stage == "place" and self.tool:
            if self._placed(self.tool):
                self.queue.remove(self.tool)
                self.record[self.tool].places += 1
                self.message = f"{self.tool} is in {self.region_of[self.tool]}"
            else:
                self.record[self.tool].notes.append("placement missed")
                self.message = f"{self.tool}: released outside its compartment, taking it again"
        self.tool = ""
        self._start_next()

    def _start_next(self) -> None:
        """Choose a tool and load the legs for it, or finish.

        Three passes, in the order that disturbs the pile least: every tool that
        can be taken cleanly, then every tool that can be taken if the fingers
        are allowed near a neighbour, and only if *neither* pass proposed
        anything for any tool, the tip-up. That ordering is what tells the two
        reasons a tool cannot be picked apart without asking a second question:
        a tool boxed in by its neighbours is picked up by the second pass or
        freed by somebody else's pick, so anything still standing when both
        passes come back empty is standing on its own geometry.
        """
        order = [tool for tool in self._order() if self.record[tool].picks < self.attempts]
        for relaxed in (False, True):
            for tool in order:
                candidates = self._candidates(tool, relaxed=relaxed)
                if not candidates:
                    continue
                grasp = candidates[0]
                attempt = self.record[tool]
                attempt.picks += 1
                self.tool = tool
                self._grasp = grasp
                self._object_at = self.env.handles.objects[tool].position(self._body_q())
                self._grasp_z = float(self._object_at[2])
                self._load("pick", self._recover_legs() + self._pick_legs(grasp))
                tilt = np.degrees(np.arccos(np.clip(-grasp.approach[2], -1.0, 1.0)))
                self.message = (
                    f"pick {attempt.picks}/{self.attempts} of {tool}: {grasp.width * 1000:.0f} mm across, "
                    f"approach {tilt:.0f} deg off vertical{', crowded' if relaxed else ''}"
                )
                return

        for tool in order:
            attempt = self.record[tool]
            # Only a tool whose *pose* refuses every grasp is pushed at, and
            # that question is the scene-free one: a flat plate proposes nothing
            # to :func:`face_grasps` at all, where a flashlight boxed in by its
            # neighbours proposes a perfectly good grasp that only the clearance
            # tests reject. Without this the routine shoves a graspable tool
            # into a wall for want of anything else to do, and one of them went
            # over the wall and onto the bench, which is a goal thrown away
            # rather than a goal not reached.
            if self._candidates_ignoring_the_cell(tool):
                continue
            if self.tip_up and attempt.tips < TIP_ATTEMPTS:
                legs = self._tip_legs(tool)
                if legs:
                    attempt.tips += 1
                    attempt.notes.append("tipped")
                    self._load("tip", legs)
                    self.message = f"{tool} offers no face the jaws fit; pushing it against the bin wall"
                    return
            attempt.notes.append("lying flat, no face fits the jaws")
            attempt.picks = self.attempts

        self.finished = True
        sorted_count = len(self.goals) - len(self.queue)
        self.message = f"routine finished: {sorted_count}/{len(self.goals)} placed, left {sorted(self.queue)}"
        self.env.request_termination(self.message)

    def _candidates_ignoring_the_cell(self, tool: str) -> list[Grasp]:
        """What this object's *pose* admits, with nothing else in the world.

        The question that separates "cannot be picked yet" from "cannot be
        picked". See :meth:`_start_next`, which is the only caller and the whole
        reason it exists.
        """
        centre, axes, half = self._box(tool)
        return face_grasps(centre, axes, half, span=self.span)

    def _order(self) -> list[str]:
        """Outstanding tools, fewest attempts first and then top of the pile.

        The height half is the baseline's and for its reason: they start
        stacked, so lifting the buried one drags whatever is on it, and every
        pick changes the answer. It matters more here than there, because the
        two grasps this routine adds are grasps of objects that are *tilted*,
        and a tilted object is one settling of the pile away from being flat.

        The attempt count in front of it is this routine's, and it is what turns
        three tries into three *different* tries. Sorted on height alone, a tool
        that just failed is still the tallest thing in the bin, so it is tried
        again against a pile that has not moved -- which is how a flashlight
        spent all three of its attempts under an overhanging multimeter that
        the very next pick would have taken away.
        """
        body_q = self._body_q()
        return sorted(
            self.queue,
            key=lambda name: (
                self.record[name].picks,
                -float(self.env.handles.objects[name].position(body_q)[2]),
            ),
        )

    def _load(self, stage: str, legs: list[_Leg]) -> None:
        self.stage = stage
        self.legs = legs
        self.index = 0
        self.frame = 0
        self.held = 0

    # ---------------------------------------------------------------- actions
    # ----------------------------------------------------------- policy shape
