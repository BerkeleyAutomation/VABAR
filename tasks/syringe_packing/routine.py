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

    env = Env("tool_sort/level1_6")     # declares joint_position, for the graph
    policy = SolutionRoutine(env)      # emits osc_pose, and says so

The environment carries both controllers and hands the arm over on the first
tagged action, re-seeded -- see
:meth:`~robosimstudio.core.env.EnvBase.use_control`. Which is the point: a graph
that plans joint trajectories and a routine that servos Cartesian deltas are two
baselines on *one* scene, and making them two scenes would have made them
incomparable.
"""

from __future__ import annotations

import numpy as np

from robosimstudio.core.math import quat_from_axis_angle, quat_multiply, quat_to_mat
from robosimstudio.solvers import ActionOsc, OscWaypoints, Waypoint, grasp_from_obb, top_down_grasp_quat
from robosimstudio_bench.goals.goal import placements

__all__ = ["RoutineSharpsDisposal"]

GRASP_HANDLE_FRAC = -0.70
"""Slide the grasp 52.5 mm along the syringe towards its plunger flange.

**Grasping the centroid stopped working when the collider stopped being a box,
and this is the whole of that story.** `75a953f` replaced the VAB export's
collision boxes with convex hulls of the drawn mesh -- right, because those boxes
stood up to 27 mm proud of what a viewer sees. The syringe's box was 23.6 mm wide
along its **entire** length, so closing to 19.6 mm bit into it anywhere,
centroid included. The hull is not:

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


class RoutineSharpsDisposal:
    """Pick each goal's tool out of the pile and place it in its region.

    Args:
        env: The environment to drive. It must offer an :data:`MODE` controller,
            which every scene with a gripper does.
        arm: Which arm sorts. Empty uses the first.
        handle_frac: How far along the long axis to slide the grasp from the
            centroid towards the light end, as a fraction of the half-extent.
            Negative slides the other way, towards the heavy end, which is what
            this task needs -- see :data:`GRASP_HANDLE_FRAC`.

            ``tool_sort`` keeps 0.0, because that is what the reference's
            *recorded baselines* use (``--handle_frac 0.0``, stated under its
            task table) rather than the 0.5 its scripts happen to carry as an
            argparse default. Comparing against a published number means running
            the configuration it was published at. This suite has no published
            number left to match: the one it had was measured against a collider
            it no longer has.
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
        handle_frac: float = GRASP_HANDLE_FRAC,
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
        ``syringe_packing/tray_pair``: grasped and lifted at a 22.7 mm opening, then
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
            legs += [Waypoint("over_hole", tuple(over), tuple(release_quat), grip, hold)]
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
        """The :class:`~robosimstudio_bench.solutions.base.SolutionBase` interface."""
        del obs
        return self.act()
