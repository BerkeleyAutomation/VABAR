# SPDX-License-Identifier: Apache-2.0
"""The cable suite's tools: cable state, the plans, the grip, and the loop's memory.

Sixteen entries in two halves, registered one call at a time so the privilege
scan can read the tag -- see :func:`register_cable_tools`, which is where the
split and what it costs on this tree are written down.

**``sim.get_cable_state``** is the deformable's ground truth: every segment's
position, the winding count about the task's peg, and how far the chain sits
from the working gripper. It is what the graph's verify reads, and what its
grasp node uses to confirm the jaws actually took the cable.

**``sim.plan_route``** solves the whole keyframe schedule into joint-space
stage trajectories with the arm's full-pose LM solver -- yaw authority
included, which the runtime's own IK trades away and a grasp across a cable
cannot afford. The wrap's grasp stages are solved against the **live** cable
(nearest segment to the bend, jaws across the local tangent): the plan is
privileged twice, once for knowing where the cable is and once for never
having to see it.

**``sim.perceive_weave``** is the ORACLE BODY of the perception slot: the
stations, their required sides, the scored seat abscissae and the rod, read out
of the simulator. It exists to be replaced -- the vision graph declares the same
four keys off the ``cable`` camera -- and confining the privileged read to one
tool is what makes "the vision body is the only thing that changed" a checkable
claim rather than a described one.

**``sim.set_gripper`` / ``sim.wait_steps``** for the same reason the USB-C
suite carries them: every grip here is a partial command (3 mm of finger
travel, on the wrap's 22 mm reference rod and the weave's 7 mm alike --
``robot.close_gripper``'s full close squirts it out), ramped because
a step input on the finger servo slams the pads.

**The ``cable.*`` half** is what a weave needs that is not a reading: the second
hand's opening grasp, the two-arm carry that pays slack out while the work hand
crosses, the crossing counter, and the two loop-carried values a graph cannot
express as edges. None of it reads the scene, save that ``cable.plan_station``
falls back to the scene's own answer for a geometry argument the caller leaves
``None``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from robosimstudio.arms import arm_id_of, arm_name as runtime_arm_name
from robosimstudio.solvers.motion.path import stream_hz as _stream_hz
from robosimstudio_materials.rod.measures import WRAP_GATE_TURNS, winding_turns
from vabar.cable.curve_tools import register_curve_tools

from .routine import SETTLE_STEPS, peg_xy_of

__all__ = ["ToolsCable", "register_cable_tools"]

WEAVE_SPRINGBACK = 0.012
"""How far past the seat a far-side crossing is placed [m].

Measured, not chosen: a segment released at the seat settles 9 mm back toward
the station line as the rod pulls straight against the spool. See
:meth:`ToolsCable.plan_station`.
"""


def register_cable_tools(registry, env, bus) -> None:
    """Register the cable suite's tools on a GaP tool registry.

    **One call per tool, with the tag written out**, which is
    ``tool_sort``'s shape rather than the loop over ``entries()`` the other
    suites use. ``tests/test_gap_privilege.py`` reads these call sites as TEXT,
    so what a reader sees at the call is what is passed -- a privileged tool
    registered under some other name and without the tag would slip through the
    runtime filter too, and the scan is what turns that hole into a decision
    someone has to make.

    **The ``cable.`` half is unprivileged, and that is the point of it.** Eight
    of these command the hand, solve IK over geometry the caller supplies, or
    hand back what the policy itself produced; none reads the simulator. The
    vision graph runs on them with the oracle withheld, which is the whole claim
    this suite's perception work makes -- ``--no-privileged`` is not a flag it
    tolerates, it is the condition the graph was written to be measured under.
    Tagging them ``privileged`` to satisfy a text scan would withhold them under
    that flag and the graph would abort at its first recovery call, which is
    exactly what it did before this was fixed: ``Tool 'cable.set_gripper' not
    found``.

    So they say what they are: a literal tag naming what the tool does, which
    ``tests/test_gap_privilege.py`` accepts as one of its four forms. The law
    asks every call site to state which side of the boundary it is on; it does
    not ask every tool to be on the privileged side.

    Args:
        registry: A ``gap_core.tools.ToolRegistry``.
        env: The toolkit-layer environment.
        bus: The bus whose snapshot carries the evaluator's metrics.
    """
    tools = ToolsCable(env, bus)
    # The unprivileged curve.* tools the perception nodes fit the rod with. The
    # RoboSimStudio this suite was written against registered them on every
    # connector; upstream does not carry them, so the suite brings its own.
    register_curve_tools(registry)
    registry.register_callable(
        "sim.get_cable_state",
        tools.get_cable_state,
        summary="Segment positions, winding turns, distance to the jaws.",
        tags=("privileged",),
    )
    registry.register_callable(
        "sim.plan_route",
        tools.plan_route,
        summary="The route as joint-space stage trajectories, grasp solved live.",
        tags=("privileged",),
    )
    registry.register_callable(
        "sim.plan_station",
        tools.plan_station,
        summary="The weave's crossings as joint-space stage trajectories, station by station.",
        tags=("privileged",),
    )
    registry.register_callable(
        "sim.perceive_weave",
        tools.perceive_weave,
        summary=(
            "The four things a weave has to see -- the stations, their sides, the scored seat "
            "abscissae and the rod -- read out of the simulator. The oracle body of the perception "
            "slot."
        ),
        tags=("privileged",),
    )
    registry.register_callable(
        "sim.set_gripper",
        tools.set_gripper,
        summary="Command a partial jaw width in metres, ramped.",
        tags=("privileged",),
    )
    registry.register_callable(
        "sim.hold_cable",
        tools.hold_cable,
        summary="Latch the pinched segment to the hand -- the suite's kinematic hold.",
        tags=("privileged",),
    )
    registry.register_callable(
        "sim.release_cable",
        tools.release_cable,
        summary="Drop the latch.",
        tags=("privileged",),
    )
    registry.register_callable(
        "sim.wait_steps",
        tools.wait_steps,
        summary="Dwell for N control steps while the policy loop holds.",
        tags=("privileged",),
    )
    registry.register_callable(
        "cable.plan_station",
        tools.plan_station_from,
        summary=(
            "One station as joint-space stage trajectories, planned against SUPPLIED geometry: it "
            "solves IK and choreography over whatever body saw the scene."
        ),
        tags=("planner",),
    )
    registry.register_callable(
        "cable.set_gripper",
        tools.set_gripper,
        summary=(
            "Command a partial jaw width in metres, ramped. A commanded width is an instruction to "
            "the hand rather than a reading of the scene; robot.close_gripper cannot carry one."
        ),
        tags=("hand",),
    )
    registry.register_callable(
        "cable.plan_support",
        tools.plan_support,
        summary=(
            "The support hand's opening move: take the cable's far end, to hold it while the work "
            "hand crosses. IK over the geometry it is handed."
        ),
        tags=("planner",),
    )
    registry.register_callable(
        "cable.carry_with_feed",
        tools.carry_with_feed,
        summary=(
            "Stream a crossing's carried legs on the work arm while the support arm pays cable out, "
            "both on the same control period."
        ),
        tags=("planner",),
    )
    registry.register_callable(
        "cable.next_station",
        tools.next_station,
        summary=(
            "The loop gate: is there another crossing to make, and which one? A policy counting its "
            "own passes, not a reading of the world."
        ),
        tags=("memory",),
    )
    registry.register_callable(
        "cable.support_memory",
        tools.support_memory,
        summary=(
            "Carry the support chain's last solved joints across a crossing -- the policy remembering "
            "where it put its own arm."
        ),
        tags=("memory",),
    )
    registry.register_callable(
        "cable.curve_memory",
        tools.curve_memory,
        summary=(
            "Carry the rod's last known centreline across a loop pass. It stores what perception "
            "already produced and reads nothing."
        ),
        tags=("memory",),
    )
    registry.register_callable(
        "cable.wait_steps",
        tools.wait_steps,
        summary="Dwell for N control steps. Waiting is not knowing.",
        tags=("hand",),
    )


class ToolsCable:
    """The suite's privileged reads, aimed at what the task actually declares.

    Every number here used to come out of ``TaskSpec.metadata``, which is gone.
    None of them was ever a parameter: the peg is a fixture with a handle, the
    goal lane is the region the goal is scored against, the work surface is the
    cell's, and the mode is the variant's own name. Reading them from those
    places rather than from a copy is what stops a graph verifying against one
    peg while the evaluator scores another -- the failure the tool_sort port
    found the expensive way, where a tool registry looked for the *previous*
    evaluator's metric key, found nothing, and reported a policy that could not
    sort one hammer.

    **Three fields of state, and they are the loop's rather than the world's.**
    A graph binds a subgraph's inputs from its upstream producers, and a
    loop-carried value has no upstream on the first pass -- the node that
    consumes it is the node that makes it. So the crossing counter, the last
    fitted centreline and the support chain's last solved joints live here, the
    way ``tool_sort`` keeps its own loop state. Nothing in them is read from the
    scene.
    """

    def __init__(self, env, bus) -> None:
        self._station = 0
        """Crossings begun. The loop gate's whole state -- see :meth:`next_station`."""
        self._curve = ""
        """The rod's last known centreline, JSON. See :meth:`curve_memory`."""
        self._support_q = ""
        """The support chain's last solved joints, JSON. See :meth:`support_memory`."""
        self.env = env
        self.bus = bus
        self.mode = env.task.name
        self.peg_xy = peg_xy_of(env)
        lane = env.handles.regions.get("goal_lane")
        # Empty for the wrap, which has no lane and no clause that wants one.
        # An invented default would be a number a graph could route on.
        self.goal_xy = tuple(float(v) for v in lane.center()[:2]) if lane is not None else ()

    # ------------------------------------------------------------------ state
    def get_cable_state(self) -> dict[str, Any]:
        def read() -> dict[str, Any]:
            env = self.env
            cable = env.scene.by_name[env.task.deformables[0].name]
            body_q = env.state.body_q.numpy()
            positions = cable.positions(body_q)
            turns, engaged = winding_turns(positions, self.peg_xy)
            from robosimstudio.core.math import transform_point  # noqa: PLC0415

            arm = env.handles.arms[self._arm_name()]
            tcp = transform_point(body_q[arm.ee_body], arm.ee_offset)
            nearest = float(np.min(np.linalg.norm(positions - tcp, axis=1)))
            mid = positions[len(positions) // 2]
            return {
                "segments": [[float(v) for v in p] for p in positions],
                "midpoint": [float(v) for v in mid],
                "turns": float(turns),
                "engaged_segments": int(engaged),
                "turns_gate": float(WRAP_GATE_TURNS),
                "nearest_to_tcp": float(nearest),
                "goal_xy": [float(v) for v in self.goal_xy],
                "peg_xy": [float(v) for v in self.peg_xy],
                "mode": self.mode,
            }

        return self.bus.call(read)

    # ------------------------------------------------------------------- plan
    def plan_route(self) -> dict[str, Any]:
        """The keyframe schedule as joint trajectories, one stage per phase.

        The routine's own plan
        (:func:`~tasks/cable_route/routine.py::plan_stages`),
        densified with the routine's own smoothstep -- a graph streaming these
        waypoints and the routine interpolating its endpoints produce the same
        motion, which is what makes their numbers comparable.

        DENSIFIED AT THE CELL'S CONTROL RATE, not at :data:`PLAN_HZ`, because
        the rate a stage is densified at *is* the speed it runs at:
        ``robot.execute_trajectory`` streams one waypoint per control step
        whenever the facade exposes ``stream_joint_trajectory``, and this one
        does. Densified at PLAN_HZ's 30 and streamed at this cell's 60, the
        wrap's 16.94 s schedule went out as 471 waypoints and ran in 7.85 s --
        every stage at 2x, the ring at 2.3x because ``int(0.193 * 30)``
        truncates 5.79 waypoints to 5, so the hand orbited the peg at 0.32 m/s
        where the script asks for 0.14. :func:`plan_station_stages` and
        ``ToolsUsbc._stream_hz`` already read the cell's rate for exactly this
        reason; this was the last planner that did not.

        WHAT FIXING IT IS AND IS NOT WORTH, measured 2026-08-12 on the wrap's
        one layout. Ten graph trials each: the median retained ``winding_turns``
        went 0.054 -> 0.212, and the outcome stopped being bimodal (eight of ten
        at <=0.084 and two at ~1.0, nothing between) and became a single spread
        of 0.011-0.574. On its own it did NOT close the gap to the routine's
        0.820 median and the success rate did not move; the settle below is what
        closed it, and the two together are what the numbers there describe.

        Neither fix is about the wrap forming. Sampled every control step, the
        winding peaks at ~1.0 turns on the ring in six of six episodes across
        both policies -- graph included, and including the ones that score 0.05.
        The wrap always forms. Everything measured here is about holding it.
        """
        # STAND STILL AND LET THE ROD SETTLE BEFORE READING IT, which is
        # `RoutineCableRoute.act`'s first act and was the one stage of its
        # decomposition the graph had no equivalent for: a graph's first node
        # runs at step ~1, and this one reads the live chain.
        #
        # It never read as a missing settle, because the grasp lands either way:
        # the jaws close on rod at 7 mm in ten of ten without it.
        #
        # AND THE OBVIOUS MECHANISM IS NOT THE ONE, which is worth recording
        # because it is the one a reader will assume. The guess was that the
        # settle moves the rod and so changes which segment gets taken. It does
        # not: measured three episodes each side, the graph takes chain segment
        # 8 at x = 0.588 with this settle and segment 8 at x = 0.585 without it,
        # while the routine takes segment 7 at x = 0.630 in all three of its own
        # -- so the settle does not reconcile the graph's grasp with the
        # routine's, and the 0.26 m of free rod on the working side is unchanged
        # by it. What changes is that the rod is at REST when the jaws close
        # rather than still moving under the anchor's pinch, and the outcome
        # moves with it. The effect is measured; this explanation is not, and
        # should not be repeated as though it were.
        #
        # THE EFFECT, ten graph trials against ten without it, both with the
        # stream-rate fix above and on the wrap's one layout: retained turns
        # went from a mean 0.221 (median 0.212, one trial over the 0.5 gate) to
        # a mean 0.773 (median 0.993, six over). Against the ORIGINAL graph the
        # distribution shift is Mann-Whitney z = +2.87, and against the scripted
        # routine's own eleven trials it is z = +0.56 -- which is to say the
        # graph is no longer distinguishable from the routine it reproduces.
        # Success counts alone do not carry that: 6/10 against 2/10 is Fisher
        # p = 0.17, because this variant's run-to-run spread is +-0.3 turns and
        # ten trials cannot resolve a rate through it. Compare the spreads, as
        # the atom's own calibration block does.
        self._wait_for_step(SETTLE_STEPS, timeout_s=60.0)

        def build() -> dict[str, Any]:
            from .routine import plan_stages  # noqa: PLC0415

            env = self.env
            ik = env.arm(self._arm_name())
            stream_hz = _stream_hz(env.task.physics)
            plan = plan_stages(env, self.mode)
            stages = []
            for name, qs, duration, grip_cmd in plan["stages"]:
                n = max(2, int(duration * stream_hz))
                waypoints = []
                for i in range(1, n + 1):
                    t = i / n
                    s = t * t * (3.0 - 2.0 * t)
                    arc = s * (len(qs) - 1)
                    lo = min(int(arc), len(qs) - 2)
                    q = qs[lo] + (arc - lo) * (qs[lo + 1] - qs[lo])
                    waypoints.append({"positions": [float(q[c]) for c in ik.arm_coords]})
                stages.append(
                    {
                        "stage": name,
                        "trajectory": {"waypoints": waypoints},
                        "duration": float(duration),
                        "grip_command_m": float(grip_cmd),
                    }
                )
            return {
                "arm_id": arm_id_of(env, self._arm_name()),
                "stages": stages,
                "grips": {
                    k: {"command_m": float(v), "fraction_closed": float(1.0 - v / self._grip_open())}
                    for k, v in plan["grips"].items()
                },
                "grasp_yaw": float(plan["grasp_yaw"]),
                "mode": self.mode,
                "errors": {k: float(v) for k, v in plan["errors"].items()},
                "worst_grasp_error": float(plan["worst_grasp_error"]),
            }

        return self.bus.call(build)

    # ----------------------------------------------------------------- weave
    def plan_station(self, index: int = 0, carry_clear: float = 0.030, approach: float = 0.055) -> dict[str, Any]:
        """The weave, station by station, as joint-space stage trajectories.

        **What a crossing is, physically.** The rod starts straight, 80 mm short
        of the station line and on the robot's side of every station. Each
        station names the side the rod has to finish on, and they alternate --
        so a station wanting the near side needs the rod pulled *to* it, and one
        wanting the far side needs it lifted over a 23.5 mm spool and set down
        beyond. One motion template serves both -- take, lift, cross, place --
        and whether a station runs it or tidies with a rest-height drag is
        decided against the **live rod**, not the side: a span left on the
        wrong side of its station must cross the fixture, and there is no
        dragging through a spool. The target x is the side's.

        **Worked from the anchored end outwards.** The support hand pins the far
        (+y) tip, so a crossing placed near it is held by cable that is not
        going anywhere, and each later crossing pulls slack from rod that has
        already been committed. Starting at the free end instead drags the whole
        rod through every station it has already passed.

        The target is the seat the *document* scores: shaft radius plus rod
        radius, on the station's named side. Read from the layout rather than
        restated, so a variant with different stations needs nothing here.

        Args:
            carry_clear: How far above the spool tops the rod is carried [m].
            approach: How far above the rod the hand stages before descending [m].

        Returns the stages in execution order, the grip commands, and which
        ``arm_id`` to drive -- the shape ``sim.plan_route`` returns, so
        ``move_stage.py`` drives either without knowing which it has.
        """

        def build() -> dict[str, Any]:
            from .routine import plan_station_stages  # noqa: PLC0415

            plan = plan_station_stages(self.env, index=index, carry_clear=carry_clear, approach=approach)
            return {"arm_id": arm_id_of(self.env, self._arm_name()), **plan}

        return self.bus.call(build)

    # ------------------------------------------------------------- perception
    def perceive_weave(self) -> dict[str, Any]:
        """The four numbers a weave is planned from, read out of the simulator.

        The oracle body of the perception slot, and it exists to be *replaced*.
        Its whole contract is the shape of its answer -- stations, sides,
        ``seats_x`` and a rod polyline -- so a vision body that fits spools and
        a centreline from RGB-D declares the same keys and every node
        downstream is unchanged.

        ``seats_x`` is the *evaluator's* box centre here, which is the one key
        a perceptual body cannot reproduce by construction: it has to derive
        the same abscissa from the post it measured, post radius plus rod
        radius on the named side. The difference between the two is the
        cleanest per-station error signal this task has.
        """

        def read() -> dict[str, Any]:
            from .assets.layout import WEAVE_LAYOUTS, WEAVE_SIDES  # noqa: PLC0415
            from .routine import station_xy  # noqa: PLC0415

            env = self.env
            variant = env.task.name
            if variant not in WEAVE_LAYOUTS:
                raise RuntimeError(f"perceive_weave is for the weave variants; this is {variant!r}")
            # Off the compiled scene, not off the layout table. This is the
            # *oracle*, so it is allowed to know exactly where the stations are
            # -- but "exactly" means this episode's, and a board variant draws
            # its stations per episode. Reading the table would have reported a
            # board the episode did not build, alongside `seats_x` read from the
            # regions, which are this episode's: an oracle disagreeing with
            # itself, and the vision body it exists to be compared against would
            # have looked wrong for being right.
            stations = station_xy(env)
            sides = [int(WEAVE_SIDES[k % len(WEAVE_SIDES)]) for k in range(len(stations))]
            rod = env.scene.by_name[env.task.deformables[0].name].positions(env.state.body_q.numpy())
            seats_x: list[float] = []
            for k, (sx, _sy) in enumerate(stations):
                region = env.handles.regions.get(f"seat{k}")
                seats_x.append(float(region.center()[0]) if region is not None else float(sx))
            return {
                "stations": [[float(sx), float(sy)] for sx, sy in stations],
                "sides": sides,
                "seats_x": seats_x,
                "rod": [[float(v) for v in p] for p in rod],
                "body": "oracle",
            }

        return self.bus.call(read)

    def plan_station_from(
        self,
        index: int = 0,
        stations: Any = None,
        sides: Any = None,
        seats_x: Any = None,
        rod: Any = None,
        carry_clear: float = 0.030,
        approach: float = 0.055,
        seat_bias: float | None = None,
        max_offset: float | None = None,
        slack_margin: float | None = None,
        final_pass: bool | None = None,
    ) -> dict[str, Any]:
        """``sim.plan_station``'s choreography over geometry the caller supplies.

        Deliberately not a ``sim.`` name: it reads nothing the simulator knows
        that a robot would not. What it does is IK and choreography -- which
        template a station runs, the pass order, the pad floor, the arc
        re-timing -- over four numbers somebody else measured.

        All four scene fields must be measured and supplied. This tool survives
        ``--no-privileged``; silently filling a missing field from simulator
        state would make a vision evaluation privileged under an innocent name.
        """
        supplied = {"stations": stations, "sides": sides, "seats_x": seats_x, "rod": rod}
        missing = [key for key, value in supplied.items() if value is None]
        if missing:
            raise ValueError(f"cable.plan_station requires measured {', '.join(missing)}")
        scene = dict(supplied)

        def build() -> dict[str, Any]:
            from .routine import plan_station_stages  # noqa: PLC0415

            plan = plan_station_stages(
                self.env,
                index=index,
                carry_clear=carry_clear,
                approach=approach,
                scene=scene,
                seat_bias=seat_bias,
                max_offset=max_offset,
                slack_margin=slack_margin,
                final_pass=final_pass,
            )
            return {"arm_id": arm_id_of(self.env, self._arm_name()), "perceived": sorted(scene), **plan}

        return self.bus.call(build)

    # ------------------------------------------------------------ the far end
    def plan_support(
        self,
        approach: float = 0.055,
        rod: Any = None,
        grasp_xy: Any = None,
        grasp_axis_z: float | None = None,
        grasp_tangent: Any = None,
        arm: str = "",
    ) -> dict[str, Any]:
        """The support hand's anchor grasp, as joint-space stages.

        **Why a weave needs a second hand at all.** ``RoutineWeave`` opens with
        this move and its own note says what happens without it: a free cable
        answers a sideways pull by *translating* rather than bending, so the
        first crossing hauls the whole cable toward one station and the next has
        none left to grasp -- measured, 26 of 42 segments piled at one station.
        The far end has to be held by something before any crossing is worth
        planning, and the graph ran single-handed until this existed, which was
        the one structural difference between it and the routine that solves
        this task.
        """

        if rod is None:
            raise RuntimeError(
                "cable.plan_support needs the measured centreline it is to grasp; "
                "it will not fall back to simulator cable state"
            )

        def build() -> dict[str, Any]:
            from .routine import plan_support_stages  # noqa: PLC0415

            cable = self.env.task.deformables[0]
            start_fixed = "start" in tuple(getattr(cable, "anchors", ()))
            name = (
                self.env.handles.work_arm
                if arm == "work"
                else self.env.handles.support_arm
                if arm in ("", "support")
                else str(arm)
            )
            if start_fixed and grasp_xy is None:
                return {
                    "arm_id": arm_id_of(self.env, name),
                    "arm_name": name,
                    "start_fixed": True,
                    "stages": [],
                    "segment": -1,
                    "worst_grasp_error": 0.0,
                }
            plan = plan_support_stages(
                self.env,
                approach=approach,
                rod=rod,
                grasp_xy=grasp_xy,
                grasp_axis_z=grasp_axis_z,
                grasp_tangent=grasp_tangent,
                arm=name,
            )
            return {
                "arm_id": arm_id_of(self.env, name),
                "arm_name": name,
                "start_fixed": False,
                **{k: v for k, v in plan.items() if k != "errors"},
            }

        return self.bus.call(build)

    def carry_with_feed(
        self,
        plan: Any = None,
        legs: str = "move,cross",
        q_support: Any = None,
        quat: Any = None,
        segment: int = -1,
    ) -> dict[str, Any]:
        """The carry, with the far hand paying cable out underneath it.

        **Why this cannot be two sequential calls.** The cable between two
        pinches is inextensible. Once the chord from the anchor to the place
        point exceeds the material run between them, something gives -- and with
        the far tip pinned rigid what gives is the work hand: the cross leg
        pulls the cable straight out of its 0.5 mm pinch. Paying out first
        slackens the cable onto the bench; paying out afterwards is too late.
        The two streams advance on the same period or the crossing fails.

        Measured, and this is why the tool exists: anchoring the far end without
        it doubled full solves (1/6 -> 2/6) and *lowered* the mean, because the
        crossings that used to merely drift now fight a taut cable.

        The feed is measured rather than scripted -- the cable's path from the
        pinch to the place point less the material run between them -- so a
        crossing that owes no slack streams nothing and the hand stands still.
        """
        import json  # noqa: PLC0415

        if (plan or {}).get("grasp_slack_m") is None:
            raise ValueError("cable.carry_with_feed requires measured grasp_slack_m")
        wanted = [t.strip() for t in str(legs).split(",") if t.strip()]
        stages = [s for s in ((plan or {}).get("stages") or []) if str(s.get("stage")) in wanted]
        if not stages:
            return {"streamed": 0, "fed": False, "reason": "no carried legs"}

        work_rows = [[float(v) for v in wp["positions"]] for st in stages for wp in st["trajectory"]["waypoints"]]

        def build_feed() -> Any:
            from .routine import plan_support_feed  # noqa: PLC0415

            if q_support is None or quat is None:
                return None
            rows, q_end = plan_support_feed(
                self.env,
                plan,
                np.asarray(q_support, dtype=np.float64),
                np.asarray(quat, dtype=np.float64),
                segment=int(segment),
            )
            return (
                None if rows is None else [[float(v) for v in r] for r in rows],
                [float(v) for v in np.asarray(q_end).reshape(-1)],
            )

        got = self.bus.call(build_feed)
        feed_rows, q_end = got if isinstance(got, tuple) else (got, None)
        if q_end:
            # Where the next pay-out seeds from. Without this every crossing
            # re-seeds from the anchor and the hand replays one motion.
            self.support_memory(q=json.dumps(q_end))
        streams = {int(arm_id_of(self.env, self._arm_name())): work_rows}
        if feed_rows:
            streams[int(arm_id_of(self.env, self.env.handles.support_arm))] = feed_rows

        # THE FACADE, off the bus, because the connector publishes it there --
        # this is the one tool in the suite that needs more of the runtime than
        # the bus's own `call`.
        facade = getattr(self.bus, "facade", None)
        if facade is None or not hasattr(facade, "stream_dual_trajectory"):
            return {"streamed": 0, "fed": False, "reason": "no multi-arm streaming on this runtime"}
        # `settle_max_steps=0` is the whole point: the facade's dual stream
        # normally settles each arm on its own end point once the tracks run
        # out, and settling here would hold the work hand still at the end of a
        # CARRY while the rod it is dragging keeps moving. The stages that want
        # a dwell ask for one (`place`), and the ones that do not must not get
        # one for free.
        facade.stream_dual_trajectory(streams, settle_max_steps=0)
        return {
            "streamed": len(work_rows),
            "fed": bool(feed_rows),
            "feed_rows": len(feed_rows or []),
            "reason": "",
        }

    # ------------------------------------------------------ the loop's memory
    def next_station(self, total: int = 0, advance: bool = True) -> dict[str, Any]:
        """Which crossing to make next, or that there are none left.

        ``tool_sort``'s ``sim.next_goal`` in the cable's own terms, and it is a
        gate rather than a branch inside the planning subgraph for the reason
        that file gives: a subgraph declares its outputs by ``$ref`` into its
        own nodes, so one that sometimes skips the node producing a plan would
        have an output with nothing behind it. Keeping the decision in a
        subgraph of its own lets the planner always run and always export.

        **The count comes from PERCEPTION, not from the layout.** ``total`` is
        however many stations the perception slot found, so a vision body that
        sees two spools of three weaves two crossings and says so, rather than
        the graph asking for a station nothing saw.

        A policy is entitled to remember how many passes it has made. Nothing
        here reads the scene.
        """
        n = max(0, int(total))
        index = int(self._station)
        if n <= 0 or index >= n:
            return {"route": "finished", "index": index, "remaining": 0, "total": n}
        if advance:
            self._station = index + 1
        return {"route": "found", "index": index, "remaining": max(0, n - index - 1), "total": n}

    def support_memory(self, q: str = "") -> dict[str, Any]:
        """The support chain's last solved joint vector, across crossings.

        **Each pay-out starts where the last one ended.** ``plan_support_feed``
        returns ``(rows, q_end)`` and the routine keeps that ``q_end`` as the
        seed for the next crossing, because the support hand has *moved*: it
        paid cable out and is now somewhere else. Re-seeding every feed from the
        anchor's original pose replays the same motion from the same start, so
        the hand walks out and snaps back once per crossing, dragging the cable
        with it.

        Measured before it was carried: six episodes, all scoring exactly 0.50
        with the same two stations short. Six identical failures is not noise.
        """
        if str(q).strip():
            self._support_q = str(q)
        return {"q": self._support_q}

    def curve_memory(self, rod: str = "") -> dict[str, Any]:
        """The rod's last known centreline, remembered across a loop pass.

        **Why a tool and not an edge.** A graph binds a subgraph's inputs from
        its *upstream* producers, and the prior curve has no upstream on the
        first pass -- the node that consumes it is the node that makes it. Left
        as a declared input it fails validation with "input 'rod' has no
        upstream producer", which is correct: a loop-carried value is state, not
        dataflow, and ``tool_sort`` keeps its own loop state the same way.

        This stores what the perception body already produced and hands it back.
        It reads nothing about the world that the policy did not itself observe.
        """
        if str(rod).strip():
            self._curve = str(rod)
        return {"rod": self._curve}

    # ------------------------------------------------------------------ roles
    def _arm_name(self, arm: str = "") -> str:
        """The named arm, or whichever one this pairing works with.

        Never a ``"right"`` literal: the two YAM cells list their arms in
        opposite orders, so a literal names a different hand on each robot and
        the binding is the only thing that knows which is meant.
        """
        return arm or self.env.handles.work_arm

    def _grip_open(self, arm: str = "") -> float:
        """This arm's finger command at full open [m], read off the model."""
        return self.env.handles.arms[self._arm_name(arm)].gripper_open

    # ------------------------------------------------------------------- grip
    def set_gripper(
        self, width: float = -1.0, arm: str = "", ramp_steps: int = 24,
        arm_id: int | None = None,
    ) -> dict[str, Any]:
        """Ramp the jaw command to ``width`` metres, 0 shut. Negative is full open.

        The default is a sentinel rather than the number, because the number is
        a property of whichever hand this pairing brought and is not known until
        the scene is compiled.
        """
        if arm_id is not None:
            selected = runtime_arm_name(self.env, arm_id)
            if arm and arm != selected:
                raise ValueError(f"arm {arm!r} conflicts with arm_id {arm_id} ({selected!r})")
            arm = selected
        arm = self._arm_name(arm)
        span = self._grip_open(arm)

        def read_current() -> float:
            return float(self.env.gripper(arm).command)

        def write_fraction(fraction: float) -> None:
            gripper = self.env.gripper(arm)
            gripper.command = float(fraction)
            gripper.apply(self.env.pipeline.control.joint_target_q)

        cmd = float(np.clip(span if width < 0.0 else float(width), 0.0, span))
        target_fraction = 1.0 - cmd / span
        start_fraction = self.bus.call(read_current)
        steps = max(1, int(ramp_steps))
        for i in range(1, steps + 1):
            fraction = start_fraction + (target_fraction - start_fraction) * (i / steps)
            self.bus.call(write_fraction, fraction)
            self._wait_for_step(1)
        return {"arm": arm, "command_m": cmd, "fraction_closed": target_fraction}

    def hold_cable(self) -> dict[str, Any]:
        """Engage the hold on whatever segment sits in the jaws.

        Privileged the way the tape latch is: a real gripper would have to
        earn the hold. Returns ``held: False`` when nothing is within reach,
        which a graph should route on rather than dragging air around a peg.
        """

        def engage() -> dict[str, Any]:
            cable = self.env.scene.by_name[self.env.task.deformables[0].name]
            seg = cable.hold_segment(True, self.env.state)
            return {"held": seg >= 0, "segment": int(seg)}

        return self.bus.call(engage)

    def release_cable(self) -> dict[str, Any]:
        def drop() -> dict[str, Any]:
            cable = self.env.scene.by_name[self.env.task.deformables[0].name]
            cable.hold_segment(False)
            return {"released": True}

        return self.bus.call(drop)

    def wait_steps(self, steps: int = 30) -> dict[str, Any]:
        return {"waited": self._wait_for_step(int(steps), timeout_s=60.0)}

    def _wait_for_step(self, steps: int, timeout_s: float = 10.0) -> int:
        """Block until the owner has stepped ``steps`` times, through the bus's own wait.

        Polling ``snapshot.step`` in a sleep loop parks nowhere, so under a
        lockstep clock or a fleet barrier -- which step only while the graph is
        parked in a bus wait -- the step it spins for never comes and every wait
        burns its whole deadline. Same shape as upstream's ``robot_tools._wait``.
        """
        from robosimstudio.core.bus import BusTimeout

        start = int(self.bus.snapshot.step)
        target = start + int(steps)
        deadline = time.monotonic() + float(timeout_s)
        while int(self.bus.snapshot.step) < target:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                self.bus.wait_for_step(after=int(self.bus.snapshot.step), timeout=remaining)
            except BusTimeout:
                break
        return int(self.bus.snapshot.step) - start
