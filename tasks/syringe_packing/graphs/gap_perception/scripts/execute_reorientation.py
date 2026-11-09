"""Execute the carry and turn the syringe upright at a wrist yaw the insertion can fly.

Canonical executor from the reusable executing-held-object-motion skill, with one
task-specific step: the carry to the box and the turn over it are chosen together,
by planning, instead of carrying and then taking the planner's smallest-turn guess.

Why (``tray_clutter_t03``, 2026-09-13): the smallest turn put the hand on the
far side of the aperture from the base. The turn itself planned, but the
insertion strokes that follow came within 2-6 mrad of a joint limit, and cuRobo
refused them ("IK line switched joint branch"). ``tools/sharps_placement_pretest.py``
found comfortable yaws at the same placement, with 0.4-0.7 rad of margin. Whether
one is reachable depends on the posture the carry ends in, and in sweeps s3/s5 a
cramped carry end (t06, t08, t11) left every in-place yaw at 1-6 mrad. At t08's box
position the pretest reaches 0.7-0.9 rad from a better carry end.

Every candidate is *planned*, from the posture it would really start in, before
anything moves:

1. **carry, then turn in place.** The planner's carry to the transit position, in
   the pick orientation (``motion.plan_to_pose``). Each yaw's in-place turn is
   probed from that carry's planned end joints (``seed_joints`` is the start
   state).
2. **carried upright.** If no in-place turn leaves real margin, each yaw is
   planned as one free move from the live joints straight to the upright transit
   pose. The arm reconfigures on the way, which is how VAB-industrial's scripted
   policy carries a syringe to the hole (``run_tool_sort.py``: "the arm
   reconfigures into that pose over the length of the move").
3. **carry, then free turn**, as a last resort.

For each candidate, the insertion strokes ``plan_visual_insert.py`` will fly and
the retract ``release_and_retract.py`` makes are probed from its end joints
(``motion.plan_linear`` with ``seed_joints``). The kept candidate has the largest
joint-limit margin over every planned waypoint, and its planned trajectories are
what get executed.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    final_pose: dict[str, Any]


YAW_STEP_DEG = 30.0
"""Candidate spacing about the aperture axis. The pretest's comfortable band is
about 50 deg wide at 10 deg resolution, so 30 deg always lands one candidate in it."""

MARGIN_ENOUGH_RAD = 0.30
"""Past this, more joint margin stops outranking a smaller wrist turn."""

MARGIN_GOOD_RAD = 0.15
"""A carry-then-turn candidate this good ends the search; below it the upright carry is tried."""

MARGIN_RETRY_RAD = 0.05
"""Below this best margin, the carry-then-free-turn candidates are tried too."""

RETRACT_PROBE_M = 0.16
"""The rise ``release_and_retract.py`` makes after opening, probed from the stroke's end."""

TURN_TIME_SCALE = 4.0
"""Playback stretch for any leg that turns the syringe, which is slower than the
plan's ``time_scale`` for the others. The syringe is pinched at one station, and
two pads resist a pivot about the pinch axis only by friction.
``tray_clutter_t24`` (2026-09-13), second syringe: a 119 deg turn played at 2x
rotated the hand at about 6.6 deg per 10 steps, and the syringe swung 60 deg in
the fingers within 30 steps. The seat check then refused it for a 44 deg tilt."""

RECONFIGURING_TURN_TIME_SCALE = 6.0
"""Playback stretch for a turn that is not in place (a free turn or an upright
carry). These reconfigure the whole arm. Sweep s7's t10 took a 139 deg free turn
at 4x that dropped the hand 116 mm, and the syringe swung 70 deg in the fingers."""

REPLAN_JOINT_TOLERANCE_RAD = 0.05
"""If the executed carry ends farther than this from its planned end joints, the
chosen turn is re-planned from the live joints."""


def _time_scale(trajectory: dict[str, Any], scale: float) -> dict[str, Any]:
    """Resample a joint path so execution duration changes, not its geometry."""
    waypoints = list(trajectory.get("waypoints") or [])
    if scale <= 1.0 or len(waypoints) < 2:
        return trajectory
    rows = np.asarray([waypoint["positions"] for waypoint in waypoints], dtype=np.float64)
    count = max(2, int(round((len(rows) - 1) * scale)) + 1)
    source = np.linspace(0.0, 1.0, len(rows))
    target = np.linspace(0.0, 1.0, count)
    scaled = np.stack(
        [np.interp(target, source, rows[:, joint]) for joint in range(rows.shape[1])],
        axis=1,
    )
    return {"waypoints": [{"positions": row.tolist()} for row in scaled]}


def _rotation(quat: dict[str, float]) -> Rotation:
    return Rotation.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]])


def _pose(position: np.ndarray, rotation: Rotation) -> dict[str, Any]:
    q = rotation.as_quat()
    return {"position": dict(zip(("x", "y", "z"), map(float, position))),
            "rotation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])}}


def _path(result: dict[str, Any]) -> np.ndarray | None:
    trajectory = result.get("trajectory") if result.get("planned") else None
    if not trajectory or not trajectory.get("waypoints"):
        return None
    return np.asarray([w["positions"] for w in trajectory["waypoints"]], dtype=np.float64)


def _best(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    feasible = ([c for c in candidates if c["stage"] == "ok"]
                or [c for c in candidates if c["stage"] == "inserted, retract refused"])
    if not feasible:
        return None
    return max(feasible, key=lambda c: (min(c["margin"], MARGIN_ENOUGH_RAD), -c["turn_deg"]))


def _report(candidates: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{c['yaw_deg']:+.0f} {c['strategy']}: {c['stage']}" + ("" if c["margin"] is None else f" {c['margin']:.3f}")
        for c in candidates
    )


class _Search:
    """Candidate carries and turns over the aperture, probed with the planners that fly them."""

    def __init__(self, ctx: NodeContext, search: dict[str, Any], carry_pose: dict[str, Any], world: Any,
                 attachment: Any) -> None:
        self.ctx, self.world, self.attachment, self.carry_pose = ctx, world, attachment, carry_pose
        limits = ctx.tool("robot.describe_arm")["joint_limits"]
        self.lower = np.asarray(limits["lower"], dtype=np.float64)
        self.upper = np.asarray(limits["upper"], dtype=np.float64)
        self.current = _rotation(ctx.tool("robot.get_ee_pose")["pose"]["rotation"])
        self.center = np.asarray(search["fixture_center"], dtype=np.float64)
        axis = np.asarray(search["axis"], dtype=np.float64)
        self.axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
        self.offset = np.asarray(search["feature_offset"], dtype=np.float64)
        held_axis = self.current.apply(_rotation(search["feature_rotation"]).apply([0.0, 0.0, 1.0]))
        correction, _ = Rotation.align_vectors(
            [self.axis], [held_axis / max(float(np.linalg.norm(held_axis)), 1.0e-12)])
        self.aligned = correction * self.current
        self.clearance = float(search.get("transit_clearance_m", 0.08))
        self.depths = [float(d) for d in search.get("insert_depths_m", (-0.020, -0.010, 0.0))]

    def margin(self, rows: np.ndarray) -> float:
        rows = rows[:, : len(self.lower)]
        return float(np.minimum(rows - self.lower, self.upper - rows).min())

    def hand(self, tip: np.ndarray, rotation: Rotation) -> dict[str, Any]:
        return _pose(tip - rotation.apply(self.offset), rotation)

    def candidate(self, yaw: float, strategy: str, start: np.ndarray | None, hold: bool,
                  base_margin: float = np.inf) -> dict[str, Any]:
        """Plan the move to one yaw's upright transit pose from ``start`` (live
        joints when None), then probe the insertion and retract from its end."""
        rotation = Rotation.from_rotvec(self.axis * np.deg2rad(yaw)) * self.aligned
        transit = self.hand(self.center - self.clearance * self.axis, rotation)
        entry = {"yaw_deg": float(yaw), "strategy": strategy, "hold": hold, "stage": "turn refused",
                 "margin": None, "pose": transit, "start": start,
                 "turn_deg": float(np.degrees((rotation * self.current.inv()).magnitude()))}
        kwargs = {"pose": transit, "world_config": self.world, "attached_object": self.attachment,
                  "hold_position": hold}
        if start is not None:
            kwargs["seed_joints"] = [float(v) for v in start]
        rows = _path(self.ctx.tool("motion.plan_to_pose", **kwargs))
        if rows is None:
            return entry
        entry.update(turn={"waypoints": [{"positions": r.tolist()} for r in rows]},
                     margin=min(base_margin, self.margin(rows)))
        joints, leg_start = rows[-1], transit
        for index, depth in enumerate(self.depths):
            target = self.hand(self.center + depth * self.axis, rotation)
            contact = index > 0
            leg = self.ctx.tool(
                "motion.plan_linear", start=leg_start, end=target, seed_joints=[float(v) for v in joints],
                orientation="lock", world_config=self.world, attached_object=self.attachment,
                allow_start_contact=contact, allow_goal_contact=contact,
                contact_margin=0.008 if contact else 0.005,
            )
            leg_rows = _path(leg)
            if leg_rows is None:
                entry["stage"] = f"insertion leg {index} refused"
                return entry
            entry["margin"] = min(entry["margin"], self.margin(leg_rows))
            joints, leg_start = leg_rows[-1], target
        # The retract after release must also be flyable from where the stroke
        # ends. tray_clutter_t11 chose a yaw whose insertion planned but left
        # the shoulder 0.1 rad from its limit, and the 160 mm retract failed.
        entry["stage"] = "inserted, retract refused"
        lifted = {"position": dict(leg_start["position"]), "rotation": dict(leg_start["rotation"])}
        lifted["position"]["z"] = float(lifted["position"]["z"]) + RETRACT_PROBE_M
        retract_rows = _path(self.ctx.tool(
            "motion.plan_linear", start=leg_start, end=lifted, seed_joints=[float(v) for v in joints],
            orientation="lock", world_config=self.world, allow_start_contact=True, contact_margin=0.008,
        ))
        if retract_rows is not None:
            entry["margin"] = min(entry["margin"], self.margin(retract_rows))
            entry["stage"] = "ok"
        return entry

    def run(self) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """The chosen carry (None when the candidate carries upright itself) and turn."""
        yaws = np.arange(-180.0, 180.0, YAW_STEP_DEG)
        candidates: list[dict[str, Any]] = []
        carry = self.ctx.tool("motion.plan_to_pose", pose=self.carry_pose, world_config=self.world,
                              attached_object=self.attachment)
        carry_rows = _path(carry)
        if carry_rows is not None:
            carry_margin = self.margin(carry_rows)
            candidates += [self.candidate(y, "turn in place", carry_rows[-1], True, carry_margin) for y in yaws]
            print(f"[carry_sg.execute] carry then turn in place -- {_report(candidates)}", flush=True)
        best = _best(candidates)
        if best is None or best["margin"] < MARGIN_GOOD_RAD:
            upright = [self.candidate(y, "carried upright", None, False) for y in yaws]
            print(f"[carry_sg.execute] carried upright -- {_report(upright)}", flush=True)
            candidates += upright
            best = _best(candidates)
        if carry_rows is not None and (best is None or best["margin"] < MARGIN_RETRY_RAD):
            free = [self.candidate(y, "free turn", carry_rows[-1], False, carry_margin) for y in yaws]
            print(f"[carry_sg.execute] carry then free turn -- {_report(free)}", flush=True)
            candidates += free
            best = _best(candidates)
        if best is None:
            raise RuntimeError(f"no carry and wrist yaw lets cuRobo turn and insert ({_report(candidates)})")
        print(f"[carry_sg.execute] chose wrist yaw {best['yaw_deg']:+.0f} {best['strategy']} "
              f"(margin {best['margin']:.3f} rad, turn {best['turn_deg']:.0f} deg)", flush=True)
        carry_trajectory = None if best["start"] is None else carry["trajectory"]
        return carry_trajectory, best


def run(ctx: NodeContext, reorientation_plan: dict[str, Any]) -> Output:
    waypoints = list(reorientation_plan.get("waypoints") or [])
    if not waypoints:
        raise ValueError("reorientation_plan must contain waypoints")
    world = reorientation_plan.get("world_config")
    attachment = reorientation_plan.get("attached_object")
    time_scale = max(1.0, float(reorientation_plan.get("time_scale", 1.0)))
    final_pose = None
    index = 0
    while index < len(waypoints):
        waypoint = waypoints[index]
        final_pose = waypoint["pose"]
        mode = waypoint.get("mode")
        if mode is None:
            mode = "contact_transition" if waypoint.get("contact_transition") else (
                "planned_linear" if waypoint.get("cartesian") else "planned_joint"
            )
        index += 1
        if mode == "contact_transition":
            ctx.tool("robot.go_to_pose_cartesian", pose=final_pose)
            continue
        waypoint_attachment = attachment if waypoint.get("use_attachment", True) else None
        following = waypoints[index] if index < len(waypoints) else None
        if mode == "planned_joint" and following is not None and following.get("turn_search"):
            # This carry and the turn after it are chosen together (module docstring).
            carry_trajectory, best = _Search(ctx, following["turn_search"], final_pose, world,
                                             waypoint_attachment).run()
            turn = best["turn"]
            if carry_trajectory is not None:
                ctx.tool("robot.execute_trajectory", trajectory=_time_scale(carry_trajectory, time_scale))
                live = np.asarray(ctx.tool("robot.forward_kinematics")["joint_config"], dtype=np.float64)
                planned_end = np.asarray(best["start"], dtype=np.float64)
                if float(np.max(np.abs(live[: len(planned_end)] - planned_end))) > REPLAN_JOINT_TOLERANCE_RAD:
                    replanned = ctx.tool("motion.plan_to_pose", pose=best["pose"], world_config=world,
                                         attached_object=waypoint_attachment, hold_position=best["hold"])
                    if replanned.get("planned") and (replanned.get("trajectory") or {}).get("waypoints"):
                        turn = replanned["trajectory"]
            turn_scale = TURN_TIME_SCALE if best["strategy"] == "turn in place" else RECONFIGURING_TURN_TIME_SCALE
            ctx.tool("robot.execute_trajectory", trajectory=_time_scale(turn, max(time_scale, turn_scale)))
            final_pose = best["pose"]
            index += 1  # the turn waypoint is done
            continue
        result = ctx.tool(
            "motion.plan_linear" if mode == "planned_linear" else "motion.plan_to_pose",
            **(
                {"end": final_pose, "orientation": "lock", "world_config": world, "attached_object": waypoint_attachment}
                if mode == "planned_linear"
                else {"pose": final_pose, "world_config": world, "attached_object": waypoint_attachment,
                      "hold_position": bool(waypoint.get("hold_position", False))}
            ),
        )
        trajectory = result.get("trajectory") if result.get("planned") else None
        if not trajectory or not trajectory.get("waypoints"):
            raise RuntimeError("cuRobo refused the syringe reorientation waypoint")
        ctx.tool("robot.execute_trajectory", trajectory=_time_scale(trajectory, time_scale))
    return {"final_pose": final_pose}
