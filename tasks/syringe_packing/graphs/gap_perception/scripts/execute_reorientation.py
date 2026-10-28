"""Canonical executor from the reusable executing-held-object-motion skill."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    final_pose: dict[str, Any]


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


def run(ctx: NodeContext, reorientation_plan: dict[str, Any]) -> Output:
    waypoints = list(reorientation_plan.get("waypoints") or [])
    if not waypoints:
        raise ValueError("reorientation_plan must contain waypoints")
    world = reorientation_plan.get("world_config")
    attachment = reorientation_plan.get("attached_object")
    time_scale = max(1.0, float(reorientation_plan.get("time_scale", 1.0)))
    final_pose = None
    for waypoint in waypoints:
        final_pose = waypoint.get("pose", final_pose)
        mode = waypoint.get("mode")
        if mode is None:
            mode = "contact_transition" if waypoint.get("contact_transition") else (
                "planned_linear" if waypoint.get("cartesian") else "planned_joint"
            )
        if mode == "contact_transition":
            ctx.tool("robot.go_to_pose_cartesian", pose=final_pose)
            continue
        if mode == "joint_path":
            # An explicit joint-space path the planner authored itself -- the
            # wrist-only turn: a few joints move, the rest hold, no re-solve.
            rows = [{"positions": [float(v) for v in row]} for row in waypoint["positions"]]
            scale = max(time_scale, float(waypoint.get("time_scale", 0.0)))
            ctx.tool("robot.execute_trajectory", trajectory=_time_scale({"waypoints": rows}, scale))
            continue
        waypoint_attachment = attachment if waypoint.get("use_attachment", True) else None
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
            # The same conservatism execute_insertion falls back for, and not a
            # TSDF artefact -- the mesh world refused these same waypoints on
            # two of the first sweep's layouts. Every reorientation waypoint is
            # authored in the cleared air above the bench with the syringe held
            # at its centre, so an orientation-locked free-space line is the
            # graph author's deliberate fallback here too.
            result = ctx.tool(
                "motion.plan_linear",
                end=final_pose, orientation="lock",
                attached_object=waypoint_attachment,
                allow_start_contact=True, allow_goal_contact=True,
            )
            trajectory = result.get("trajectory") if result.get("planned") else None
        if not trajectory or not trajectory.get("waypoints"):
            raise RuntimeError("cuRobo refused the syringe reorientation waypoint")
        # A waypoint may ask to be flown slower than the plan's own scale: the
        # in-place upright turn is a rotation about the pinch axis, the one
        # axis a two-pad grip resists only by friction, so its speed is what
        # decides whether the syringe comes round with the hand or pivots.
        scale = max(time_scale, float(waypoint.get("time_scale", 0.0)))
        ctx.tool("robot.execute_trajectory", trajectory=_time_scale(trajectory, scale))
    return {"final_pose": final_pose}
