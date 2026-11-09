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
        final_pose = waypoint["pose"]
        mode = waypoint.get("mode")
        if mode is None:
            mode = "contact_transition" if waypoint.get("contact_transition") else (
                "planned_linear" if waypoint.get("cartesian") else "planned_joint"
            )
        if mode == "contact_transition":
            ctx.tool("robot.go_to_pose_cartesian", pose=final_pose)
            continue
        waypoint_attachment = attachment if waypoint.get("use_attachment", True) else None
        result = ctx.tool(
            "motion.plan_linear" if mode == "planned_linear" else "motion.plan_to_pose",
            **(
                {"end": final_pose, "orientation": "lock", "world_config": world, "attached_object": waypoint_attachment}
                if mode == "planned_linear"
                else {"pose": final_pose, "world_config": world, "attached_object": waypoint_attachment}
            ),
        )
        trajectory = result.get("trajectory") if result.get("planned") else None
        if not trajectory or not trajectory.get("waypoints"):
            raise RuntimeError("cuRobo refused the syringe reorientation waypoint")
        ctx.tool(
            "robot.execute_trajectory",
            trajectory=_time_scale(trajectory, time_scale),
            max_steps_per_waypoint=60,
        )
    return {"final_pose": final_pose}
