"""Execute orientation-locked insertion waypoints with the reusable fixture skill."""

import os
import sys
from typing import Any, TypedDict

from gap import NodeContext

# Helpers shared by this graph's scripts live beside them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_checks import checked_execute  # noqa: E402


class Output(TypedDict):
    final_pose: dict[str, Any]


def run(ctx: NodeContext, placement_plan: dict[str, Any]) -> Output:
    waypoints = list(placement_plan.get("waypoints") or [])
    world = placement_plan.get("world_config")
    attachment = placement_plan.get("attached_object")
    if not waypoints:
        raise ValueError("placement_plan must contain waypoints")
    final_pose = None
    for waypoint in waypoints:
        final_pose = waypoint["pose"]
        mode = waypoint.get("mode") or (
            "planned_linear" if waypoint.get("cartesian", False) else "planned_joint"
        )
        result = ctx.tool(
            "motion.plan_linear" if mode == "planned_linear" else "motion.plan_to_pose",
            **(
                {"end": final_pose, "orientation": "lock", "world_config": world,
                 "attached_object": attachment,
                 "allow_start_contact": bool(waypoint.get("allow_start_contact", False)),
                 "allow_goal_contact": bool(waypoint.get("allow_goal_contact", False)),
                 "contact_margin": float(waypoint.get("contact_margin", 0.005))}
                if mode == "planned_linear"
                else {"pose": final_pose, "world_config": world, "attached_object": attachment}
            ),
        )
        trajectory = result.get("trajectory") if result.get("planned") else None
        if not trajectory or not trajectory.get("waypoints"):
            raise RuntimeError("cuRobo refused the aperture approach waypoint")
        # Release is forbidden unless proprioception confirms the arm actually
        # reached the insertion target. One failed episode opened 66 mm high.
        final_pose = checked_execute(
            ctx, trajectory, final_pose, scale=3.0,
            position_tolerance=0.004, angle_tolerance=0.04,
        )
    return {"final_pose": final_pose}

