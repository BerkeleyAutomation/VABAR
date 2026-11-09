"""Canonical clearance planner from planning-held-object-motion."""

from typing import Any, TypedDict
from gap import NodeContext


class Output(TypedDict):
    reorientation_plan: dict[str, Any]


def run(ctx: NodeContext, approach_pose: dict[str, Any], world_config: dict[str, Any],
        attached_object: dict[str, Any]) -> Output:
    del ctx
    return {"reorientation_plan": {
        "time_scale": 2.0,
        "waypoints": [{"pose": approach_pose, "mode": "planned_joint", "cartesian": False}],
        "world_config": world_config,
        "attached_object": attached_object,
    }}
