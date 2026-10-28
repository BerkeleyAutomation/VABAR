"""The route, planned by the simulator: joint-space stages with yaw authority.

Privileged twice over: the wrap's grasp stages are solved against the live
cable (nearest segment to the bend, jaws across the local tangent), and every
waypoint carries full orientation -- the runtime's own IK leaves wrist roll
free, and a grasp across a cable cannot afford a free roll.
"""

from typing import TypedDict

from gap import NodeContext

PLAN_ERROR_LIMIT_M = 0.025


class Output(TypedDict):
    route: str
    stages: list
    arm_id: int
    grips: dict
    mode: str
    worst_grasp_error_mm: float


def run(ctx: NodeContext) -> Output:
    plan = ctx.tool("sim.plan_route")
    worst = float(plan["worst_grasp_error"])
    return {
        "route": "planned" if worst <= PLAN_ERROR_LIMIT_M else "unreachable",
        "stages": plan["stages"],
        "arm_id": int(plan["arm_id"]),
        "grips": plan["grips"],
        "mode": str(plan["mode"]),
        "worst_grasp_error_mm": worst * 1e3,
    }
