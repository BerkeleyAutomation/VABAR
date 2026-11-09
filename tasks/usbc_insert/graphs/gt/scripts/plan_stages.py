"""The joint-space schedule, from the simulator's own plan.

Privileged twice over: the plan knows where the plug and the bore are without
looking, and it is solved with the arm's full-pose LM solver -- position AND
orientation -- because a keyed connector cannot afford the free wrist roll the
runtime's own go_to_pose IK trades away.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    stages: list
    arm_id: int
    grips: dict


def run(ctx: NodeContext) -> Output:
    plan = ctx.tool("sim.plan_insert")
    return {
        "route": "planned",
        "stages": plan["stages"],
        "arm_id": int(plan["arm_id"]),
        "grips": plan["grips"],
    }
