"""Run one crossing's legs, with the grips the plan calibrated.

Execution only: the legs arrive solved, and what this adds is the *boundary*
work a trajectory cannot carry — when the jaws open, when they close, and how
long to stand still while they travel.

**Grip before the leg, never after.** The jaws open ahead of the leg that lifts
away; opening after it means the leg lifts with the rod still pinched. And a
finger command is a target, not a state: stepping straight into the carry leaves
the pads mid-travel, so each boundary dwells for the jaws' own travel time.

**The pinch carries the rod — there is no latch.** ``RoutineWeave`` dropped
``sim.hold_cable`` once the hand actually reached the rod, and measured the
pinch holding it: 266 contacts at peak, 390 mm of rod moved. A kinematic hold
here would be doing the work the grasp is supposed to be judged on.
"""

import json
from typing import TypedDict

from gap import NodeContext

WEAVE_CLOSED = ("move", "cross", "place")
"""Legs flown with the rod in the jaws."""


class Output(TypedDict):
    route: str
    stages: int


def run(
    ctx: NodeContext,
    plan: str = "",
    arm_id: int = 0,
    grip_open_m: float = 0.0475,
    grip_close_m: float = 0.003,
    settle_steps: int = 24,
) -> Output:
    stages = (json.loads(plan or "{}") or {}).get("stages") or []
    if not stages:
        return {"route": "blocked", "stages": 0}
    for stage in stages:
        name = str(stage["stage"])
        if name == "clear":
            ctx.tool("cable.set_gripper", width=float(grip_open_m))
            ctx.tool("cable.wait_steps", steps=int(settle_steps))
        ctx.tool(
            "robot.execute_trajectory",
            trajectory=stage["trajectory"],
            arm_id=int(arm_id),
            max_steps_per_waypoint=15,
        )
        if name == "take":
            ctx.tool("cable.set_gripper", width=float(grip_close_m))
            ctx.tool("cable.wait_steps", steps=int(settle_steps))
    return {"route": "crossed", "stages": len(stages)}
