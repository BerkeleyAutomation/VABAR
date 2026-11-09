"""Re-plan the hang around where the tool **actually** ended up in the jaws.

The closed-loop node, and the one thing this graph has that the scripted routine
does not. The routine computes every station -- pre-hook, over-hook, lower --
from the pose it measured *before* it closed the jaws, and then never looks
again. Two things happen in between that it therefore pays for at the hook: the
close shifts the tool by up to a few millimetres as the pads bottom out on it,
and the turn to vertical swings whatever the pads did not hold.

This runs at the turn station with the tool in hand, reads the tool's live pose,
computes where its landmark rides relative to the tool centre *now*, and puts
the stations where that landmark lands on the holder. ``slip_mm`` is the
difference from what the pick predicted -- the quantity the routine cannot see.

``lost`` rather than a plan when the tool is not in the hand: the stations are
computed from where the tool is being carried, and a hand that is carrying
nothing would be given a plan aimed at a tool lying on the table.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    object_name: str
    arm_id: int
    phases: dict
    slip_mm: list
    worst_mm: float
    worst_deg: float


def run(ctx: NodeContext) -> Output:
    state = ctx.tool("sim.get_hang_state")
    if not bool(state["held"]):
        return {
            "route": "lost",
            "object_name": str(state["object_name"]),
            "arm_id": int(state["arm_id"]),
            "phases": {},
            "slip_mm": [],
            "worst_mm": 0.0,
            "worst_deg": 0.0,
        }
    plan = ctx.tool("sim.plan_place")
    errors = plan.get("errors_mm_deg") or {}
    return {
        "route": "carried",
        "object_name": str(plan["object_name"]),
        "arm_id": int(plan["arm_id"]),
        "phases": dict(plan["phases"]),
        "slip_mm": [round(float(v), 2) for v in plan["slip_mm"]],
        "worst_mm": max((float(v[0]) * 1e3 for v in errors.values()), default=0.0),
        "worst_deg": max((float(v[1]) for v in errors.values()), default=0.0),
    }
