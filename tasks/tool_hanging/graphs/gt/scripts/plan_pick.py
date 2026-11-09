"""Measure the outstanding tool and solve the reach onto it.

The pick half of the plan: approach, descend, lift and the turn to vertical,
solved against the arm's own full-pose objective and densified at the cell's
control rate, so streaming a stage runs it at its declared duration.

Full pose rather than position-plus-approach-axis, and that is the whole reason
this is a suite tool rather than ``robot.go_to_pose``: the runtime's own IK
leaves wrist roll free, and a tool standing on end *is* a roll. A spanner turned
to the wrong roll presents its ring edge-on to the hook and misses by the width
of the ring.

The errors come back per stage, in millimetres and degrees, because a plan that
cannot be held is the failure worth telling apart from a hang that missed --
``routing`` on it is cheaper than watching an arm run a path it never solved.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    object_name: str
    arm_id: int
    phases: dict
    lift_mm: float
    worst_mm: float
    worst_deg: float


def run(ctx: NodeContext) -> Output:
    plan = ctx.tool("sim.plan_pick")
    errors = plan.get("errors_mm_deg") or {}
    worst_mm = max((float(v[0]) * 1e3 for v in errors.values()), default=0.0)
    worst_deg = max((float(v[1]) for v in errors.values()), default=0.0)
    return {
        "route": "found",
        "object_name": str(plan["object_name"]),
        "arm_id": int(plan["arm_id"]),
        "phases": dict(plan["phases"]),
        "lift_mm": float(plan["lift_m"]) * 1e3,
        "worst_mm": worst_mm,
        "worst_deg": worst_deg,
    }
