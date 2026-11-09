"""Score the route the way the evaluator will.

Mode-aware: pickplace checks the chain's midpoint against the goal lane, wrap
checks the winding count against the suite's gate (carried in the state so
the graph and the score cannot disagree).
"""

from typing import TypedDict

from gap import NodeContext

PLACE_TOLERANCE_M = 0.05


class Output(TypedDict):
    route: str
    turns: float
    goal_distance_mm: float


def run(ctx: NodeContext) -> Output:
    """The distance is computed inside the branch that has a lane.

    It used to be computed first, against a ``goal_xy`` the wrap variant carried
    and had no use for -- a leftover from when both modes read one metadata
    block and every key came back whether it meant anything or not. The lane is
    a declared region now and the wrap declares none, so asking for it there is
    asking for a place that does not exist.
    """
    state = ctx.tool("sim.get_cable_state")
    distance = float("nan")
    if state["mode"] == "wrap":
        ok = float(state["turns"]) >= float(state["turns_gate"])
    else:
        mid, goal = state["midpoint"], state["goal_xy"]
        distance = ((mid[0] - goal[0]) ** 2 + (mid[1] - goal[1]) ** 2) ** 0.5
        ok = distance <= PLACE_TOLERANCE_M
    return {
        "route": "routed" if ok else "short",
        "turns": float(state["turns"]),
        "goal_distance_mm": float(distance) * 1e3,
    }
