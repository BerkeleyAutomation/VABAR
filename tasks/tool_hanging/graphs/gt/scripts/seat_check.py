"""With the jaws still on it: is the tool where the goal wants it?

Read before the release rather than after, on purpose. The goal's three clauses
are geometry, borne load and rest, and only the first of them can be true while
the hand is still holding the tool -- so this reads the geometry alone and lets
the release supply the other two. A node that waited for the score to turn true
would be waiting for something it has to let go of first.

The verdict is the ``threaded`` atom over the same declared features the
evaluator reads, so this node and the score cannot disagree about what hung
means. ``offset`` comes with it because a near miss and a wild one want
different answers: still on the axis and merely short is worth pushing again,
and tens of millimetres off is a tool that is going to be dropped wherever it is.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    object_name: str
    threaded: bool
    crossings: bool
    offset_mm: float
    clearance_mm: float
    held: bool


def run(ctx: NodeContext, allow_retry: bool = False) -> Output:
    state = ctx.tool("sim.get_hang_state")
    seated = bool(state["threaded"])
    route = "seated"
    if not seated:
        # Worth one more push only while the tool is still in the hand and the
        # loop and the rod are at least crossing: the stations are all computed
        # from a carried tool, and re-running them over an empty gripper mimes
        # a hang. Anything else releases where it is and lets the cursor decide
        # whether the job gets another whole pass.
        route = "again" if (allow_retry and bool(state["held"])) else "short"
    return {
        "route": route,
        "object_name": str(state["object_name"]),
        "threaded": seated,
        "crossings": bool(state["crossings"]),
        "offset_mm": float(state["offset_mm"]),
        "clearance_mm": float(state["clearance_mm"]),
        "held": bool(state["held"]),
    }
