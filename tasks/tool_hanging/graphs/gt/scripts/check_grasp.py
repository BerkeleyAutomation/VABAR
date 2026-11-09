"""Did the close find anything, or shut on the table beside the tool?

Read from the **jaw gap**, not from contact, and that choice is the whole point
of this node. A tool still lying on the table loads neither pad, so the contact
test the evaluator's grasp predicate uses reads false on a perfectly good pinch
-- measured across a clearance sweep, it read false at every instant sampled on
grasps that then lifted a spanner 30 cm off the table. What *is* readable before
the lift is that the pads stalled: the same command reaches 0.7-1.2 mm on air
and stops at 13-17 mm on a spanner's shaft.

It earns its place on this suite specifically. The README's one open routine item
is the 17 mm spanner, whose jaws "close ~1 mm above its thinner shaft and the
wrench stays on the table": the scripted routine then flies the empty hand
through the whole hang and reports a wrench on a hook that nothing ever left the
table for. Routing to the give-up branch instead spends the remaining budget on
a second pass, or on the tool that can still be hung.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    object_name: str
    pinched: bool
    jaw_gap_mm: float
    jaw_shut_mm: float


def run(ctx: NodeContext) -> Output:
    state = ctx.tool("sim.get_hang_state")
    pinched = bool(state["pinched"])
    return {
        "route": "grasped" if pinched else "missed",
        "object_name": str(state["object_name"]),
        "pinched": pinched,
        "jaw_gap_mm": float(state["jaw_gap_mm"]),
        "jaw_shut_mm": float(state["jaw_shut_mm"]),
    }
