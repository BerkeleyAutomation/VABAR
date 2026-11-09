"""The loop gate: is there another goal to do, and which one?

A sorting problem is a *conjunction* of goals, so a workflow that runs pick,
transport and place once solves a single-goal variant and leaves an eight-goal
one at reward 0.125. This node is what makes the difference -- it is entered at
the top of every pass, and its ``route`` decides whether the graph goes round
again or finishes.

It is a gate rather than a branch inside ``target_sg`` deliberately: a subgraph
declares its outputs by ``$ref`` into its own nodes, so one that sometimes skips
the node producing ``target_obb`` would have an output with nothing behind it.
Keeping the decision in a subgraph of its own means ``target_sg`` always runs
its OBB node and always has something to export.

``remaining`` and ``given_up`` are returned for the trace: on a variant with a
deliberately ungraspable tool, "finished with 2 remaining, both given up" and
"finished with 0 remaining" are very different runs and the exit status alone
cannot tell them apart.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    object_name: str
    region_name: str
    remaining: int


def run(ctx: NodeContext) -> Output:
    # The one caller that charges an attempt: this node is entered once per pass,
    # so it is the only place a retry count means passes rather than questions.
    goal = ctx.tool("sim.next_goal", count_attempt=True)
    return {
        "route": "finished" if goal["done"] else "found",
        "object_name": str(goal["object_name"]),
        "region_name": str(goal["region_name"]),
        "remaining": int(goal["remaining"]),
    }
