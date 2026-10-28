"""Which tool is still outstanding -- the cursor the whole workflow loops on.

These documents hang two tools, and the goal states them as two clauses in its
own order. Asking the suite which one is still unsatisfied, rather than
unrolling both into the graph, is what lets one workflow serve three variants
whose tools, holders and stage shapes all differ -- and what makes the loop a
loop: the back edge from ``release`` re-enters here, and the answer has changed
because a tool got hung in between.

``count_attempt`` is passed only here. Every other node asks the same question
to learn *which* tool it is working on, and charging those as attempts would
abandon each job on its first pass. Two passes then a give-up is what stops a
tool that slides off its hook from costing the other one its turn.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    object_name: str
    holder_name: str
    arm_id: int
    remaining: int
    attempt: int


def run(ctx: NodeContext, count_attempt: bool = True) -> Output:
    job = ctx.tool("sim.next_job", count_attempt=bool(count_attempt))
    return {
        "route": str(job["route"]),
        "object_name": str(job["object_name"]),
        "holder_name": str(job["holder_name"]),
        "arm_id": int(job["arm_id"]),
        "remaining": int(job["remaining"]),
        "attempt": int(job["attempt"]),
    }
