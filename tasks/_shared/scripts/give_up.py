"""Abandon the goal that just failed, so the rest of the variant still runs.

The reference workflow's failure branch is ``abort``: open the hand, go home,
end the episode. That is right for a single-goal problem, where there is nothing
else to do. On an eight-goal problem it means one tool the gripper cannot close
on costs the other seven -- and three of these variants ship such a tool on
purpose (the utility knife at 10.8 mm, the tape measure at 161 mm across, the
adjustable wrench longer than a compartment's diagonal). A baseline that stops
at the first of them is not scoring the task, it is scoring the ordering.

So this marks the goal given up for the rest of the episode (``sim.skip_goal``
holds that set connector-side, since a subgraph re-entered by a loop edge gets a
fresh scope) and hands control back to the gate, which picks the next one.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    object_name: str
    given_up: list


def run(ctx: NodeContext, object_name: str = "") -> Output:
    result = ctx.tool("sim.skip_goal", object_name=object_name)
    return {
        "object_name": str(result["object_name"]),
        "given_up": list(result["given_up"]),
    }
