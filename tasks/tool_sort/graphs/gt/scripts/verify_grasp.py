"""Did the hand actually take the tool, or did it close on air?

``robot.close_gripper`` returning means the fingers finished travelling. It does
not mean anything is between them, and the difference is not cosmetic: without
this node the graph routes to its transport branch regardless, flies an empty
gripper to the compartment, opens it, and reports ``exit_status: success`` on an
episode that scored zero. That failure is invisible in the trace -- every node
ran, nothing raised -- and it is what ``tool_sort/sort_new`` did on its first
end-to-end run, carrying nothing while its screwdriver never left the pile.

``sim.check_grasp`` is the contact test the evaluators use, not a height or a
finger-width heuristic: an impulse on each side of the hand pushing outwards.
So "held" here means the same thing it means when the task is scored.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    held: bool
    object_name: str


def run(ctx: NodeContext, object_name: str = "") -> Output:
    if not object_name:
        object_name = str(ctx.tool("sim.next_goal")["object_name"])
    held = bool(ctx.tool("sim.check_grasp", object_name=object_name)["held"])
    return {
        "route": "grasped" if held else "failed",
        "held": held,
        "object_name": object_name,
    }
