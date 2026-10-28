"""Did the weave hold once the hand was off it?

Reads the **evaluator**, not a measure of this script's own. The document scores
a seat per station under a ``worst_of`` against release-and-rest, and a graph
that re-derived "seated" here would be free to disagree with the thing that
decides the episode -- which is the failure the tool_sort port found the
expensive way, where a registry looked for the previous evaluator's metric key,
found nothing, and reported a policy that could not sort one hammer.

Nothing latches in this task, so this is a reading of where the rod *ended up*,
not of where it once was. A weave that sprang apart on release scores where it
sprang to, and this reports that.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    exit: str
    success: bool
    reward: float


def run(ctx: NodeContext) -> Output:
    state = ctx.tool("sim.check_success")
    success = bool(state["success"])
    return {
        "exit": "woven" if success else "short",
        "success": success,
        "reward": float(state.get("reward", 0.0)),
    }
