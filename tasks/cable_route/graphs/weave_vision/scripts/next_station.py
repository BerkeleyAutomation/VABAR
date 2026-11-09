"""The loop gate: is there another crossing to make, and which one?

Entered at the top of every pass. Its ``route`` decides whether the graph goes
round again or goes to the verify — the shape ``tool_sort``'s ``next_goal`` uses,
and for its reason: a weave is a *conjunction* of crossings, so a workflow that
plans, takes and seats once solves a one-station variant and leaves a
three-station one at a third.

A gate of its own rather than a branch inside the planner, because a subgraph
declares its outputs by ``$ref`` into its own nodes: one that sometimes skipped
the node producing a plan would export a name with nothing behind it.

``remaining`` is carried for the trace. "Finished with 0 remaining" and
"finished after the plan refused twice" are very different runs and an exit
status cannot tell them apart.
"""

import json
from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    index: int
    remaining: int


def run(ctx: NodeContext, scene: str = "") -> Output:
    # The station count is PERCEPTION's, not the layout's: a body that found two
    # spools of three should weave two crossings and say so.
    seen = json.loads(scene or "{}")
    total = len(seen.get("stations") or [])
    gate = ctx.tool("cable.next_station", total=int(total))
    print(f"[station] {gate['route']} index={gate['index']} remaining={gate['remaining']}"
          f" of {total}", flush=True)
    return {
        "route": str(gate["route"]),
        "index": int(gate["index"]),
        "remaining": int(gate["remaining"]),
    }
