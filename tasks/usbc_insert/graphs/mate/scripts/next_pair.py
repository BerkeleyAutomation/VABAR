"""The mating loop's gate: is there another connector to mate, and which one?

Entered at the top of every pass. Its ``route`` decides whether the graph goes
round again or goes to ``done`` -- the shape ``cable_route``'s ``next_station``
uses, and for its reason: a two-connector scene is a *conjunction* of mates, so
a workflow that grounds, aims, strokes and verifies once solves a one-port
variant and leaves a two-port one at half.

A gate of its own rather than a branch inside the grounding subgraph, because a
subgraph declares its outputs by ``$ref`` into its own nodes: one that sometimes
skipped the node producing a reading would export a name with nothing behind it.

``remaining`` and ``done`` are carried for the trace, and ``done`` is load
bearing beyond that -- the plug of a completed pass is still in the scene, in a
port, looking exactly like the one this pass wants.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    index: int
    remaining: int
    receptacle: str


def run(ctx: NodeContext) -> Output:
    gate = ctx.tool("usbc.next_pair")
    print(f"[pair] {gate['route']} index={gate['index']} remaining={gate['remaining']}"
          f" of {gate['total']} -> {gate['receptacle'] or '(none)'}", flush=True)
    return {
        "route": str(gate["route"]),
        "index": int(gate["index"]),
        "remaining": int(gate["remaining"]),
        "receptacle": str(gate["receptacle"]),
    }
