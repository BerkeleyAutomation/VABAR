"""Where the stations, the seats and the rod are, read from the simulator.

The privileged body of the perception slot. It is one tool call and a shape
check: ``sim.perceive_weave`` already answers in the layout the slot declares,
and what this node adds is the refusal -- a body that returned a rod of one
point or a side per nothing would plan a weave against garbage several nodes
later, where it reads as a motion failure.

The vision body beside this one declares the same keys. Nothing downstream
knows which ran.
"""

import json
from typing import Any, TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    scene: str
    stations: int
    body: str


def run(ctx: NodeContext) -> Output:
    seen: dict[str, Any] = ctx.tool("sim.perceive_weave")
    stations = [[float(v) for v in st] for st in seen["stations"]]
    sides = [int(v) for v in seen["sides"]]
    seats_x = [float(v) for v in seen["seats_x"]]
    # The same three gates the vision body will need, stated once here so the
    # two bodies fail the same way rather than each inventing a tolerance.
    if not stations:
        return {"route": "not_found", "scene": "{}", "stations": 0, "body": "oracle"}
    if not (len(sides) == len(seats_x) == len(stations)):
        raise ValueError(
            f"perception answered {len(stations)} stations, {len(sides)} sides, {len(seats_x)} seats"
        )
    # ONE JSON STRING, and not because a dict would be nicer. gap's graph type
    # registry has no list type -- a subgraph input is one of its named
    # messages or a scalar -- so a scene whose length is a property of the
    # variant (three stations here, four next door) has no typed spelling. The
    # alternative idioms are worse in ways that matter: a fixed set of Vec3
    # inputs pins the graph to one variant, and publishing into a suite-side
    # store hides the dataflow the graph exists to make visible. A string keeps
    # the edge on the diagram and the parse in one place.
    scene = {"stations": stations, "sides": sides, "seats_x": seats_x}
    return {
        "route": "found",
        "scene": json.dumps(scene),
        "stations": len(stations),
        "body": str(seen.get("body", "oracle")),
    }
