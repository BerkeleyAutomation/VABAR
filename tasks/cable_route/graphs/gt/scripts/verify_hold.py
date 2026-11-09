"""Did the close actually take the cable?

The chain's nearest segment to the tool-centre point: inside the jaw span
means the rod is between the pads; a hand span away means the close shut on
air. Routing to failure beats dragging nothing around a peg for forty stages.
"""

from typing import TypedDict

from gap import NodeContext

HOLD_RADIUS_M = 0.04
"""A segment centre within this of the TCP counts as in the jaws [m]."""


class Output(TypedDict):
    route: str
    nearest_mm: float


def run(ctx: NodeContext) -> Output:
    state = ctx.tool("sim.get_cable_state")
    nearest = float(state["nearest_to_tcp"])
    return {
        "route": "held" if nearest < HOLD_RADIUS_M else "missed",
        "nearest_mm": nearest * 1e3,
    }
