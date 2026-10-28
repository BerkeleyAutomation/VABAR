"""Did the close actually take the plug?

The signal is the one a real linear gripper has: the jaws were commanded 2 mm
past the part's width, so a measured gap *inside* that width means they are
stalled on the part. Empty jaws reach their command exactly. Routing to failure
on a miss beats carrying an empty gripper through a 3.4 s insertion stroke and
reporting the seat that never happened.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    held: bool
    jaw_gap_mm: float


def run(ctx: NodeContext) -> Output:
    state = ctx.tool("sim.get_insertion_state")
    held = bool(state["held"])
    return {
        "route": "held" if held else "missed",
        "held": held,
        "jaw_gap_mm": float(state["jaw_gap"]) * 1e3,
    }
