"""Take (or release) the cable: a ramped partial width, a latch, a dwell.

The close is 3 mm on an 11 mm cable -- ``robot.close_gripper``'s full close
squirts it out, and a step input on the finger servo slams the pads, so the
suite tool ramps. ``latch="hold"`` then pins the pinched segment to the hand
(the suite's kinematic hold -- a friction pinch bleeds under the drag both
variants apply), and a missed latch raises rather than dragging air. The
dwell lets the rod settle before the next stage pulls on it.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    command_m: float
    latch: str
    waited: int


def run(ctx: NodeContext, width_m: float, settle_steps: int = 30, latch: str = "") -> Output:
    # Negative width_m means "full open", and the tool resolves it against
    # whichever hand this pairing brought. The workflow passes -1 rather than a
    # number for every open: 0.0475 is the i2rt gripper's travel, and a graph
    # that states it is pinned to that one hand for no reason -- the close is a
    # fact about the cable (its diameter), the open never was.
    result = ctx.tool("sim.set_gripper", width=float(width_m))
    if latch == "hold":
        held = ctx.tool("sim.hold_cable")
        if not bool(held.get("held")):
            raise RuntimeError("latch missed: nothing within reach of the jaws")
    elif latch == "release":
        ctx.tool("sim.release_cable")
    waited = ctx.tool("sim.wait_steps", steps=int(settle_steps))
    return {"command_m": float(result["command_m"]), "latch": latch, "waited": int(waited["waited"])}
