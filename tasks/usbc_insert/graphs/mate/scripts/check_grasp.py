"""Did the close actually take the plug?

The signal is the one a real linear gripper has and nothing more: the jaws were
commanded past the part's width, so fingers that stall short of their command
are stopped on something. Empty jaws reach their command exactly.

**``usbc.grip_state``, not ``sim.get_insertion_state``.** The privileged read
answers a stronger question -- are the jaws stalled *at the plug's own width* --
and it is strictly better, and only a scene can answer it. What this node has is
the weaker half, and the weakness is bounded in the direction that matters: it
cannot tell you *what* is between the pads, and it can tell you that an empty
hand is not about to fly a 3.4 s insertion stroke into a 0.15 mm fit and report
the seat that never happened.

The gap is also reported, because a stall at the wrong width is the failure this
node cannot route on and the trace should still show. On this rig the pads reach
their target within servo noise on air and stall about 2 mm short of it on a
10 mm plug.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    stalled: bool
    gap_mm: float
    lag_mm: float


def run(ctx: NodeContext) -> Output:
    state = ctx.tool("usbc.grip_state")
    stalled = bool(state["stalled"])
    print(
        f"[grasp] {'held' if stalled else 'MISSED'}: jaws at {float(state['gap_m']) * 1e3:.2f} mm, "
        f"trailing their command by {float(state['lag_m']) * 1e3:.2f} mm",
        flush=True,
    )
    return {
        "route": "held" if stalled else "missed",
        "stalled": stalled,
        "gap_mm": float(state["gap_m"]) * 1e3,
        "lag_mm": float(state["lag_m"]) * 1e3,
    }
