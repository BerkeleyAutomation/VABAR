"""Is the plug actually seated -- released, in the bore, past the gate?

Reads the same quantities the evaluator gates on (``sim.get_insertion_state``
carries the evaluator's own gate values), so the graph and the score cannot
disagree about what "inserted" means. Reported in millimetres because that is
the scale the task lives at.
"""

import math
from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    depth_mm: float
    gate_mm: float
    lat_mm: float
    mate_gap_mm: float
    tilt_deg: float
    tilt_gate_deg: float
    held: bool


def run(ctx: NodeContext, settle_steps: int = 48) -> Output:
    # Let the mate ring down before reading it. The verify runs one instant
    # after the retract, and a freshly released pair is still oscillating on
    # the support brace: measured 5.81 deg of tilt at that instant on an
    # episode whose settled mate scored 5.0-gated reward 1.00 -- the one
    # honest seat this graph ever aborted for being read too early.
    ctx.tool("sim.wait_steps", steps=int(settle_steps))
    state = ctx.tool("sim.get_insertion_state")
    inserted = bool(state["inserted"]) and not bool(state["held"])
    return {
        "route": "inserted" if inserted else "short",
        "depth_mm": float(state["depth"]) * 1e3,
        "gate_mm": float(state["depth_gate"]) * 1e3,
        "lat_mm": float(state["lat"]) * 1e3,
        "mate_gap_mm": float(state["mate_gap"]) * 1e3,
        # Tilt, now that the insertion atom returns it. Reading a key the
        # measurement did not carry once killed the whole workflow here -- every
        # episode ended `policy_request` right after the push, which read as a
        # short stroke -- so this reports it only because the atom declares it.
        # It is worth the line: depth and lat both pass on a plug hanging out of
        # the mouth, and tilt is the clause that actually fails, so a verify that
        # omitted it said `short` while naming two quantities that were fine.
        "tilt_deg": math.degrees(float(state.get("tilt", 0.0))),
        "tilt_gate_deg": math.degrees(float(state.get("tilt_gate", 0.0))),
        "held": bool(state["held"]),
    }
