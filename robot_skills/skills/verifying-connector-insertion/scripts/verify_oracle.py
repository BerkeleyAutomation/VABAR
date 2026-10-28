"""Is the plug seated -- released, in the bore, past the gate?

The **oracle body** of the verify slot. Reads the same quantities the evaluator
gates on (``sim.get_insertion_state`` carries the *document's* own thresholds,
not the asset's defaults), so the graph and the score cannot disagree about what
"inserted" means. Reported in millimetres because that is the scale the task
lives at.

The tilt reading is here because reading a key the measurement did not carry
once killed a whole workflow: every episode ended right after the push and read
as a short stroke. It is worth the line for the opposite reason too -- depth and
lat both pass on a plug hanging out of the mouth, and tilt is the clause that
actually fails, so a verify that omitted it would say ``short`` while naming two
quantities that were fine.
"""

import math
from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    verify_body: str
    depth_mm: float
    gate_mm: float
    lat_mm: float
    mate_gap_mm: float
    hand_clear: bool


def run(ctx: NodeContext, settle_steps: int = 48) -> Output:
    # Let the mate ring down before reading it. The verify runs one instant
    # after the retract, and a freshly released pair is still oscillating on the
    # support brace: measured 5.81 deg of tilt at that instant on an episode
    # whose settled mate scored reward 1.00 -- the one honest seat this suite
    # ever aborted for being read too early.
    ctx.tool("robot.wait_steps", steps=int(settle_steps), timeout_s=60.0)
    state = ctx.tool("sim.get_insertion_state")
    clear = not bool(state["held"])
    inserted = bool(state["inserted"]) and clear
    print(
        f"[verify:oracle] depth {float(state['depth']) * 1e3:.2f} mm "
        f"(gate {float(state['depth_gate']) * 1e3:.2f})  "
        f"lat {float(state['lat']) * 1e3:.2f} mm  "
        f"tilt {math.degrees(float(state.get('tilt', 0.0))):.2f} deg  "
        f"hand clear={clear}",
        flush=True,
    )
    return {
        "route": "inserted" if inserted else "short",
        "verify_body": "oracle",
        "depth_mm": float(state["depth"]) * 1e3,
        "gate_mm": float(state["depth_gate"]) * 1e3,
        "lat_mm": float(state["lat"]) * 1e3,
        "mate_gap_mm": float(state["mate_gap"]) * 1e3,
        "hand_clear": clear,
    }
