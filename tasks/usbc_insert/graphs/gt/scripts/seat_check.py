"""After a push: seated, worth one more try, or gone?

Three outcomes, and routing between them is what makes the graph a policy
rather than a script:

* **seated** -- past the evaluator's gate: release and verify.
* **again** -- still held but short. A wedge is often not repeatable: back off
  a few centimetres, re-measure, re-correct, push again. The workflow unrolls
  exactly one retry; a second never paid on the tool-sort suite and there is
  no reason to expect connectors to differ.
* **lost** -- not held and not seated: the part was squeezed out or dropped,
  and the nominal stations know nothing about where it landed. Route to
  failure rather than pantomiming the rest of the script over an empty
  gripper.
"""

import math
from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    depth_mm: float
    held: bool
    lat_mm: float
    tilt_deg: float


def run(ctx: NodeContext, allow_retry: bool = True) -> Output:
    state = ctx.tool("sim.get_insertion_state")
    seated = bool(state["inserted"])
    held = bool(state["held"])
    # **On the axis, not clamped in the pads.** This routed on `held` -- the
    # measured-stall grasp signal -- which made the retry branch UNREACHABLE:
    # `ease` deliberately opens the jaws to 0.6 mm of clearance before the push,
    # so the fingers reach their command and `held` is False by the time this
    # node runs, in every trace. A plug that jammed half way therefore fell to
    # `lost` and aborted the episode, and the one recovery the workflow ships --
    # the very thing the README credits with turning partial seats into seats --
    # never ran once.
    #
    # What actually distinguishes "worth another push" from "gone" is whether
    # the plug is still on the bore axis. Measured on a tight fit that jams at
    # 6.6 mm: lat 0.335 mm, i.e. dead on axis and simply short. A genuinely lost
    # plug reads tens of millimetres off (11.1 mm when the fit refused it
    # outright), which the lateral guard separates cleanly.
    on_axis = float(state["lat"]) <= float(state["lateral_guard"])
    if seated:
        route = "seated"
    elif on_axis and allow_retry:
        route = "again"
    elif on_axis:
        # Retries exhausted: release what we have and let the verify score it.
        route = "seated"
    else:
        route = "lost"
    return {
        "route": route,
        "depth_mm": float(state["depth"]) * 1e3,
        "held": held,
        "lat_mm": float(state["lat"]) * 1e3,
        # Reported because this node routes on it: `inserted` folds tilt in,
        # so a push that reads seated-then-not is otherwise unattributable.
        "tilt_deg": math.degrees(float(state.get("tilt", 0.0))),
    }
