"""One gripper move: a derived partial width, then a real dwell.

``robot.close_gripper`` closes fully, and driving the pads to zero on a 10 mm
part ejects it -- every grip here is a calibrated partial command
(``sim.set_gripper``), which writes targets and returns without advancing
time. The dwell (``sim.wait_steps``) is what lets the fingers actually travel
and the part settle between stages: the reference gives its pinch-close-hold
sequence 1.8 s for exactly this reason.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    preset: str
    command_m: float
    waited: int


def run(ctx: NodeContext, preset: str, settle_steps: int = 36) -> Output:
    result = ctx.tool("sim.set_gripper", width=preset)
    waited = ctx.tool("sim.wait_steps", steps=int(settle_steps))
    return {"preset": preset, "command_m": float(result["command_m"]), "waited": int(waited["waited"])}
