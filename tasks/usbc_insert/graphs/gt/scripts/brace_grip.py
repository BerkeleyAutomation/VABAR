"""Close the support arm on the barrel, at a width the plan derived.

``grip_step.py`` next door commands a *named preset* on the working hand -- the
presets are the plug's, derived from its 10 mm width. This one takes metres,
because the width is the barrel's and the plan is what measured it.

``arm`` is the name the plan resolved, never a literal: the two YAM cells order
their arms oppositely, so ``"left"`` written here would brace with a different
physical hand depending on which robot the task is standing on.

A real dwell for the same reason its sibling has one: ``sim.set_gripper`` writes
targets and returns without advancing time, so without a wait the next node
starts moving before the pads have travelled. On this grip that matters more
than on the plug's -- the arm is about to be asked to hold 90 g against an 80 mm
push, and a brace that has not closed yet is a brace that is not there.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    command_m: float
    waited: int


def run(ctx: NodeContext, width: float, arm: str, settle_steps: int = 48) -> Output:
    result = ctx.tool("sim.set_gripper", width=float(width), arm=arm)
    waited = ctx.tool("sim.wait_steps", steps=int(settle_steps))
    return {"command_m": float(result["command_m"]), "waited": int(waited["waited"])}
