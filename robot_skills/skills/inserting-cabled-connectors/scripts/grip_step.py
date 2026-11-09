"""One gripper move: a derived partial width, then a real dwell.

``robot.close_gripper`` closes fully, and driving the pads to zero on a 10 mm
part ejects it -- every grip here is a calibrated partial command, which writes
targets and returns without advancing time. The dwell is what lets the fingers
actually travel and the part settle between stages: the reference gives its
pinch-close-hold sequence 1.8 s for exactly this reason.

``usbc.set_gripper`` rather than the ``sim.`` spelling ``graphs/gt`` uses: it
is the same callable under a second name -- see ``tools.py``'s registrar -- and
this graph reaches it by the unprivileged one because it is meant to run under
``--no-privileged``. Commanding a width is an instruction to the hand.

The dwell is the runtime's own ``robot.wait_steps``, with a stated timeout: its
derived budget is half a second per control step, and this scene renders five
cameras behind a hydroelastic narrow phase. Waiting on the clock names
nothing, which is why that tool is a ``robot.`` one and this suite ships no
copy of it.

``arm`` is empty for the work hand and a resolved name for the support hand,
never a literal: the two YAM cells order their arms oppositely, so ``"left"``
written here would grip with a different physical hand per robot.
"""

from typing import Any, TypedDict

from gap import NodeContext


class Output(TypedDict):
    width: str
    command_m: float
    waited: int


def run(ctx: NodeContext, width: Any = "open", arm: str = "", settle_steps: int = 36) -> Output:
    result = ctx.tool("usbc.set_gripper", width=width, arm=str(arm))
    waited = ctx.tool("robot.wait_steps", steps=int(settle_steps), timeout_s=60.0)
    return {"width": str(width), "command_m": float(result["command_m"]), "waited": int(waited["waited"])}
