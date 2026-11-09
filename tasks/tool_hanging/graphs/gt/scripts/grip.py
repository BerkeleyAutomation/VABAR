"""One gripper move on the hand that is working, then a real dwell.

``robot.set_grip`` rather than ``robot.open_gripper``/``close_gripper``, for two
reasons. It takes an ``arm_id``, and these variants alternate hands -- the second
tool is picked up by the arm the first one was not, so a verb that always drives
arm 0 would close the idle hand on nothing while the working one stayed open.
And it ramps: a step input on a stiff finger servo slams the pads, which is how
a thin spanner shaft gets flicked out from between them rather than held.

The presets are the *hand's*, derived from its own measured span rather than
tabulated here -- ``open`` is its span, ``close`` its own minimum. A node that
names a width in millimetres is a node that works on one gripper.

The dwell is what lets the fingers actually travel and the tool settle in them
before the next stage starts moving: ``set_grip`` returns when it has finished
writing the ramp, not when the pads have arrived.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    preset: str
    arm_id: int
    width_mm: float
    waited: int


def run(ctx: NodeContext, preset: str, arm_id: int = -1, ramp_steps: int = 30, settle_steps: int = 20) -> Output:
    # Whichever hand the plan resolved for the job in progress, never a literal:
    # the two YAM cells order their arms oppositely, and these variants alternate
    # hands between the two jobs anyway. Asked of the *state* rather than of the
    # cursor -- by the time the release opens the jaws, the tool in them has
    # already satisfied its clause, and the cursor would answer with the next one.
    arm = int(arm_id) if arm_id >= 0 else int(ctx.tool("sim.get_hang_state")["arm_id"])
    result = ctx.tool(
        "robot.set_grip",
        preset=str(preset),
        arm_id=arm,
        ramp_steps=int(ramp_steps),
        settle_steps=int(settle_steps),
    )
    return {
        "preset": str(preset),
        "arm_id": arm,
        "width_mm": float(result["command_width_m"]) * 1e3,
        "waited": int(result.get("waited") or 0),
    }
