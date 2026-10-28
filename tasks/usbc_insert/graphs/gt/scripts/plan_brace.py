"""Ask the suite where the support arm takes the receptacle, and how hard.

The bimanual half of this task, and the reason it exists: the cradle the
bulkhead rests in has **no axial stop**. That is deliberate -- reacting the
insertion push is the support arm's job, and it is what makes this variant
bimanual rather than a plug going into scenery -- but it means an insertion run
without a brace does not fail by jamming. It fails by shoving a 90 g receptacle
along its own axis until the stroke runs out, which reads exactly like a stroke
that was too short.

``sim.plan_brace`` knows the barrel band off the same asserted constants the
mesh loader verifies, so the grasp cannot drift from the part. It returns the
same stage shape ``sim.plan_insert`` does, which is why ``move_stage.py`` drives
both without knowing which arm it is moving.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    stages: list
    arm_id: int
    arm_name: str
    grip_m: float


def run(ctx: NodeContext) -> Output:
    plan = ctx.tool("sim.plan_brace")
    return {
        "route": "planned",
        "stages": plan["stages"],
        "arm_id": int(plan["arm_id"]),
        # The NAME as well as the id: `sim.set_gripper` takes a name, and the
        # two YAM cells order their arms oppositely, so a literal would command
        # a different physical hand per robot.
        "arm_name": str(plan["arm_name"]),
        "grip_m": float(plan["grip"]["command_m"]),
    }
