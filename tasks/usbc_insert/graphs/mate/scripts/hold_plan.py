"""Take the receptacle, or decline to -- and route on which.

**The bimanual half, and it is conditional on the scene rather than on a
variant name.** The bulkhead's cradle has no axial stop. That is deliberate --
reacting the insertion push is the support arm's job, and it is what makes that
variant bimanual rather than a plug going into scenery -- but it means an
insertion run without a brace does not fail by jamming. It fails by shoving a
90 g receptacle along its own axis until the stroke runs out, which reads
exactly like a stroke that was too short. The block receptacle is bolted to the
bench and there is nothing to hold: a support arm sent to grip it would be
reaching into a fixture.

So the branch is `receptacle_held`, which the perception slot answers, and this
node is where a workflow that serves both variants stops needing to know which
one it is running.

`usbc.plan_hold` rather than `sim.plan_brace`, and that is the whole point of
the pair: the privileged one aims at the barrel position the assembly was
authored with, this one aims at the point that was handed to it. Everything
else about the approach -- the side grasp across the barrel, the standoff, the
two legs, the commanded gap -- is a fact about this hand and this part shape and
is the same either way.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    stages: list
    hold_arm_id: int
    hold_arm_name: str
    hold_grip_m: float


NOWHERE: Output = {
    "route": "fixed",
    "stages": [],
    "hold_arm_id": 0,
    "hold_arm_name": "",
    "hold_grip_m": 0.0,
}


def run(ctx: NodeContext, receptacle_held: bool = False, barrel_point: list | None = None) -> Output:
    if not bool(receptacle_held):
        print("[hold] the receptacle is scenery; no brace", flush=True)
        return dict(NOWHERE)  # type: ignore[return-value]
    if not barrel_point:
        # Held and nowhere to take it. Refused rather than defaulted: falling
        # back to the authored barrel here would let a perception body that
        # found nothing produce a brace that worked, which is the one failure
        # this whole pairing exists to make visible.
        raise RuntimeError("the receptacle is held but perception returned no barrel point to take it by")
    plan = ctx.tool("usbc.plan_hold", grasp_point=list(barrel_point))
    print(f"[hold] bracing with {plan['arm_name']!r} at {plan['grasp_point']}", flush=True)
    return {
        "route": "held",
        "stages": plan["stages"],
        "hold_arm_id": int(plan["arm_id"]),
        # The NAME as well as the id: the gripper tools take a name, and the two
        # YAM cells order their arms oppositely, so a literal would command a
        # different physical hand per robot.
        "hold_arm_name": str(plan["arm_name"]),
        "hold_grip_m": float(plan["grip"]["command_m"]),
    }
