"""Take the cable's far end with the support hand, before any crossing.

**The one structural difference between this graph and the routine that solves
the task.** ``RoutineWeave`` opens with this move; the graph has not, and has
been weaving single-handed the whole time.

Why it matters is a property of the material, not of the plan. A cable held at
one end and pulled sideways *bends*; a free cable **translates**. Without an
anchor the first crossing hauls the whole cable toward one station and the next
has nothing left to grasp -- the routine records 26 of 42 segments piled at a
single station. Every later crossing then works on a cable that has already
moved out from under the model that was measured.

Two legs and a close: enter, take, pinch. The pinch is a contact grasp at the
same half-millimetre engagement the work hand uses -- no latch, because the hold
the score sees has to be the hold the fingers earn.

Failure is fatal here in a way a missed look was not. A weave without an anchor
is not a degraded weave, it is a different and much worse task, so this routes
``unanchored`` to abort rather than carrying on and reporting the result as a
policy failure.

**On a run that is already anchored, this node does nothing at all.** The
argument above is about the material, and the material is different when the
bench is holding it: the terminated tiers weld the rod's first body to the world
(``SpecCable.pin_start``), so a sideways pull bends the cable without any hand in
it. Those runs weave single-handed and the phase is skipped -- see the branch on
``start_fixed`` below. The weaves, whose rods are free at both ends, still open
with this move for exactly the reason they always did.
"""

import json
from typing import Any, TypedDict

from gap import NodeContext

#: Refuse a grasp the solver could not reach. The routine's own limit: past this
#: the chain is in a different basin and what streams is a whip with the cable
#: near the jaws.
PLAN_ERROR_LIMIT = 0.05

SETTLE_STEPS = 24


class Output(TypedDict):
    route: str
    segment: int
    worst_error_m: float
    q_support: str
    quat: str


def run(ctx: NodeContext, approach: float = 0.055, scene: str = "") -> Output:
    # THE CURVE THIS HAND IS TO GRASP, PERCEIVED. `cable.plan_support` now
    # refuses without one rather than quietly reading the simulator's own chain,
    # which is what it did for every "vision" run before this -- under a
    # `cable.` name that `--no-privileged` does not filter.
    #
    # FROM `scene`, NOT FROM `cable.curve_memory`. This is the first node after
    # `posts` and the memory is still empty here: only `perceive_rod` writes it,
    # and that is the `track` node, which runs after this one. Reading the
    # memory instead cost a whole calibration pass -- every episode printed "no
    # perceived centreline to take hold of" and aborted at `anchor_sg` before a
    # single crossing. `perceive_weave_vision` puts the fitted curve straight
    # into its `scene` output, which is the same string `plug_port` already
    # takes its stations from.
    seen = json.loads(scene or "{}")
    rod = seen.get("rod") or []
    if len(rod) < 2:
        print(f"[anchor] no perceived centreline to take hold of "
              f"({len(rod)} point(s) in the scene read)", flush=True)
        return {"route": "unanchored", "segment": -1, "worst_error_m": 0.0,
                "q_support": "null", "quat": "null"}
    plan: dict[str, Any] = ctx.tool("cable.plan_support", approach=float(approach), rod=rod)

    # A RUN THAT IS ALREADY HELD DOES NOT NEED A HAND TO HOLD IT.
    #
    # Everything this node exists for is in the docstring above, and all of it
    # is about a cable that is free at both ends. The terminated tiers are not:
    # the rod's first body is welded to the bench, so the sideways pull the
    # first crossing makes bends the cable exactly as the far-end grip was
    # meant to make it. Taking the far end anyway costs the episode a whole
    # arm -- it stands in the run for the rest of the weave, every carry has to
    # pay cable out underneath it, and the hand that finally has to plug the
    # free end into the port is the hand that has been gripping it.
    #
    # So the phase is skipped, and the graph goes on single-handed. `q_support`
    # comes back EMPTY rather than a null-shaped string, because that is what
    # `run_legs` tests to decide whether the carry is bimanual: "null" is
    # truthy and would send every crossing back through `cable.carry_with_feed`
    # looking for a support chain that was never solved.
    if plan.get("start_fixed"):
        print("[anchor] the run's start is welded to the bench; no hand needed to hold it "
              "-- weaving single-handed", flush=True)
        return {"route": "anchored", "segment": -1, "worst_error_m": 0.0,
                "q_support": "", "quat": ""}

    worst = float(plan.get("worst_grasp_error", 0.0))
    stages = plan.get("stages") or []
    arm_id = int(plan.get("arm_id", 1))
    arm_name = str(plan.get("arm_name", ""))
    print(f"[anchor] arm={arm_name} segment={plan.get('segment')} "
          f"worst_ik={worst * 1000:.1f}mm legs={[s.get('stage') for s in stages]}", flush=True)
    if not stages or worst > PLAN_ERROR_LIMIT:
        print(f"[anchor] REFUSED ({worst * 1000:.0f}mm)", flush=True)
        return {"route": "unanchored", "segment": -1, "worst_error_m": worst,
                "q_support": "null", "quat": "null"}

    for stage in stages:
        ctx.tool(
            "robot.execute_trajectory",
            trajectory=stage["trajectory"],
            arm_id=arm_id,
            max_steps_per_waypoint=15,
        )
    # Close on the far end, then dwell for the jaws' own travel: a finger
    # command is a target, not a state, and the first crossing must not start
    # while the pads are still moving.
    ctx.tool("cable.set_gripper", width=float(plan.get("close", 0.0045)), arm=arm_name)
    ctx.tool("cable.wait_steps", steps=SETTLE_STEPS)
    print(f"[anchor] held segment {plan.get('segment')}", flush=True)
    # The solved chain and the grasp orientation travel with the exit: every
    # later pay-out seeds its IK from them, and re-deriving them per crossing
    # would let the support chain drift into a different basin mid-weave.
    #
    # `json` is imported at module scope now, because this node reads the
    # perceived centreline before it plans. A function-local `import json` down
    # here made the name local to the whole of `run`, so the read at the top
    # raised `cannot access local variable 'json'` -- every episode aborting at
    # `anchor_sg.take_far_end` before a single crossing.
    return {
        "route": "anchored",
        "segment": int(plan.get("segment", -1)),
        "worst_error_m": worst,
        "q_support": json.dumps([float(v) for v in plan.get("q_end", [])]),
        "quat": json.dumps([float(v) for v in plan.get("quat", [])]),
    }
