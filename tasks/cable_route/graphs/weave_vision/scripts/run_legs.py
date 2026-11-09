"""Run a named subset of a crossing's legs, with the grip work at its boundary.

**One script, four phases.** ``take``, ``carry``, ``seat`` and ``release`` are
four nodes in the graph with four skills and four exits, and they share this
implementation because they are the same operation on different legs -- stream
these waypoints, and open or close the jaws at this boundary. Four near-identical
files would make the graph look more decomposed than it is; four nodes over one
script says exactly how much is shared.

**Grip before the leg it enables, never after.** The jaws open ahead of the leg
that lifts away, because opening after it means the leg lifts with the rod still
pinched. They close after the leg that arrives, because closing before it means
closing on the air the hand is still travelling through. Each boundary then
dwells for the jaws' own travel: a finger command is a target, not a state, and
stepping straight into the next leg leaves the pads mid-stroke.

**No latch.** The pinch carries the rod. ``RoutineWeave`` dropped
``sim.hold_cable`` once the hand actually reached the rod and measured the grasp
holding it -- 266 contacts at peak, 390 mm of rod moved. A kinematic hold here
would be doing the work the grasp is supposed to be judged on.
"""

import json
from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    legs: int
    ran: list


def run(
    ctx: NodeContext,
    plan: str = "",
    arm_id: int = 0,
    legs: str = "",
    open_before: bool = False,
    close_after: bool = False,
    grip_open_m: float = 0.0475,
    grip_close_m: float = 0.003,
    settle_steps: int = 24,
    q_support: str = "",
    quat: str = "",
    segment: int = -1,
) -> Output:
    stages = (json.loads(plan or "{}") or {}).get("stages") or []
    wanted = [s.strip() for s in str(legs).split(",") if s.strip()]
    mine = [s for s in stages if str(s.get("stage")) in wanted]
    print(f"[{wanted[0] if wanted else 'legs'}] want={wanted} have="
          f"{[str(s.get('stage')) for s in stages]}", flush=True)
    if not mine:
        # Not every layout runs every leg: a station whose rod already lies on
        # its required side tidies with a drag and never crosses, so the cross
        # legs are simply absent. That is a no-op, not a failure.
        return {"route": "done", "legs": 0, "ran": []}

    # The CARRY is the one phase that must move both arms at once: the support
    # hand pays cable out while the work hand crosses. Everything else is
    # single-armed and streams normally.
    if str(legs).startswith("move") and q_support:
        import json as _json  # noqa: PLC0415

        # The chain the LAST pay-out ended at, falling back to the anchor's on
        # the first crossing. See ``cable.support_memory``.
        carried = ctx.tool("cable.support_memory").get("q") or ""
        seed = _json.loads(carried) if carried.strip() else _json.loads(q_support or "null")
        got = ctx.tool(
            "cable.carry_with_feed",
            plan=_json.loads(plan or "{}"),
            legs=legs,
            q_support=seed,
            quat=_json.loads(quat or "null"),
            segment=int(segment),
        )
        print(f"[move] bimanual streamed={got.get('streamed')} fed={got.get('fed')}"
              f" feed_rows={got.get('feed_rows', 0)} lead_rows={got.get('feed_lead_rows', 0)} {got.get('reason', '')}", flush=True)
        if got.get("streamed"):
            return {"route": "done", "legs": len(mine), "ran": [str(s.get("stage")) for s in mine]}

    # THE JAWS ON THE ARM THE LEGS ARE FLYING. `cable.set_gripper` defaults to
    # the work arm, which was the same hand for every crossing until the station
    # planner started choosing one per station -- and a leg streamed to the left
    # arm with the right arm's jaws closing is not a weaker grasp, it is two
    # hands each doing half of one motion. The node already knows the arm as an
    # id and has no name to pass, so the tool takes the id.
    if open_before:
        ctx.tool("cable.set_gripper", width=float(grip_open_m), arm_id=int(arm_id))
        ctx.tool("cable.wait_steps", steps=int(settle_steps))
    for stage in mine:
        ctx.tool(
            "robot.execute_trajectory",
            trajectory=stage["trajectory"],
            arm_id=int(arm_id),
            max_steps_per_waypoint=15,
        )
    if close_after:
        ctx.tool("cable.set_gripper", width=float(grip_close_m), arm_id=int(arm_id))
        ctx.tool("cable.wait_steps", steps=int(settle_steps))
    ran = [str(s.get("stage")) for s in mine]
    print(f"[{wanted[0]}] ran={ran}"
          f"{' open' if open_before else ''}{' close' if close_after else ''}", flush=True)
    return {"route": "done", "legs": len(mine), "ran": ran}
