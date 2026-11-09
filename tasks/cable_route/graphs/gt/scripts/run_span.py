"""Stream a contiguous span of planned stages, named by boundaries.

One executor for both variants -- pickplace's stage list and wrap's differ in
length and names, but both are "everything up to the close", "everything
between the close and the release", "everything after". Naming the boundaries
rather than the stages is what lets one workflow serve a 11-stage placement
and a 40-stage wrap.

Each stage streams through ``robot.execute_trajectory``: the paths are dense
full-pose solves at 30 Hz, and a per-waypoint converge-and-stop would turn the
wrap's ring into a polygon of lunges.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    ran: list
    waypoints: int


def run(
    ctx: NodeContext,
    stages: list,
    arm_id: int = 0,
    start: str = "",
    start_after: str = "",
    stop_before: str = "",
    merge: bool = False,
) -> Output:
    names = [s["stage"] for s in stages]
    lo = 0
    if start:
        lo = names.index(start)
    elif start_after:
        lo = names.index(start_after) + 1
    hi = names.index(stop_before) if stop_before else len(stages)
    if lo >= hi:
        raise RuntimeError(f"empty span [{lo}:{hi}] over {names}")

    span = stages[lo:hi]
    if merge:
        # One continuous stream with one settle at the end. The wrap's ring is
        # 32 stages; per-stage settles turn its orbit into stop-and-go, which
        # bleeds winding into slack and burns most of the episode budget
        # standing still.
        waypoints = [w for stage in span for w in stage["trajectory"]["waypoints"]]
        ctx.tool(
            "robot.execute_trajectory",
            trajectory={"waypoints": waypoints},
            arm_id=int(arm_id),
            max_steps_per_waypoint=15,
        )
        return {"ran": [s["stage"] for s in span], "waypoints": len(waypoints)}

    ran, total = [], 0
    for stage in span:
        ctx.tool(
            "robot.execute_trajectory",
            trajectory=stage["trajectory"],
            arm_id=int(arm_id),
            max_steps_per_waypoint=15,
        )
        ran.append(stage["stage"])
        total += len(stage["trajectory"]["waypoints"])
    return {"ran": ran, "waypoints": total}
