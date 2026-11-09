"""Drive one phase of the plan, in joint space, along its whole path.

``robot.execute_trajectory`` rather than ``robot.move_to_joints`` because a
phase is a *path*, not a destination: a stepped target makes a position-servoed
arm lunge, and a lunge with a 300 g spanner hanging out of the jaws throws it.
The plan's waypoints are densified at the cell's control rate and the streaming
path feeds them one per control step, so a phase runs at its declared duration.

A **phase**, not a named stage, and that is what lets one workflow drive three
variants. Every variant reaches, carries, places and retreats; what is *inside*
a place is three legs onto a hook for a spanner and three quite different ones
into a rack for a screwdriver. The graph can name the phase; it could only name
one of the two stage lists. It is also what keeps the episode inside its step
budget -- the phase arrives as one trajectory, so it pays one end-of-move settle
rather than one per leg.

The path is fetched from the planner rather than carried in from the node that
planned it. A dense phase is a few hundred joint vectors; threading that through
subgraph inputs makes the plan a value the *graph* has to route correctly
instead of a thing the planner owns -- and the planner is where a re-plan
overwrites it, which is the whole point of re-planning.
"""

from typing import TypedDict

from gap import NodeContext

SETTLE_TOLERANCE = 0.004
"""Joint-space L2 the end-of-phase settle has to reach [rad].

Not the 0.01 default, which on a loaded arm is worth about a centimetre at the
tool centre -- a phase would report arrival with the ring still short of the
hook tip it was about to be pushed over. Not the USB-C graph's 0.001 either:
that suite pays a 120-step settle for a 0.15 mm-class fit, and a hook mouth is
millimetres wide rather than tenths. 0.004 exits in a handful of steps on a path
that already ended where it was going, which is what a dense phase does.
"""

SETTLE_WAYPOINT_BUDGET = 15
"""Sizes the end-of-phase settle at ``max(4 * this, 60)`` = 60 control steps.

Only the settle is sized by this on the streaming path -- it is not a
per-waypoint stand-still. It is a ceiling and not a cost when the phase arrives;
it is also the floor the runtime imposes, which is why the plan hands over one
trajectory per phase rather than one per leg.
"""


class Output(TypedDict):
    phase: str
    stages: list
    waypoints: int


def run(ctx: NodeContext, phase: str) -> Output:
    plan = ctx.tool("sim.get_phase", phase=str(phase))
    ctx.tool(
        "robot.execute_trajectory",
        trajectory=plan["trajectory"],
        arm_id=int(plan["arm_id"]),
        max_steps_per_waypoint=SETTLE_WAYPOINT_BUDGET,
        tolerance=SETTLE_TOLERANCE,
    )
    return {
        "phase": str(phase),
        "stages": list(plan["stages"]),
        "waypoints": len(plan["trajectory"]["waypoints"]),
    }
