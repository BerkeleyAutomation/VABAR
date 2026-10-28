"""Drive one planned stage, in joint space, along its whole path.

``robot.execute_trajectory`` rather than ``robot.move_to_joints`` because a
stage is a *path*, not a destination: a stepped target makes a position-servoed
arm lunge, and a lunge mid-insertion is how a 0.15 mm-a-side fit gets wedged.
The plan's waypoints are the full-pose solve densified at the cell's **control
rate** (``ToolsUsbc._stream_hz``), and the streaming path feeds them one per
control step, so streaming them runs each stage at its declared duration.
"""

from typing import TypedDict

from gap import NodeContext

SETTLE_TOLERANCE = 0.001
"""Joint-space L2 the end-of-stage settle has to reach before the next node [rad].

**Not the 0.01 default, which does not mean what it looks like.** The settle
exits on a joint-space norm, and what that norm is worth at the tool centre
depends on which joints carry it: measured on a free arm 0.01 rad is ~1.3 mm,
and on the loaded arm at the top of the retract it is ~13. So the stage reported
arrival while the hand stood a centimetre short of its own plan, which is a
silent failure in exactly the quantity the goal's release clause reads.
"""

SETTLE_WAYPOINT_BUDGET = 30
"""Sizes the settle at ``max(4 * this, 60)`` = 120 control steps.

Only the end-of-stage settle is sized by this on the streaming path -- it is not
a per-waypoint stand-still. 120 rather than the 60 the shipped 15 gave, because
a tighter tolerance needs room to reach: the two are one fix and neither works
alone. Measured across a 2x2 on ``tcp-distance`` against its 50 mm gate, where a
plan-perfect retract is ~64 mm:

    settle 60,  tol 0.01   (shipped)  50.95 - 53.85 mm
    settle 800, tol 0.01              52.31 - 52.47 mm   exits early on tolerance
    settle 60,  tol 0.001             50.91 - 53.80 mm   capped by the budget
    settle 800, tol 0.0005            64.43 - 64.45 mm

Fixing the plan's 2x densification (see ``ToolsUsbc._stream_hz``) is what makes
this affordable rather than what replaces it: paced correctly but left on the
shipped settle the retract came back 64.92 and then 51.80 -- it can now arrive,
but only this makes it arrive every time.
"""


class Output(TypedDict):
    stage: str
    moved: bool
    waypoints: int


def run(ctx: NodeContext, stages: list, stage: str, arm_id: int = 0) -> Output:
    target = next((s for s in stages if s["stage"] == stage), None)
    if target is None:
        raise RuntimeError(f"the plan has no stage {stage!r}; has {[s['stage'] for s in stages]}")
    ctx.tool(
        "robot.execute_trajectory",
        trajectory=target["trajectory"],
        arm_id=int(arm_id),
        max_steps_per_waypoint=SETTLE_WAYPOINT_BUDGET,
        tolerance=SETTLE_TOLERANCE,
    )
    return {"stage": stage, "moved": True, "waypoints": len(target["trajectory"]["waypoints"])}
