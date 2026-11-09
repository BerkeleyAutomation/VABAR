"""Abandon one tool and get the hand out of the way.

Two tools, one budget. A tool the arm cannot grasp -- or cannot reach a plan for
-- costs the *other* tool its turn if the cursor keeps returning it, because the
episode truncates before the second one is ever tried. Giving up marks it for
the rest of the episode so the next pass through the cursor moves on.

The hand opens and homes on the way out. Opening because whatever went wrong may
have left the pads part-closed on something, and homing because a hand parked
over the table is a hand the *other* arm's plan has to route around -- and
nothing here routes around anything.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    object_name: str
    given_up: list


def run(ctx: NodeContext, arm_id: int = -1) -> Output:
    state = ctx.tool("sim.get_hang_state")
    arm = int(arm_id) if arm_id >= 0 else int(state["arm_id"])
    ctx.tool("robot.set_grip", preset="open", arm_id=arm, settle_steps=10)
    ctx.tool("robot.go_home", arm_id=arm)
    dropped = ctx.tool("sim.give_up", object_name=str(state["object_name"]))
    return {
        "route": "skipped",
        "object_name": str(dropped["object_name"]),
        "given_up": list(dropped["given_up"]),
    }
