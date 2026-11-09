"""How many stations this weave has, and what a grip is worth.

The count only. **The stations themselves are planned one at a time**, by
``run_weave.py``, because each grasp has to be solved against where the rod is
*now*: the first crossing moves the rod, so a plan made up front puts the second
grasp where the rod used to be and the latch closes on nothing. Measured, that
is ``hold_segment`` returning -1 at two stations out of three.

So this asks the suite for station 0 and keeps only the numbers that do not
change -- how many there are, and the two grip commands.
"""

import json
from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    stations: int
    grip_open_m: float
    grip_close_m: float


def run(ctx: NodeContext, scene: str) -> Output:
    seen = json.loads(scene or "{}")
    # Station 0 planned against the geometry the perception slot handed over,
    # and only the numbers that do not change kept. The count now comes from
    # what was *seen* rather than from the layout table, which is the whole
    # difference between this and the privileged version: a vision body that
    # finds two spools of three plans a two-station weave and says so, instead
    # of planning three and missing one silently.
    plan = ctx.tool(
        "cable.plan_station",
        index=0,
        stations=seen["stations"],
        sides=seen["sides"],
        seats_x=seen["seats_x"],
    )
    return {
        "route": "planned",
        "stations": int(plan["stations"]),
        "grip_open_m": float(plan["grips"]["open"]["command_m"]),
        "grip_close_m": float(plan["grips"]["close"]["command_m"]),
    }
