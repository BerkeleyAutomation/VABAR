"""Is this goal an open compartment or a box with a hole in the lid?

The one branch the reference workflow has no need for, because it only ever
targeted a compartment. Two of these six variants do not: `syringe` and the
third goal of `sort_power` name a sealed sharps box, where releasing a tool from
above leaves it sitting on the lid.

The question is answered from the task rather than from geometry -- the export
records which regions belong to containers -- so this node is a lookup, and the
branch it drives is the difference between `descend_release.py` (the reference's
own, unchanged) and `post_upright.py`.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    region_name: str
    is_container: bool


def run(ctx: NodeContext, region_name: str = "") -> Output:
    if not region_name:
        region_name = str(ctx.tool("sim.next_goal")["region_name"])
    is_container = bool(ctx.tool("sim.get_container", region_name=region_name)["is_container"])
    return {
        "route": "container" if is_container else "compartment",
        "region_name": region_name,
        "is_container": is_container,
    }
