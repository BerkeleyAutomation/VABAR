"""The goal compartment as an oriented bounding box.

The benchmark's goal is a BDDL *region* -- an xy rectangle in a bin's frame plus
a height window -- not a physical container, so there is nothing to perceive
here even in principle. Asking the simulator for the region's box is therefore
not a shortcut around perception; it is reading the task definition, the same
way a human demonstrator is told which compartment to aim for.

The box comes back with an identity rotation (regions are axis-aligned), which
lets the shared ``compute_drop_pose`` script serve a region and a real container
without branching.

An empty ``region_name`` means ask ``sim.next_goal`` for the region belonging to
the outstanding goal, the same way :file:`target_obb.py` picks its object. The
two nodes query independently rather than the second reading the first's answer:
nothing between them moves an object, so they cannot disagree, and keeping them
independent means neither subgraph has to declare a cross-subgraph input for a
string.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox


class Output(TypedDict):
    container_obb: OrientedBoundingBox
    region_name: str


def run(ctx: NodeContext, region_name: str = "") -> Output:
    if not region_name:
        goal = ctx.tool("sim.next_goal")
        if goal["done"]:
            raise RuntimeError("every goal of this variant is already satisfied")
        region_name = goal["region_name"]

    result = ctx.tool("sim.get_region_obb", region_name=region_name)
    return {"container_obb": result["obb"], "region_name": region_name}
