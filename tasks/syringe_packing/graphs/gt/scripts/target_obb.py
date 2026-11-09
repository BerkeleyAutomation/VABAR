"""The tool's grasp box, from the simulator rather than from vision.

This is the **one node** that differs between this ground-truth graph and a
perception-driven one. The reference implementation of this workflow hardcoded
a literal OBB measured at a fixed seed, because no VLM credentials were
available; this asks the simulator instead, which is both live (it tracks a
re-sampled layout) and honest about what it is (a privileged call, available
only to the in-process executor).

A perception graph replaces this script with a ``perceive`` node producing a
point cloud and ``geometry.filter_and_compute_obb``. Everything downstream --
the grasp candidates, the align pose, the transport -- is unchanged, which is
the property that makes a ground-truth graph a useful reference for a real one
rather than a separate program.

``handle_frac`` is the benchmark's own heuristic and is not optional for every
tool: a claw hammer's bounding box is 13.2 cm across against an 8 cm gripper
span, so the fingers cannot close on it at the centroid at all. Sliding the
grasp along the long axis towards the light end is what the reference baseline
does, and what makes the difference between a grip that survives transport and
one that catches the claw by friction and drops it. ``sim.get_object_obb``
reports ``fits_gripper`` so the choice is made on the object rather than
assumed.

An empty ``object_name`` means **ask the task which goal is outstanding**
(``sim.next_goal``), which is what lets one workflow serve all six variants
instead of only the one whose object it happened to name. The literal stays
available because a graph aimed at a particular tool is a legitimate thing to
want -- but it is no longer the default, since naming ``hammer_0`` here is
exactly what made this graph unrunnable on the four variants that contain no
hammer.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox


class Output(TypedDict):
    target_obb: OrientedBoundingBox
    fits_gripper: bool
    grasp_width: float
    object_name: str
    region_name: str


def run(ctx: NodeContext, object_name: str = "", handle_frac: float = 0.6) -> Output:
    region_name = ""
    if not object_name:
        goal = ctx.tool("sim.next_goal")
        if goal["done"]:
            raise RuntimeError("every goal of this variant is already satisfied")
        object_name, region_name = goal["object_name"], goal["region_name"]

    result = ctx.tool("sim.get_object_obb", object_name=object_name, handle_frac=handle_frac)
    fits = bool(result["fits_gripper"])
    # Plan against the whole box when the gripper can close on it, and against
    # the graspable segment when it cannot.
    obb = result["obb"] if fits else result["grasp_obb"]
    return {
        "target_obb": obb,
        "fits_gripper": fits,
        "grasp_width": float(result["grasp_width"]),
        # Reported so a trace says which tool this run was about -- with the
        # object chosen at runtime, the workflow file no longer tells you.
        "object_name": object_name,
        "region_name": region_name,
    }
