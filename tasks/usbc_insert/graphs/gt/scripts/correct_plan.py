"""Measure the plug in the jaws, and re-plan the stroke around what it says.

The whole reason this graph seats where the open-loop script jams. Closing the
jaws shifts the plug of order a millimetre -- against a 0.15 mm-a-side fit --
and the script never re-measures. This node reads the plug's true pose (held,
after the lift), computes where it sits relative to the bore axis, and re-plans
the carry/insert/retract stages displaced by exactly that error:

* laterally (x, z): the plug must arrive *on* the bore axis, so the arm's
  stroke shifts by ``mated - plug`` in the plane normal to the axis;
* axially (y): a plug that sits shallower in the jaws than the nominal grasp
  station needs the stroke to run correspondingly further, or the seat ends
  short by the slip.

This is the kamino reference's ``regrasp-correct`` (its docstring credits it
with the difference between jamming and seating), restated as two tool calls.
"""

from typing import TypedDict

from gap import NodeContext

GRASP_DY_FALLBACK = -0.0199
"""Where the plug rides relative to the TCP at the nominal grasp station [m].

``sim.get_insertion_state`` reports this as ``grasp_offset`` now, off the suite's
own stage -- so the number here is only what an older connector without that key
falls back to. It used to be the only copy, restated because a node script has no
package to import from, and it would have gone silently wrong the day the station
or the hand moved."""


class Output(TypedDict):
    stages: list
    grips: dict
    correct_x_mm: float
    correct_y_mm: float
    correct_z_mm: float
    lat_before_mm: float


def run(ctx: NodeContext, axial: bool = True, retry_from: float = 0.0) -> Output:
    state = ctx.tool("sim.get_insertion_state")
    plug = state["plug_position"]
    mated = state["mated_position"]

    # Lateral: put the held plug on the bore axis.
    correct_x = float(mated[0] - plug[0])
    correct_z = float(mated[2] - plug[2])

    # Axial: the in-jaw slip. The nominal plan believes the plug rides
    # -GRASP_DY ahead of the TCP; however far it actually sits behind that is
    # how far short the stroke would end.
    correct_y = 0.0
    if axial:
        ee = ctx.tool("robot.get_ee_pose")
        position = (ee.get("pose") or ee)["position"]
        grasp_dy = float(state.get("grasp_offset", GRASP_DY_FALLBACK))
        expected_plug_y = float(position["y"]) - grasp_dy
        correct_y = expected_plug_y - float(plug[1])

    plan = ctx.tool(
        "sim.plan_insert",
        correct_x=correct_x,
        correct_y=correct_y,
        correct_z=correct_z,
        retry_from=float(retry_from),
    )
    return {
        "stages": plan["stages"],
        "grips": plan["grips"],
        "correct_x_mm": correct_x * 1e3,
        "correct_y_mm": correct_y * 1e3,
        "correct_z_mm": correct_z * 1e3,
        "lat_before_mm": float(state["lat"]) * 1e3,
    }
