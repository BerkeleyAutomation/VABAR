"""Re-plan the stroke around what the re-look measured.

**The node that makes this a policy rather than a script**, and it is the same
correction ``graphs/gt`` credits with the difference between jamming and
seating -- earned from whichever body of the perception slot ran.

Two things move between the first plan and this one, and both are of the order
the fit cannot absorb:

* **The plug shifts in the jaws.** Closing a parallel gripper on a 10 mm part
  moves it by about a millimetre, against a 0.15 mm-a-side clearance. Passing
  the plug's *measured* position re-bases the whole schedule on where the part
  actually is, which is what the reference's open-loop stroke never does -- it
  seats 3.6 mm of 13 for exactly this reason.
* **The receptacle shifts under the brace.** The support jaws close 3 mm a side
  onto a free 90 g barrel whose cradle has no axial stop.

``usbc.plan_mate`` and not ``sim.plan_insert``: the privileged twin fills the
frame in from the authored assembly, which would silently discard everything
this node was given and produce a plan that worked for the wrong reason.

The ground yaw describes the original grasp. When RGB-D resolves the held
plug's top face, its measured frame supplies a small wrist correction before
entry. The TCP translation accounts for rotating the shoulder about the grip.
An axial reseat preserves the wrist's actual attitude inside the bore.

**``receptacle_held`` picks the overshoot.** A push is commanded past the seat
because a position servo settles short of its command -- but how far past
depends on what it is over-travelling into. A bolted fixture takes it; a barrel
held in a friction pinch is walked across the bench by it. The planner owns both
numbers; this only says which case it is, and the perception slot already
answered that.

**``axial_only`` is what keeps the second pass from levering the mate.** With
the tab already in the bore the plug rests on the floor of a slot twice its own
height, so a re-plan measures it 1.5 mm under the bore axis and asks the hand to
lift it there -- through jaws that have just closed firm. Set, the stroke keeps
only its axial component, which is the one direction still free once the socket
is holding the part.

The axial reseat commands at most 3 mm forward and preserves the current
wrist attitude. It never withdraws a tab that may already be seated: the
shoulder can be occluded at this station, so its distance estimate is not a
reliable reason to pull the connector back out.
"""

from typing import TypedDict
import sys
from pathlib import Path

import numpy as np

from gap import NodeContext
from robosimstudio.core.math import mat_to_quat, quat_to_mat

# Sibling helpers resolve inside the GaP skill package.
from .held_frame import alignment_delta


class Output(TypedDict):
    route: str
    stages: list
    arm_id: int
    insert_dist_mm: float
    lift_mm: float


def run(
    ctx: NodeContext,
    plug_point: list,
    mouth_point: list,
    bore_axis: list,
    yaw_deg: float = 180.0,
    label: str = "stroke",
    with_inspect: bool = False,
    anchored: bool = False,
    retry_from: float = 0.0,
    axial_only: bool = False,
    receptacle_held: bool = False,
    plug_axis: list | None = None,
    plug_up: list | None = None,
) -> Output:
    if not plug_point or not mouth_point or not bore_axis:
        # Refused rather than defaulted. `usbc.plan_mate` falls back to the
        # authored assembly for any part of the frame left unstated, which is a
        # convenience for its privileged twin and would be a lie here: a
        # perception body that measured nothing would produce a working plan.
        raise RuntimeError(
            "the stroke needs a measured frame; perception returned "
            f"plug={plug_point!r} mouth={mouth_point!r} axis={bore_axis!r}"
        )
    # THE TOOL CENTRE, when the hand is already holding something. The plan is
    # then a displacement from where the hand stands rather than a schedule
    # rebuilt around a grasp station that no longer exists -- see
    # `ToolsUsbc._mate` for the 17.5 mm that assumption was worth on a drawn
    # layout with the plug five millimetres off the mate line.
    tcp = None
    correction = {}
    if anchored:
        ee = ctx.tool("robot.get_ee_pose")
        pos = (ee.get("pose") or ee)["position"]
        tcp = [float(pos["x"]), float(pos["y"]), float(pos["z"])]
        rotation = (ee.get("pose") or ee)["rotation"]
        current_quat = np.array([rotation[key] for key in ('x', 'y', 'z', 'w')])
        if axial_only:
            # Keep the actual wrist attitude during a push inside the bore.
            correction['mate_quat'] = current_quat.tolist()
        elif plug_axis and plug_up:
            delta = alignment_delta(plug_axis, plug_up, bore_axis)
            correction['mate_quat'] = mat_to_quat(delta @ quat_to_mat(current_quat)).tolist()
            offset = np.asarray(plug_point) - np.asarray(tcp)
            shift = offset - delta @ offset
            correction.update(correct_x=float(shift[0]), correct_y=float(shift[1]), correct_z=float(shift[2]))
            angle = np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1)))
            print(f'[{label}] correcting measured in-jaw orientation by {angle:.2f} deg', flush=True)
    plan = ctx.tool(
        "usbc.plan_mate",
        plug_point=list(plug_point),
        tcp_point=tcp,
        mouth_point=list(mouth_point),
        bore_axis=list(bore_axis),
        yaw_deg=float(yaw_deg),
        retry_from=float(retry_from),
        axial_only=bool(axial_only),
        receptacle_held=bool(receptacle_held),
        # Only the plan that DRIVES the empty-hand look asks for it; see
        # `ToolsUsbc._mate` for why an unplayed stage is not free here.
        with_inspect=bool(with_inspect),
        **correction,
    )
    print(
        f"[{label}] {plan['insert_dist'] * 1e3:.1f} mm along {[round(v, 3) for v in plan['axis']]}, "
        f"rising {plan['lift_m'] * 1e3:.1f} mm, jaws at {plan['yaw_deg']:.1f} deg",
        flush=True,
    )
    return {
        "route": "planned",
        "stages": plan["stages"],
        "arm_id": int(plan["arm_id"]),
        "insert_dist_mm": float(plan["insert_dist"]) * 1e3,
        "lift_mm": float(plan["lift_m"]) * 1e3,
    }
