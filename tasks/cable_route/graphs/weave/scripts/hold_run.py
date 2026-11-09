"""Pin the run with the support hand, just past the crossing being worked.

**THIS NODE IS NOT WIRED IN.** ``workflow.json`` has no ``hold`` node and never
has; the file is written and unused. It is left as it is rather than quietly
connected, because adding a second support-hand grip mid-weave is a behaviour
change that wants its own before-and-after -- and because it carries the same
pass-versus-station indexing defect ``seat_station.py`` was just fixed for
(``k = index % len(stations)``, where ``index`` counts passes and the planner
works them out of order). Fix that before running it.

**Why a hold and not only an anchor.** The support hand already takes the rod's
far tip before the first crossing, which stops a free cable translating instead
of bending. That is a grip on the END, and it leaves the whole span between the
tip and the station free to be dragged: a crossing that carries 90 mm of rod
across a post pulls the material either side of it along too, and what the
previous crossing seated comes back out. Measured on the terminated run, that is
exactly the failure -- the seats went from held to wrong-side while the end
itself landed correctly.

A hold is a grip on the RUN, placed between the station being worked and
everything already done. It gives the carry something local to react against, so
the material it borrows comes from the free side rather than from the finished
side.

**Which hand, and why it is not a choice.** Measured on this bench with a 0.70 m
rod, the two arms cover complementary halves: the work arm reaches -y to -0.40
and stops at about +0.20, and the support arm reaches +y past +0.28 and cannot
get to -0.40 at all. The stations sit in the middle, where both can reach. So
the support hand holds on the +y side and the work hand crosses -- not a
convention, an envelope.

**Placed on the cable, not on the bench.** The hold point is a node of the
fitted centreline, so it lands on the rod wherever the rod actually is. A fixed
bench coordinate would be a hold that misses once the first crossing has moved
the material.
"""

import json
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

#: How far along the run from the worked station the hold is placed [m].
#:
#: Far enough that the two hands do not share a workspace -- they are solid to
#: each other, and a plan that streams one forearm through the other is a plan
#: that only worked because the model was not looking. Near enough that the span
#: it isolates is the one being crossed: a hold out at the tip is the anchor
#: again, and the anchor is what this exists because of.
HOLD_AHEAD = 0.300
"""How far along the run from the worked station the hold is placed [m].

**Raised from 105 mm, because that was the pulling.** The cable between two
pinches is inextensible: hold it a hand's width from where the other hand is
about to carry, and the carry spends its travel dragging the held span rather
than bending the free one. What it looks like from outside is a robot pulling
the cable straight instead of wrapping it, which is exactly what it is.

300 mm is a little over a third of this 820 mm run, and it is bounded on both
sides. Too short and the two grips fight each other -- at 105 mm the carry spent
its travel dragging the held span, which is what "the robot is pulling the cable"
looks like. Too long and the hold is out past the free tail, holding nothing the
crossing is about.

With the stations on a 155 mm pitch the holds land from +0.145 in y, which is
clear of the work arm by more than a hand's width at every station and still
inside the support arm's own reach past +0.28.

**Both fractions in this note were written against a 0.70 m rod** -- "a little
under half this 700 mm run" -- and the terminated variants carry 0.82 m. The same
drift put the anchor bracket 60 mm along the cable and the support hand in the
middle of the board. Nothing here has been re-measured at the new length, because
nothing here runs; see the note at the top of this file.
"""

#: The dividing line between the two hands' workspaces [m].
#:
#: **The arms are solid to each other and were reaching through one another.**
#: Holding a fixed distance ahead of the crossing is not enough on its own: when
#: a station sits high in +y the hold lands only a hand's width from the work
#: arm, and both are about 100 mm across. What stops it is a line neither hand
#: crosses -- the support stays above it, the work arm below -- so their
#: envelopes overlap nowhere regardless of which station is being worked.
#:
#: Placed at +0.02, which is below the nearest hold ``HOLD_AHEAD`` produces and
#: above the highest station a crossing is worked at. It is a floor, not the
#: usual case: with the holds landing at +0.045 and beyond it rarely binds, and
#: it exists for the draw where a station sits high enough that the two hands
#: would otherwise share a workspace.
WORKSPACE_SPLIT = 0.02

#: Height above the rod's own axis the pinch closes at [m]. The pads straddle
#: the rod, so this is a clearance for the approach and not a grasp depth.
HOLD_LIFT = 0.055

SETTLE = 26


class Output(TypedDict):
    exit: str
    detail: str
    holding: bool


def run(ctx: NodeContext, scene: str = "", index: int = 0,
        grip_open_m: float = 0.0475, grip_close_m: float = 0.004) -> Output:
    seen = json.loads(scene or "{}")
    stations = [(float(a), float(b)) for a, b in (seen.get("stations") or [])]
    rod = json.loads(ctx.tool("cable.curve_memory").get("rod") or "null")
    if not stations or not rod or len(rod) < 4:
        print("[hold] no stations or no centreline; not holding", flush=True)
        return {"exit": "held", "detail": "nothing to hold", "holding": False}

    curve = np.asarray(rod, dtype=np.float64).reshape(-1, 3)
    k = int(index) % len(stations)
    sx, sy = stations[k]

    # The support hand owns +y on this bench, so the hold goes on that side of
    # the station -- between the crossing and the run's free tail.
    # AHEAD OF THE CROSSING **AND** ABOVE THE DIVIDING LINE. Whichever is
    # further; see WORKSPACE_SPLIT.
    want_y = max(sy + HOLD_AHEAD, WORKSPACE_SPLIT)
    j = int(np.argmin(np.abs(curve[:, 1] - want_y)))
    hx, hy, hz = (float(v) for v in curve[j])
    if hy < WORKSPACE_SPLIT:
        print(f"[hold] station {k}: the rod does not reach above the split at "
              f"y={WORKSPACE_SPLIT:+.2f} (nearest node {hy:+.3f}); not holding", flush=True)
        return {"exit": "held", "detail": "no rod above the workspace split", "holding": False}
    if abs(hy - sy) < 0.5 * HOLD_AHEAD:
        # The rod does not reach far enough past this station to hold clear of
        # the hand. Better to skip than to close the two grippers together.
        print(f"[hold] station {k}: nearest rod node is only "
              f"{abs(hy - sy) * 1000:.0f} mm past it; skipping the hold", flush=True)
        return {"exit": "held", "detail": "no room for a hold", "holding": False}

    plan = ctx.tool("cable.plan_support", rod=rod)
    arm_id = int(plan.get("arm_id", 1))
    print(f"[hold] station {k}: pinning the run at ({hx:.3f}, {hy:+.3f}), "
          f"{(hy - sy) * 1000:+.0f} mm along from the crossing", flush=True)

    def _go(z, what):
        try:
            ctx.tool("robot.go_to_pose", arm_id=arm_id, max_steps=140, pose={
                "position": {"x": hx, "y": hy, "z": float(z)},
                "rotation": ctx.tool("robot.grasp_frame", arm_id=int(arm_id))["rotation"]})
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[hold]   {what} refused: {type(exc).__name__}", flush=True)
            return False

    # EVERY CALL GUARDED. The hold is an ASSIST: the crossing it precedes is
    # perfectly runnable without it, so nothing this node does may end the run.
    # Measured before this: an unguarded grip call took two episodes down at 143
    # and 106 steps, and the trace stopped mid-node with no reason recorded.
    arm_name = str(plan.get("arm_name", ""))
    try:
        ctx.tool("cable.set_gripper", width=float(grip_open_m), arm=arm_name)
        ctx.tool("cable.wait_steps", steps=SETTLE)
        if not _go(hz + HOLD_LIFT, "approach"):
            return {"exit": "held", "detail": "hold approach unreachable", "holding": False}
        if not _go(hz + 0.002, "descend"):
            return {"exit": "held", "detail": "hold descent unreachable", "holding": False}
        ctx.tool("cable.set_gripper", width=float(grip_close_m), arm=arm_name)
        ctx.tool("cable.wait_steps", steps=SETTLE)
    except Exception as exc:  # noqa: BLE001 -- an assist may not end the episode
        print(f"[hold] station {k}: giving up on the hold ({type(exc).__name__}); "
              f"the crossing runs without it", flush=True)
        return {"exit": "held", "detail": f"hold abandoned: {type(exc).__name__}",
                "holding": False}
    print(f"[hold] station {k}: holding", flush=True)
    return {"exit": "held", "detail": f"holding the run {HOLD_AHEAD * 1000:.0f} mm past station {k}",
            "holding": True}
