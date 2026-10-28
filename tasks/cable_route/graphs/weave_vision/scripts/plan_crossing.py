"""One crossing, as joint-space stages, over perceived geometry.

The cable's answer to ``computing-feature-mating-poses``: given where the posts
are, which side this one wants, and where the rod lies *now*, produce the legs
that take the rod and put it on that side. Geometry and kinematics only —
nothing here moves the arm, which is what lets the same plan be inspected in a
trace before anything commits to it.

``cable.plan_station`` is deliberately not a ``sim.`` name: it solves IK over
geometry the caller supplies and reads nothing the simulator knows that a robot
would not. Under ``--no-privileged`` it survives, which is what makes "only the
perception body changed" a checkable claim.
"""

import json
from typing import Any, TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    plan: str
    arm_id: int
    grip_open_m: float
    grip_close_m: float
    worst_error_m: float


#: A leg the solver could not reach is refused rather than streamed. The wrap
#: routine's own threshold: past this the chain is in a different basin and what
#: streams is a whip with the rod in the jaws.
PLAN_ERROR_LIMIT = 0.05


def run(ctx: NodeContext, index: int = 0, scene: str = "", rod: str = "") -> Output:
    seen = json.loads(scene or "{}")
    curve = json.loads(rod or "null")
    stations=seen.get("stations") or []
    sides=seen.get("sides") or []
    desired=None;planned_index=int(index)
    if len(stations) in (2, 4) and len(sides)==len(stations):
        by_y=sorted(range(len(stations)),key=lambda j:float(stations[j][1]))
        default=[j for j in reversed(by_y) if sides[j]>0]+[j for j in reversed(by_y) if sides[j]<0]
        # Work from the fixed anchor toward the free end. Revisiting the
        # anchor seat after the outer bends can demand material that is
        # already wrapped around the intervening pegs.
        desired=by_y
        planned_index=default.index(desired[int(index)%len(stations)])
    plan: dict[str, Any] = ctx.tool(
        "cable.plan_station",
        index=planned_index,
        stations=seen.get("stations"),
        sides=seen.get("sides"),
        seats_x=seen.get("seats_x"),
        **({"rod": curve} if curve else {}),
        **({"final_pass": int(index) % len(stations) == len(stations)-1} if stations else {}),
    )
    if desired is not None:
        # The default planner's preceding-pass list no longer describes this
        # schedule. Feed around the measured seats already worked in this order,
        # including the interior near-side bend when restoring the anchor seat.
        seat=float(plan["seat_m"]);z=float(plan["grasp_m"][2])
        plan["feed_via"]=[
            [float(stations[j][0])+float(sides[j])*seat,float(stations[j][1]),z]
            for j in desired[:int(index)%len(stations)]
        ]
        print(f"[mate] outward order={desired}, pass={index}, planner pass={planned_index}",flush=True)
    worst = float(plan.get("worst_grasp_error", 0.0))
    legs = [str(s.get("stage")) for s in plan.get("stages", [])]
    # PASS AND STATION, because they are not the same number and this line used
    # to print the pass under the word "station". The planner works far-side
    # crossings before near-side seats, so on a three-spool board pass 0 is
    # station 1 -- and a log that says "station=0" beside a lift-over template
    # sends a reader looking for a bug in the side decision, which is where an
    # afternoon went.
    side = plan.get("side")
    side_txt = f"{side:+d}" if isinstance(side, int) else "?"
    print(f"[mate] pass={index} station={plan.get('station', '?')} side={side_txt} "
          f"legs={legs} worst_ik={worst * 1000:.1f}mm"
          f" grip={float(plan['grips']['close']['command_m']) * 1000:.1f}mm", flush=True)
    if worst > PLAN_ERROR_LIMIT:
        return {
            "route": "infeasible", "plan": "null", "arm_id": 0,
            "grip_open_m": 0.0, "grip_close_m": 0.0, "worst_error_m": worst,
        }
    return {
        "route": "computed",
        # THE WHOLE PLAN, not just its legs. The support hand's pay-out is
        # arithmetic on where this crossing takes the cable and where it puts it
        # down (``grasp_m``, ``place_m``) and which seated crossings the cable
        # already bends around on the way (``feed_via``). Carrying only the
        # stages costs nothing until something needs the geometry, and then it
        # fails as a bare KeyError several nodes later.
        "plan": json.dumps({
            k: plan[k] for k in ("stages", "grasp_m", "place_m", "feed_via", "station", "side",
                                "grasp_ok", "grasp_slack_m", "grasp_index", "grasp_reason")
            if k in plan
        }),
        "arm_id": int(plan["arm_id"]),
        "grip_open_m": float(plan["grips"]["open"]["command_m"]),
        "grip_close_m": float(plan["grips"]["close"]["command_m"]),
        "worst_error_m": worst,
    }
