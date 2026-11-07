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

import numpy as np
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

PEG_KEEPOUT = 0.032
"""How close any part of a finger's stroke may come to a perceived peg's axis [m].

**The take's open jaws were coming down on the NEXT spool.** A crossing grasps the
rod across it with the jaws fully open, 47.5 mm a finger, and because the rod's
start is welded to the anchor the planner walks the take point away from the
anchor until enough cable lies behind it -- on cable_routing/L2 that is 34-143 mm
past the station, while the pegs stand 72-80 mm apart. Station 1's take therefore
lands abreast of peg 2, and on 24-26 of 30 baseline episodes a finger's stroke came
within 32 mm of peg 2's axis. Replayed over three baseline cells (89 episodes),
seat 1 held on 13 of 13 crossings whose open jaws cleared every peg by this much,
and on 13 of 76 that did not; seat 2, 60 of 63 against 12 of 26. Recorded, the
finger lands on the spool: the jaws close above the rod, or the finger hooks the
flange and twists the wrist 0.66 rad off its command, and the carry moves nothing.

32 mm is the spool's 12.5 mm rim plus the clearance the L1 port keep-out implies
(``plug_port.PORT_KEEPOUT``: 40 mm from the centre of a 22.6 mm half-diagonal).
L1's pegs are 240 mm apart, and none of an L1 cell's 60 takes comes this close,
so L1 runs exactly as before."""

JAW_MIN_REACH = 0.015
"""Narrowest pre-grasp opening per finger a crossing will use [m] -- the terminal
take's own floor (``plug_port.JAW_MIN_REACH``): 30 mm between the fingertips
around a 7 mm rod."""

JAW_CLOSED_REACH = 0.004
"""Where a closed finger stands off the grasp line [m]: about the rod's radius."""

REGRASP_MARGINS = (0.035, 0.050, 0.065, 0.080, 0.095, 0.110)
"""Slack margins the planner is asked again at, in turn, when no opening down to
:data:`JAW_MIN_REACH` clears the pegs [m]. The planner's own is 20 mm; each larger
one walks the take further from the anchor, off the peg it was beside. The first
plan whose take clears is kept; when none does, the first plan runs at the
narrowest opening."""


def rod_tangent(rod, point_xy):
    """The perceived cable direction at a point, as ``routine.cable_tangent`` measures it."""
    pts = np.asarray(rod, dtype=float).reshape(-1, 3)
    i = int(np.argmin(np.linalg.norm(pts[:, :2] - np.asarray(point_xy, dtype=float)[:2], axis=1)))
    local = pts[max(0, i - 2): min(len(pts), i + 3), :2]
    return np.linalg.svd(local - local.mean(axis=0), full_matrices=False)[2][0]


def _segment_distance(point, a, b) -> float:
    ab = b - a
    t = float(np.clip(np.dot(point - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


def nearest_tip_to_post(grasp_xy, tangent, posts, reach) -> float:
    """How close either finger's stroke, open at ``reach`` to closed, comes to any post axis [m]."""
    t = np.asarray(tangent, dtype=float).reshape(-1)[:2]
    if not np.all(np.isfinite(t)) or float(np.linalg.norm(t)) < 1e-9 or not len(posts):
        return float("inf")
    t = t / np.linalg.norm(t)
    normal = np.array([-t[1], t[0]])
    g = np.asarray(grasp_xy, dtype=float)[:2]
    inner = min(JAW_CLOSED_REACH, float(reach))
    return min(_segment_distance(np.asarray(p, dtype=float)[:2], g + s * inner * normal, g + s * float(reach) * normal)
               for s in (1.0, -1.0) for p in posts)


def jaws_clear_posts(grasp_xy, tangent, posts, reach) -> bool:
    """Whether both fingers, over their whole stroke, stay :data:`PEG_KEEPOUT` off every post.

    A finger comes down at ``reach`` and closes to the rod, so what has to clear is
    the segment from the open tip in to the closed jaw, not the tip alone: a post
    between the tip and the rod blocks the close as surely as one under the tip
    blocks the descent.
    """
    return nearest_tip_to_post(grasp_xy, tangent, posts, reach) >= PEG_KEEPOUT


def jaw_reach_clearing_posts(grasp_xy, tangent, posts, reach_max):
    """The widest opening per finger, at most ``reach_max``, whose stroke clears every post; None if none does."""
    for k in range(round((float(reach_max) - JAW_MIN_REACH) / 0.001) + 1):
        r = round(float(reach_max) - 0.001 * k, 4)
        if jaws_clear_posts(grasp_xy, tangent, posts, r):
            return r
    return None


def take_opening(curve, plan, posts):
    """``(opening per finger [m], note)`` for a planned take, keeping its grasp point.

    The widest opening, up to the plan's own, whose fingers clear every post by
    :data:`PEG_KEEPOUT`; :data:`JAW_MIN_REACH` when none does.
    """
    reach_max = float(plan["grips"]["open"]["command_m"])
    if not curve or not posts:
        return reach_max, {}
    tangent = rod_tangent(curve, plan["grasp_m"][:2])
    reach = jaw_reach_clearing_posts(plan["grasp_m"][:2], tangent, posts, reach_max)
    opening = JAW_MIN_REACH if reach is None else float(reach)
    return opening, {"take_open_m": round(opening, 4), "clears": reach is not None,
                     "tip_to_post_m": round(nearest_tip_to_post(plan["grasp_m"][:2], tangent, posts, opening), 4)}


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
    def _plan(**extra) -> dict[str, Any]:
        return ctx.tool(
            "cable.plan_station",
            index=planned_index,
            stations=seen.get("stations"),
            sides=seen.get("sides"),
            seats_x=seen.get("seats_x"),
            **({"rod": curve} if curve else {}),
            **({"final_pass": int(index) % len(stations) == len(stations)-1} if stations else {}),
            **extra,
        )

    plan: dict[str, Any] = _plan()
    # KEEP THE OPEN JAWS OFF THE PEGS. See PEG_KEEPOUT. Narrow the take's opening
    # first; only when no opening clears does the take point move.
    reach_max = float(plan["grips"]["open"]["command_m"])
    take_open = reach_max
    jaw: dict[str, Any] = {}
    if curve and stations:
        tangent = rod_tangent(curve, plan["grasp_m"][:2])
        full_open = nearest_tip_to_post(plan["grasp_m"][:2], tangent, stations, reach_max)
        reach = jaw_reach_clearing_posts(plan["grasp_m"][:2], tangent, stations, reach_max)
        jaw["full_open_tip_to_post_m"] = round(full_open, 4)
        if reach is None:
            for margin in REGRASP_MARGINS:
                try:
                    alt = _plan(slack_margin=margin)
                except (ValueError, RuntimeError) as exc:
                    jaw["regrasp_refused"] = f"slack margin {margin}: {exc}"[:200]
                    break
                if float(alt.get("worst_grasp_error", 0.0)) > PLAN_ERROR_LIMIT or not alt.get("grasp_ok", True):
                    continue
                alt_tangent = rod_tangent(curve, alt["grasp_m"][:2])
                alt_reach = jaw_reach_clearing_posts(alt["grasp_m"][:2], alt_tangent, stations, reach_max)
                if alt_reach is not None:
                    print(f"[mate] jaws: no opening clears the pegs here; take moved along the rod "
                          f"(slack margin {margin * 1000:.0f} mm)", flush=True)
                    jaw.update(moved=True, slack_margin_m=margin)
                    plan, reach, tangent = alt, alt_reach, alt_tangent
                    break
        take_open = JAW_MIN_REACH if reach is None else float(reach)
        jaw.update(take_open_m=round(take_open, 4), clears=reach is not None,
                   tip_to_post_m=round(nearest_tip_to_post(plan["grasp_m"][:2], tangent, stations, take_open), 4))
        print(f"[mate] jaws: open {take_open * 1000:.0f} mm a finger "
              f"(fully open {full_open * 1000:.0f} mm off the nearest peg, now "
              f"{jaw['tip_to_post_m'] * 1000:.0f} mm)", flush=True)
    plan["take_open_m"] = take_open
    plan["jaw"] = jaw
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
                                "grasp_ok", "grasp_slack_m", "grasp_index", "grasp_reason",
                                "take_open_m", "jaw")
            if k in plan
        }),
        "arm_id": int(plan["arm_id"]),
        "grip_open_m": float(plan["grips"]["open"]["command_m"]),
        "grip_close_m": float(plan["grips"]["close"]["command_m"]),
        "worst_error_m": worst,
    }
