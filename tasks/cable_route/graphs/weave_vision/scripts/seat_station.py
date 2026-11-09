"""Seat one crossing, then LOOK, and correct it if it did not take.

Modelled on ``tool_hanging``'s ``place_sg``, which is
``hang -> check -> replan -> hang_again -> check_again`` rather than a single
open-loop hang. Every manipulation skill in that graph verifies its own outcome;
ours streamed legs and hoped, and the whole difference between 1.00 and 0.33 on
this bench is one crossing that did not take and was never looked at again.

**What "did not take" means here is a fraction, not a touch.** The document scores
``containment cable seatN >= 0.047`` -- 4.7% of the rod inside a box 42 mm wide
and 70 mm long. Measured on the failures, the losing station is almost never on
the wrong side: it is on the right side with too little of it there. So the check
counts rod inside the box, and the correction pushes the place point deeper along
the seat's own axis rather than pressing harder.

**IT MEASURES AND DOES NOT CORRECT, and the correction it was written for is
gone rather than disabled.** The obvious fix for "not enough rod in the box" is
to push the hand further onto the required side, and it was built and measured:
on ``port3`` station 0 the coverage went from 0.016 to **0.000** after a 14 mm
nudge. The cable is not short of the box, it is crossing it at an angle, and
driving the contact point further in rotates it further rather than laying more
of it down.

What is kept is the half that works: the skill knows whether its own crossing
took, reports the fraction, and hands an honest number to whatever comes next.
The correction wants a different move -- arriving ALONG the seat rather than
across it -- and that belongs in the carry.
"""

import json
from typing import TypedDict

import numpy as np
from gap import NodeContext

CAMERA = "cable"
ROD_QUERY = "thin white cable"
ROD_SCORE = 0.20
CURVE_NODES = 42

#: Fraction of the rod that must be inside the seat box, the document's own.
SEAT_FRACTION = 0.047

#: The box, off the station and on its required side [m]. The document draws it
#: 3 to 45 mm out; both numbers are read here rather than assumed.
SEAT_INNER, SEAT_OUTER, SEAT_HALF_Y = 0.003, 0.045, 0.035

SETTLE = 30


class Output(TypedDict):
    route: str
    seated: bool
    fraction: float
    detail: str


def _frame(ctx: NodeContext):
    obs = ctx.tool("robot.get_observation")
    frames = obs["cameras"] if isinstance(obs, dict) else obs.cameras
    block = frames.get(CAMERA) if isinstance(frames, dict) else next(
        (f for f in (frames or ()) if str(f.get("name", "")) == CAMERA), None)
    if block is None:
        return None
    images = block.get("images") if isinstance(block.get("images"), dict) else {}

    def _first(*keys):
        for src in (images, block):
            for k in keys:
                v = src.get(k)
                if v is not None:
                    return v
        return None

    depth = _first("depth", "depth_data")
    if depth is None:
        return None
    rgb = np.asarray(_first("rgb"), dtype=np.uint8)
    K = np.asarray(block["intrinsics"], dtype=np.float64).reshape(3, 3)
    pose = block["pose"]
    if isinstance(pose, dict):
        p, r = pose.get("position", {}), pose.get("rotation", {})
        px, py, pz = (float(p[k]) for k in ("x", "y", "z"))
        w, x, y, z = (float(r[k]) for k in ("w", "x", "y", "z"))
    else:
        flat = np.asarray(pose, dtype=np.float64).reshape(-1)
        px, py, pz = flat[:3]
        w, x, y, z = flat[3:7]
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, (px, py, pz)
    return rgb, np.asarray(depth, dtype=np.float64), K, T


def _measure(ctx: NodeContext, station, side: float) -> float:
    """What fraction of the rod is inside this station's seat box, from pixels."""
    got = _frame(ctx)
    if got is None:
        return -1.0
    rgb, depth, K, T = got
    found = ctx.tool("sam3.segment_text", image=rgb, query=ROD_QUERY, max_results=1)
    masks, scores = found.get("masks") or [], found.get("scores") or []
    if not masks or float(scores[0]) < ROD_SCORE:
        return -1.0
    fit = ctx.tool("curve.fit_centerline", mask=np.asarray(masks[0]), depth=depth,
                   intrinsics=K, camera_pose=T, nodes=CURVE_NODES)
    pts = np.asarray(fit.get("points") or [], dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return -1.0
    # Densified, so the count is a faithful fraction of the CURVE rather than of
    # its knots -- the same reason ``verify_weave_vision`` densifies.
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    want = np.arange(0.0, s[-1], 0.002)
    dense = np.stack([np.interp(want, s, pts[:, k]) for k in range(3)], axis=1)
    sx, sy = float(station[0]), float(station[1])
    near = dense[np.abs(dense[:, 1] - sy) <= SEAT_HALF_Y]
    if not len(near):
        return 0.0
    off = (near[:, 0] - sx) * side
    return float(np.count_nonzero((off >= SEAT_INNER) & (off <= SEAT_OUTER)) / len(dense))


def run(ctx: NodeContext, plan: str = "", scene: str = "", index: int = 0,
        arm_id: int = 0, grip_open_m: float = 0.0475, grip_close_m: float = 0.003) -> Output:
    made = json.loads(plan or "{}") or {}
    stages = made.get("stages") or []
    mine = [s for s in stages if str(s.get("stage")) == "place"]
    seen = json.loads(scene or "{}")
    stations = seen.get("stations") or []
    sides = seen.get("sides") or []
    if not mine or not stations:
        print("[seat] nothing to place", flush=True)
        return {"route": "done", "seated": False, "fraction": 0.0, "detail": "no place leg"}

    # THE STATION THE PLAN WORKED, not the pass number.
    #
    # ``index`` counts PASSES and the planner does not work them in order: it
    # sorts far-side crossings first and near-side seats last, each from the
    # anchored end outwards, because a far-side crossing drags the whole rod
    # forward and would push a near-side seat done earlier out of its own box.
    # ``plan_station_stages`` publishes the station it resolved to as
    # ``plan["station"]``; this node was ignoring it and indexing the perceived
    # list by the pass number instead.
    #
    # On ``port3`` -- sides (-1, +1, -1) -- the order is (1, 2, 0), so every
    # pass measured a DIFFERENT station's box from the one it had just worked.
    # Pass 0 crossed station 1 and reported station 0; pass 1 dragged station 2
    # and reported station 1; pass 2 dragged station 0 and reported station 2.
    #
    # That is where "station 1 is 0.000 in every single trial" came from, and
    # it was never a finding about station 1: the run those numbers were read
    # off finished with station 1 the BEST-seated of the three (0.074 against
    # 0.044 and 0.044 at the evaluator's own reckoning). A per-crossing
    # measurement indexed by something other than the crossing is worse than no
    # measurement, because it reads like evidence.
    k = int(made.get("station", int(index) % len(stations))) % len(stations)
    station = stations[k]
    side = float(made.get("side", sides[k] if k < len(sides) else -1.0))

    def _place(stage) -> None:
        ctx.tool("robot.execute_trajectory", trajectory=stage["trajectory"],
                 arm_id=int(arm_id), max_steps_per_waypoint=15)

    _place(mine[0])
    ctx.tool("cable.wait_steps", steps=SETTLE)
    frac = _measure(ctx, station, side)
    need = SEAT_FRACTION
    print(f"[seat] station {k}: {frac:.3f} of the rod in the box (need {need:.3f})", flush=True)
    if frac < 0.0:
        return {"route": "done", "seated": False, "fraction": 0.0,
                "detail": "could not measure the seat"}
    if frac >= need:
        return {"route": "done", "seated": True, "fraction": round(frac, 4),
                "detail": f"seated, {frac:.3f} in the box"}

    # Short, and no correction is attempted -- see the module docstring for the
    # nudge that was built, measured and taken back out.
    return {"route": "done", "seated": False, "fraction": round(frac, 4),
            "detail": f"short at {frac:.3f}, no correction attempted"}
