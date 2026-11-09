"""Weave the rod, one station at a time, re-planning at each.

**A loop rather than a node per station**, because the station count is the
variant's and not this file's: ``weave3`` has three and ``weave_stagger`` four,
and a workflow that named them would serve one member of a family whose whole
point is that its members differ in exactly one dimension.

**Re-planned per station, and that is not an optimisation.** Each crossing moves
the rod, so the next grasp has to be solved against where the rod ended up.
Planned once up front instead, the second and third grasps descend on the rod's
*starting* line, close on nothing, and the latch refuses -- ``hold_segment``
returning -1 while the hand drags air. ``sim.plan_station`` reads the live rod.

**The latch, not the pinch, carries it.** ``sim.hold_cable`` pins the grasped
segment to the hand kinematically; the jaws close too, but a compliant
parallel-jaw grip on a 10 mm chain under drag does not hold on its own, and the
cable suite has said so since its first port. The latch is dropped *before* the
clear leg, or the rod is lifted away with the hand.

**This path is single-handed, and the routine it shares its planner with is
not (2026-08-11).** ``RoutineWeave`` weaves with the support hand anchoring
the far end by contact and paying out slack, and that is what its 3/3 stands
on; ``sim.plan_station`` now aims where the seats actually are and grasps the
live rod, so THIS graph inherits grasps that land -- on a rod nobody anchors,
which is the measured haul-away configuration, and with none of the routine's
plan-error refusals (the tool returns ``errors`` and ``worst_grasp_error``;
nothing here reads them). A bimanual graph needs support-arm tools that do
not exist yet; until they do, the routine is the demonstration and this
script is the tool-driving skeleton.
"""

import json
from typing import TypedDict

import numpy as np

from gap import NodeContext

CLOSED = ("move", "cross", "place")
"""The legs the rod is carried through. Everything else runs with open jaws."""


CAMERA = "cable"
ROD_QUERY = "white rod"
ROD_SCORE = 0.20
CURVE_NODES = 42

#: How far a re-observed centreline may differ in length from the opening one
#: before it is refused. A fit made while the gripper covers most of the rod
#: collapses onto what is still visible -- measured on ``weave3``, arclength
#: swinging 118-526 mm across one episode, and on ``weave_arc``, 101-873 mm with
#: a quarter of frames lost outright. A static fitter cannot tell a shortened
#: rod from a shortened *view* of one, so the gate is length: keep the previous
#: curve rather than plan a grasp against half a rod.
#:
#: This is a crude stand-in for what TrackDLO's motion-coherence step does
#: properly -- impute the occluded stretch instead of discarding the frame --
#: and it is stated as crude because the difference matters: this keeps a stale
#: curve where a tracker would keep a current one.
ARCLENGTH_TOLERANCE = 0.25


def _camera(observation, name):
    frames = observation["cameras"] if isinstance(observation, dict) else observation.cameras
    if isinstance(frames, dict):
        return frames.get(name)
    for frame in frames or ():
        if str(frame.get("name", "")) == name:
            return frame
    return None


def _frame(block):
    images = block.get("images") if isinstance(block.get("images"), dict) else {}

    def _first(*keys):
        for src in (images, block):
            for key in keys:
                v = src.get(key)
                if v is not None:
                    return v
        return None

    rgb = np.asarray(_first("rgb"), dtype=np.uint8)
    depth = _first("depth", "depth_data")
    if depth is None:
        return None
    K = np.asarray(block["intrinsics"], dtype=np.float64).reshape(3, 3)
    pose = block["pose"]
    if isinstance(pose, dict):
        pos, rot = pose.get("position", {}), pose.get("rotation", {})
        px, py, pz = (float(pos[k]) for k in ("x", "y", "z"))
        w, x, y, z = (float(rot[k]) for k in ("w", "x", "y", "z"))
    else:
        flat = np.asarray(pose, dtype=np.float64).reshape(-1)
        px, py, pz = flat[:3]
        w, x, y, z = flat[3:7]
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, (px, py, pz)
    return rgb, np.asarray(depth, dtype=np.float64), K, T


def _observe_rod(ctx, body):
    """The rod's centreline right now, or ``None`` if this body cannot see it.

    The vision body re-segments and re-fits; the oracle body re-reads the
    simulator. Both are re-observations of a shape that the crossing just made
    has changed, which is the whole reason the station loop asks again rather
    than planning once.
    """
    if str(body) != "vision":
        try:
            return [list(map(float, p)) for p in ctx.tool("sim.perceive_weave")["rod"]]
        except Exception:  # noqa: BLE001 -- withheld under --no-privileged, which is fine
            return None
    observation = ctx.tool("robot.get_observation")
    block = _camera(observation, CAMERA)
    if block is None:
        return None
    got = _frame(block)
    if got is None:
        return None
    rgb, depth, K, T = got
    found = ctx.tool("sam3.segment_text", image=rgb, query=ROD_QUERY, max_results=1)
    masks, scores = found.get("masks") or [], found.get("scores") or []
    if not masks or float(scores[0]) < ROD_SCORE:
        return None
    fit = ctx.tool(
        "curve.fit_centerline",
        mask=np.asarray(masks[0]),
        depth=depth,
        intrinsics=K,
        camera_pose=T,
        nodes=CURVE_NODES,
    )
    points = fit.get("points") or []
    return [list(map(float, p)) for p in points] if len(points) >= 2 else None


def _plausible(curve, reference_len):
    """Whether a re-observed curve is long enough to have been the whole rod."""
    if not curve or reference_len <= 0.0:
        return False
    pts = np.asarray(curve, dtype=np.float64)
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return abs(length - reference_len) <= ARCLENGTH_TOLERANCE * reference_len


class Output(TypedDict):
    route: str
    stations: int
    crossed: list


def run(
    ctx: NodeContext,
    stations: int,
    scene: str,
    grip_open_m: float = 0.0475,
    grip_close_m: float = 0.003,
    settle_steps: int = 24,
    body: str = "oracle",
) -> Output:
    seen = json.loads(scene or "{}")
    crossed: list = []
    # The opening observation, and the length every later one is judged against.
    curve = [list(map(float, p)) for p in (seen.get("rod") or [])] or None
    reference_len = 0.0
    if curve:
        pts = np.asarray(curve, dtype=np.float64)
        reference_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    for n in range(int(stations)):
        # RE-OBSERVED PER STATION, because the shape is an outcome of the
        # crossing just made. Planning every grasp against the rod's opening
        # shape puts the second and third descents where the rod used to be --
        # measured on the privileged path as `hold_segment` returning -1 at two
        # stations of three.
        #
        # A script node may call tools, so this needs no graph restructure: the
        # detector and the fitter are `sam3.segment_text` and
        # `curve.fit_centerline`, and both are unprivileged.
        fresh = _observe_rod(ctx, body)
        if _plausible(fresh, reference_len) or (fresh and reference_len <= 0.0):
            curve = fresh
            if reference_len <= 0.0:
                pts = np.asarray(curve, dtype=np.float64)
                reference_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        plan = ctx.tool(
            "cable.plan_station",
            index=n,
            stations=seen["stations"],
            sides=seen["sides"],
            seats_x=seen["seats_x"],
            **({"rod": curve} if curve else {}),
        )
        arm_id = int(plan["arm_id"])
        for stage in plan["stages"]:
            name = stage["stage"]
            if name == "clear":
                # Before the leg, not after: a latch still on while the hand
                # lifts away takes the rod with it.
                ctx.tool("cable.set_gripper", width=float(grip_open_m))
                ctx.tool("cable.wait_steps", steps=int(settle_steps))
            ctx.tool(
                "robot.execute_trajectory",
                trajectory=stage["trajectory"],
                arm_id=arm_id,
                max_steps_per_waypoint=15,
            )
            if name == "take":
                ctx.tool("cable.set_gripper", width=float(grip_close_m))
                ctx.tool("cable.wait_steps", steps=int(settle_steps))
        crossed.append({"station": int(plan["station"]), "side": int(plan["side"])})
    return {"route": "woven", "stations": len(crossed), "crossed": crossed}
