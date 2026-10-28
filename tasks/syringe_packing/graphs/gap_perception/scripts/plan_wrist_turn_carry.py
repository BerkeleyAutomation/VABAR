"""Stand the syringe up with the wrist alone, then carry it to the box on straight lines.

Replaces the free joint-space reorientation, which let the planner re-solve the
whole arm for a quarter turn of the hand -- measured swinging the elbow through
a radian, dipping the hand 30 cm into the tray, and pivoting the syringe in the
pinch. Pre-tested offline against the executing planner
(``tools/sharps_wrist_turn_pretest.py``); this is the same procedure, live:

1. **Turn with two joints.** Find, by forward kinematics over the arm's own
   chain, a setting of the wrist's last joint that lays the next joint's axis
   along the pinch, then sweep that joint alone until the held tip points down
   the fixture axis. Every other joint holds, so nothing re-solves; the arc the
   hand sweeps is checked clear of the tray by the held object's sweep radius.
2. **Yaw to a reachable heading.** A rotation about the vertical twists the
   syringe about its own axis, which the pads hold easily -- only a pitch about
   the pinch axis can make it pivot -- so the heading is free to be chosen for
   reachability at the box. Candidates are probed with the executing planner.
3. **Carry on straight lines.** Orientation-locked ``plan_linear`` legs only:
   up to transport height, across to above the aperture, down to the transit
   pose. The approach and stroke that follow are the insertion node's, also
   straight.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation

SET_JOINT = 6
"""0-based index of the joint set first, to align the turning joint's axis with the pinch (Panda: joint 7)."""
TURN_JOINT = 5
"""0-based index of the joint that makes the turn (Panda: joint 6, the wrist pitch)."""
SET_GRID_DEG = np.arange(-180.0, 181.0, 5.0)
TURN_GRID_DEG = np.arange(-120.0, 121.0, 2.0)
TIP_DOWN_TOLERANCE_DEG = 4.0
TRAY_TOP_ABOVE_SURFACE_M = 0.04
JOINT_MARGIN_MIN_RAD = 0.15
ROOM_ENOUGH_RAD = 0.30
"""Joint room along the whole chain at which the search stops looking for a roomier heading."""
YAWS_DEG = (0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0, -120.0, 120.0, -150.0, 150.0, 180.0)
YAWS_FINE_DEG = (-15.0, 15.0, -45.0, 45.0, -75.0, 75.0)
"""Tried only when no heading on the coarse grid plans the whole chain."""
CARRY_LEGS = 4
"""up, across, down, approach: the legs this node flies itself. A heading that
plans these but not the stroke is still a carry to the approach pose, from
which the insertion node re-plans the stroke against the real posture (with
its own contact-tolerant fallback) -- better odds than an abort, which is a
certain failure. Measured on trial t03 (2026-09-09): the one heading whose
carry planned lost only the last 55 mm, and the graph gave up."""
TRANSPORT_CLEARANCE_M = 0.05
TURN_TIME_SCALE = 4.0
YAW_TIME_SCALE = 2.0
TURN_STEP_DEG = 1.0


class Output(TypedDict):
    reorientation_plan: dict[str, Any]


def _rotation(pose: dict[str, Any]) -> Rotation:
    q = pose["rotation"]
    return Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]])


def _vec(value: dict[str, float]) -> np.ndarray:
    return np.array([value[key] for key in ("x", "y", "z")], dtype=np.float64)


def _pose(position: np.ndarray, rotation: Rotation) -> dict[str, Any]:
    q = rotation.as_quat()
    return {"position": dict(zip(("x", "y", "z"), map(float, position))),
            "rotation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])}}


def _end_joints(result: dict[str, Any]) -> np.ndarray | None:
    trajectory = result.get("trajectory") if result.get("planned") else None
    if not trajectory or not trajectory.get("waypoints"):
        return None
    return np.asarray(trajectory["waypoints"][-1]["positions"], dtype=np.float64)


def run(ctx: NodeContext, held_feature_in_tcp: dict[str, Any],
        fixture_feature: dict[str, Any], relation: str,
        world_config: dict[str, Any], attached_object: dict[str, Any],
        approach_clearance_m: float = 0.08) -> Output:
    if relation not in {"shaft_into_aperture", "tip_through_aperture", "insert_through",
                        "loop_over_shaft", "feature_to_fixture"}:
        raise ValueError(f"unsupported feature relation {relation!r}")
    axis = _vec(fixture_feature["axis"]); axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    fixture_center = _vec(fixture_feature["pose"]["position"])
    feature_target = fixture_center - max(0.02, float(approach_clearance_m)) * axis
    feature_offset = _vec(held_feature_in_tcp["position"])
    tip_in_tcp = _rotation(held_feature_in_tcp).apply([0.0, 0.0, 1.0])
    tip_in_tcp /= max(float(np.linalg.norm(tip_in_tcp)), 1.0e-12)

    arm = ctx.tool("robot.describe_arm")
    lower = np.asarray(arm["joint_limits"]["lower"], dtype=np.float64)
    upper = np.asarray(arm["joint_limits"]["upper"], dtype=np.float64)
    surface_z = float(ctx.tool("robot.describe_workspace")["surface_z"])
    floor_z = surface_z + TRAY_TOP_ABOVE_SURFACE_M
    q0 = np.asarray(ctx.tool("robot.forward_kinematics")["joint_config"], dtype=np.float64)

    def sphere_extent(sphere: dict[str, Any]) -> float:
        center = sphere.get("center") or {}
        offset = np.array([float(center.get(k, 0.0)) for k in ("x", "y", "z")])
        return float(np.linalg.norm(offset)) + float(sphere.get("radius", 0.0))
    sweep = max([sphere_extent(s) for s in attached_object.get("spheres", [])], default=0.10)

    def margin_of(q: np.ndarray) -> float:
        return float(np.minimum(q - lower, upper - q).min())

    def fk(configs: list[np.ndarray]) -> list[tuple[np.ndarray, Rotation]]:
        poses = ctx.tool("robot.forward_kinematics", joint_configs=[[float(v) for v in c] for c in configs])["poses"]
        return [(_vec(p["position"]), _rotation(p)) for p in poses]

    # -- 1. the wrist-only turn, by search over the arm's own kinematics
    configs, index = [], []
    for d_set in np.deg2rad(SET_GRID_DEG):
        q_set = q0.copy(); q_set[SET_JOINT] = q0[SET_JOINT] + d_set
        if not lower[SET_JOINT] <= q_set[SET_JOINT] <= upper[SET_JOINT]:
            continue
        for theta in np.deg2rad(TURN_GRID_DEG):
            q = q_set.copy(); q[TURN_JOINT] = q_set[TURN_JOINT] + theta
            if lower[TURN_JOINT] <= q[TURN_JOINT] <= upper[TURN_JOINT]:
                configs.append(q); index.append((float(d_set), float(theta)))
    poses = fk(configs)
    best_by_set: dict[float, tuple[float, int]] = {}
    for i, ((d_set, _theta), (_p, rot)) in enumerate(zip(index, poses)):
        tip_world = rot.apply(tip_in_tcp)
        angle = float(np.degrees(np.arccos(np.clip(tip_world @ axis, -1.0, 1.0))))
        # Either end may go down: the wrist pitches one way within its range,
        # and a 150 mm syringe fits a 52 mm hole flange-first as well as
        # tip-first. Which end is down decides which end the carry aims.
        off_axis = min(angle, 180.0 - angle)
        if off_axis <= TIP_DOWN_TOLERANCE_DEG and (d_set not in best_by_set or off_axis < best_by_set[d_set][0]):
            best_by_set[d_set] = (off_axis, i)
    candidates = []
    for d_set, (_angle, i) in best_by_set.items():
        q_turned = configs[i]
        if margin_of(q_turned) < JOINT_MARGIN_MIN_RAD:
            continue
        q_set = q_turned.copy(); q_set[TURN_JOINT] = q0[TURN_JOINT]
        arc = [q_set + np.eye(len(q0))[TURN_JOINT] * t for t in np.linspace(0.0, q_turned[TURN_JOINT] - q0[TURN_JOINT], 24)]
        lowest = min(p[2] for p, _ in fk(arc)) - sweep
        if lowest < floor_z:
            continue
        candidates.append((abs(d_set) + abs(index[i][1]), d_set, q_set, q_turned, poses[i]))
    if not candidates:
        raise RuntimeError("no wrist-only turn stands the syringe up clear of the tray with joint room")
    candidates.sort(key=lambda c: c[0])

    # -- 2 + 3. a heading the linear carry can be planned for, by the planner that will fly it
    chosen = None
    partial = None  # (depth, room, candidate): the deepest chain short of the stroke
    for _cost, _d_set, q_set, q_turned, (p_turned, rot_turned) in candidates[:3]:
        # The end pointing down. The grasp is at the barrel's centre, so the
        # far end sits at the tip's offset mirrored through the TCP.
        tip_down = float(rot_turned.apply(tip_in_tcp) @ axis) > 0.0
        end_offset = feature_offset if tip_down else -feature_offset
        yaws = YAWS_DEG + (() if chosen is not None else YAWS_FINE_DEG)
        # The yaw-in-place turn only decides a seed for the linear chain
        # below (its own trajectory is replanned at execution time, never
        # streamed from here -- see execute_reorientation.py), so every
        # yaw's turn is independent IK from the same q_turned seed: exactly
        # what a batched solve answers in one GPU call instead of one
        # per yaw (measured 3.2x at 12 candidates; see MotionBackendCuRobo.
        # solve_ik_batch). yaw 0 needs no turn -- it already IS q_turned.
        turn_seed_by_yaw: dict[float, np.ndarray | None] = {0.0: q_turned}
        non_zero_yaws = [yaw_deg for yaw_deg in yaws if yaw_deg != 0.0]
        if non_zero_yaws:
            rotations_by_yaw = {
                yaw_deg: Rotation.from_euler("z", yaw_deg, degrees=True) * rot_turned
                for yaw_deg in non_zero_yaws
            }
            batch = ctx.tool(
                "motion.solve_ik_batch",
                poses=[_pose(p_turned, rotations_by_yaw[yaw_deg]) for yaw_deg in non_zero_yaws],
                attached_object=attached_object,
                seed_joints=[float(v) for v in q_turned],
            )["results"]
            for yaw_deg, result in zip(non_zero_yaws, batch, strict=True):
                turn_seed_by_yaw[yaw_deg] = (
                    np.asarray(result["joint_config"], dtype=np.float64) if result else None
                )
        for yaw_deg in yaws:
            rotation = Rotation.from_euler("z", yaw_deg, degrees=True) * rot_turned
            seed = turn_seed_by_yaw[yaw_deg]
            if seed is None:
                continue
            transit = feature_target - rotation.apply(end_offset)
            transport_z = max(float(p_turned[2]), float(transit[2])) + TRANSPORT_CLEARANCE_M
            approach = (fixture_center - 0.020 * axis) - rotation.apply(end_offset)
            stroke = (fixture_center + 0.035 * axis) - rotation.apply(end_offset)
            legs = [
                np.array([p_turned[0], p_turned[1], transport_z]),
                np.array([transit[0], transit[1], transport_z]),
                transit,
                approach,
                stroke,
            ]
            # The whole chain, through the stroke into the aperture: a heading
            # whose carry plans but leaves the shoulder against its stop at the
            # approach still cannot make the last 55 mm (measured: 0.18 rad of
            # room on joint 2 there, stroke refused, syringe dropped from 9 cm).
            start, room, depth = p_turned, margin_of(seed), 0
            for target in legs:
                planned = ctx.tool("motion.plan_linear", end=_pose(target, rotation), start=_pose(start, rotation),
                                   orientation="lock", attached_object=attached_object,
                                   seed_joints=[float(v) for v in seed],
                                   allow_start_contact=(target is stroke), allow_goal_contact=(target is stroke))
                seed = _end_joints(planned)
                if seed is None:
                    break
                depth += 1
                room = min(room, margin_of(seed))
                start = target
            candidate = (room, q_set, q_turned, p_turned, rotation, yaw_deg, legs[:3], tip_down, end_offset)
            if depth == len(legs):
                if chosen is None or room > chosen[0]:
                    chosen = candidate
                if room >= ROOM_ENOUGH_RAD:
                    break
            elif depth >= CARRY_LEGS and (partial is None or (depth, room) > partial[:2]):
                partial = (depth, room, candidate)
        if chosen is not None and chosen[0] >= ROOM_ENOUGH_RAD:
            break
    chain_complete = chosen is not None
    if chosen is None and partial is not None:
        print(f"[wrist_turn] no heading plans the whole chain; carrying on yaw {partial[2][5]:+.0f} deg, "
              f"whose chain reached the approach ({partial[0]}/{len(legs)} legs, room {partial[1]:.2f} rad) -- "
              "the stroke is left to the insertion node")
        chosen = partial[2]
    if chosen is None:
        raise RuntimeError("no heading lets the syringe be carried to the box on straight lines from the wrist turn")
    _room, q_set, q_turned, p_turned, rotation, yaw_deg, (up, across, transit), tip_down, end_offset = chosen

    # the turn as an explicit joint path: set the aligning joint, then sweep the turning joint, a degree a row
    rows = []
    for a, b in ((q0, q_set), (q_set, q_turned)):
        count = max(2, int(np.ceil(np.degrees(np.abs(b - a).max()) / TURN_STEP_DEG)) + 1)
        rows.extend(a + (b - a) * t for t in np.linspace(0.0, 1.0, count)[1:])
    waypoints: list[dict[str, Any]] = [
        {"mode": "joint_path", "positions": [[float(v) for v in row] for row in rows], "time_scale": TURN_TIME_SCALE},
    ]
    if yaw_deg != 0.0:
        waypoints.append({"pose": _pose(p_turned, rotation), "mode": "planned_joint", "cartesian": False,
                          "hold_position": True, "time_scale": YAW_TIME_SCALE})
    waypoints += [
        {"pose": _pose(up, rotation), "mode": "planned_linear", "cartesian": True},
        {"pose": _pose(across, rotation), "mode": "planned_linear", "cartesian": True},
        {"pose": _pose(transit, rotation), "mode": "planned_linear", "cartesian": True},
    ]
    return {"reorientation_plan": {
        "time_scale": 2.0,
        "waypoints": waypoints,
        "world_config": world_config,
        "attached_object": attached_object,
        "wrist_turn": {"set_joint": SET_JOINT + 1, "turn_joint": TURN_JOINT + 1, "yaw_deg": float(yaw_deg), "tip_down": bool(tip_down), "chain_complete": bool(chain_complete), "chain_min_margin_rad": float(_room)},
        "end_offset_in_tcp": {"x": float(end_offset[0]), "y": float(end_offset[1]), "z": float(end_offset[2])},
    }}
