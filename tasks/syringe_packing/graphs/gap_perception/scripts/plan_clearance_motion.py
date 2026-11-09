"""Plan lift, a carry in the pick orientation, and reorientation at the box.

The reference this suite reproduces (VAB-industrial's own scripted policy,
``docs/generated_videos/syringe_sharps_box/run_video.mp4``) grasps, lifts,
and covers essentially all of the lateral distance to the box within about a
second (frame 170 to frame 190 of 400, at 20 fps) *before* any reorientation
is visible -- the wrist only turns the syringe upright once it is already
parked next to the box, where it then holds for the rest of the clip. That is
the opposite order from an earlier version of this file, which rotated the
syringe upright immediately after lift-off (still over the tray) and only
then carried it, already vertical, to the box. This plan now matches the
reference's own order: escape contact, carry to the box in the pick
orientation, then rotate in place once there.

The transit leg (waypoint 2 below) plans a free joint-space goal
(``motion.plan_to_pose``) rather than an axis-locked ``motion.plan_linear``.
That choice is independent of the reordering above and is not this leg's
fix to make: ``motion.plan_linear`` was measured here across five separate
runs (``outputs/ten_syringes-linear-transit-v1`` through ``-v4b``) and every
one executed a real, repeatable climb to z=1.49-1.505 m, 390-570 mm above
either endpoint, regardless of a plain diagonal, a split vertical-lift-then
-carry, or either of two collision-world fixes tried (a phantom-obstacle
clearing cylinder, a taller ESDF ceiling -- neither moved the number). FK-ing
cuRobo's own planned joint waypoints (not the executed trajectory) shows the
*plan* itself only bows to z=1.33 m; the further climb to 1.5 m opens up
between that plan and its *execution*, in the feed-forward joint streaming
this repo does not own (``gap.connector.core.stream_trajectory`` /
``env.stream_joint_trajectory``, one physics tick per waypoint). Even
``motion.plan_to_pose`` is not fully clean of this -- six repeated runs of
the unchanged original code split 3 flat (z stays 0.83-1.10 m) to 3 spiking
the same way -- but it spikes less often (3/6 against plan_linear's 4/4), so
it remains the mode used here until the execution-side gap above has an
actual fix.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    reorientation_plan: dict[str, Any]


def _rotation(pose: dict[str, Any]) -> Rotation:
    q = pose["rotation"]
    return Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]])


def _pose(position: np.ndarray, rotation: Rotation) -> dict[str, Any]:
    q = rotation.as_quat()
    return {"position": dict(zip(("x", "y", "z"), map(float, position))),
            "rotation": {"w": float(q[3]), "x": float(q[0]),
                         "y": float(q[1]), "z": float(q[2])}}


def _vec(value: dict[str, float]) -> np.ndarray:
    return np.array([value[key] for key in ("x", "y", "z")], dtype=np.float64)


def run(ctx: NodeContext, held_feature_in_tcp: dict[str, Any],
        fixture_feature: dict[str, Any], relation: str,
        world_config: dict[str, Any], attached_object: dict[str, Any],
        support_normal: dict[str, float] | None = None,
        approach_clearance_m: float = 0.08) -> Output:
    if relation not in {"shaft_into_aperture", "tip_through_aperture", "insert_through",
                        "loop_over_shaft", "feature_to_fixture"}:
        raise ValueError(f"unsupported feature relation {relation!r}")
    axis_value = fixture_feature.get("axis")
    if not axis_value:
        raise ValueError("fixture feature has no directed axis")
    fixture_pose = fixture_feature["pose"]
    axis = _vec(axis_value); axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    normal = _vec(support_normal or {"x": 0.0, "y": 0.0, "z": 1.0})
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)

    ee = ctx.tool("robot.get_ee_pose")["pose"]
    current_rotation = _rotation(ee)
    feature_rotation = _rotation(held_feature_in_tcp)
    feature_offset = _vec(held_feature_in_tcp["position"])
    current_axis = current_rotation.apply(feature_rotation.apply([0.0, 0.0, 1.0]))
    current_axis /= max(float(np.linalg.norm(current_axis)), 1.0e-12)
    correction, _ = Rotation.align_vectors([axis], [current_axis])
    aligned = correction * current_rotation

    fixture_center = _vec(fixture_pose["position"])
    feature_target = fixture_center - max(0.02, float(approach_clearance_m)) * axis
    candidates = []
    unchecked = []
    for angle in np.deg2rad([0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0]):
        rotation = Rotation.from_rotvec(axis * float(angle)) * aligned
        hand_position = feature_target - rotation.apply(feature_offset)
        candidate = _pose(hand_position, rotation)
        # Geodesic TCP rotation is the stable, robot-independent score. It
        # prevents symmetry from producing an arbitrary large wrist turn.
        turn = float((rotation * current_rotation.inv()).magnitude())
        unchecked.append((turn, abs(float(angle)), candidate, rotation))
        try:
            check = ctx.tool("motion.plan_joint", pose=candidate, orientation="lock")
        except Exception:
            continue
        if not check.get("planned"):
            continue
        if float(check.get("position_error_m", 0.0)) > 0.006:
            continue
        if float(check.get("rotation_error_rad", 0.0)) > np.deg2rad(4.0):
            continue
        candidates.append((turn, abs(float(angle)), candidate, rotation))
    if not candidates:
        # Not fatal any more. This pose only fixes where the carry ends; the
        # wrist yaw is chosen later, with cuRobo, by execute_reorientation.py's
        # turn search. Sweep s5's t02 aborted here when the Newton IK member
        # (the lenient probe above) rejected all eight headings.
        candidates = unchecked
    _, _, transit_pose, goal_rotation = min(candidates, key=lambda item: item[:2])

    def sphere_extent(sphere: dict[str, Any]) -> float:
        center = sphere.get("center") or {}
        offset = (np.asarray(center, dtype=float).reshape(3) if isinstance(center, (list, tuple))
                  else np.array([float(center.get(k, 0.0)) for k in ("x", "y", "z")]))
        return float(np.linalg.norm(offset)) + float(sphere.get("radius", 0.0))
    extents = [sphere_extent(s) for s in attached_object.get("spheres", [])]
    escape_distance = max(0.04, max(extents, default=0.025) + 0.015)
    current_position = _vec(ee["position"])
    escape_position = current_position + escape_distance * normal
    # Same hand position transit_pose already computed for the box, held at
    # the pick orientation rather than goal_rotation -- the carry happens
    # before the rotation, matching the reference's own order (see module
    # docstring). Reusing transit_pose's position rather than recomputing it
    # from current_rotation keeps this a same-place turn in the waypoint
    # below, not a second lateral move disguised as a rotation.
    carry_position = _vec(transit_pose["position"])
    carry_pose = _pose(carry_position, current_rotation)
    return {"reorientation_plan": {
        "time_scale": 2.0,
        "waypoints": [
            {"pose": _pose(escape_position, current_rotation),
             "mode": "contact_transition", "cartesian": True},
            # Carry to the box first, still in the pick orientation --
            # free joint-space, not an axis-locked plan_linear; see the
            # module docstring for the measurements behind that call.
            {"pose": carry_pose, "mode": "planned_joint", "cartesian": False},
            # Then rotate upright in place, next to the box. carry_pose and
            # transit_pose share the same xyz -- hold_position keeps the
            # solver from wandering position while it searches orientation,
            # rather than re-resolving the whole arm into an unrelated elbow
            # configuration for what is actually a pure in-place turn. Without
            # this, the free re-solve measurably lands somewhere the very
            # next node (insert_sg.execute's final engagement stroke) cannot
            # always plan from -- one joint alone landed 35 deg away from the
            # equivalent solve after the old (rotate-then-carry) ordering,
            # and cost 4/4 fresh runs the insertion outright.
            #
            # The turn's wrist yaw is *not* final here. The smallest-turn
            # candidate above is only checked with the lenient Newton IK at
            # this transit height; on tray_clutter_t03 it left the insertion
            # strokes 2-6 mRad from a joint limit and cuRobo refused them
            # (tools/sharps_placement_pretest.py: 6% of tray grasps get a
            # comfortable yaw from this rule, while yaws +110..+160 have
            # 0.4-0.7 rad of margin). execute_reorientation.py searches the yaw
            # live from the carry's end with the planners that fly it, using
            # this geometry; transit_pose stays the fallback.
            {"pose": transit_pose, "mode": "planned_joint", "cartesian": False,
             "hold_position": True,
             "turn_search": {
                 "fixture_center": [float(v) for v in fixture_center],
                 "axis": [float(v) for v in axis],
                 "feature_offset": [float(v) for v in feature_offset],
                 "feature_rotation": dict(held_feature_in_tcp["rotation"]),
                 "transit_clearance_m": max(0.02, float(approach_clearance_m)),
                 # plan_visual_insert.py's own stroke: 20 mm above the
                 # aperture plane, then 10 mm phases down to it.
                 "insert_depths_m": [-0.020, -0.010, 0.0],
             }},
        ],
        "world_config": world_config,
        "attached_object": attached_object,
    }}
