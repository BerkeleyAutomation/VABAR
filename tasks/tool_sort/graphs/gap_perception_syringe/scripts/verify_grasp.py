"""Lift, reobserve the syringe from the wrist, and reject an empty grasp."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    verified: bool
    observed_center: dict[str, float]
    point_count: int
    camera: str


# The held syringe occupies only a few dozen wrist-camera pixels. Recorded
# true-positive wrist views score 0.006--0.010; the old 0.02 gate routed those
# successful grasps to abort/release. Confidence only admits a candidate here:
# metric depth, table clearance, and hand proximity still verify the grasp.
HELD_OBJECT_SCORE_MIN = 0.005


def run(ctx: NodeContext) -> Output:
    ee = ctx.tool("robot.get_ee_pose")["pose"]
    lifted = {
        "position": dict(ee["position"]),
        "rotation": dict(ee["rotation"]),
    }
    lifted["position"]["z"] = float(lifted["position"]["z"]) + 0.04
    ctx.tool("robot.go_to_pose_cartesian", pose=lifted)

    observation = ctx.tool("robot.get_observation")
    wrist = next(
        (camera for camera in observation["cameras"] if "eye_in_hand" in camera.get("name", "")),
        None,
    )
    if wrist is None:
        raise RuntimeError("syringe grasp verification requires an eye-in-hand RGB-D camera")

    candidates = [wrist]
    overhead = next(
        (camera for camera in observation["cameras"] if camera.get("name") == "overhead"),
        None,
    )
    if overhead is not None:
        candidates.append(overhead)
    camera = None
    result = None
    for candidate in candidates:
        for query in ("syringe", "red needle cap of syringe"):
            detected = ctx.tool(
                "sam3.segment_text", image=candidate["rgb"], query=query, max_results=3,
            )
            if (detected.get("masks") and detected.get("scores")
                    and float(detected["scores"][0]) >= HELD_OBJECT_SCORE_MIN):
                camera, result = candidate, detected
                break
        if camera is not None:
            break
    if camera is None or result is None:
        raise RuntimeError("neither wrist nor overhead camera sees a syringe after lifting")

    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=np.asarray(result["masks"][0], dtype=np.uint8),
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
    )["points"]
    points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
    if len(points) < 20:
        raise RuntimeError("wrist syringe mask contains too few valid depth points")

    center = np.median(points, axis=0)
    hand = np.array([lifted["position"][key] for key in ("x", "y", "z")], dtype=np.float64)
    workspace = ctx.tool("robot.describe_workspace")
    above_table = float(center[2] - workspace["surface_z"])
    hand_distance = float(np.linalg.norm(center - hand))
    if above_table < 0.03 or hand_distance > 0.20:
        raise RuntimeError(
            "wrist reobservation shows the syringe was not lifted with the hand "
            f"(above table {above_table * 1000.0:.1f} mm, hand distance {hand_distance * 1000.0:.1f} mm)"
        )

    return {
        "verified": True,
        "observed_center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "point_count": int(len(points)),
        "camera": str(camera.get("name", "unknown")),
    }
