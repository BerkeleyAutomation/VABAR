"""Lift, reobserve the syringe from the wrist, and reject an empty grasp."""

from typing import TypedDict

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

QUERIES = ("medical syringe", "syringe")
"""Asked in this order on each camera; every returned mask is a candidate."""

MASKS_PER_QUERY = 0
"""How many masks to consider per query, not just the best one (0 = every mask).

Every mask, not the top 8: with 6-8 syringes still in the tray, the lifted wrist
sees 6+ tray syringes scoring 0.8-0.93. On ``tray_clutter_t00``/``t01``/``t07``
(sweep s2, 2026-09-13), syringe_0 was lifted to z 0.99 while the 8 returned masks
were all tray syringes (nearest 215 mm from the hand), so a held syringe was
declared dropped.

From the lifted wrist the tray's syringes are large, well-lit and score higher
than the one held between the fingers, which is foreshortened and seen end-on.
``tray_clutter_t03`` (tiled, 2026-09-13): with the syringe held 1.4 cm from the
TCP at +19.9 cm, "syringe" returned three tray syringes (scores 0.052/0.050/
0.041, 217-281 mm from the hand) and the held one only 4th (0.040, 23 mm from
the hand); "medical syringe" ranked it 3rd (0.32). Checking only masks[0]
rejected a successful grasp, and the abort's open_gripper dropped it."""

MAX_SYRINGE_SPAN_M = 0.25
"""A candidate's point cloud longer than this is not one syringe (a 150 mm
barrel plus depth noise). Keeps the arm itself -- a 395k-px mask 176 mm from
the hand in the same view -- from passing as the held object."""

MAX_SYRINGE_WIDTH_M = 0.06
"""A candidate wider than this across its main axis is not one syringe."""

MAX_HAND_DISTANCE_M = 0.20
MIN_ABOVE_TABLE_M = 0.03


def _points(ctx, camera, mask):
    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=np.asarray(mask, dtype=np.uint8),
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
    )["points"]
    return np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)


def _syringe_sized(points) -> bool:
    centered = points - np.median(points, axis=0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    along = centered @ basis[0]
    across = centered - np.outer(along, basis[0])
    span = float(np.percentile(along, 97) - np.percentile(along, 3))
    width = 2.0 * float(np.percentile(np.linalg.norm(across, axis=1), 90))
    return span <= MAX_SYRINGE_SPAN_M and width <= MAX_SYRINGE_WIDTH_M


def run(ctx: NodeContext) -> Output:
    lifted = ctx.tool("robot.get_ee_pose")["pose"]
    observation = ctx.tool("robot.get_observation")
    wrist = next(
        (camera for camera in observation["cameras"] if "eye_in_hand" in camera.get("name", "")),
        None,
    )
    if wrist is None:
        raise RuntimeError("syringe grasp verification requires an eye-in-hand RGB-D camera")

    cameras = [wrist]
    overhead = next(
        (camera for camera in observation["cameras"] if camera.get("name") == "overhead"),
        None,
    )
    if overhead is not None:
        cameras.append(overhead)
    hand = np.array([lifted["position"][key] for key in ("x", "y", "z")], dtype=np.float64)
    surface_z = float(ctx.tool("robot.describe_workspace")["surface_z"])

    nearest = None  # (hand_distance, above_table, center, point_count, camera name)
    seen_any = False
    for camera in cameras:
        for query in QUERIES:
            detected = ctx.tool(
                "sam3.segment_text", image=camera["rgb"], query=query, max_results=MASKS_PER_QUERY,
            )
            for mask, score in zip(detected.get("masks") or [], detected.get("scores") or [], strict=False):
                if float(score) < HELD_OBJECT_SCORE_MIN:
                    continue
                points = _points(ctx, camera, mask)
                if len(points) < 20 or not _syringe_sized(points):
                    continue
                seen_any = True
                center = np.median(points, axis=0)
                candidate = (
                    float(np.linalg.norm(center - hand)),
                    float(center[2] - surface_z),
                    center,
                    int(len(points)),
                    str(camera.get("name", "unknown")),
                )
                if nearest is None or candidate[0] < nearest[0]:
                    nearest = candidate
            if nearest is not None and nearest[0] <= MAX_HAND_DISTANCE_M and nearest[1] >= MIN_ABOVE_TABLE_M:
                hand_distance, _, center, count, name = nearest
                return {
                    "verified": True,
                    "observed_center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
                    "point_count": count,
                    "camera": name,
                }
    if not seen_any or nearest is None:
        raise RuntimeError("neither wrist nor overhead camera sees a syringe after lifting")
    raise RuntimeError(
        "wrist reobservation shows the syringe was not lifted with the hand "
        f"(nearest syringe: above table {nearest[1] * 1000.0:.1f} mm, "
        f"hand distance {nearest[0] * 1000.0:.1f} mm)"
    )
