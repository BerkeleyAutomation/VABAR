"""Localize the syringe and its directed insertion feature from overhead RGB-D."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    target_obb: dict[str, Any]
    target_mask: np.ndarray
    target_cloud: dict[str, np.ndarray]
    insertion_pose: dict[str, Any]
    cap_center: dict[str, float]
    functional_feature: dict[str, Any]


def _overhead(observation):
    return next(c for c in observation["cameras"] if c.get("name") == "overhead")


def _mask(ctx, image, query, threshold, optional=False):
    result = ctx.tool("sam3.segment_text", image=image, query=query, max_results=3)
    if not result.get("masks") or float(result["scores"][0]) < threshold:
        if optional:
            return None
        raise ValueError(f"could not perceive {query!r}")
    return np.asarray(result["masks"][0], dtype=np.uint8)


RESCUE_THRESHOLD = 0.02
"""Floor for a candidate that still gets the red-cap test. SAM's score for a
thin syringe on the textured mat collapses to ~0.04 when the parked wrist
encroaches on the tray -- measured across the 20-layout sweep, where 7 of 20
episodes aborted at the 0.08 gate with the syringe plainly in frame. A mask
that *contains the red needle cap's pixels* is the instruction's syringe
whatever SAM scored it, so low-score candidates are kept alive long enough to
take that test rather than refused on score alone."""


def _candidate_masks(ctx, image, query, threshold):
    """Every mask above the rescue floor, best first, with its score.

    The caller applies ``threshold`` only after the red-cap evidence has had
    its chance; see :data:`RESCUE_THRESHOLD`.
    """
    result = ctx.tool("sam3.segment_text", image=image, query=query, max_results=3)
    pairs = [
        (np.asarray(m, dtype=np.uint8), float(s))
        for m, s in zip(result.get("masks") or [], result.get("scores") or [], strict=False)
        if float(s) >= RESCUE_THRESHOLD
    ]
    if not pairs:
        raise ValueError(f"could not perceive {query!r}")
    return pairs


def _red_fraction(rgb, mask) -> float:
    """How much of a mask is red-cap-coloured -- the goal syringe's signature.

    The scene's one deliberately red thing is the goal syringe's needle cap
    (the untextured spare has none), so redness inside a candidate mask is
    direct evidence of identity that no SAM score can fake. Plain channel
    arithmetic rather than another SAM call: the cap query itself is what
    failed in the runs this exists for.
    """
    pixels = np.asarray(rgb, dtype=np.int16)[np.asarray(mask) > 0]
    if not len(pixels):
        return 0.0
    red = (pixels[:, 0] > 110) & (pixels[:, 0] > pixels[:, 1] + 40) & (pixels[:, 0] > pixels[:, 2] + 40)
    return float(red.mean())


def _points(ctx, camera, mask):
    cloud = ctx.tool(
        "geometry.mask_to_world_points", mask=mask, depth=camera["depth"],
        intrinsics=camera["intrinsics"], camera_pose=camera["pose"],
    )["points"]
    return np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)


def _narrow_endpoint(points):
    origin = np.median(points, axis=0)
    _, _, basis = np.linalg.svd(points - origin, full_matrices=False)
    axis = basis[0]
    along = (points - origin) @ axis
    tails = [points[along <= np.percentile(along, 15)], points[along >= np.percentile(along, 85)]]
    widths = []
    for tail in tails:
        centered = tail - np.median(tail, axis=0)
        perpendicular = centered - np.outer(centered @ axis, axis)
        widths.append(float(np.percentile(np.linalg.norm(perpendicular, axis=1), 90)))
    return np.median(tails[int(np.argmin(widths))], axis=0)


def run(ctx: NodeContext, observation: dict[str, Any],
        object_description: str = "syringe",
        direction_marker_description: str = "red needle cap of syringe",
        feature_type: str = "tip") -> Output:
    camera = _overhead(observation)
    # The direction marker doubles as the *identity* marker: a scene may hold
    # lookalike syringes (the packing tray's spares), and the instruction's
    # syringe is the one wearing the red needle cap. Identity is decided by
    # **red pixels inside the candidate mask** first -- direct evidence no
    # detection score can fake -- with the perceived cap's location as the
    # tie-breaker. Score order alone flipped to a spare when the room dressing
    # changed the background contrast, and the cap *query* itself dips under
    # its gate in exactly the layouts where the target is hardest to rank, so
    # neither is trusted on its own (measured: 5 of 20 sweep episodes posted
    # the spare, 7 more aborted on score, zero of either mode with this).
    cap_mask = _mask(ctx, camera["rgb"], direction_marker_description, 0.06, optional=True)
    cap_points = _points(ctx, camera, cap_mask) if cap_mask is not None else np.empty((0, 3))
    cap_hint = np.median(cap_points, axis=0) if len(cap_points) >= 8 else None

    candidates = _candidate_masks(ctx, camera["rgb"], object_description, 0.08)
    scored = []
    for mask, score in candidates:
        redness = _red_fraction(camera["rgb"], mask)
        scored.append((mask, score, redness))
    # Red pixels establish identity, not how much of the syringe was found:
    # a cap-only fragment has a higher red fraction than the whole object.
    marked = [entry for entry in scored if entry[2] >= 0.01]
    target_mask, points = None, None
    if marked:
        # The cap holds a few percent of the syringe's own pixels, so 1% red
        # inside the mask is decisive -- accept this candidate whatever its
        # detection score.
        target_mask = max(marked, key=lambda entry: entry[1])[0]
        points = _points(ctx, camera, target_mask)
        if len(points) < 30:
            target_mask, points = None, None
    if points is None and cap_hint is not None:
        best = None
        for mask, score, _ in scored:
            cloud = _points(ctx, camera, mask)
            if len(cloud) < 30:
                continue
            gap = float(np.linalg.norm(np.median(cloud, axis=0)[:2] - cap_hint[:2]))
            if best is None or gap < best:
                best, target_mask, points = gap, mask, cloud
    if points is None:
        # No red evidence and no cap: only now does the score gate apply.
        strong = [(m, s) for m, s, _ in scored if s >= 0.08]
        if not strong:
            raise ValueError(f"could not perceive {object_description!r}")
        target_mask = strong[0][0]
        points = _points(ctx, camera, target_mask)
    if len(points) < 30:
        raise ValueError("syringe mask contains too few depth points")
    obb = ctx.tool(
        "geometry.filter_and_compute_obb", points={"points": points.astype(np.float32)}
    )["obb"]
    cap = cap_hint if cap_hint is not None else _narrow_endpoint(points)
    center = np.array([obb["center"][k] for k in ("x", "y", "z")], dtype=np.float64)
    axis = cap - center
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    along = (points - center) @ axis
    tip = np.median(points[along >= np.percentile(along, 98)], axis=0)
    reference = np.array([0.0, 0.0, 1.0]) if abs(float(axis[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(reference, axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(axis, x_axis)
    quat = Rotation.from_matrix(np.column_stack((x_axis, y_axis, axis))).as_quat()
    insertion_pose = {
        "position": {"x": float(tip[0]), "y": float(tip[1]), "z": float(tip[2])},
        "rotation": {"w": float(quat[3]), "x": float(quat[0]), "y": float(quat[1]), "z": float(quat[2])},
    }
    centered = points - tip
    radial = centered - np.outer(centered @ axis, axis)
    radius = float(np.percentile(np.linalg.norm(radial, axis=1), 35))
    return {
        "target_obb": obb,
        "target_mask": target_mask,
        "target_cloud": {"points": points.astype(np.float32)},
        "cap_center": {"x": float(cap[0]), "y": float(cap[1]), "z": float(cap[2])},
        "insertion_pose": insertion_pose,
        "functional_feature": {"kind": feature_type, "pose": insertion_pose,
            "axis": {"x": float(axis[0]), "y": float(axis[1]), "z": float(axis[2])},
            "radius_outer": max(radius, 1.0e-4), "confidence": 1.0,
            "description": direction_marker_description},
    }
