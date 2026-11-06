"""Localize the syringe and its directed insertion feature from overhead RGB-D."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy import ndimage
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    target_obb: dict[str, Any]
    target_mask: np.ndarray
    target_cloud: dict[str, np.ndarray]
    insertion_pose: dict[str, Any]
    cap_center: dict[str, float]
    functional_feature: dict[str, Any]
    support_z: float
    jaw_clearance_m: float


def _overhead(observation):
    return next(c for c in observation["cameras"] if c.get("name") == "overhead")


def _mask(ctx, image, query, threshold, optional=False):
    result = ctx.tool("sam3.segment_text", image=image, query=query, max_results=0)
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
    result = ctx.tool("sam3.segment_text", image=image, query=query, max_results=0)
    pairs = [
        (np.asarray(m, dtype=np.uint8), float(s))
        for m, s in zip(result.get("masks") or [], result.get("scores") or [], strict=False)
        if float(s) >= RESCUE_THRESHOLD
    ]
    if not pairs:
        raise ValueError(f"could not perceive {query!r}")
    return pairs


AWAY_FROM_APERTURE_M = 0.12
"""Exclude a syringe already released into the sharps box.

The all-syringe graph loops back through target perception after each release.
A syringe standing in the opening remains highly visible and often has the
strongest red-cap signal, but it is no longer a source target.
"""


def _red_pixels(rgb, mask) -> int:
    """How many of a mask's pixels are red-cap-coloured -- the goal syringe's signature.

    The scene's one deliberately red thing is the goal syringe's needle cap
    (the untextured spare has none), so red pixels inside a candidate mask are
    direct evidence of identity that no SAM score can fake. Plain channel
    arithmetic rather than another SAM call: the cap query itself is what
    failed in the runs this exists for.

    A *count*, not a fraction: ranking by the red fraction preferred SAM's
    low-score sliver around the cap alone over the whole syringe that also
    contains it. ``tray_clutter_t03``: a 165-px, score-0.06 cap-end fragment
    (18 red px, 10.9 %) beat the 1764-px, score-0.79 syringe (36 red px, 2.0 %),
    so the target cloud spanned 51 mm and the grasp landed 9 mm from the tip.
    """
    pixels = np.asarray(rgb, dtype=np.int16)[np.asarray(mask) > 0]
    if not len(pixels):
        return 0
    red = (pixels[:, 0] > 110) & (pixels[:, 0] > pixels[:, 1] + 40) & (pixels[:, 0] > pixels[:, 2] + 40)
    return int(red.sum())


MIN_CAP_PIXELS = 8
"""Red pixels a candidate must hold to count as containing the needle cap."""
RING_OUTER_PX = 14
"""How far past the target mask [px] the support-surface ring reaches."""
RING_GAP_PX = 4
"""How far the ring stays off every detected syringe [px], so it samples the
surface they rest on rather than a neighbour's barrel or a mask's soft edge."""


def _support_z(ctx, camera, target_mask, occupied, points) -> float:
    """The height of the surface the target rests on, from depth beside it.

    The target's own cloud is a poor floor estimate: transparent-looking
    barrels and mask edges mix in floor pixels, which drags an OBB bottom (and
    the grasp height derived from it) down onto or below the floor. A thin ring
    just outside the target mask -- clear of every detected syringe -- sees the
    tray floor itself. Falls back to the cloud's low percentile when the ring
    is too small to trust.
    """
    target = np.asarray(target_mask) > 0
    ring = ndimage.binary_dilation(target, iterations=RING_OUTER_PX) & ~ndimage.binary_dilation(
        occupied | target, iterations=RING_GAP_PX
    )
    if int(ring.sum()) >= 50:
        z = _points(ctx, camera, ring.astype(np.uint8))[:, 2]
        if len(z) >= 50:
            return float(np.median(z))
    return float(np.percentile(points[:, 2], 5))


MIN_SYRINGE_SPAN_M = 0.08
"""A candidate shorter than this is a fragment, not a syringe: the whole barrel is
seen at 113-117 mm. On the second pick of sweep s2's t02, t03, t04 and t15, the
kept mask was SAM's sliver around a red cap. Its cloud was 28-43 mm long, with
its centre on the cap, so the grasp went to the needle end instead of the centre
of mass."""
MAX_SYRINGE_SPAN_M = 0.20
MAX_SYRINGE_WIDTH_M = 0.045
"""A candidate's cloud longer or wider than these is not one syringe. The 150 mm
syringe is seen as 113-117 mm; its flange is 23.6 mm wide. In a packed tray, SAM
also returns masks spanning several neighbouring syringes. On
``tray_clutter_t16``'s second pick, one such mask held the red cap and scored
0.84, and its 250 x 155 mm OBB put the grasp between syringes."""


MIN_MASK_PIXELS = 500
"""A syringe on the tray covers 1300-1700 overhead pixels. Sweep s4's ``t08`` replay
had a 61-pixel line on the bench past the tray, 80 mm long, that passed the span
gate. It ranked first because nothing stood near it."""


def _syringe_sized(points) -> bool:
    centered = points - np.median(points, axis=0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    along = centered @ basis[0]
    across = centered - np.outer(along, basis[0])
    span = float(np.percentile(along, 97) - np.percentile(along, 3))
    width = 2.0 * float(np.percentile(np.linalg.norm(across, axis=1), 90))
    return MIN_SYRINGE_SPAN_M <= span <= MAX_SYRINGE_SPAN_M and width <= MAX_SYRINGE_WIDTH_M


JAW_CLEARANCE_FREE_M = 0.021
"""Distance across the jaw to the nearest raised surface (neighbour barrel or tray
wall) at which a pre-closed Robotiq still fits. It allows a 21 mm jaw:
2 x (21 - 8.5 band - 2 margin). See compute_grasp.py."""
JAW_CLEARANCE_ENOUGH_M = 0.045
"""Past this, more room stops outranking the other preferences."""
STATION_HALF_LENGTH_M = 0.035
"""Half the stretch of barrel around the station whose surroundings the fingers can touch."""
NO_NEIGHBOUR_CLEARANCE_M = 0.08
CLEARANCE_RING_PX = 60
CLEARANCE_RING_GAP_PX = 6
"""The depth ring the clearance is read from: 6-60 px outside the mask. The overhead
view is about 1 mm per pixel at tray height."""
OWN_BODY_HALF_WIDTH_M = 0.013
"""Raised points this close to the candidate's axis are its own body: SAM's mask
stops short of the rounded barrel edge and the 23.6 mm flange. An offline replay
of sweep s4's ``t08`` observation read 8.5 mm of clearance for ``syringe_0``,
whose nearest neighbour lies 57.6 mm across its axis. A neighbour this close
could not be straddled by any jaw width anyway."""
PROTRUSION_M = 0.004
"""Height above the ring's floor that counts as something a fingertip lands on. A
lying barrel stands 11 mm proud and the tray's end wall 9 mm."""
CAP_MAX_LATERAL_M = 0.010
CAP_MIN_ALONG_M = 0.025
"""A candidate's red pixels must lie on its own axis and toward one end to count as
its needle cap. Sweep s3 ``t12``: a syringe-sized mask along the tray's end wall,
22.7 mm beside ``syringe_0``, held that syringe's cap pixels. It ranked first on
clearance, since nothing lay on its wall side, and the grasp landed on the wall."""


def _red_mask(rgb) -> np.ndarray:
    pixels = np.asarray(rgb, dtype=np.int16)
    return (pixels[..., 0] > 110) & (pixels[..., 0] > pixels[..., 1] + 40) & (pixels[..., 0] > pixels[..., 2] + 40)


def _distinct(scored):
    """SAM returns overlapping masks for one syringe; keep the best-scored of each."""
    kept = []
    for entry in sorted(scored, key=lambda e: e[1], reverse=True):
        mask = np.asarray(entry[0]) > 0
        if all(np.logical_and(mask, np.asarray(k[0]) > 0).sum() < 0.3 * min(mask.sum(), (np.asarray(k[0]) > 0).sum())
               for k in kept):
            kept.append(entry)
    return kept


def _horizontal_axis(points):
    center = np.median(points, axis=0)
    _, _, basis = np.linalg.svd(points - center, full_matrices=False)
    axis = basis[0][:2] / max(float(np.linalg.norm(basis[0][:2])), 1.0e-9)
    return center, axis


def _cap_on_axis(ctx, camera, mask, points) -> bool:
    red = _red_mask(camera["rgb"]) & (np.asarray(mask) > 0)
    if int(red.sum()) < MIN_CAP_PIXELS:
        return False
    cap = _points(ctx, camera, red.astype(np.uint8))
    if len(cap) < 4:
        return False
    center, axis = _horizontal_axis(points)
    delta = np.median(cap, axis=0)[:2] - center[:2]
    return abs(float(delta @ [-axis[1], axis[0]])) <= CAP_MAX_LATERAL_M and abs(float(delta @ axis)) >= CAP_MIN_ALONG_M


FLANGE_TO_CAP_M = 0.088
"""Along-axis distance from the finger flange to the red cap's centre. From the
mesh survey, the flange is 23.6 mm wide at +45 mm on the body's y axis, and the
cap centre sits 43 mm on the needle side of the centre of mass (``compute_grasp``)."""
FLANGE_WIDTH_RATIO_MIN = 1.5
"""The flange must be at least this much wider than the barrel's median width to be trusted."""


def _red_cap_point(ctx, camera, mask, points):
    """The needle cap's world point from red pixels on the target's own axis, or None."""
    red = _red_mask(camera["rgb"]) & (np.asarray(mask) > 0)
    if int(red.sum()) < MIN_CAP_PIXELS:
        return None
    cap = _points(ctx, camera, red.astype(np.uint8))
    if len(cap) < 4:
        return None
    center, axis = _horizontal_axis(points)
    point = np.median(cap, axis=0)
    delta = point[:2] - center[:2]
    if abs(float(delta @ [-axis[1], axis[0]])) > CAP_MAX_LATERAL_M or abs(float(delta @ axis)) < CAP_MIN_ALONG_M:
        return None
    return point


def _cap_from_flange(points):
    """Where the needle cap is, from the flange: the widest stretch of the cloud."""
    center = np.median(points, axis=0)
    _, _, basis = np.linalg.svd(points - center, full_matrices=False)
    axis = basis[0]
    along = (points - center) @ axis
    lateral = np.linalg.norm((points - center) - np.outer(along, axis), axis=1)
    edges = np.linspace(np.percentile(along, 3), np.percentile(along, 97), 11)
    widths, mids = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        inside = (along >= lo) & (along < hi)
        if inside.sum() >= 8:
            widths.append(2.0 * float(np.percentile(lateral[inside], 90)))
            mids.append(0.5 * (lo + hi))
    if len(widths) < 5:
        return None
    widths = np.asarray(widths)
    flange = int(np.argmax(widths))
    if widths[flange] < FLANGE_WIDTH_RATIO_MIN * float(np.median(widths)):
        return None
    toward_flange = axis if mids[flange] > 0 else -axis
    return center + toward_flange * mids[flange] * np.sign(mids[flange]) - toward_flange * FLANGE_TO_CAP_M


MIN_HEIGHT_ABOVE_FLOOR_M = 0.005
"""A lying syringe's top stands 11 mm proud of what it rests on. Sweep s7's t09
(and s6's t23) grasped a 170 x 40 mm "syringe" in the sharps box's shadow on the
bench. Its cloud was 4 mm thick at bench height, and the hand drove into the box
beside it."""


def _raised_above_floor(ctx, camera, mask, points) -> bool:
    distance = ndimage.distance_transform_edt(~(np.asarray(mask) > 0))
    ring = (distance > CLEARANCE_RING_GAP_PX) & (distance <= CLEARANCE_RING_PX)
    around = _points(ctx, camera, ring.astype(np.uint8))
    if len(around) < 50:
        return True
    floor = float(np.percentile(around[:, 2], 10))
    return float(np.percentile(points[:, 2], 90)) - floor >= MIN_HEIGHT_ABOVE_FLOOR_M


def _jaw_clearance(ctx, camera, mask, points) -> float:
    """Free distance [m] across the jaw from the barrel's axis to the nearest surface
    standing proud of the floor (a neighbour or the tray wall), within the stretch
    of barrel the fingers straddle. Read from depth, so undetected neighbours and
    walls count too."""
    distance = ndimage.distance_transform_edt(~(np.asarray(mask) > 0))
    ring = (distance > CLEARANCE_RING_GAP_PX) & (distance <= CLEARANCE_RING_PX)
    around = _points(ctx, camera, ring.astype(np.uint8))
    if len(around) < 50:
        return NO_NEIGHBOUR_CLEARANCE_M
    floor = float(np.percentile(around[:, 2], 10))
    raised = around[around[:, 2] > floor + PROTRUSION_M]
    center, axis = _horizontal_axis(points)
    delta = raised[:, :2] - center[:2]
    across = np.abs(delta @ [-axis[1], axis[0]])
    near = (np.abs(delta @ axis) <= STATION_HALF_LENGTH_M) & (across > OWN_BODY_HALF_WIDTH_M)
    if not near.any():
        return NO_NEIGHBOUR_CLEARANCE_M
    return float(np.min(across[near]))


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
        aperture: dict[str, float], rim_z: float,
        object_description: str = "medical syringe",
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
    aperture_xyz = np.array([aperture[k] for k in ("x", "y", "z")], dtype=np.float64)
    if cap_hint is not None and np.linalg.norm(cap_hint[:2] - aperture_xyz[:2]) <= AWAY_FROM_APERTURE_M:
        cap_hint = None

    candidates = _candidate_masks(ctx, camera["rgb"], object_description, 0.08)
    scored = []
    occupied = None
    for mask, score in candidates:
        if score >= 0.08:
            # Every confident detection -- merged ones included -- is somewhere
            # the support ring must not sample.
            occupied = (np.asarray(mask) > 0) if occupied is None else (occupied | (np.asarray(mask) > 0))
        if int((np.asarray(mask) > 0).sum()) < MIN_MASK_PIXELS:
            continue
        cloud = _points(ctx, camera, mask)
        if len(cloud) < 30:
            continue
        if not _syringe_sized(cloud):
            continue
        center = np.median(cloud, axis=0)
        if np.linalg.norm(center[:2] - aperture_xyz[:2]) <= AWAY_FROM_APERTURE_M:
            continue
        # Source syringes are below the box rim. Keep a small tolerance for
        # transparent-surface depth noise, but reject detections above it.
        if center[2] > float(rim_z) + 0.02:
            continue
        red = _red_pixels(camera["rgb"], mask)
        scored.append((mask, score, red, cloud))
    if not scored:
        raise ValueError(f"could not perceive an unplaced {object_description!r}")
    # Every syringe in the tray is a goal, so which one to take first is a
    # grasp-feasibility choice. A mask holding red pixels off its own axis is
    # not a syringe (see CAP_MAX_LATERAL_M) and goes last. A mask holding none
    # is neutral: SAM's syringe masks often stop short of the thin cap, and in
    # the s4 replay only 1 of 5 real syringes carried its cap. Then take the one
    # with the most room across the jaw. The Robotiq's fingertip band reaches
    # ~8.5 mm past each side of its jaw opening, and a neighbour or wall closer
    # than that is landed on (tray_clutter_t16/t17: the hand stopped 7-9 mm high
    # on the next barrel). Then SAM's score.
    syringes = _distinct(scored)
    if not any(entry[2] >= MIN_CAP_PIXELS or entry[1] >= 0.08 for entry in syringes):
        raise ValueError(f"could not perceive {object_description!r}")
    ranked = []
    syringes = [entry for entry in syringes if _raised_above_floor(ctx, camera, entry[0], entry[3])]
    if not syringes:
        raise ValueError(f"could not perceive an unplaced {object_description!r} standing above its support")
    for index, (mask, score, red, cloud) in enumerate(syringes):
        clearance = _jaw_clearance(ctx, camera, mask, cloud)
        not_vetoed = red < MIN_CAP_PIXELS or _cap_on_axis(ctx, camera, mask, cloud)
        ranked.append((not_vetoed, clearance >= JAW_CLEARANCE_FREE_M,
                       min(clearance, JAW_CLEARANCE_ENOUGH_M), score, index, clearance))
    best_rank = max(ranked)
    target_mask, points = syringes[best_rank[4]][0], syringes[best_rank[4]][3]
    jaw_clearance = float(best_rank[5])
    if len(points) < 30:
        raise ValueError("syringe mask contains too few depth points")
    obb = ctx.tool(
        "geometry.filter_and_compute_obb", points={"points": points.astype(np.float32)}
    )["obb"]
    shaft_center = np.median(points, axis=0)
    _, _, basis = np.linalg.svd(points - shaft_center, full_matrices=False)
    shaft_axis = basis[0]
    if cap_hint is not None:
        delta = cap_hint - shaft_center
        lateral = np.linalg.norm(delta - shaft_axis * float(delta @ shaft_axis))
        nearest = np.min(np.linalg.norm(points - cap_hint, axis=1))
        if lateral > 0.012 or nearest > 0.025:
            cap_hint = None
    # Which end is the needle decides which end goes into the box first. A
    # syringe posted plunger-first stands on its flat 13.6 mm thumb press under
    # the hole, and the next one jams on top of it (sweep s5 t05/t06: posted
    # plunger-down, then the second syringe was stuck 10 mm above the rim).
    # The narrow-tail guess cannot tell the needle hub from the plunger rod in
    # overhead depth, so it comes last.
    cap = _red_cap_point(ctx, camera, target_mask, points)
    if cap is None:
        cap = cap_hint
    if cap is None:
        cap = _cap_from_flange(points)
    if cap is None:
        cap = _narrow_endpoint(points)
    center = np.array([obb["center"][k] for k in ("x", "y", "z")], dtype=np.float64)
    axis = shaft_axis if float(shaft_axis @ (cap - center)) > 0 else -shaft_axis
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
    if occupied is None:
        occupied = np.zeros(np.asarray(target_mask).shape, dtype=bool)
    support_z = _support_z(ctx, camera, target_mask, occupied, points)
    return {
        "target_obb": obb,
        "support_z": support_z,
        "jaw_clearance_m": jaw_clearance,
        "target_mask": target_mask,
        "target_cloud": {"points": points.astype(np.float32)},
        "cap_center": {"x": float(cap[0]), "y": float(cap[1]), "z": float(cap[2])},
        "insertion_pose": insertion_pose,
        "functional_feature": {"kind": feature_type, "pose": insertion_pose,
            "axis": {"x": float(axis[0]), "y": float(axis[1]), "z": float(axis[2])},
            "radius_outer": max(radius, 1.0e-4), "confidence": 1.0,
            "description": direction_marker_description},
    }

