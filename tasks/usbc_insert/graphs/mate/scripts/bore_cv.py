# SPDX-License-Identifier: Apache-2.0
"""Find a USB-C bore in a wrist frame with OpenCV, and nothing learned.

**Why this exists when the suite already has a segmenter.** SAM3 finds a
*receptacle* -- a blue object -- and the graph then infers where the hole in it
must be from the part's silhouette and its catalogue dimensions. That works and
it is measured (see `relook_vision`), but it never looks at the hole. This does:
given a frame in which the bore is actually facing the lens, it reads the
aperture itself, and needs no model of the part beyond "a bore is a recess".

**What makes the frame possible** is not this file. It is the pair of changes in
`ToolsUsbc.INSPECT_PITCH_DEG`, `INSPECT_RISE` and `INSPECT_BACK`: a station
visited with the hand EMPTY, on the mate line, pitched 20 degrees below
horizontal and lifted 50 mm so the support arm is not between the lens and the
mouth. From the shipped stations the bore is 87 degrees off the view and this
finds nothing; from that one it is head-on.

Measured against ground truth over eight drawn layouts, from the mate line:

    found            8/8
    mouth            1.16 mm mean, 1.27 worst
    bore axis        0.15 deg mean, 0.90 worst

Three things it deliberately does not use: a learned segmenter, the slot's
dimensions, and the receptacle's pose. The blue threshold is a colour, the
recess depth is a property of "a hole", and the shape test is that an aperture
is a filled rectangle nearer the middle of a face than the barrel's own step is.
"""

from __future__ import annotations

import numpy as np

BLUE_LO, BLUE_HI = (85, 40, 35), (128, 255, 255)
"""HSV bounds for the receptacle's own colour. A threshold, not a detector: the
slot shares the barrel's hue (measured 106 on both) and differs only in value,
which is why the hole is found by DEPTH below rather than by colour here."""

SLOT_BEHIND_M = 0.012
"""How far behind its own surroundings a recess has to sit to be the bore [m].

Swept against ground truth: 4 mm and 6 mm swallow the barrel's step down from
the flange and read 1300+ px of "slot"; 20 mm keeps only the deepest sliver and
loses the aperture's edges. Twelve is the middle of the range where the blob is
the aperture and nothing else -- at the true mouth pixel the floor sits 29.6 mm
back, and the step behind the flange only 5-10."""

SEARCH_PX = 34
"""How far from the face's centroid to look for the aperture [px]. At this
station the mouth lands within 20 px of frame centre on every layout drawn."""

SLOT_ASPECT = (1.3, 8.0)
MIN_FACE_PX, MIN_SLOT_PX = 600, 40
RING_PX = 7
"""How wide a ring around the aperture the mouth PLANE is measured over [px].

The mouth is on the face the plug arrives at, and that face is the ring
immediately around the hole -- not the whole silhouette. A depth percentile over
the whole blue blob lands on the flange rim, which at 30 mm of radius seen from
90 mm off axis is 12 mm nearer than the mouth plane, and everything measured
from it inherits that."""


def recess_frame(mask, depth, K, T, axis_hint):
    """Fit a visible front face and find its recessed opening in RGB-D.

    A cropped/occluded face need not have the aperture at its visible centroid.
    RANSAC separates the front from the side wall; the front's image footprint
    excludes that side wall from the subsequent recess search.
    """
    import cv2
    from scipy.ndimage import binary_fill_holes

    mask = binary_fill_holes(np.asarray(mask) > 0)
    valid = mask & np.isfinite(depth) & (depth > 1.e-4)
    v, u = np.nonzero(valid)
    if len(u) < 40:
        return None, None, "no supported recessed opening"
    z = depth[v, u]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(z)], 1)
    cloud = (rays * z[:, None]) @ T[:3, :3].T + T[:3, 3]
    axis = np.asarray(axis_hint, dtype=float)
    axis /= max(np.linalg.norm(axis), 1.e-12)
    along = cloud @ axis
    front = cloud[along < np.percentile(along, 1.) + 0.012]
    if len(front) < 40:
        return None, None, "no supported recessed opening"
    sample = front[::max(1, len(front) // 5000)]
    rng = np.random.default_rng(0)
    best, normal, origin = 0, None, None
    for _ in range(96):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length < 1.e-10:
            continue
        n /= length
        if abs(n @ axis) < 0.8:
            continue
        score = int((np.abs((sample - a) @ n) < 0.0005).sum())
        if score > best:
            best, normal, origin = score, n, a
    if normal is None or best < 40:
        return None, None, "no supported recessed opening"
    plane = np.abs((cloud - origin) @ normal) < 0.0005
    face = cloud[plane]
    origin = face.mean(0)
    normal = np.linalg.svd(face - origin, full_matrices=False)[2][2]
    if normal @ axis < 0:
        normal = -normal
    footprint = np.zeros(mask.shape, np.uint8)
    hull = cv2.convexHull(np.column_stack([u[plane], v[plane]]).astype(np.int32))
    cv2.fillConvexPoly(footprint, hull, 1)
    footprint = cv2.erode(footprint, np.ones((5, 5), np.uint8))
    residual = (cloud - origin) @ normal
    recessed = (residual > 0.002) & (residual < 0.05) & (footprint[v, u] > 0)
    holes = np.zeros(mask.shape, np.uint8)
    holes[v[recessed], u[recessed]] = 1
    count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    candidates = []
    for i in range(1, count):
        x, y, width, height, area = stats[i]
        aspect = max(width, height) / max(1, min(width, height))
        if area < 12 or not 1.2 <= aspect <= 12 or area / (width * height) < 0.3:
            continue
        pixels = labels[v, u] == i
        world_rays = rays[pixels] @ T[:3, :3].T
        denom = world_rays @ normal
        if np.any(np.abs(denom) < 1.e-6):
            continue
        scale = ((origin - T[:3, 3]) @ normal) / denom
        opening = T[:3, 3] + world_rays * scale[:, None]
        across = np.cross(normal, [0., 0., 1.])
        across /= max(np.linalg.norm(across), 1.e-12)
        up = np.cross(across, normal)
        local = opening - origin
        mid = [np.mean(np.percentile(local @ direction, [2., 98.])) for direction in (across, up)]
        spans = [np.ptp(np.percentile(local @ direction, [2., 98.])) for direction in (across, up)]
        if min(spans) < 1.e-6 or max(spans) / min(spans) > 6.:
            # A very oblique camera may see only one inner wall. Its midpoint
            # is not the center of the opening, even if it looks like a slot.
            continue
        method = "robust face plane"
        # On a chamfer, projecting recessed pixels onto the mouth plane shifts
        # their midpoint toward the eye. Opposite rim points at the SAME depth
        # instead measure the cross-section center directly in 3D.
        shallow_centers = []
        for lo in np.arange(.002, .004, .0005):
            band = pixels & (residual >= lo) & (residual < lo + .0005)
            if int(band.sum()) < 12:
                continue
            section = cloud[band] - origin
            shallow_centers.append([np.mean(np.percentile(section @ direction, [2., 98.]))
                                    for direction in (across, up)])
        if len(shallow_centers) >= 3:
            centers = np.asarray(shallow_centers)
            if np.max(np.std(centers, axis=0)) < .0005:
                mid = np.median(centers, axis=0)
                method = "front plane and chamfer sections"
        candidates.append((area, origin + mid[0] * across + mid[1] * up, method))
    if not candidates:
        return None, None, "no supported recessed opening"
    area, mouth, method = max(candidates, key=lambda c: c[0])
    camera_point = T[:3, :3].T @ (mouth - T[:3, 3])
    return mouth, normal, {"slot_px": int(area), "face_mm": float(camera_point[2]) * 1000.,
                           "method": method}


def find_bore(rgb, depth, K, T):
    import cv2  # noqa: PLC0415 -- a heavy optional import, kept out of module load
    from scipy.ndimage import binary_fill_holes

    h, w = depth.shape
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    blue = cv2.inRange(hsv, np.array(BLUE_LO, np.uint8), np.array(BLUE_HI, np.uint8))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(blue, 8)
    if n <= 1:
        return None, None, "no blue region"
    pick = sorted(range(1, n), key=lambda i: ((cent[i][0] - w/2)**2 + (cent[i][1] - h/2)**2)
                  / max(stats[i, cv2.CC_STAT_AREA], 1))[0]
    if stats[pick, cv2.CC_STAT_AREA] < MIN_FACE_PX:
        return None, None, f"blue region only {stats[pick, cv2.CC_STAT_AREA]} px"
    # RTX can make the recessed interior dark and nearly achromatic. A hole
    # enclosed by the detected face still belongs to the aperture search.
    body = binary_fill_holes(lab == pick)
    d = np.where(np.isfinite(depth) & (depth > 1e-4), depth, np.nan)

    # THE SLOT IS A LOCAL HOLE, so it is found against its OWN surroundings.
    # A depth percentile over the whole silhouette lands on the flange rim --
    # 30 mm of radius seen from 70 mm off axis is 12 mm nearer than the mouth
    # plane, and everything measured from it inherits that.
    cx0, cy0 = cent[pick]
    yy, xx = np.ogrid[:h, :w]
    near = (xx - cx0) ** 2 + (yy - cy0) ** 2 < SEARCH_PX ** 2
    local = np.nanmedian(d[body & near])
    if not np.isfinite(local):
        return None, None, "no depth on the face"
    slot = body & near & (d > local + SLOT_BEHIND_M)
    # NOT a morphological close. The aperture sometimes renders as two blobs --
    # a filled rectangle and a thin sliver of the wall beside it -- and closing
    # merges them into one long shape that no aspect test recognises. Left
    # separate, the scoring below picks the filled one on its own merits.
    slot = slot.astype(np.uint8)
    sn, slab, sstats, _ = cv2.connectedComponentsWithStats(slot, 8)
    best, best_i, score = None, -1, -1e9
    for i in range(1, sn):
        area = int(sstats[i, cv2.CC_STAT_AREA])
        if area < MIN_SLOT_PX:
            continue
        (bx, by), (bw, bh), _ = cv2.minAreaRect(cv2.findNonZero((slab == i).astype(np.uint8)))
        lo, hi = min(bw, bh), max(bw, bh)
        if lo < 1.0 or not (SLOT_ASPECT[0] <= hi / lo <= SLOT_ASPECT[1]):
            continue
        # A SLOT IS A FILLED RECTANGLE NEAR THE MIDDLE OF THE FACE. Fill is what
        # separates it from a wall sliver, which is long, thin and hollow; the
        # distance term breaks ties toward the aperture rather than the barrel's
        # own step. Neither term is a dimension of this particular connector.
        s = (area / max(bw * bh, 1.0)) * 100.0 + area / 20.0 - np.hypot(bx - cx0, by - cy0)
        if s > score:
            best, best_i, score = (bx, by, area, hi / lo), i, s
    if best is None:
        return None, None, "no slot-shaped recess"
    bx, by, area, aspect = best

    # THE MOUTH PLANE IS THE RING AROUND THE SLOT, not the whole face: it is the
    # surface the plug actually arrives at, and it is the only part of this
    # silhouette guaranteed to be at the mouth's own depth.
    sel = (slab == best_i).astype(np.uint8)
    ring = cv2.dilate(sel, np.ones((2 * RING_PX + 1,) * 2, np.uint8)) - sel
    ring = (ring > 0) & body & np.isfinite(d)
    if int(ring.sum()) < 40:
        return None, None, "no ring around the slot"
    face_d = float(np.nanmedian(d[ring]))
    ys, xs = np.nonzero(ring)
    zz = d[ys, xs]
    keep = np.abs(zz - face_d) < 0.004
    pts = np.stack([(xs[keep] - K[0, 2]) / K[0, 0] * zz[keep],
                    (ys[keep] - K[1, 2]) / K[1, 1] * zz[keep], zz[keep]], 1)
    if len(pts) < 30:
        return None, None, "ring too thin to fit a plane"
    world = pts @ np.asarray(T[:3, :3]).T + np.asarray(T[:3, 3])
    normal = np.linalg.svd(world - world.mean(0), full_matrices=False)[2][2]

    # Intersect the aperture-centre ray with the fitted face. Median ring
    # depth is biased on an oblique face, especially when half is occluded.
    ray = np.asarray(T[:3, :3]) @ np.array([(bx - K[0, 2]) / K[0, 0],
                                           (by - K[1, 2]) / K[1, 1], 1.])
    eye = np.asarray(T[:3, 3])
    denom = float(normal @ ray)
    if abs(denom) < 1.e-6:
        return None, None, "face is edge-on"
    distance = float(normal @ (world.mean(0) - eye)) / denom
    if not np.isfinite(distance) or distance <= 0:
        return None, None, "invalid face intersection"
    mouth = eye + distance * ray
    if float(normal @ (np.asarray(T[:3, 3]) - mouth)) < 0:
        normal = -normal
    return mouth, -normal, {"px": (bx, by), "face_mm": face_d * 1e3, "slot_px": area,
                            "aspect": aspect, "ring_px": int(ring.sum())}
