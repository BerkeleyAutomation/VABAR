# SPDX-License-Identifier: Apache-2.0
# Vendored from KeplerC/RoboSimStudio@eeb2aa88, robosimstudio_gap/curve_tools.py (sha256 0ccd42d4...),
# which upstream RoboSimStudio does not carry. Edits: the TrackDLO import path, and the
# registration now runs from cable_route's (:tools ...) hook rather than the connector.
"""Unprivileged ``curve.*`` tools: a mask and a depth frame in, a centreline out.

The primitive the rigid geometry bundle does not have. ``geometry.fit_linear_feature``
reduces a cloud to its principal axis, which is exactly right for a rod lying
straight and exactly wrong for one that has been woven: the principal axis of a
curve through four posts is a chord that passes through none of the places a hand
needs to go. Fitting a line to a bent cable throws away the state.

What replaces it is an **ordered polyline**, parameterised by arc length on
``s in [0, 1]``. That parameterisation is not invented here -- it is the suite's
own. ``tasks/cable_route/assets/cable.py`` already declares the rod's features as
``end_a (s=0.0)``, ``midpoint (s=0.5)`` and ``end_b (s=1.0)``, so a fitted
centreline reproduces a coordinate the asset publishes and per-node error is
scorable against a declared feature rather than only against task reward.

**The ordering is the hard part, and it is TrackDLO's.** Skeletonising a mask is
one call; recovering a traversal order through occlusion breaks and
self-crossings is not. That work is vendored under
:mod:`~vabar.cable.trackdlo_skeleton` with its own MIT
licence and citation. What this module adds is the metric half: back-project the
ordered pixels through depth, fit a smoothing spline in arc length, and resample
uniformly.

**Unprivileged, structurally.** No environment, no bus, no simulator -- a mask,
a depth image, an intrinsic matrix and a camera pose, all of which a real cell
has. Registered before the privileged branch for the same reason ``geometry.*``
is.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["CurveTools", "fit_centerline", "register_curve_tools", "track_centerline"]

#: Nodes the centreline is resampled to. TrackDLO's own default region: enough
#: that a 500 mm rod is described every ~12 mm, few enough that the tracker
#: this feeds stays real-time. A caller wanting the rod's own discretisation
#: passes its segment count.
DEFAULT_NODES = 40

#: How far the mask is UPscaled before TrackDLO's skeletoniser sees it.
#:
#: Upstream divides by 10 and mode-filters with a 15-pixel window, which suits
#: the frame it was written for: a rope ~40 px across in 1280x720. This bench
#: renders 640x480 with a rod ~6 px across, so that preprocessing erases the rod
#: outright -- measured, the chain came back empty and the PCA fallback silently
#: took over, returning all 1651 mask pixels ordered along their principal axis
#: rather than a skeleton. The mean-width estimate built on that read 0.6 px
#: instead of 5.6.
#:
#: Upscaling four times and asking for no further downscale puts the rod at
#: ~24 px, which is the regime the mode filter expects. The chain comes back in
#: upscaled pixels and is divided down again before use.
UPSCALE = 4

#: Mean squared metric residual allowed by ``splprep`` (1 mm RMS).
#: The points are in metres. A pixel-scale value of 2 allowed a single cubic
#: to flatten the weave and amplify an erroneous background depth sample.
SMOOTH = 1.0e-6


def register_curve_tools(registry) -> None:
    """Register the unprivileged curve tools on a GaP tool registry."""
    tools = CurveTools()
    for name, fn, summary in tools.entries():
        registry.register_callable(name, fn, summary=summary, tags=("geometry",))


def _to_world(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    v, u = np.nonzero(np.asarray(mask) > 0)
    if len(u) == 0:
        return np.zeros((0, 3))
    z = depth[v, u]
    ok = np.isfinite(z) & (z > 1.0e-4)
    u, v, z = u[ok], v[ok], z[ok]
    if len(z) == 0:
        return np.zeros((0, 3))
    cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)], axis=1)
    return (T @ cam.T).T[:, :3]


_MERGE_WARNED: list[bool] = []
"""Whether the chain-merge fallback has already been reported. Once is enough:
this runs every frame of every episode."""


def _connect_fragments(graph, u, v, ends, max_gap):
    """Join nearby, facing ends across an occlusion without closing a loop."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    count, labels = connected_components(graph, directed=False)
    if count < 2:
        return graph
    sizes = np.bincount(labels)
    ends = np.asarray([i for i in ends if sizes[labels[i]] >= max(12, int(max_gap*.4))], dtype=int)
    if len(ends) < 2:
        return graph
    xy = np.column_stack((u, v)).astype(float)
    directions = {}
    for end in ends:
        previous, current = -1, int(end)
        for _ in range(12):
            candidates = [int(i) for i in graph.indices[graph.indptr[current]:graph.indptr[current+1]] if i != previous]
            if len(candidates) != 1:
                break
            previous, current = current, candidates[0]
        tangent = xy[end]-xy[current]
        directions[int(end)] = tangent/max(float(np.linalg.norm(tangent)), 1e-9)
    links = []
    for ai, bi in cKDTree(xy[ends]).query_pairs(float(max_gap)):
        a, b = int(ends[ai]), int(ends[bi])
        if labels[a] == labels[b]:
            continue
        delta = xy[b]-xy[a];distance = float(np.linalg.norm(delta))
        unit = delta/max(distance,1e-9)
        facing_a = float(directions[a]@unit);facing_b = float(directions[b]@(-unit))
        if min(facing_a, facing_b) < -.25:
            continue
        links.append((distance*(2.-.5*(facing_a+facing_b)), distance, a, b))
    parent = list(range(count));used = set();out = graph.tolil()
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]];i = parent[i]
        return i
    for _, distance, a, b in sorted(links):
        first, second = find(int(labels[a])), find(int(labels[b]))
        if first == second or a in used or b in used:
            continue
        out[a,b] = distance;out[b,a] = distance
        parent[first] = second;used.update((a,b))
    return out.tocsr()


def _ordered_pixels(mask: np.ndarray, max_gap_px: float = 0.0) -> np.ndarray:
    """The mask's skeleton as one ordered ``(N, 2)`` pixel chain.

    TrackDLO's merge first, which is the only part that can order a skeleton
    broken by occlusion or crossing itself. Its failure is not an error -- a
    single clean blob gives the assignment nothing to assign -- so a thinned
    fallback follows, ordered along its principal axis. The fallback
    skeletonises too: ordering the *filled* mask by PCA returns every pixel,
    which is not a centreline and quietly poisons anything measured from its
    length.
    """
    import cv2  # noqa: PLC0415
    from skimage.morphology import skeletonize  # noqa: PLC0415

    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    # Thinning a 4x full camera frame for each small mask is unnecessarily
    # expensive. Preserve a zero border and restore image coordinates below.
    active_v, active_u = np.nonzero(m)
    if len(active_u) < 4:
        return np.zeros((0, 2))
    u0, v0 = max(0, int(active_u.min())-2), max(0, int(active_v.min())-2)
    u1, v1 = min(m.shape[1], int(active_u.max())+3), min(m.shape[0], int(active_v.max())+3)
    m = m[v0:v1, u0:u1]
    offset = np.asarray([u0, v0], dtype=np.float64)
    big = cv2.resize(m, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_NEAREST)

    # A clean cable mask is a one-pixel graph with two ends. Traverse that
    # graph instead of sorting pixels by a principal axis: PCA interleaves
    # neighbouring rows of a curved skeleton and creates large zigzags. The
    # longest finite endpoint-to-endpoint geodesic is the cable's ordered
    # centreline. Masks broken by occlusion fall through to TrackDLO below.
    thin = skeletonize(big > 0).astype(np.uint8)
    vv, uu = np.nonzero(thin)
    if len(uu) >= 4:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        ids = np.full(thin.shape, -1, dtype=np.int32)
        ids[vv, uu] = np.arange(len(uu), dtype=np.int32)
        rows: list[int] = []
        cols: list[int] = []
        weights: list[float] = []
        for dv, du, cost in ((0, 1, 1.0), (1, -1, 2.0**0.5), (1, 0, 1.0), (1, 1, 2.0**0.5)):
            valid = (vv + dv < thin.shape[0]) & (uu + du >= 0) & (uu + du < thin.shape[1])
            src = np.nonzero(valid)[0]
            dst = ids[vv[src] + dv, uu[src] + du]
            keep = dst >= 0
            for a, b in zip(src[keep], dst[keep], strict=True):
                rows.extend((int(a), int(b)))
                cols.extend((int(b), int(a)))
                weights.extend((cost, cost))
        graph = coo_matrix((weights, (rows, cols)), shape=(len(uu), len(uu))).tocsr()
        degree = np.diff(graph.indptr)
        ends = np.flatnonzero(degree == 1)
        if max_gap_px > 0.0:
            graph = _connect_fragments(graph, uu, vv, ends, float(max_gap_px) * UPSCALE)
            ends = np.flatnonzero(np.diff(graph.indptr) == 1)
        if len(ends) >= 2:
            distances, predecessors = dijkstra(
                graph, directed=False, indices=ends, return_predecessors=True
            )
            distances = np.where(np.isfinite(distances), distances, -1.0)
            ei, target = np.unravel_index(int(np.argmax(distances)), distances.shape)
            source = int(ends[ei])
            current = int(target)
            path = [current]
            while current != source and current >= 0:
                current = int(predecessors[ei, current])
                if current >= 0:
                    path.append(current)
            if len(path) >= 4 and path[-1] == source:
                path.reverse()
                return np.stack([uu[path], vv[path]], axis=1).astype(np.float64) / float(UPSCALE) + offset

    from .trackdlo_skeleton import extract_connected_skeleton  # noqa: PLC0415

    try:
        chains = extract_connected_skeleton(False, big, img_scale=1)
    except Exception as exc:
        # SAY SO, ONCE. A silent fallback here cost a long time: an unrelated
        # skimage API change made the merge raise on *every* frame, the except
        # swallowed it, and the PCA fallback ordered every centreline in the
        # suite while the logs showed a healthy fit. The failure is legitimate
        # on a degenerate mask, so this stays a fallback rather than an error --
        # but it no longer happens quietly.
        if not _MERGE_WARNED:
            _MERGE_WARNED.append(True)
            print(
                f"[curve]   TrackDLO chain merge unavailable ({type(exc).__name__}: {exc}); "
                f"ordering by principal axis instead",
                flush=True,
            )
        chains = []
    pts = [p for chain in (chains or []) for p in chain]
    if len(pts) >= 8:
        return np.asarray(pts, dtype=np.float64) / float(UPSCALE) + offset

    v, u = np.nonzero(thin)
    if len(u) < 4:
        return np.zeros((0, 2))
    xy = np.stack([u, v], axis=1).astype(np.float64) / float(UPSCALE)
    centred = xy - xy.mean(axis=0)
    axis = np.linalg.eigh(centred.T @ centred)[1][:, -1]
    return xy[np.argsort(centred @ axis)] + offset


def _cloud_radius(mask, depth, K: np.ndarray, arclength: float) -> float:
    """Mean radius of the masked object, by area over length -- see the note in
    :func:`fit_centerline` on why this ratio rather than a width measurement."""
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    v, u = np.nonzero(np.asarray(mask) > 0)
    if len(u) == 0 or arclength <= 1.0e-6:
        return 0.0
    z = d[v, u]
    z = z[np.isfinite(z) & (z > 1.0e-4)]
    if len(z) == 0:
        return 0.0
    pixel_m = float(np.median(z)) / float(K[0, 0])
    return 0.5 * (len(u) * pixel_m * pixel_m) / float(arclength)


def _to_axis(curve: np.ndarray, T: np.ndarray, radius: float) -> np.ndarray:
    """Push a surface-fitted centreline onto the object's AXIS.

    Depth returns the range to the rod's camera-facing *surface*, so a
    centreline back-projected from it is a line on the skin of the cylinder,
    displaced one radius toward the camera. Nothing in the fit corrects for
    that, and the resulting error is a systematic bias rather than noise --
    which makes it invisible in a scatter and fully present in every number
    derived from it.

    Measured on ``weave3`` from the 55-degree cable camera: the node abreast of
    each station read +2.5 to +3.9 mm (median 3.1) toward +x of the rod's own
    body, at all three stations, in every pose. Geometry predicts exactly that
    -- a 5 mm rod seen from 55 degrees of elevation in +x puts the surface
    ``R cos(55) = 2.87`` mm out in x and ``R sin(55) = 4.10`` mm up in z.

    The z half matters as much as the x half and is easier to overlook: a rod
    read 4 mm high is a grasp planned 4 mm shallow.

    The correction is one radius along each point's own view ray, away from the
    camera. Per-point rather than one direction for the whole curve, because a
    500 mm rod spans enough of the frame that the ray turns noticeably across it.
    """
    pts = np.asarray(curve, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0 or radius <= 0.0:
        return pts
    eye = np.asarray(T, dtype=np.float64).reshape(4, 4)[:3, 3]
    ray = pts - eye
    n = np.linalg.norm(ray, axis=1, keepdims=True)
    return pts + radius * np.divide(ray, n, out=np.zeros_like(ray), where=n > 1.0e-9)


def fit_centerline(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    nodes: int = DEFAULT_NODES,
    smooth: float = SMOOTH,
    merge_gap_m: float = 0.0,
) -> dict[str, Any]:
    """Ordered 3D centreline of a deformable linear object. See the module doc."""
    from scipy.interpolate import splev, splprep

    mask = np.asarray(mask)
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    T = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)

    active_depth = depth[(mask > 0) & np.isfinite(depth) & (depth > 1.0e-4)]
    gap_px = float(merge_gap_m) * K[0, 0] / float(np.median(active_depth)) if len(active_depth) and merge_gap_m > 0 else 0.0
    chain = _ordered_pixels(mask, max_gap_px=gap_px) if gap_px > 0 else _ordered_pixels(mask)
    if len(chain) < 4:
        return {"points": [], "arclength_m": 0.0, "radius_m": 0.0, "nodes": 0, "ordered": False}

    # Back-project the ordered *chain*, not the whole mask: the chain is the
    # medial axis, so its depth samples are the rod's own surface rather than a
    # mix of surface and silhouette edge, where the edge pixels straddle the
    # background and read metres away.
    h, w = depth.shape[:2]
    uu = np.clip(np.round(chain[:, 0]).astype(int), 0, w - 1)
    vv = np.clip(np.round(chain[:, 1]).astype(int), 0, h - 1)
    z = depth[vv, uu]
    # Upscaled skeleton samples can round outside the original mask. Their
    # depths then belong to the bench or robot, even though the source mask
    # was height-gated. Never fit those pixels into the cable.
    ok = np.isfinite(z) & (z > 1.0e-4) & (mask[vv, uu] > 0)
    if int(ok.sum()) < 4:
        return {"points": [], "arclength_m": 0.0, "radius_m": 0.0, "nodes": 0, "ordered": False}
    uu, vv, z = uu[ok], vv[ok], z[ok]
    cam = np.stack([(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)], axis=1)
    world = (T @ cam.T).T[:, :3]

    # Drop consecutive duplicates -- splprep refuses a repeated knot, and a
    # downsampled skeleton produces them wherever two pixels share a depth.
    keep = np.concatenate([[True], np.linalg.norm(np.diff(world, axis=0), axis=1) > 1.0e-6])
    world = world[keep]
    if len(world) < 4:
        return {"points": [], "arclength_m": 0.0, "radius_m": 0.0, "nodes": 0, "ordered": False}

    k = min(3, len(world) - 1)
    try:
        tck, _ = splprep(world.T, s=float(smooth) * len(world), k=k)
        curve = np.asarray(splev(np.linspace(0.0, 1.0, int(nodes)), tck)).T
    except Exception:
        curve = world

    seglen = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    arclength = float(seglen.sum())

    # RADIUS FROM THE MEDIAL AXIS' LOCAL HALF-WIDTH.
    #
    # Three wrong ways were measured before this one, and each failed for its
    # own reason worth keeping:
    #
    # * Halving the cloud's spread along a world axis measures thickness only
    #   while the object lies along that axis. On ``weave_arc`` it read 4.14 mm
    #   for a 5 mm rod, because the spread was mostly the curve.
    # * Perpendicular distance from the centreline to the back-projected cloud
    #   is worse: a silhouette edge pixel straddles the object and the bench, so
    #   its depth is the bench's and it lands centimetres out. 8.6-9.0 mm.
    # * Mask area over spline length works only when the mask contains *only*
    #   the rod. RTX's soft lab lighting connected white jacket highlights to
    #   another pale object; the geodesic still followed the cable, but the
    #   extra area inflated 3.5 mm to 15.7 mm and moved every seat by 12 mm.
    #
    # A distance transform sampled on the geodesic measures the local
    # half-width of the curve itself, so disconnected or attached area away
    # from the traversed cable cannot change its radius.
    pixel_m = float(np.median(z)) / float(K[0, 0])
    import cv2  # noqa: PLC0415

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    cu = np.clip(np.round(chain[:, 0]).astype(int), 0, binary.shape[1] - 1)
    cv = np.clip(np.round(chain[:, 1]).astype(int), 0, binary.shape[0] - 1)
    half_width_px = distance[cv, cu]
    half_width_px = half_width_px[np.isfinite(half_width_px) & (half_width_px > 0.0)]
    radius = float(np.median(half_width_px) * pixel_m) if len(half_width_px) else 0.0

    # Surface -> axis. See :func:`_to_axis`; the radius this uses is the one
    # just measured from the same mask, so no dimension is assumed.
    curve = _to_axis(curve, T, radius)

    return {
        "points": [[float(v) for v in p] for p in curve],
        "arclength_m": arclength,
        "radius_m": radius,
        "nodes": int(len(curve)),
        "ordered": bool(len(chain) >= 4),
    }


# ---------------------------------------------------------------- tracking
#: Motion-coherence width, as a fraction of the rod's own length. TrackDLO's
#: ``beta``: the kernel over which one node's motion is shared with its
#: neighbours. Wide enough that an occluded stretch is carried by the visible
#: rod either side of it; narrow enough that a real bend is not flattened.
TRACK_BETA = 0.18

#: How far a node moves toward its observation each iteration. TrackDLO's
#: ``alpha``. Below one because a correspondence built from a single frame's
#: mask is noisy, and a node that jumps onto its nearest pixel every frame
#: chatters along the rod.
TRACK_ALPHA = 0.6

#: Correspondence width [m]. Observed points beyond roughly this distance from a
#: node contribute almost nothing to it, which is what stops the far strand of a
#: doubled-back rod capturing nodes from the near one.
TRACK_SIGMA = 0.020

#: Visibility below which a node is treated as unobserved and moved only by its
#: neighbours. TrackDLO's ``k_vis`` plays this role: the gripper covering a
#: stretch of rod must not drag those nodes onto the gripper.
TRACK_VIS_FLOOR = 0.05

TRACK_ITERS = 6

#: How hard each node is pulled toward the midpoint of its neighbours, per
#: iteration. A BENDING PRIOR, and without one the tracker coils.
#:
#: Measured, ``weave3`` episode 4 step 660: with the arm across the rod the
#: occluded nodes folded into a knot while a cold fit followed the visible rod
#: correctly -- tracked error 14.3 mm against the fit's 6.0. Nothing in the
#: correspondence-plus-coherence update penalises curvature, so a stretch with
#: no observation of its own is free to double back, and re-spacing then
#: preserves the fold at the right length rather than removing it.
#:
#: A rod resists bending. This is the cheapest statement of that, and it is the
#: term TrackDLO gets for free from its non-Gaussian kernel over displacement
#: derivatives -- the part of the method this compact version does not have.
TRACK_STIFFNESS = 0.25


def track_centerline(
    prior: Any,
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    beta: float = TRACK_BETA,
    alpha: float = TRACK_ALPHA,
    sigma: float = TRACK_SIGMA,
    vis_floor: float = TRACK_VIS_FLOOR,
    iters: int = TRACK_ITERS,
    stiffness: float = TRACK_STIFFNESS,
    radius: float | None = None,
) -> dict[str, Any]:
    """Advance a known centreline onto a new frame. TrackDLO's idea, compactly.

    **Why this is not just fitting again.** A cold fit sees only what is visible
    now, so a gripper covering half the rod produces half a rod -- measured on
    ``weave3``, arclength collapsing to 118 mm mid-carry, and on ``weave_arc``,
    a quarter of frames lost outright. Nothing in a single frame says whether a
    short rod is short or merely half-seen.

    A tracker answers from the *previous* shape. Each node takes a Gaussian-
    weighted correspondence to the new cloud; the resulting displacement field
    is then smoothed along the rod, so a node with no observation of its own
    moves with the stretch either side of it. That is Motion Coherence Theory's
    contribution and it is the whole reason occlusion stops being fatal:
    velocity is imputed rather than the frame discarded.

    Inextensibility closes it. A cable does not stretch, so after each update
    the nodes are re-spaced to the length they had -- which is what stops a
    partly-observed rod from being pulled bodily into the visible stretch.

    Faithful in structure to TrackDLO (Xiang et al., RA-L 2023), not in
    numerics: the full method solves a CPD-style EM with a non-Gaussian kernel
    over the deformation field, where this takes a fixed number of damped
    correspondence steps. It buys the occlusion behaviour, not the paper's
    accuracy, and the difference is worth stating rather than implying.

    Returns the same shape :func:`fit_centerline` does, plus ``visibility`` --
    the per-node correspondence mass, which is what a caller should route on
    when deciding whether it still knows where the rod is.
    """
    Y = np.asarray(prior, dtype=np.float64).reshape(-1, 3)
    if len(Y) < 3:
        return fit_centerline(mask, depth, intrinsics, camera_pose, nodes=max(len(Y), DEFAULT_NODES))
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    T = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)
    X = _to_world(mask, np.asarray(depth, dtype=np.float64), K, T)
    # The observation is a cloud on the rod's SKIN, and the model it corrects is
    # an axis. Left alone the tracker pulls every node one radius toward the
    # camera each pass -- the same surface-vs-axis bias :func:`_to_axis`
    # documents, applied repeatedly rather than once. Corrected on the way in so
    # the correspondence is axis-to-axis.
    #
    # THE RADIUS IS THE ONE THE OPENING SURVEY MEASURED, not this frame's.
    # :func:`_cloud_radius` is area over length, and both of its terms are wrong
    # on a tracking frame in the same direction: a mask that caught a fingertip,
    # a bench highlight or the arm's own shadow gains area, while an occluded rod
    # loses the length that area is divided by. Measured on a saved carry frame,
    # that ratio returned 23.15 mm against the opening camera's 3.47 mm on a rod
    # whose radius is a constant of the scene -- and _to_axis then pushed every
    # node 23 mm along its view ray, which put the reconstructed cable through
    # the bench it is lying on. The rod does not change radius during an episode,
    # so the opening measurement is the better estimator on every later frame,
    # and a caller that has one should pass it.
    #
    # ``None`` keeps the old per-frame behaviour for a caller that never surveyed.
    if len(X):
        skin = float(radius) if radius is not None and radius > 0.0 else _cloud_radius(
            mask, depth, K, _arclength(Y)
        )
        X = _to_axis(X, T, skin)
    if len(X) < 8:
        return {
            "points": [[float(v) for v in p] for p in Y],
            "arclength_m": _arclength(Y),
            "radius_m": 0.0,
            "nodes": int(len(Y)),
            "ordered": True,
            "visibility": [0.0] * len(Y),
            "tracked": True,
        }

    target_len = _arclength(Y)
    # The coherence kernel is over ARC LENGTH ALONG THE ROD, not over space. Two
    # points that a fold has brought close together are not neighbours, and a
    # spatial kernel would couple their motion -- exactly the self-intersection
    # case TrackDLO uses a geodesic distance for.
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Y, axis=0), axis=1))])
    s = s / max(s[-1], 1.0e-9)
    G = np.exp(-((s[:, None] - s[None, :]) ** 2) / (2.0 * float(beta) ** 2))

    vis = np.zeros(len(Y))
    for _ in range(int(iters)):
        d2 = np.sum((Y[:, None, :] - X[None, :, :]) ** 2, axis=2)
        w = np.exp(-d2 / (2.0 * float(sigma) ** 2))
        vis = w.sum(axis=1)
        mass = np.maximum(vis, 1.0e-12)
        targets = (w @ X) / mass[:, None]
        raw = np.where((vis > float(vis_floor))[:, None], targets - Y, 0.0)
        # Share the motion along the rod. Occluded nodes contribute nothing and
        # receive the field their visible neighbours generate.
        seen = (vis > float(vis_floor)).astype(np.float64)
        denom = np.maximum(G @ seen, 1.0e-9)
        Y = Y + float(alpha) * ((G @ raw) / denom[:, None])
        # Bending prior, applied where the rod is NOT observed. A visible node
        # should answer to its pixels; an occluded one has nothing else to
        # answer to, and left alone it folds.
        smooth = Y.copy()
        smooth[1:-1] = 0.5 * (Y[:-2] + Y[2:])
        weight = float(stiffness) * (vis <= float(vis_floor)).astype(np.float64)[:, None]
        Y = (1.0 - weight) * Y + weight * smooth
        Y = _respace(Y, target_len)

    area_m2 = float(np.count_nonzero(np.asarray(mask) > 0)) * (float(np.median(X[:, 2])) / K[0, 0]) ** 2
    length = _arclength(Y)
    return {
        "points": [[float(v) for v in p] for p in Y],
        "arclength_m": length,
        "radius_m": 0.5 * area_m2 / length if length > 1.0e-6 else 0.0,
        "nodes": int(len(Y)),
        "ordered": True,
        "visibility": [float(v) for v in (vis / max(float(np.max(vis)), 1.0e-9))],
        "tracked": True,
    }


def _arclength(Y: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(Y, axis=0), axis=1)))


def _respace(Y: np.ndarray, target_len: float) -> np.ndarray:
    """Re-space nodes uniformly along the polyline, restoring its length.

    A cable is inextensible, and the update above is not: nodes pulled toward a
    partly-observed cloud bunch into the visible stretch. Rescaling to the
    length the rod had is the cheapest statement of that constraint, and the one
    that keeps an occluded tip where the material says it must be.
    """
    seg = np.linalg.norm(np.diff(Y, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1.0e-9:
        return Y
    Y = Y[0] + (Y - Y[0]) * (target_len / total)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Y, axis=0), axis=1))])
    want = np.linspace(0.0, s[-1], len(Y))
    return np.stack([np.interp(want, s, Y[:, k]) for k in range(3)], axis=1)


class CurveTools:
    """The centreline implementations. Stateless."""

    def entries(self) -> list[tuple[str, Any, str]]:
        return [
            (
                "curve.fit_centerline",
                self.fit_centerline,
                "Fit an ordered, arc-length-parameterised 3D centreline to a deformable linear "
                "object from its mask and a depth frame.",
            ),
            (
                "curve.track_centerline",
                self.track_centerline,
                "Advance a known centreline onto a new frame, imputing motion for occluded "
                "stretches from their visible neighbours.",
            ),
        ]

    def fit_centerline(
        self,
        mask: Any = None,
        depth: Any = None,
        intrinsics: Any = None,
        camera_pose: Any = None,
        nodes: int = DEFAULT_NODES,
        smooth: float = SMOOTH,
        merge_gap_m: float = 0.0,
    ) -> dict[str, Any]:
        if mask is None or depth is None or intrinsics is None or camera_pose is None:
            raise ValueError("curve.fit_centerline needs mask, depth, intrinsics and camera_pose")
        return fit_centerline(mask, depth, intrinsics, camera_pose, nodes=nodes, smooth=smooth, merge_gap_m=merge_gap_m)

    def track_centerline(
        self,
        prior: Any = None,
        mask: Any = None,
        depth: Any = None,
        intrinsics: Any = None,
        camera_pose: Any = None,
        radius: Any = None,
    ) -> dict[str, Any]:
        """``radius`` is the opening survey's, in metres, and should be passed.

        Optional only so that a caller with no survey behind it still works; see
        the note in :func:`track_centerline` on why this frame's own area-over-
        length estimate is the worse one on every frame after the first.
        """
        if prior is None or mask is None or depth is None or intrinsics is None or camera_pose is None:
            raise ValueError("curve.track_centerline needs prior, mask, depth, intrinsics and camera_pose")
        return track_centerline(
            prior, mask, depth, intrinsics, camera_pose,
            radius=None if radius is None else float(radius),
        )
