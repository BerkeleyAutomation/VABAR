"""Measure the held plug and socket from the current wrist RGB-D frame.

Movable sockets are registered against their known local surface geometry, so
mouth position and bore direction share one fresh 3D measurement. Fixed sockets
retain their earlier aperture measurement. The plug's shoulder and top plane
are measured again after grasping to account for slip in the fingers.
"""

import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from held_frame import fit_top_frame
from surface_frame import fit_plug, fit_receptacle

REQUIRED_TOOLS = ("sam3.segment_text", "usbc.receptacle_shape", "usbc.plug_shape")
WHY_ABSENT = (
    "the sam3 segmenter is not resolvable here, so the vision body of the relook slot is "
    "declared and skipped. It is not a task failure and it is not a perception result: run "
    "the oracle body (--perception oracle, the default), or install transformers and torch."
)

CAMERA = "wrist"
"""The eye-in-hand view, riding the work arm's tool body.

Named rather than passed in, because the *role* is already what the rig
declares: ``parent_frame="work"`` resolves to whichever arm is doing the work.
A cell with two hand cameras would name one here.
"""

PLUG_QUERY = "orange plug"
RECEPTACLE_QUERY = "blue object"
PLUG_SCORE = 0.20
RECEPTACLE_SCORE = 0.20
MIN_MASK_PIXELS = 40
"""The ground body's prompts and floors, restated -- same scene, same palette,
and its docstring carries the measured table these came from. A body that
grounded on different words would be measuring a different thing."""

SHOULDER_BINS = 40
SHOULDER_DROP = 0.7
"""How the plug's own origin is found: the step where its tab meets its body.

The bench body's constants, restated, because the two slots have to mean the
same thing by "the plug" -- see `ground_vision.SHOULDER_BINS` for why a centroid
is 10 mm wrong on this part, and this file's own history for what happened when
only one of the two was fixed.
"""

PROFILE_BIN = 0.002
"""How finely the mask is sliced along the bore axis [m].

2 mm is four to five pixels of the cloud at this standoff, which is fine enough
to resolve the flange's 6 mm band and coarse enough that a slice still holds
hundreds of points.
"""

FACE_LEVEL = 0.5
BODY_PERCENTILE = 75.0
"""How the near face is found: the HALF-DENSITY CROSSING of the leading ramp.

This was ``FACE_PERCENTILE = 5``, a 5th percentile of the along-axis coordinate,
and it carried a bias the size of the failure it caused. Measured over ten drawn
layouts against ground truth, at the look station:

=======================  ============  ====
estimator                mean error    sd
=======================  ============  ====
5th percentile           **+3.13 mm**  0.07
half-density crossing    **+0.75 mm**  0.07
=======================  ============  ====

A percentile is a fraction of *the whole cloud*, so where it lands depends on
how long the part is and how much of the tail the mask caught -- on this
receptacle the mask runs 84 mm because it takes the trailing cable with it,
and 5% of 84 mm is 4 mm of pure offset. The direction is always the same, and
it is the wrong one: a mouth read 3 mm deeper than it is makes the stroke push
3 mm less far, straight out of a fit with 0.15 mm a side.

A density crossing is a property of the EDGE and nothing else. Slice the cloud
along the axis, take the interior level as the 75th percentile of the per-slice
counts, and interpolate where the leading ramp first reaches half of it. The
residual +0.75 mm is the mask's own dilation -- about a pixel and a half at
0.55 mm/px -- and it is left uncorrected rather than subtracted, because a
constant fitted here would be a fact about this standoff and not about the part.
"""

MIN_SLICE = 6
"""Points below which a slice states nothing rather than a noisy something."""


class Output(TypedDict):
    route: str
    relook_body: str
    mate_plug_point: list
    mate_mouth_point: list
    mate_bore_axis: list
    mate_plug_axis: list
    mate_plug_up: list
    mate_moved_mm: float


NOTHING: Output = {
    "route": "lost",
    "relook_body": "vision",
    "mate_plug_point": [],
    "mate_mouth_point": [],
    "mate_bore_axis": [],
    "mate_plug_axis": [],
    "mate_plug_up": [],
    "mate_moved_mm": 0.0,
}


def _missing(ctx: NodeContext) -> list[str]:
    registry = getattr(ctx, "_tool_registry", None)
    if registry is None:
        return []
    return [name for name in REQUIRED_TOOLS if name not in registry]


def _camera(observation: Any, name: str) -> dict[str, Any]:
    """One camera's frame, whichever shape the runtime publishes them in."""
    frames = observation["cameras"] if isinstance(observation, dict) else observation.cameras
    if isinstance(frames, dict):
        if name in frames:
            return frames[name]
        raise ValueError(
            f"no camera {name!r} in the observation; have {sorted(frames)}. A vision body needs "
            "a renderer: --renderer tiled."
        )
    for frame in frames or ():
        if str(frame.get("name", "")) == name:
            return frame
    raise ValueError(f"no camera {name!r} in the observation")


def _frame(block: dict[str, Any], name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(rgb, depth, K, world_from_camera)`` out of one camera block.

    The camera's pose is published per frame, which is what makes a *moving*
    camera usable at all: the arm put the lens somewhere and the observation
    says where, so nothing here has to know what the arm did.
    """
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
        raise ValueError(f"camera {name!r} carries no depth; this body measures in metres")
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


def _detect(ctx: NodeContext, rgb: np.ndarray, query: str, floor: float) -> np.ndarray | None:
    result = ctx.tool("sam3.segment_text", image=rgb, query=query, max_results=0)
    masks = result.get("masks") or []
    scores = [float(s) for s in (result.get("scores") or [])]
    for i in sorted(range(len(scores)), key=lambda k: -scores[k]):
        if scores[i] < floor:
            break
        mask = np.asarray(masks[i])
        if int(np.count_nonzero(mask)) >= MIN_MASK_PIXELS:
            return mask
    return None


def _principal_xy(points: np.ndarray) -> np.ndarray:
    """The horizontal direction a cloud is longest along, as a unit ``[x, y]``.

    A plain PCA on the two bench-plane coordinates. Not on all three: every part
    here lies on a horizontal mate line, and letting a 150 mm-tall slab's own
    height win the first component would return "up".
    """
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    axis = np.asarray(vt[0], dtype=np.float64)
    return axis / max(float(np.linalg.norm(axis)), 1.0e-12)


def _shoulder(points: np.ndarray, axis: np.ndarray) -> float:
    """Where along its own axis a part steps from body width to tab width [m].

    Returns the coordinate along ``axis``, which for this connector is its
    origin: the plug's declared geometry puts the shoulder at zero, with 34 mm
    of body behind it and 13 mm of tab in front. See :data:`SHOULDER_BINS`.

    Falls back to the midpoint of the extent when no step is found, which is
    what a part with no shoulder looks like and is at least not a centroid.
    """
    perp = np.array([-axis[1], axis[0], 0.0])
    along = points @ axis
    across = points @ perp
    lo, hi = float(np.percentile(along, 2.0)), float(np.percentile(along, 98.0))
    if hi - lo < 1.0e-4:
        return 0.5 * (lo + hi)
    edges = np.linspace(lo, hi, int(SHOULDER_BINS) + 1)
    widths, centres = [], []
    for i in range(int(SHOULDER_BINS)):
        sel = (along >= edges[i]) & (along < edges[i + 1])
        if int(np.count_nonzero(sel)) < 3:
            continue
        w = float(np.percentile(across[sel], 95.0) - np.percentile(across[sel], 5.0))
        widths.append(w)
        centres.append(0.5 * (edges[i] + edges[i + 1]))
    if len(widths) < 6:
        return 0.5 * (lo + hi)
    w = np.asarray(widths)
    # The step, found WITHOUT knowing which end is the tab. A principal axis has
    # no sign -- an SVD returns whichever direction it likes -- so a scan that
    # walks "from the rear" walks from the tip half the time, and on this plug
    # that put the origin 10.4 mm out, near the tab tip. Splitting on the
    # largest mean-width difference finds the same boundary from either side.
    splits = range(2, len(w) - 2)
    scored = [(abs(float(w[:i].mean() - w[i:].mean())), i) for i in splits]
    best_drop, best = max(scored)
    if best_drop < (1.0 - SHOULDER_DROP) * float(np.percentile(w, 90.0)):
        # No step worth the name: a part with a constant section, which this
        # connector is not. Its middle is at least not its centre of area.
        return 0.5 * (lo + hi)
    return float(centres[best])


def _slices(points: np.ndarray, axis: np.ndarray, perp: np.ndarray) -> dict[str, np.ndarray]:
    """The mask's profile along the bore axis: where it starts, how wide, how high.

    One row per :data:`PROFILE_BIN` slice, holding the slice's centre along the
    axis, how many points fell in it, the lateral MIDPOINT of its extent and the
    top of its surface. Everything the receptacle read needs is a statistic of
    this table, and stating it once is what keeps the three coordinates measured
    off the same cloud in the same way.

    The lateral value is a midpoint of the 2nd/98th percentiles rather than a
    median, and the difference is not cosmetic: a barrel lit from one side shows
    more of the wall the camera can see, so the median of its points sits toward
    that wall. Over ten layouts the median scattered by 0.75 mm sd and the
    extent midpoint by 0.23 -- same mean, a third of the spread.
    """
    along = points @ axis
    edges = np.arange(float(along.min()), float(along.max()) + PROFILE_BIN, PROFILE_BIN)
    if len(edges) < 3:
        return {"s": np.zeros(0), "n": np.zeros(0), "mid": np.zeros(0), "top": np.zeros(0)}
    idx = np.clip(np.digitize(along, edges) - 1, 0, len(edges) - 2)
    n = np.zeros(len(edges) - 1)
    mid = np.full(len(edges) - 1, np.nan)
    top = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        take = idx == b
        n[b] = int(take.sum())
        if n[b] < MIN_SLICE:
            continue
        across = points[take] @ perp
        lo, hi = np.percentile(across, 2.0), np.percentile(across, 98.0)
        mid[b] = 0.5 * (lo + hi)
        top[b] = float(np.percentile(points[take][:, 2], 98.0))
    return {"s": edges[:-1] + PROFILE_BIN / 2, "n": n, "mid": mid, "top": top}


def _face(profile: dict[str, np.ndarray]) -> float:
    """Where the near face is, along the axis: the leading half-density crossing.

    See :data:`FACE_LEVEL` for the measurement that replaced a percentile with
    this, and by how much.
    """
    n, s = profile["n"], profile["s"]
    if not len(n):
        raise RuntimeError("the receptacle's mask sliced to nothing along its own axis")
    body = float(np.percentile(n[n > 0], BODY_PERCENTILE))
    level = FACE_LEVEL * body
    first = int(np.argmax(n >= level))
    if first == 0:
        return float(s[0])
    y0, y1 = n[first - 1], n[first]
    return float(s[first - 1] + (level - y0) / max(y1 - y0, 1.0e-9) * PROFILE_BIN)


def run(
    ctx: NodeContext,
    scene_mouth_point: list | None = None,
    scene_bore_axis: list | None = None,
    scene_plug_half_h: float = 0.0,
    inspect_mouth_point: list | None = None,
    inspect_bore_axis: list | None = None,
    label: str = "relook",
    camera: str = CAMERA,
    plug_query: str = PLUG_QUERY,
    receptacle_query: str = RECEPTACLE_QUERY,
) -> Output:
    absent = _missing(ctx)
    if absent:
        raise RuntimeError(f"the relook slot's vision body needs {list(absent)}: {WHY_ABSENT}")
    if not scene_mouth_point:
        raise RuntimeError(
            "the relook needs the ground read's mouth to carry its height through; the ground slot produced none"
        )

    measured_bore_axis = list(inspect_bore_axis or [])
    if inspect_bore_axis:
        heading = np.asarray([inspect_bore_axis[0], inspect_bore_axis[1], 0.0], dtype=float)
        inspect_bore_axis = (heading / np.linalg.norm(heading)).tolist()
    observation: Any = ctx.tool("robot.get_observation")
    rgb, depth, K, T = _frame(_camera(observation, camera), camera)
    ground_mouth = np.asarray([float(v) for v in scene_mouth_point], dtype=np.float64)

    # Preserve a prior camera measurement of a fixed socket when occluded.
    # A movable bulkhead is measured again after bracing and insertion.
    kept_static = None
    shape = None
    try:
        shape = ctx.tool("usbc.receptacle_shape")
    except Exception:
        kept_axis = inspect_bore_axis or scene_bore_axis
        if not kept_axis:
            return dict(NOTHING)
        kept_static = {"mouth_point": inspect_mouth_point or scene_mouth_point, "bore_axis": kept_axis}
    if kept_static is not None:
        rec_pts = None
    else:
        rec_mask = _detect(ctx, rgb, receptacle_query, RECEPTACLE_SCORE)
        if rec_mask is None:
            print(f"[{label}:vision] no {receptacle_query!r} in {camera!r}", flush=True)
            return dict(NOTHING)  # type: ignore[return-value]
        rec_pts = _to_world(_foreground_mask(rec_mask, depth), depth, K, T)
        if len(rec_pts) < MIN_MASK_PIXELS:
            return dict(NOTHING)  # type: ignore[return-value]

    if rec_pts is None:
        frame = kept_static
        axis = np.asarray([float(v) for v in frame["bore_axis"]], dtype=np.float64)
        mouth = np.asarray([float(v) for v in frame["mouth_point"]], dtype=np.float64)
        source = "earlier camera frame of bolted fixture"
    else:
        fit = fit_receptacle(
            rec_pts,
            shape,
            inspect_mouth_point or scene_mouth_point,
            measured_bore_axis or scene_bore_axis,
        )
        if fit is None:
            print(f"[{label}:vision] no supported fresh socket surface fit", flush=True)
            return dict(NOTHING)
        mouth, axis = np.asarray(fit["mouth"]), np.asarray(fit["axis"])
        source = "current RGB-D surface fit"
        print(
            f"[{label}:vision] socket surface: {fit['inlier_count']} points, "
            f"residual={fit['residual_m'] * 1000:.3f} mm, "
            f"axis={np.round(axis, 4).tolist()}",
            flush=True,
        )

    plug_mask = _detect(ctx, rgb, plug_query, PLUG_SCORE)
    side_read = None
    if plug_mask is None and label != "relook":
        side_read = _side_shoulder(ctx, observation, axis, scene_plug_half_h, plug_query)
    if plug_mask is None and side_read is None:
        print(
            f"[{label}:vision] no supported {plug_query!r} read; the plug is occluded or the jaws may be empty",
            flush=True,
        )
        return dict(NOTHING)
    if side_read is not None:
        plug_point, plug_pts = side_read
        print(f"[{label}:vision] shoulder from side RGB-D and known plug body length", flush=True)
    else:
        plug_mask = _foreground_mask(plug_mask, depth)
        plug_pts = _to_world(plug_mask, depth, K, T)
        if len(plug_pts) < MIN_MASK_PIXELS:
            return dict(NOTHING)  # type: ignore[return-value]
        # THE ORIGIN, THE SAME WAY THE BENCH READ FINDS IT -- the shoulder step
        # where the tab meets the body. This took a median of the mask, and a
        # median is the plug's centre of *area*, which sits about 10 mm behind the
        # point every offset in the planner is stated from. The bench body was
        # fixed for exactly this and the re-look was not, so the two slots disagreed
        # about what "the plug" meant by a centimetre -- and since the stroke is a
        # displacement from here, that centimetre went straight into the mate.
        # Measured: the mate landed 15-17 mm off axis on drawn layouts whose plug
        # perception was good to 1.2 mm.
        #
        # The step is measured along the plug's own long axis, and the wrist reads
        # it at 0.33 mm/px -- the finest view of the part anything in this cell has.
        plug_axis_xy = _principal_xy(plug_pts)
        plug_axis = np.array([float(plug_axis_xy[0]), float(plug_axis_xy[1]), 0.0])
        plug_perp = np.array([-plug_axis[1], plug_axis[0], 0.0])
        plug_point = (
            _shoulder(plug_pts, plug_axis) * plug_axis
            + float(np.median(plug_pts @ plug_perp)) * plug_perp
            # THE HEIGHT, FROM THE TOP FACE AND THE PART'S OWN HALF-HEIGHT.
            # A median z of a part seen from above is a reading of its top surface,
            # not its centre -- measured, the held plug came back 8 mm high, and the
            # stroke is a displacement so all 8 went into the mate. The bench read
            # could bisect against the surface the plug was resting on; nothing is
            # under it now, so the half-height it measured back there is carried
            # here instead.
            + np.array([0.0, 0.0, float(np.percentile(plug_pts[:, 2], 90.0)) - float(scene_plug_half_h)])
        )

    held_frame = None
    if label == "relook":
        import cv2  # noqa: PLC0415

        # The old channel-ratio gate rejects the pale gold produced by RTX,
        # even when the segmenter correctly finds the held connector.
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
        orange = cv2.inRange(hsv, np.array([5, 24, 48], np.uint8), np.array([40, 255, 255], np.uint8)) > 0
        held_frame = fit_top_frame(_to_world(np.asarray(plug_mask, dtype=bool) & orange, depth, K, T), axis)
        if held_frame:
            normal = np.asarray(held_frame["up"])
            face = np.asarray(held_frame["point"])
            top = plug_point.copy()
            top[2] = (face @ normal - top[:2] @ normal[:2]) / normal[2]
            plug_point = top - float(scene_plug_half_h) * normal
            refined = fit_plug(
                plug_pts,
                ctx.tool("usbc.plug_shape"),
                plug_point,
                held_frame["axis"],
                held_frame["up"],
            )
            if refined is None:
                print(f"[{label}:vision] no supported held-plug surface fit", flush=True)
                return dict(NOTHING)
            held_frame = refined
            plug_point = np.asarray(refined["point"])
            print(
                f"[{label}:vision] held plug surface fit: {held_frame['inlier_count']} points, "
                f"axis={np.round(held_frame['axis'], 4).tolist()}, "
                f"residual={held_frame['residual_m'] * 1000:.3f} mm",
                flush=True,
            )
    if not _consistent_with_grasp(plug_point, observation):
        print(
            f"[{label}:vision] inferred shoulder is outside the held-plug envelope; "
            "the visible fragment cannot support a correction",
            flush=True,
        )
        return dict(NOTHING)
    moved = float(np.linalg.norm(mouth - ground_mouth) * 1e3)
    print(
        f"[{label}:vision] mouth {_mm(mouth)} [{source}] ({moved:.2f} mm from the bench read)  "
        f"plug in the jaws {_mm(plug_point)} from {len(plug_pts)} px  "
        f"axis ({axis[0]:+.2f}, {axis[1]:+.2f})",
        flush=True,
    )
    return {
        "route": "found",
        "relook_body": "vision",
        "mate_plug_point": [float(v) for v in plug_point],
        "mate_mouth_point": [float(v) for v in mouth],
        "mate_bore_axis": [float(v) for v in axis],
        "mate_plug_axis": held_frame["axis"] if held_frame else [],
        "mate_plug_up": held_frame["up"] if held_frame else [],
        "mate_moved_mm": moved,
    }


def _side_shoulder(ctx, observation, axis, half_height, query):
    """Axial correction when the wrist cannot see past its own fingers.

    The orange rear of the 34 mm connector body remains visible in profile.
    Its body length is a part dimension; all position data comes from RGB-D.
    Requiring most of the body span rejects isolated glints and short fragments.
    """
    try:
        rgb, depth, K, T = _frame(_camera(observation, "side"), "side")
    except (KeyError, ValueError):
        return None
    result = ctx.tool("sam3.segment_text", image=rgb, query=query, max_results=0)
    masks = [
        np.asarray(m, dtype=bool)
        for m, score in zip(result.get("masks") or [], result.get("scores") or [], strict=True)
        if float(score) >= PLUG_SCORE
    ]
    if not masks:
        return None
    points = _to_world(_foreground_mask(np.logical_or.reduce(masks), depth), depth, K, T)
    if len(points) < MIN_MASK_PIXELS:
        return None
    along = points @ axis
    rear, front = np.percentile(along, [2.0, 98.0])
    if not 0.030 <= front - rear <= 0.055:
        return None
    perp = np.array([-axis[1], axis[0], 0.0])
    shoulder = (
        (rear + 0.034) * axis
        + float(np.mean(np.percentile(points @ perp, [5.0, 95.0]))) * perp
        + np.array([0.0, 0.0, float(np.percentile(points[:, 2], 90.0)) - half_height])
    )
    return (shoulder, points) if _consistent_with_grasp(shoulder, observation) else None


def _foreground_mask(mask, depth):
    """Reject background depth assigned to colored silhouette edge pixels.

    The 47 mm plug fits within a 70 mm depth band around its median surface.
    RTX edge pixels can retain the object's color with the bench's depth;
    even a few then dominate PCA and the shoulder-width profile.
    """
    if depth is None:
        return np.asarray(mask, dtype=bool)
    depth = np.asarray(depth).squeeze()
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
    if not valid.any():
        return valid
    return valid & (abs(depth - np.median(depth[valid])) < 0.035)


def _mm(v) -> str:
    return "(" + ", ".join(f"{float(x) * 1e3:.1f}" for x in v) + ") mm"


def _consistent_with_grasp(point, observation):
    """Reject an occluded mask whose inferred origin cannot be in these jaws.

    A valid relook has the shoulder roughly 32-45 mm ahead of the TCP.
    This loose bound permits slip and tilt; it rejects, for example, a thin
    orange sliver back-projected 52 mm above the hand during a reseat.
    """
    arms = observation.get("arms") or []
    if not arms:
        return True
    pose = arms[0].get("ee_pose", {}).get("position", {})
    if not all(key in pose for key in ("x", "y", "z")):
        return True
    delta = np.asarray(point) - np.array([pose[key] for key in ("x", "y", "z")])
    return bool(abs(delta[2]) <= 0.025 and np.linalg.norm(delta) <= 0.065)
