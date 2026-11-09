"""Is the plug seated -- measured, not scored.

The non-privileged body of the verify slot, and the one place where the two
bodies of a slot are **not** measuring the same quantity. That is worth saying
plainly rather than papering over.

The oracle reads the evaluator's own numbers: tab-tip depth past the mouth plane
along the bore axis, computed from two body poses and a declared cavity. No
camera can see a tab tip that is inside a socket, and none of the three
quantities the goal gates on is visible from outside the mate.

**What a camera can see is the gap.** When the plug is home its shoulder is on
the mouth ring and there is nothing between the two parts; when it is short
there is a bright slot of bench between them, and its width along the bore axis
is exactly how far the stroke fell short. So this body measures ``mate_gap`` --
which the documents already record every step as an observation, precisely
because it is the quantity that shows a mate losing the fit gradually rather
than at once -- and reads a seat as a gap that has closed.

That is a *different instrument reading the same event*, not a reproduction of
the score, and the two will not agree to the tenth of a millimetre. What they
should agree about is the verdict, and where they do not, the pair is the
measurement: a vision verify that says seated on an episode the evaluator scores
zero is the honest report of a perception limit, and the last thing it should do
is quietly consult ``sim.*`` to avoid saying so.

Two more clauses, and both are genuinely visible:

``hand_clear``
    The goal wants the hand *away*, not merely open -- a plug still between the
    fingers at depth is held in position by two servos rather than retained by
    the socket, which is precisely what is under test. The tool centre is
    proprioception and the plug is in the frame, so the distance between them is
    a real measurement.

``lat``
    How far the plug sits off the receptacle's centre, across the mate line.
    **The guard the gap cannot do without**: a plug beside the mouth shows no
    gap at all, because there is nothing between the two parts to measure.

``released``
    The fingers reaching their command, which is what an empty hand does.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

REQUIRED_TOOLS = ("sam3.segment_text",)
WHY_ABSENT = (
    "the sam3 segmenter is not resolvable here, so the vision body of the verify slot is "
    "declared and skipped. It is not a task failure and it is not a perception result: run "
    "the oracle body (--perception oracle, the default), or install transformers and torch."
)

CAMERA = "side"
"""Which view the gap is measured in.

The placed profile view, looking along the mate line's own perpendicular from
160 mm. It is in this rig because the reference put it there to *read insertion
depth in profile*, which is the same job: a gap between two parts is widest and
least foreshortened when the line between them lies across the image.

``mating`` was the other candidate and is worse for this one thing. It looks
down the mate line at a shallow angle, so a 2 mm gap projects to under a
pixel-and-a-half and a plug that has stopped short looks like a plug that has
not.
"""

PLUG_QUERY = "orange plug"
RECEPTACLE_QUERY = "blue object"
PLUG_SCORE = 0.20
RECEPTACLE_SCORE = 0.20
MIN_MASK_PIXELS = 40
"""The ground body's prompts and floors, restated -- same scene, same palette,
and its docstring carries the measured table these came from. A body that
grounded on different words would be measuring a different thing."""

SEATED_GAP_M = 0.004
"""How wide the gap between the parts may be and still read as seated [m].

**Not the goal's threshold, and it cannot be.** The goal gates tab-tip depth at
10.4 mm of a 13 mm bore, which is a quantity inside the socket. What is outside
is the shoulder-to-face distance, and the two are related by the plug's own
geometry: a plug 2.6 mm short of home is 2.6 mm short of the gate as well,
because both measure the same translation along the same axis.

4 mm is therefore a *deliberately loose* stand-in for 2.6 -- loose because a
mask edge on a 0.4 mm/px view carries a pixel or two of slop at each of the two
edges it is differencing, and a verify that failed inside its own resolution
would report a perception limit as a task failure. What it still refuses is the
reference's own outcome: an open-loop stroke that stops 5-9 mm out.
"""

SEATED_LAT_M = 0.010
"""How far off the receptacle's own centre the plug may sit and still be in it [m].

**The guard the gap reading cannot do without**, and it was missing: a plug
resting *beside* the mouth shows no gap along the mate line at all, because
there is nothing between the two parts to measure. Measured on a failing
bulkhead episode -- this body reported ``gap 1.76 mm (seated)`` and
``off-axis 15.50 mm`` in the same breath, called it inserted, and the evaluator
scored the episode 0.00. It had the number and did not gate on it.

10 mm rather than the document's 8.79 mm half-diagonal, and the difference is
honest: the goal measures the plug against the *bore's* axis, and what a camera
has is the plug's median against the **receptacle's** median, which is that
part's whole silhouette -- a barrel with a flange at one end and a boot at the
other is not centred on its own bore. So this is a looser test of a nearby
quantity, sized to refuse a plug beside the mouth without failing one that is
in it.
"""

JOINT_SPAN_M = 0.020
"""How much of each part, back from the joint, the across-the-axis reading uses [m].

Twenty millimetres is long enough that a slice holds hundreds of points and
short enough to stay on the connector: on the receptacle it covers the nose and
the flange, which are bodies of revolution about the bore and so are centred on
it; on the plug it covers the tab and the shoulder. Neither window reaches the
cable, which is the thing that made the old whole-mask median useless.
"""

CLEAR_M = 0.05
"""How far the tool centre must be from the plug to count as let go [m].

The document's own release clause, restated because a camera has to be told what
"away" means and the number is the task's rather than the instrument's.
# portable: a task threshold on the release, not a property of any hand.
"""


class Output(TypedDict):
    route: str
    verify_body: str
    depth_mm: float
    gate_mm: float
    lat_mm: float
    mate_gap_mm: float
    hand_clear: bool


NOTHING: Output = {
    "route": "short",
    "verify_body": "vision",
    "depth_mm": 0.0,
    "gate_mm": 0.0,
    "lat_mm": 0.0,
    "mate_gap_mm": 0.0,
    "hand_clear": False,
}


def _missing(ctx: NodeContext) -> list[str]:
    registry = getattr(ctx, "_tool_registry", None)
    if registry is None:
        return []
    return [name for name in REQUIRED_TOOLS if name not in registry]


def _camera(observation: Any, name: str) -> dict[str, Any]:
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
    cam = np.stack(
        [(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)], axis=1
    )
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


def _tcp(observation: Any, ctx: NodeContext) -> np.ndarray | None:
    """The work arm's tool centre, from proprioception. ``None`` if unavailable."""
    del observation
    try:
        ee = ctx.tool("robot.get_ee_pose")
    except Exception:  # noqa: BLE001 -- a cell without this tool is not a failure here
        return None
    pose = (ee.get("pose") or ee) if isinstance(ee, dict) else ee
    position = pose.get("position", pose) if isinstance(pose, dict) else pose
    if isinstance(position, dict):
        return np.array([float(position["x"]), float(position["y"]), float(position["z"])])
    flat = np.asarray(position, dtype=np.float64).reshape(-1)
    return flat[:3] if flat.size >= 3 else None


def run(
    ctx: NodeContext,
    scene_bore_axis: list | None = None,
    scene_mouth_point: list | None = None,
    scene_receptacle_held: bool = True,
    settle_steps: int = 48,
    camera: str = CAMERA,
    plug_query: str = PLUG_QUERY,
    receptacle_query: str = RECEPTACLE_QUERY,
) -> Output:
    absent = _missing(ctx)
    if absent:
        raise RuntimeError(f"the verify slot's vision body needs {list(absent)}: {WHY_ABSENT}")
    # Let the mate ring down before reading it, exactly as the oracle body does.
    # A freshly released pair is still oscillating on the support brace, and the
    # two bodies have to be reading the same instant of the same episode or the
    # comparison between them is a comparison of settling times.
    ctx.tool("robot.wait_steps", steps=int(settle_steps), timeout_s=60.0)

    observation: Any = ctx.tool("robot.get_observation")
    rgb, depth, K, T = _frame(_camera(observation, camera), camera)

    plug_mask = _detect(ctx, rgb, plug_query, PLUG_SCORE)
    rec_mask = _detect(ctx, rgb, receptacle_query, RECEPTACLE_SCORE)
    if plug_mask is None or rec_mask is None:
        print(f"[verify:vision] {camera!r} shows no plug or no receptacle", flush=True)
        return dict(NOTHING)  # type: ignore[return-value]
    plug_pts = _to_world(plug_mask, depth, K, T)
    rec_pts = _to_world(rec_mask, depth, K, T)
    if len(plug_pts) < MIN_MASK_PIXELS or len(rec_pts) < MIN_MASK_PIXELS:
        return dict(NOTHING)  # type: ignore[return-value]

    axis = np.asarray([float(v) for v in (scene_bore_axis or (0.0, 1.0, 0.0))], dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
    perp = np.array([-axis[1], axis[0], 0.0])

    # The gap: from the plug's leading edge to the receptacle's near face, along
    # the mate line. Percentiles at both ends rather than extremes, because each
    # mask carries a pixel or two of whatever is behind it and an extreme is
    # exactly the pixel most likely to be wrong.
    plug_lead = float(np.percentile(plug_pts @ axis, 95.0))
    face = float(np.percentile(rec_pts @ axis, 5.0))
    fixed_mouth = (np.asarray(scene_mouth_point, dtype=float)
                   if scene_mouth_point and not scene_receptacle_held else None)
    if fixed_mouth is not None:
        # The large block's visible silhouette can be far from its aperture.
        # A fixed socket retains the opening measured by the earlier cameras.
        face = float(fixed_mouth @ axis)
    gap = max(0.0, face - plug_lead)

    # Off-axis: how far the plug sits across the mate line from the receptacle's
    # own centre. This is the guard the depth reading cannot do without -- a plug
    # beside the socket looks, along the axis alone, exactly like one inside it.
    #
    # MEASURED WHERE THE TWO PARTS MEET, not over the whole of either mask, and
    # the difference is the whole reading. Both these masks carry a CABLE: SAM3
    # takes the flex rooted on each connector along with it, which is why the
    # receptacle's outline runs 84 mm on a 66 mm part. A cable snaking off across
    # the bench drags a median sideways, and a median of the whole silhouette is
    # then a statistic about the flex rather than about the mate. Measured on
    # this variant it read **14.8-16.3 mm off-axis on a plug whose gap was
    # 0.00 mm** -- flush and fully seated -- and this body called those short and
    # aborted the episode. The threshold was already loosened to 10 mm once, to
    # cover the same effect, and 10 mm was not enough because the fault is not a
    # tolerance.
    #
    # So each part is taken over a slice near the joint: the receptacle from its
    # own near face back one FLANGE_SPAN, the plug from its leading edge forward
    # the same. And the statistic is the midpoint of the 2/98 extent rather than
    # a median -- a silhouette lit from one side has more points on the lit wall,
    # which is the bias `relook_vision._slices` records at three times the spread.
    def _centre(points: np.ndarray, lo: float, hi: float) -> float | None:
        take = points[(points @ axis >= lo) & (points @ axis <= hi)]
        if len(take) < MIN_MASK_PIXELS:
            return None
        across = take @ perp
        return 0.5 * float(np.percentile(across, 2.0) + np.percentile(across, 98.0))

    rec_centre = (float(fixed_mouth @ perp) if fixed_mouth is not None
                  else _centre(rec_pts, face, face + JOINT_SPAN_M))
    plug_centre_across = _centre(plug_pts, plug_lead - JOINT_SPAN_M, plug_lead)
    if rec_centre is None or plug_centre_across is None:
        print(f"[verify:vision] {camera!r} shows too little of the joint to measure across it",
              flush=True)
        return dict(NOTHING)  # type: ignore[return-value]
    lat = abs(plug_centre_across - rec_centre)
    if fixed_mouth is not None:
        low, high = np.percentile(plug_pts[:, 2], [5., 95.])
        vertical_gap = max(float(low - fixed_mouth[2]), float(fixed_mouth[2] - high), 0.)
        lat = float(np.hypot(lat, vertical_gap))

    tcp = _tcp(observation, ctx)
    plug_centre = np.median(plug_pts, axis=0)
    distance = float(np.linalg.norm(tcp - plug_centre)) if tcp is not None else float("nan")
    grip = ctx.tool("usbc.grip_state")
    released = not bool(grip["stalled"])
    hand_clear = bool(released and (not np.isfinite(distance) or distance > CLEAR_M))

    on_axis = bool(lat <= SEATED_LAT_M)
    seated = bool(gap <= SEATED_GAP_M and on_axis and hand_clear)
    print(
        f"[verify:vision] gap {gap * 1e3:.2f} mm (under {SEATED_GAP_M * 1e3:.1f})  "
        f"off-axis {lat * 1e3:.2f} mm (under {SEATED_LAT_M * 1e3:.1f})  "
        f"tcp {distance * 1e3:.1f} mm  released={released}  -> "
        f"{'inserted' if seated else 'short'}",
        flush=True,
    )
    return {
        "route": "inserted" if seated else "short",
        "verify_body": "vision",
        # There is no camera reading of tab-tip depth, and inventing one would be
        # the oracle's number wearing a camera's name. What is reported instead
        # is how much of the stroke is visibly left: zero when home.
        "depth_mm": 0.0,
        "gate_mm": SEATED_GAP_M * 1e3,
        "lat_mm": lat * 1e3,
        "mate_gap_mm": gap * 1e3,
        "hand_clear": hand_clear,
    }
