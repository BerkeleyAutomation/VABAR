"""Locate the connector pair from ONE calibrated RGB-D view of the bench.

The top view supplies everything this body publishes: the plug's shoulder, the
receptacle's heading, and a rough mouth taken as the receptacle's own front
along that heading. It is deliberately the only camera here. A second placed
view aimed down the mate line used to measure the opening, and being handed a
camera pointed at the answer is not perception -- so the opening is now measured
by ``inspect_vision`` from the WRIST, after ``aim`` has pointed it, with the
same ``bore_cv`` plane fit and depth recess on both tiers.

What this body owes the graph is therefore a pose good enough to aim at, not a
mate-grade one. No authored fixture positions or simulator object states are
consumed.
"""

import re
import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

# Sibling helpers resolve inside the GaP skill package.
from .bore_cv import recess_frame

REQUIRED_TOOLS = ("sam3.segment_text", "usbc.instruction",
                  "usbc.receptacle_shape")
"""What this body needs beyond the runtime's own.

``robot.get_observation`` is the runtime's and is always present. The
back-projection is done here in numpy rather than through a bundle's
``geometry.mask_to_world_points``: it is eight lines of intrinsics arithmetic,
and depending on a second distribution to do it would make this body
un-runnable wherever that distribution is missing for a reason that has nothing
to do with perception.
"""

WHY_ABSENT = (
    "the sam3 segmenter is not resolvable here, so the vision body of the ground slot is "
    "declared and skipped. It is not a task failure and it is not a perception result: run "
    "the oracle body (--perception oracle, the default), or install transformers and torch "
    "and let robosimstudio_gap.perception_tools load the weights."
)

BENCH_CAMERA = "topdown"
"""The one view this body reads.

It was two until the rig was cut to three cameras. The other was ``mating``, and
its own docstring said why it had to go: it "was aimed by somebody who knew where
the mate line was, so what this body earns is finding the parts in the frame and
not choosing where to look from."
"""

PLUG_QUERY = "orange plug"
RECEPTACLE_QUERY = "blue object"
"""The prompts, and the colour word in each is the whole of it.

**Measured on this render**, SAM3 through
:mod:`~robosimstudio_gap.perception_tools`, best-detection score in
each of the four frames a vision body of this graph reads:

=========================  ==========  ==========  ==========  ==========
Prompt                     port        bulkhead    port        bulkhead
                           topdown     topdown     mating      mating
=========================  ==========  ==========  ==========  ==========
``orange plug``            **0.727**   **0.606**   **0.822**   **0.832**
``blue object``            **0.930**   **0.962**   **0.885**   **0.952**
``blue block``             0.866       --          --          --
``blue connector socket``  0.045       --          --          --
``blue connector``         --          0.921       0.072       0.478
``socket``                 0.130       --          --          --
``receptacle``             0.141       --          --          --
=========================  ==========  ==========  ==========  ==========

Read the bottom three rows. ``socket`` and ``receptacle`` -- the words for what
the thing *is* -- score 0.13 and 0.14 and would be rejected by any sane floor,
while ``blue object`` -- a word for what it *looks like*, carrying no idea of a
connector at all -- scores 0.93. The detector is keying on the palette, which is
a property of a flat-shaded synthetic render rather than of SAM3, and it is the
first thing that breaks when the bench is re-lit or repainted. The weave's port
measured the same effect on its own scene and wrote it down; this is that
finding reproduced on a second suite.

``blue object`` over ``blue block`` for a second reason: the two receptacles are
different shapes -- a rectangular block and a barrel with a flange -- and a
shape word that fits one is a prompt that has learned which variant it is
running. The colours are
:data:`~robosimstudio.core.render.palette.ACCENT_PLUG` and ``ACCENT_PORT``,
which the loader states on both parts precisely so a frame shows where one is
relative to the other.
"""

PLUG_SCORE = 0.20
RECEPTACLE_SCORE = 0.20
"""Detection floors. Low, deliberately: on a render like this the true
detections score in the eighties and nineties while the false ones do not
appear at all, so a high floor buys nothing and a low one keeps a dimmer
re-lit scene working."""

MIN_MASK_PIXELS = 40
"""How many pixels a detection needs before it is measured.

The plug is ~63 x 29 px in the top-down view, so a real one clears this by a
factor of forty. What it rejects is a sliver of highlight on the bench.
"""

SHOULDER_BINS = 40
SHOULDER_DROP = 0.7
"""How the plug's own origin is found: the step where its tab meets its body.

**A centroid is not a datum.** The mask's centre of area sits about 10 mm behind
the plug's origin, because the part is 34 mm of body and 13 mm of tab about that
origin -- measured against ground truth, a median centroid lands 1.7 mm from the
*centroid* and 12 mm from the *origin*, and the planner's grasp offset is stated
from the origin. Handing over the wrong one is a systematic 12 mm error that no
amount of pixels fixes.

What a camera can find instead is the **shoulder**: the plug is 10 mm across its
body and 4.8 mm across its tab, and that step is where the origin is. So the
mask is binned along its own long axis, the cross-axis width is measured per
bin, and the origin is the first bin (walking from the rear) whose width falls
under ``SHOULDER_DROP`` of the body's. 40 bins over a 47 mm part is about a
millimetre each, which is three pixels at this standoff.
"""

SUPPORT_RING_PX = 12
"""How far outside the plug's mask the surface it rests on is sampled [px].

**Measured around the part, not across the frame**, and the difference was a
25 mm error. The first version took a far percentile of the whole top-down
depth image on the reasoning that the deepest thing in view is what everything
is standing on -- and in this cell the deepest thing in view is the *table*,
50 mm below the bench the plug is actually on. The plug's centre came out
25 mm low, every stage of the stroke inherited it, and the schedule rose 36.8 mm
where it should have risen 12.4.

A ring just outside the part's own silhouette cannot make that mistake: whatever
a thing is resting on is touching it. 12 px is about 4.7 mm at this standoff --
wide enough to clear the mask's own soft edge and narrow enough to stay on the
bench rather than reaching past it.
"""

HELD_WORDS = ("holding", "held", "bulkhead")
"""Words in the instruction that mean the far end of the mate is not bolted down.

Language, and it is the honest limit of what a camera plus a command gives you.
See the module docstring: a receptacle resting in a cradle and one bolted to the
bench are the same picture.
"""


class Output(TypedDict):
    route: str
    scene_body: str
    scene_plug_point: list
    scene_plug_yaw_deg: float
    scene_plug_half_h: float
    scene_mouth_point: list
    scene_bore_axis: list
    scene_barrel_point: list
    scene_receptacle_held: bool


NOTHING: Output = {
    "route": "not_found",
    "scene_body": "vision",
    "scene_plug_point": [],
    "scene_plug_yaw_deg": 0.0,
    "scene_plug_half_h": 0.0,
    "scene_mouth_point": [],
    "scene_bore_axis": [],
    "scene_barrel_point": [],
    "scene_receptacle_held": False,
}
"""Every declared key, empty -- the same contract the oracle's miss path keeps."""


# --------------------------------------------------------------- the plumbing
def _missing(ctx: NodeContext) -> list[str]:
    """Which required tools this registry does not carry.

    The weave's probe and the peel's, deliberately identical: a body that cannot
    run must fail as a missing dependency and never as a perception result. The
    alternative was measured elsewhere in this corpus and written down -- with
    the bundle absent a graph reached its detector, took the miss exit for every
    object in turn, and reported 0.00 with a plausible-looking trace.
    """
    registry = getattr(ctx, "_tool_registry", None)
    if registry is None:
        return []
    return [name for name in REQUIRED_TOOLS if name not in registry]


def _camera(observation: Any, name: str) -> dict[str, Any]:
    """One camera's frame, whichever shape the runtime publishes them in.

    ``observation["cameras"]`` is a dict keyed by camera name on the runtime
    this repo pins and a list of name-carrying frames on others. Both are
    handled because the difference is a runtime version, not a decision this
    body gets to make.
    """
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
    have = [str(f.get("name", "?")) for f in (frames or ())]
    raise ValueError(f"no camera {name!r} in the observation; have {have}")


def _frame(block: dict[str, Any], name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(rgb, depth, K, world_from_camera)`` out of one camera block."""
    images = block.get("images") if isinstance(block.get("images"), dict) else {}

    def _first(*keys):
        # `or`-chaining these would ask numpy for the truth value of an array.
        for src in (images, block):
            for key in keys:
                v = src.get(key)
                if v is not None:
                    return v
        return None

    rgb = np.asarray(_first("rgb"), dtype=np.uint8)
    depth = _first("depth", "depth_data")
    if depth is None:
        raise ValueError(
            f"camera {name!r} carries no depth. A 0.15 mm-class fit is measured in metres off a "
            "bench, so a colour-only rig cannot run this body."
        )
    K = np.asarray(block["intrinsics"], dtype=np.float64).reshape(3, 3)
    # The runtime hands a camera pose as a typed Se3Pose -- `position` and
    # `rotation` dicts, wxyz -- not as the flat 7-vector the env publishes
    # underneath. Both spellings are read because a body written against one and
    # run on the other fails with "float() argument must be ... not 'dict'",
    # which names the symptom and not the seam.
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
    """Masked pixels to world metres, in the renderer's +z-forward/+y-down basis."""
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
    """The best mask for one text prompt above ``floor``, or ``None``."""
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


# ------------------------------------------------------------- the geometry
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


def _yaw_across(axis_xy: np.ndarray, reference: np.ndarray) -> float:
    """The heading the jaws close along, given a part's own long axis [deg].

    A quarter turn off the part's heading, which is what "close across it"
    means, resolved against ``reference`` -- the bore axis. The same rule
    ``ToolsUsbc._yaw_across`` applies to the plug's quaternion, and it has to
    be: two conventions here would rotate the wrist between the two bodies and
    read as a perception error.

    A parallel jaw is symmetric under a half turn, so ``yaw`` and ``yaw + 180``
    are the **same grasp** -- and they are not the same *wrist*. This hand
    carries its camera 70 mm off to one side, so the two differ by swinging the
    lens to the other side of the gripper: measured, the eye-in-hand at the look
    station showed the receptacle under one and showed the gripper body and an
    empty bench under the other, and the re-look reported "no receptacle" on a
    scene where nothing was wrong.

    The ambiguity is real and it is unavoidable in the *measurement*: a
    principal axis has no sign, and an SVD returns whichever direction it likes.
    So it is resolved against the one direction in this scene that does have a
    sign -- the bore axis, which points from the plug into the receptacle. The
    representative kept is the one within a quarter turn of "across the mate
    line", which is where the reference schedule's own 180 degrees sits.
    """
    yaw = float(np.degrees(np.arctan2(float(axis_xy[1]), float(axis_xy[0]))) + 90.0)
    across = float(np.degrees(np.arctan2(float(reference[1]), float(reference[0]))) + 90.0)
    return yaw if abs((yaw - across + 180.0) % 360.0 - 180.0) <= 90.0 else yaw + 180.0


def _support_z(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, T: np.ndarray) -> float | None:
    """World z of the surface a masked part is resting on.

    A ring just outside the part's own silhouette, back-projected and taken at
    its median height. Whatever a thing rests on is touching it, so the pixels
    immediately around it are that surface -- which a percentile over the whole
    frame is not, in any cell that has more than one level in it.
    """
    m = np.asarray(mask) > 0
    v, u = np.nonzero(m)
    if len(u) == 0:
        return None
    pad = int(SUPPORT_RING_PX)
    v0, v1 = max(0, int(v.min()) - pad), min(m.shape[0] - 1, int(v.max()) + pad)
    u0, u1 = max(0, int(u.min()) - pad), min(m.shape[1] - 1, int(u.max()) + pad)
    box = np.zeros_like(m)
    box[v0 : v1 + 1, u0 : u1 + 1] = True
    ring = box & ~m
    pts = _to_world(ring, depth, K, T)
    if len(pts) < MIN_MASK_PIXELS:
        return None
    return float(np.median(pts[:, 2]))


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


PROFILE_BIN = 0.002
FACE_LEVEL = 0.5
BODY_PERCENTILE = 75.0
MIN_SLICE = 6
"""How a BARREL's near face is found, and why it is not `_recess_mouth`'s job.

`_recess_mouth` looks for the points lying behind a face and takes the midpoint
of their extent. That is right for the block port, whose bore is a hole in a
slab the `mating` camera sees square on. It is wrong for the bulkhead, and the
error is not small: measured against ground truth over two drawn layouts,

    dx +9.80 mm      slot half-width  x2 = 9.80
    dz +3.80 mm      slot half-height x2 = 4.00

identical to a hundredth of a millimetre across episodes. That is not noise and
it is not a centroid pulled toward a lit wall -- it is the whole half-extent,
which is what you get when only ONE inner wall of the aperture is visible and
the midpoint of "the extent" collapses onto the edge that is. This slot is
19.6 x 8.0 mm against the port's, and from the `mating` camera's angle the far
wall is behind the near one.

The bulkhead does not need a recess found. Its mouth is on the barrel axis at
the front face, and both of those are things a silhouette gives: the face from
where the outline starts, the axis height from the flange's top surface minus
its catalogue radius. Same three statistics the re-look reads off the wrist --
see `relook_vision` for the table -- and on the top-down at the ground station
they come back to **1.00 mm** where `_recess_mouth` was 10.5 out.
"""


def _slices(points: np.ndarray, axis: np.ndarray, perp: np.ndarray) -> dict[str, np.ndarray]:
    """The mask's profile along the bore axis. `relook_vision._slices`, verbatim."""
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
    """The leading half-density crossing. `relook_vision._face`, verbatim."""
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


def _barrel_mouth(points: np.ndarray, axis: np.ndarray, shape: dict) -> np.ndarray | None:
    """A barrel receptacle's mouth, from its silhouette rather than its recess."""
    perp = np.array([-axis[1], axis[0], 0.0])
    profile = _slices(points, axis, perp)
    face = _face(profile)
    lo, hi = (float(v) for v in shape["flange_from_mouth"])
    band = (profile["s"] > face + lo) & (profile["s"] < face + hi) & np.isfinite(profile["mid"])
    if not bool(band.any()):
        return None
    return (
        face * axis
        + float(np.nanmean(profile["mid"][band])) * perp
        + np.array([0.0, 0.0, float(np.nanmean(profile["top"][band])) - float(shape["flange_radius"])])
    )


def _recess_mouth(mask, depth, K, T, axis):
    """Compatibility helper returning the camera-measured opening centre."""
    return recess_frame(mask, depth, K, T, axis)[0]


def run(
    ctx: NodeContext,
    instruction: str = "",
    bench_camera: str = BENCH_CAMERA,
    plug_query: str = PLUG_QUERY,
    receptacle_query: str = RECEPTACLE_QUERY,
) -> Output:
    absent = _missing(ctx)
    if absent:
        raise RuntimeError(f"the ground slot's vision body needs {list(absent)}: {WHY_ABSENT}")

    observation: Any = ctx.tool("robot.get_observation")
    # The instruction comes from a TOOL, not from a subgraph input and not off
    # the observation, and both of those were tried.
    #
    # A subgraph's declared inputs must bind to an upstream *subgraph's*
    # outputs; a top-level workflow input does not satisfy the graph validator,
    # which refuses the whole workflow with W8 before anything runs. And the
    # observation does carry it -- the facade publishes it and the privilege
    # contract lists it by name -- right up until the runtime assembles its
    # typed Observation out of cameras and arms and drops it, silently, so this
    # body read the empty string and reported a held receptacle as bolted down.
    #
    # See `ToolsUsbc.instruction`, which says where the real fix belongs.
    if not str(instruction).strip():
        instruction = str(ctx.tool("usbc.instruction").get("instruction") or "")

    rgb, depth, K, T = _frame(_camera(observation, bench_camera), bench_camera)

    # -------------------------------------------------------------- the plug
    plug_mask = _detect(ctx, rgb, plug_query, PLUG_SCORE)
    if plug_mask is None:
        print(f"[ground:vision] no {plug_query!r} in {bench_camera!r}", flush=True)
        return dict(NOTHING)  # type: ignore[return-value]
    plug_pts = _to_world(plug_mask, depth, K, T)
    if len(plug_pts) < MIN_MASK_PIXELS:
        return dict(NOTHING)  # type: ignore[return-value]

    # What it is standing on, sampled around the part itself -- see
    # SUPPORT_RING_PX for the 25 mm this used to be wrong by.
    bench_z = _support_z(plug_mask, depth, K, T)
    if bench_z is None:
        print("[ground:vision] no support surface around the plug", flush=True)
        return dict(NOTHING)  # type: ignore[return-value]

    plug_axis_xy = _principal_xy(plug_pts)
    plug_axis = np.array([float(plug_axis_xy[0]), float(plug_axis_xy[1]), 0.0])
    plug_perp = np.array([-plug_axis[1], plug_axis[0], 0.0])
    # The 90th percentile rather than the max: a mask carries a few pixels of
    # bench at its rim, and a max reads those as the part's own surface.
    plug_top_z = float(np.percentile(plug_pts[:, 2], 90.0))
    # The ORIGIN, not the centroid -- the shoulder where the tab meets the body.
    # See SHOULDER_BINS for why a centroid is 12 mm wrong on this part.
    origin_along = _shoulder(plug_pts, plug_axis)
    plug_point = (
        origin_along * plug_axis
        + float(np.median(plug_pts @ plug_perp)) * plug_perp
        # Midway between what it is standing on and its own top face. Right for
        # anything resting on a plane, and it needs no knowledge of how tall
        # this particular part is.
        + np.array([0.0, 0.0, 0.5 * (bench_z + plug_top_z)])
    )

    # The receptacle pose is measured for fixed and movable parts alike.
    rec_mask = _detect(ctx, rgb, receptacle_query, RECEPTACLE_SCORE)
    if rec_mask is None:
        print(f"[ground:vision] no {receptacle_query!r} in {bench_camera!r}", flush=True)
        return dict(NOTHING)  # type: ignore[return-value]
    rec_pts = _to_world(rec_mask, depth, K, T)
    if len(rec_pts) < MIN_MASK_PIXELS:
        return dict(NOTHING)  # type: ignore[return-value]
    try:
        shape = ctx.tool("usbc.receptacle_shape")
    except Exception:
        shape = None
    axis_xy = _principal_xy(rec_pts)
    # Sign it so the axis runs INTO the receptacle -- away from the plug, which
    # is the direction the stroke pushes. The two parts' own separation is what
    # says which way that is, and it is the one thing in this scene that cannot
    # be ambiguous: the plug is on the near side of a mate it has not made yet.
    toward = rec_pts[:, :2].mean(axis=0) - plug_point[:2]
    if float(axis_xy @ toward) < 0.0:
        axis_xy = -axis_xy
    axis = np.array([float(axis_xy[0]), float(axis_xy[1]), 0.0])

    source = "camera"
    if shape is not None:
        found_mouth = _barrel_mouth(rec_pts, axis, shape)
        if found_mouth is None:
            print(
                f"[ground:vision] no flange band behind the face of {shape['receptacle']!r}",
                flush=True,
            )
            return dict(NOTHING)  # type: ignore[return-value]
        mouth = found_mouth
        source = "silhouette"

    # A ROUGH MOUTH, from the bench view alone.
    #
    # This used to read a second PLACED camera aimed down the mate line, and
    # that camera is gone: it had been pointed by somebody who already knew
    # where the answer was, so a body scored on perception was being handed its
    # own result. What grounding actually owes the rest of the graph is a mouth
    # good enough to AIM at -- `aim` points the wrist at it, and
    # `inspect_vision` then measures the real thing from that pose with
    # `bore_cv`'s plane fit and depth recess, the same skill on both tiers.
    #
    # The bench cloud already carries a rough mouth: the receptacle's own front
    # along its bore axis. `axis` points INTO the part (signed against the plug
    # above), so the face the plug arrives at is the LOW end of that projection.
    # The 5th percentile rather than the min, because a mask carries a few
    # pixels of bench at its rim and a min reads those as the part.
    #
    # It is deliberately crude -- it is a silhouette front, not an aperture, and
    # on the block port it is the block's near face rather than the hole in it.
    # Crude is enough to point a camera; `inspect` is what has to be right.
    if shape is None:
        perp = np.array([-axis[1], axis[0], 0.0])
        along_rec = rec_pts @ axis
        front = float(np.percentile(along_rec, 5.0))
        lateral = float(np.median(rec_pts @ perp))
        # HEIGHT COMES FROM THE PLUG, not from the receptacle's own silhouette.
        #
        # A top-down camera cannot see the height of a horizontal bore at all --
        # it photographs the receptacle's roof. The obvious guess, half way
        # between what the part stands on and its own top, is wrong by 26 mm on
        # the block port, because that part's bounding box is mostly the tall
        # flange ABOVE the hole. `aim` builds its station from this point, so
        # the error went straight into the approach: measured, the wrist rose
        # 39.8 mm where the same station on the old rig rose 12.5, and the plug
        # ended up held 117 mm over the mouth.
        #
        # What can be measured is the plug, and a mate is COAXIAL: whatever
        # height the bore is at, the plug has to arrive there, and the plug is
        # lying in plain view of this same camera. Its own centre is 12 mm under
        # the bore on this bench against the silhouette guess's 26 above -- and
        # unlike the guess it degrades sensibly, because it is a statement about
        # the mate rather than about one receptacle's proportions.
        mouth = (
            front * axis
            + lateral * perp
            + np.array([0.0, 0.0, float(plug_point[2])])
        )
        source = "topdown front, plug height"

    # ------------------------------------------------------------ the barrel
    # Where a side grasp takes a free receptacle: the middle of the part, on its
    # own axis, at the axis height the mouth reading gives. Measured rather than
    # offset from the mouth by a constant -- a constant would be this mesh's.
    along = rec_pts @ axis
    mid = 0.5 * (float(np.percentile(along, 5.0)) + float(np.percentile(along, 95.0)))
    barrel = mouth + (mid - float(mouth @ axis)) * axis

    # ---------------------------------------------------------------- the yaw
    # Resolved here rather than with the plug, because it needs the bore axis --
    # see _yaw_across for the half turn this settles and what it cost.
    plug_yaw = _yaw_across(plug_axis_xy, axis)

    # -------------------------------------------------------------- the held
    text = str(instruction).lower()
    held = any(re.search(rf"\b{word}\b", text) for word in HELD_WORDS)

    print(
        f"[ground:vision] bench z {bench_z * 1e3:.1f} mm  "
        f"plug {_mm(plug_point)} yaw {plug_yaw:.1f} deg ({len(plug_pts)} px)  "
        f"mouth {_mm(mouth)} [{source}]  "
        f"axis ({axis[0]:+.2f}, {axis[1]:+.2f})  "
        f"held={held} (language)",
        flush=True,
    )
    return {
        "route": "found",
        "scene_body": "vision",
        "scene_plug_point": [float(v) for v in plug_point],
        "scene_plug_yaw_deg": plug_yaw,
        # Half the part's own height, measured here because here it is
        # standing on something. See the re-look for what it is for.
        "scene_plug_half_h": float(0.5 * (plug_top_z - bench_z)),
        "scene_mouth_point": [float(v) for v in mouth],
        "scene_bore_axis": [float(v) for v in axis],
        "scene_barrel_point": [float(v) for v in barrel] if held else [],
        "scene_receptacle_held": bool(held),
    }


def _mm(v) -> str:
    return "(" + ", ".join(f"{float(x) * 1e3:.1f}" for x in v) + ") mm"
