"""What the overhead camera says the board and the gears are, in world metres. No simulator reads.

The paper-table vision graph measured everything with hand-rolled blob CV over
one RGB-D frame. This one asks the shared perception stack the other suites'
vision graphs ask -- ``sam3.segment_text`` -- and keeps only the arithmetic a
mask needs to become a place: the frame is back-projected once through its
own intrinsics, and every mask is a set of world points.

* **Gears and the pinion** are whatever SAM3 calls a ``gear``. The tooth class
  is the tip radius (module 2.5 mm: tip radii 22.5 / 27.5 / 32.5 mm for 16 / 20
  / 24 teeth, ten pixels apart at this camera), and the hue -- cobalt, teal,
  plum -- is asked to agree; the orange one is the pinion.
* **The button** is the ``red circle`` on the plate, colour-gated.
* **The pegs** are the ``pin`` tips SAM3 finds, and whatever else stands 25-45 mm
  proud of the bench inside the plate (a peg seen from above is a 1 mm cone
  tip, and SAM3 does not always call that a pin); each reading says which
  source found each peg.
* **Station tooth counts** come from the peg spacing (``module * (z_a + z_b) / 2``
  between meshing centres), seeded by the pinion's own radius; the peg line
  says which way the chain runs.
* **Heights** -- bench, plate, mat -- are measured in the frame: the bench is
  the depth histogram's mode, the plate is a ring around the button, the mat
  a ring around a loose gear.

Board geometry (pegs, plate, button) is cached after the first frame in which
every peg was seen, because a later frame is taken with the hand over the
chain and the hand hides whatever is under it; loose and held gears are
re-measured on every call. Everything is in metres and scaled by the frame's
own metres-per-pixel, so the rig's resolution is not assumed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

CAMERA = "overhead"
MODULE = 0.0025
"""The catalogue's module: what a tip radius reads as a tooth count on."""
FACE = 0.016
"""A gear's thickness: what a disc one face above a surface means."""
TOOTH_CLASSES = (16, 20, 24)
BUTTON_TRAVEL = 0.008
PEG_ABOVE_BENCH = 0.030
"""A peg top is at least this far above the bench; nothing else on the bench is."""
HELD_ABOVE_BENCH = 0.050
"""A disc this high is in the hand."""
MIN_DISC_M2 = 6.0e-4
"""Least top-face area a disc is believed at [m^2]: a 16-tooth gear's is 1.6e-3."""
MIN_PEG_M2 = 3.0e-6
MAX_PEG_M2 = 1.5e-4
GEAR_SCORE = 0.30
PIN_SCORE = 0.10
BUTTON_SCORE = 0.20
HELD_BAND = 0.006
SEAT_XY_TOL = 0.006
SEAT_Z_TOL = 0.0035

_CACHE: dict[str, Any] = {}


# ------------------------------------------------------------------ the frame
def frame(ctx, camera: str = CAMERA):
    """``(rgb, depth, K, world_from_camera)`` for one camera of a fresh observation."""
    obs = ctx.tool("robot.get_observation")
    frames = obs["cameras"] if isinstance(obs, dict) and "cameras" in obs else obs
    if isinstance(frames, dict):
        if camera not in frames:
            raise ValueError(f"no camera {camera!r} in the observation; have {sorted(frames)} (needs --renderer tiled)")
        block = frames[camera]
    else:
        block = next((f for f in frames or () if str(f.get("name", "")) == camera), None)
        if block is None:
            raise ValueError(f"no camera {camera!r} in the observation")
    images = block.get("images") if isinstance(block.get("images"), dict) else {}

    def first(*keys):
        for src in (images, block):
            for k in keys:
                if src.get(k) is not None:
                    return src[k]
        return None

    rgb = np.asarray(first("rgb"), dtype=np.uint8)
    depth = first("depth", "depth_data")
    if depth is None:
        raise ValueError(f"camera {camera!r} carries no depth")
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(block["intrinsics"], dtype=np.float64).reshape(3, 3)
    pose = block["pose"]
    if isinstance(pose, dict):
        p, r = pose.get("position", {}), pose.get("rotation", {})
        px, py, pz = (float(p[k]) for k in ("x", "y", "z"))
        w, x, y, z = (float(r[k]) for k in ("w", "x", "y", "z"))
    else:
        flat = np.asarray(pose, dtype=np.float64).reshape(-1)
        px, py, pz = flat[:3]
        w, x, y, z = flat[3:7]
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, (px, py, pz)
    return rgb, depth, K, T


def world_cloud(depth, K, T):
    """Every pixel as a world point, ``(H, W, 3)``; invalid depth becomes NaN."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    Z = np.where(np.isfinite(depth) & (depth > 1.0e-4), depth, np.nan)
    cam = np.stack([(u - K[0, 2]) * Z / K[0, 0], (v - K[1, 2]) * Z / K[1, 1], Z], axis=-1)
    return cam @ T[:3, :3].T + T[:3, 3]


class View:
    """One frame, back-projected, with the measurements the scripts ask for."""

    def __init__(self, rgb, depth, K, T):
        self.rgb = rgb
        self.depth = depth
        self.world = world_cloud(depth, K, T)
        self.z = self.world[..., 2]
        finite = np.isfinite(self.z)
        zs = self.z[finite]
        hist, edges = np.histogram(zs, bins=int((zs.max() - zs.min()) / 0.0005) + 1)
        self.bench_z = float(0.5 * (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1]))
        at_bench = finite & (np.abs(self.z - self.bench_z) < 0.003)
        self.px_m = float(np.median(depth[at_bench]) / K[0, 0]) if at_bench.sum() > 100 else float(np.nanmedian(depth[finite]) / K[0, 0])
        """Metres per pixel at the bench (measured there, not at whatever the arm puts in front of the camera)."""

    def area(self, px: int) -> float:
        return float(px) * self.px_m * self.px_m

    def disc(self, mask) -> dict[str, Any] | None:
        """A round thing under ``mask``: fitted centre, tip radius, top height, colour, fill."""
        m = np.asarray(mask) > 0
        pts = self.world[m]
        keep = np.isfinite(pts).all(axis=1)
        pts = pts[keep]
        if len(pts) < 12:
            return None
        # The top face only: a mask that took a gear's shadow or its rim with it
        # would fit a larger, off-centre circle. The top is the highest mode.
        top = float(np.percentile(pts[:, 2], 90))
        face = pts[pts[:, 2] > top - 0.004]
        if len(face) < 12:
            face = pts
        xy = face[:, :2]
        c = _circle_fit(xy) or (float(xy[:, 0].mean()), float(xy[:, 1].mean()))
        r = np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1])
        r_m = float(np.percentile(r, 98))
        px = int(len(face))
        fill = self.area(px) / max(math.pi * r_m * r_m, 1.0e-9)
        colour = self.rgb[m][keep][pts[:, 2] > top - 0.004] if len(face) < len(pts) else self.rgb[m][keep]
        return {
            "xy": (float(c[0]), float(c[1])),
            "top_z": top,
            "r_m": r_m,
            "px": px,
            "rgb": [float(v) for v in colour.mean(axis=0)],
            "fill": float(fill),
            "y_extent": (float(np.percentile(xy[:, 1], 1)), float(np.percentile(xy[:, 1], 99))),
        }

    def _blobs(self, mask, min_m2, max_m2=None):
        lab, n = ndimage.label(mask)
        out = []
        for i in range(1, n + 1):
            m = lab == i
            a = self.area(int(m.sum()))
            if a < min_m2 or (max_m2 is not None and a > max_m2):
                continue
            pts = self.world[m]
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) < 3:
                continue
            out.append((m, pts))
        return out

    def high_blobs(self):
        """Small things standing 30-50 mm proud of the bench: peg shafts, seen from above."""
        high = np.isfinite(self.z) & (self.z > self.bench_z + PEG_ABOVE_BENCH) & (self.z < self.bench_z + HELD_ABOVE_BENCH)
        pegs, mask = [], np.zeros_like(high)
        for m, pts in self._blobs(high, MIN_PEG_M2, MAX_PEG_M2):
            pegs.append({"xy": (float(pts[:, 0].mean()), float(pts[:, 1].mean())), "top_z": float(np.percentile(pts[:, 2], 90)), "source": "depth"})
            mask |= m
        return pegs, mask

    def discs_in_band(self, z_lo, z_hi, exclude=None, min_m2=MIN_DISC_M2):
        """Round blobs whose top lies in ``[z_lo, z_hi]`` above the bench, without a segmenter."""
        band = np.isfinite(self.z) & (self.z > self.bench_z + z_lo) & (self.z < self.bench_z + z_hi)
        if exclude is not None:
            band &= ~exclude
        out = []
        for m, _pts in self._blobs(band, min_m2):
            d = self.disc(m)
            if d is not None:
                out.append(d)
        return out

    def ring_height(self, xy, r_in, r_out, below=None):
        """Median surface height in an annulus around ``xy`` (what a part is standing on).

        ``below`` drops anything at or above that height from the ring first. What a
        disc stands on is under its top face by definition, and the ring around a gear
        lying beside the parked hand is half hand (gearmesh-easy-single-02: the
        fingertips a centimetre from the rim, the ring's median at 0.92, the gear
        "standing" 15 cm below its surface and thrown out as not a gear).
        """
        d = np.hypot(self.world[..., 0] - xy[0], self.world[..., 1] - xy[1])
        m = np.isfinite(self.z) & (d > r_in) & (d < r_out)
        if below is not None:
            m &= self.z < float(below)
        if m.sum() < 20:
            return float("nan")
        return float(np.median(self.z[m]))


def _circle_fit(xy):
    """Algebraic least-squares circle through the outer rim of a point set (robust to a hand over its middle)."""
    if len(xy) < 12:
        return None
    c0 = xy.mean(axis=0)
    r = np.hypot(xy[:, 0] - c0[0], xy[:, 1] - c0[1])
    rim = xy[r >= np.percentile(r, 85)]
    if len(rim) < 8:
        return None
    A = np.column_stack([2 * rim[:, 0], 2 * rim[:, 1], np.ones(len(rim))])
    b = (rim**2).sum(axis=1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return (float(sol[0]), float(sol[1]))


# ---------------------------------------------------------------- the segmenter
SEGMENT_TOP_K = 48
"""How many masks a query asks SAM3 for, best first.

Not a change of what the graph sees, by a factor of three: over 2,073 recorded
calls of the first sweep (``outputs/gear_levels/round1``), the most masks that
ever passed a query's score floor was 15 (``pin`` at 0.10; ``gear`` 4, ``red
circle`` 2), and the tool sorts by score before it caps, so everything this
drops is below the floor :func:`segment` discards anyway. What it changes is
the cost: at ``max_results=0`` every call returned 200 full-frame masks -- 246 MB
over the bundle's pipe, decoded in the simulator's process and written to the
trace as 200 PNGs -- and took 2-4 s of an episode's wall clock, 40-70 % of it
on the tiled frames. Measured through the same rpc client on the same frame:
0.79 s a call at 200 masks, 0.27 s at 20."""


def segment(ctx, rgb, query: str, score: float):
    """``[(mask, score)]`` from ``sam3.segment_text``, best first, at or above ``score``."""
    result = ctx.tool("sam3.segment_text", image=rgb, query=query, max_results=SEGMENT_TOP_K)
    masks = result.get("masks") or []
    scores = result.get("scores") or []
    out = []
    for m, s in zip(masks, scores, strict=False):
        if float(s) >= score:
            out.append((np.asarray(m), float(s)))
    return out


def _dedupe(discs, radius=0.010):
    """SAM3 returns a mask per instance and sometimes a second, weaker one for the same thing."""
    kept: list[dict[str, Any]] = []
    for d in sorted(discs, key=lambda d: -d.get("score", 0.0)):
        if all(math.hypot(d["xy"][0] - k["xy"][0], d["xy"][1] - k["xy"][1]) > radius for k in kept):
            kept.append(d)
    return kept


# ---------------------------------------------------------------- the reading
def teeth_from_radius(r_tip: float) -> int:
    n = 2.0 * r_tip / MODULE - 2.0
    return int(min(TOOTH_CLASSES, key=lambda c: abs(c - n)))


def r_tip(teeth: int) -> float:
    return 0.5 * MODULE * (teeth + 2)


def _is_orange(rgb):
    """The pinion's orange (palette ACCENT_PLUG, 229/158/51 on the tiled frame) and not the brass gear's yellow.

    **By the shape of the chroma, not its size.** The 20-tooth gear became yellow brass
    (199/168/82) when the gears went from three hues to three alloys, and it passed the
    old test -- red over 170, green over 100, blue under 110 -- so the cursor read it as
    the pinion, anchored the whole chain on a gear lying on the mat and never saw it as a
    gear at all (a one-gear board reported no gears; a three-gear board seated its steel
    gear on the brass gear's peg). What tells orange from brass is where the green sits
    between red and blue: (r-g)/(r-b) is 0.40 on the pinion, 0.26 on brass, 0.64 on
    bronze and 0.69 on the red button. Measured on the tiled frame; a ratio of chromas
    survives a change of exposure that the absolute thresholds would not.
    """
    r, g, b = rgb
    chroma = r - b
    if not (r > 170 and g > 100 and b < 110 and chroma > 80):
        return False
    return 0.32 < (r - g) / chroma < 0.52


def _is_red(rgb):
    r, g, b = rgb
    return r > 150 and r - g > 80 and r - b > 120


def hue_class(rgb) -> int | None:
    """Which tooth class a gear's ALLOY says it is: steel 16, brass 20, bronze 24; ``None`` if unsure.

    A weak cross-check on the radius, not a substitute for it -- three alloys
    under one brushed-metal finish are much closer together than three
    saturated hues were, and a gear half in the key's shadow moves further than
    the gap between them. Warmth orders them: steel is neutral, brass is yellow
    (green well above blue), bronze is red (red well above green).
    """
    r, g, b = rgb
    if max(r, g, b) < 30.0:
        return None
    warm = r - b
    if warm < 0.10 * max(r, 1.0):
        # Neutral, and bright: steel reads 173/178/189 on the tiled frame, the plate 56/56/60 and the mat
        # 103/103/112. A dark neutral disc is the plate around an empty peg, which SAM3 has called a gear
        # (gearmesh-easy-pair-03: a "gear16" at plate height on station 0, picked twice, grasping air).
        return 16 if max(r, g, b) >= 130.0 else None
    if g - b > 0.55 * warm:
        return 20
    if r - g > 0.45 * warm:
        return 24
    return None


def read(ctx) -> dict[str, Any]:
    """One frame, everything in it: bench, pegs, pinion, button, loose / seated / perched / held gears."""
    tcp = ctx.tool("robot.get_ee_pose")["pose"]["position"]
    tcp_xyz = (float(tcp["x"]), float(tcp["y"]), float(tcp["z"]))
    rgb, depth, K, T = frame(ctx)
    view = View(rgb, depth, K, T)
    held_top = tcp_xyz[2] + _CACHE.get("held_top_above_tcp", 0.002)

    # -- gears and the pinion, by name
    discs = []
    for mask, score in segment(ctx, rgb, "gear", GEAR_SCORE):
        d = view.disc(mask)
        if d is None or d["fill"] < 0.35 or d["r_m"] < 0.012 or d["r_m"] > 0.045:
            continue
        d["score"] = score
        discs.append(d)
    discs = _dedupe(discs)
    pinion = [d for d in discs if _is_orange(d["rgb"])]
    gears, held = [], []
    for d in discs:
        if _is_orange(d["rgb"]) or _is_red(d["rgb"]):
            continue
        d["r_tip"] = d["r_m"] + 0.5 * view.px_m
        d["teeth"] = teeth_from_radius(d["r_tip"])
        d["hue_teeth"] = hue_class(d["rgb"])
        if abs(d["top_z"] - held_top) < HELD_BAND and math.hypot(d["xy"][0] - tcp_xyz[0], d["xy"][1] - tcp_xyz[1]) < 0.05:
            held.append(d)
        elif d["top_z"] < view.bench_z + 0.045:
            d["under_z"] = view.ring_height(d["xy"], d["r_tip"] + 0.004, d["r_tip"] + 0.012, below=d["top_z"] - 0.004)
            # A gear is one of the three hues and stands one face proud of whatever it lies on. SAM3 also
            # answers `gear` with a gear's own shadow -- a flat dark disc at plate height (measured on the
            # one-gear smoke: the pinion's shadow read as a 16-tooth gear) -- and that is neither.
            proud = float(d["top_z"]) - float(d["under_z"]) if np.isfinite(d["under_z"]) else FACE
            # Whatever its colour: a gear stands one face proud of what it lies on. The hue used to excuse
            # a disc from this, and a neutral patch of plate got through as steel (pair-03, above).
            if abs(proud - FACE) > 0.006:
                continue
            gears.append(d)
    # A held gear the segmenter did not name: the disc under the hand at carry height.
    if not held and held_top > view.bench_z + 0.05:
        for d in view.discs_in_band(held_top - HELD_BAND - view.bench_z, held_top + HELD_BAND - view.bench_z):
            if d["r_m"] > 0.045 or math.hypot(d["xy"][0] - tcp_xyz[0], d["xy"][1] - tcp_xyz[1]) > 0.05:
                continue
            d["r_tip"] = d["r_m"] + 0.5 * view.px_m
            d["teeth"] = teeth_from_radius(d["r_tip"])
            d["hue_teeth"] = hue_class(d["rgb"])
            d["source"] = "depth"
            held.append(d)
    for h in held:
        ys = h["y_extent"]
        h["in_jaw_offset"] = (0.0, float(0.5 * (ys[0] + ys[1]) - tcp_xyz[1]) if (ys[1] - ys[0]) > 0.030 else 0.0)

    # A gear the segmenter did not name. `gear` finds a coloured disc on a grey mat every time and a
    # STEEL one only sometimes -- measured, when the gears went from three hues to the palette's metal:
    # the cursor saw one of two on boards it had been seeing both of. A gear is also just a disc one face
    # proud of the mat, which the depth says without being asked, so the frame is swept for those and any
    # that SAM3 missed are added. Every gear records how it was found.
    for d in gears:
        d.setdefault("source", "sam3")
    known = [d["xy"] for d in gears] + [d["xy"] for d in held] + ([pinion[0]["xy"]] if pinion else [])
    for d in view.discs_in_band(0.010, 0.030):
        if d["fill"] < 0.35 or not (0.012 < d["r_m"] < 0.045):
            continue
        if any(math.hypot(d["xy"][0] - k[0], d["xy"][1] - k[1]) < 0.012 for k in known):
            continue
        if _is_orange(d["rgb"]):
            # The pinion the segmenter did not name: an orange disc one face proud of the plate. Under the
            # path tracer SAM3 missed the 14-tooth pinion on gearmesh-easy-single-03 and the board could not
            # be read at all; the depth already had it.
            if not pinion and abs(float(d["top_z"]) - (view.bench_z + 0.022)) < 0.008:
                d["source"] = "depth"
                pinion.append(d)
                known.append(d["xy"])
            continue
        if _is_red(d["rgb"]):
            continue
        d["r_tip"] = d["r_m"] + 0.5 * view.px_m
        d["teeth"] = teeth_from_radius(d["r_tip"])
        d["hue_teeth"] = hue_class(d["rgb"])
        d["under_z"] = view.ring_height(d["xy"], d["r_tip"] + 0.004, d["r_tip"] + 0.012, below=d["top_z"] - 0.004)
        proud = float(d["top_z"]) - float(d["under_z"]) if np.isfinite(d["under_z"]) else FACE
        if abs(proud - FACE) > 0.006:
            continue
        d["source"] = "depth"
        gears.append(d)
        known.append(d["xy"])

    # -- the button: a red circle on the plate
    button = None
    for query in ("red circle", "orange disc"):
        cands = []
        for mask, score in segment(ctx, rgb, query, BUTTON_SCORE):
            d = view.disc(mask)
            if d is None or not _is_red(d["rgb"]) or d["r_m"] > 0.016:
                continue
            d["score"] = score
            cands.append(d)
        if cands:
            button = max(cands, key=lambda d: d["score"])
            button["query"] = query
            break

    # -- the pegs: what SAM3 calls a pin, and what stands proud of the bench inside the plate
    pegs = []
    for mask, score in segment(ctx, rgb, "pin", PIN_SCORE):
        m = np.asarray(mask) > 0
        a = view.area(int(m.sum()))
        if a < MIN_PEG_M2 or a > MAX_PEG_M2:
            continue
        pts = view.world[m]
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < 3:
            continue
        top = float(np.percentile(pts[:, 2], 90))
        if not (view.bench_z + PEG_ABOVE_BENCH < top < view.bench_z + HELD_ABOVE_BENCH):
            continue
        pegs.append({"xy": (float(pts[:, 0].mean()), float(pts[:, 1].mean())), "top_z": top, "source": "sam3", "score": score})
    high, high_mask = view.high_blobs()
    for p in high:
        near = [q for q in pegs if math.hypot(q["xy"][0] - p["xy"][0], q["xy"][1] - p["xy"][1]) < 0.008]
        if near:
            # The depth blob's centre, not the mask's: the blob is the shaft's own cross-section, and a `pin`
            # mask takes the peg's shadow or highlight with it -- measured under the path tracer on
            # gearmesh-easy-pair-06, the mask's centre 8-9 mm off the peg (the hand went there, the bore
            # landed on the cone) while the blob's was within 1 mm on both tracers.
            near[0]["xy"] = p["xy"]
            near[0]["top_z"] = p["top_z"]
            near[0]["source"] = "both"
            continue
        # Not a gear in the hand, not the hand itself: pegs are small.
        if any(math.hypot(h["xy"][0] - p["xy"][0], h["xy"][1] - p["xy"][1]) < 0.06 for h in held):
            continue
        pegs.append(p)

    return {
        "bench_z": view.bench_z,
        "px_m": view.px_m,
        "pegs": pegs,
        "pinion": max(pinion, key=lambda d: d["px"]) if pinion else None,
        "button": button,
        "gears": gears,
        "held": min(held, key=lambda d: math.hypot(d["xy"][0] - tcp_xyz[0], d["xy"][1] - tcp_xyz[1])) if held else None,
        "tcp": tcp_xyz,
        "plate_z": _plate_height(view, pegs, button),
        "view": view,
    }


def _plate_height(view, pegs, button):
    """The plate's top: the ring around the button (nothing else ever sits there), else the rings around the pegs.

    A peg's ring is the plate only until a gear is seated on it; from then on
    it is the gear's top face, one face higher, and a plate height taken there
    would make every seated gear read as sitting on the plate.
    """
    cached = _CACHE.get("plate_z")
    if button:
        z = view.ring_height(button["xy"], 0.014, 0.024)
        if np.isfinite(z):
            if cached is None or abs(z - cached) < 0.004:
                _CACHE["plate_z"] = float(z)
                return float(z)
    if cached is not None:
        return float(cached)
    z = float(np.nanmedian([view.ring_height(p["xy"], 0.006, 0.014) for p in pegs])) if pegs else float("nan")
    if np.isfinite(z):
        _CACHE["plate_z"] = z
    return z


def station_state(view, peg_xy, r_tip_m, plate_z):
    """What sits at a peg, from the height of an annulus around it: ``(state, top_z, coverage)``.

    Segmentation-free on purpose: a seated gear's teeth interleave with its
    neighbour's, so as blobs the two are one. The annulus (from just outside
    the bore to just inside the tip circle) is all gear top face when a gear is
    on the peg: ``seated`` at one face above the plate, ``perched`` when
    higher, ``empty`` at plate height, ``hidden`` when something far above
    (the arm) covers it.
    """
    d = np.hypot(view.world[..., 0] - peg_xy[0], view.world[..., 1] - peg_xy[1])
    ring = np.isfinite(view.z) & (d > 0.008) & (d < r_tip_m - 0.004)
    if ring.sum() < 30:
        return "hidden", float("nan"), 0.0
    zs = view.z[ring]
    top = float(np.median(zs))
    face_top = plate_z + FACE
    coverage = float(np.mean(np.abs(zs - face_top) < SEAT_Z_TOL))
    if top > plate_z + 0.06:
        return "hidden", top, coverage
    if coverage > 0.5 and abs(top - face_top) < SEAT_Z_TOL:
        return "seated", top, coverage
    if top > face_top + SEAT_Z_TOL:
        return "perched", top, coverage
    return "empty", top, coverage


def perched_centre(view, peg_xy, top, r_tip_m):
    """Where a gear perched on its peg's cone actually is: the disc at its own (higher) top height, nearest the peg."""
    cands = [
        d
        for d in view.discs_in_band(top - 0.003 - view.bench_z, top + 0.004 - view.bench_z, min_m2=8.0e-4)
        if math.hypot(d["xy"][0] - peg_xy[0], d["xy"][1] - peg_xy[1]) < 0.02 and 0.5 * r_tip_m < d["r_m"] < r_tip_m + 0.008
    ]
    if not cands:
        return tuple(peg_xy)
    best = min(cands, key=lambda d: math.hypot(d["xy"][0] - peg_xy[0], d["xy"][1] - peg_xy[1]))
    return (float(best["xy"][0]), float(best["xy"][1]))


def geometry(reading: dict[str, Any]) -> dict[str, Any] | None:
    """Stations from the pegs, ordered outward from the pinion, with the tooth count each spacing implies, and the chain's heading."""
    pegs = reading["pegs"]
    if not pegs:
        return None
    anchor = reading["pinion"]["xy"] if reading["pinion"] else (reading["button"]["xy"] if reading["button"] else None)
    if anchor is None:
        return None
    # WALK the chain from the pinion rather than sorting by distance from it. On a straight board the two
    # give the same answer; on a bent one they do not -- a chain that turns back brings a far station
    # nearer the pinion than the one before it, and a station read out of order is a tooth count read off
    # the wrong spacing. Each step takes the unused peg nearest the one just placed.
    order, rest, here = [], list(pegs), anchor
    while rest:
        nxt = min(rest, key=lambda p: math.hypot(p["xy"][0] - here[0], p["xy"][1] - here[1]))
        order.append(nxt)
        rest.remove(nxt)
        here = nxt["xy"]
    if reading["pinion"]:
        n_prev = max(6, int(round(2.0 * (reading["pinion"]["r_m"] + 0.5 * reading["px_m"]) / MODULE - 2.0)))
        prev_xy = reading["pinion"]["xy"]
    else:
        n_prev, prev_xy = None, None
    stations = []
    for k, p in enumerate(order):
        if n_prev is not None and prev_xy is not None:
            d = math.hypot(p["xy"][0] - prev_xy[0], p["xy"][1] - prev_xy[1])
            n = int(round(2.0 * d / MODULE - n_prev))
            teeth = int(min(TOOTH_CLASSES, key=lambda c: abs(c - n)))
        else:
            teeth = TOOTH_CLASSES[min(k, len(TOOTH_CLASSES) - 1)]
        # The heading of the link this station sits on: the hand comes at it square to that, not square
        # to the board, because a bent chain points its neighbours in different directions.
        upstream = prev_xy if prev_xy is not None else anchor
        link_yaw = math.atan2(p["xy"][1] - upstream[1], p["xy"][0] - upstream[0])
        stations.append(
            {
                "index": k,
                "peg_xy": p["xy"],
                "peg_top_z": p["top_z"],
                "teeth": teeth,
                "peg_source": p.get("source", "?"),
                "link_yaw": float(link_yaw),
            }
        )
        n_prev, prev_xy = teeth, p["xy"]
    # The chain's heading: from the pinion (else the first peg) to the last peg.
    tail = order[-1]["xy"]
    head = reading["pinion"]["xy"] if reading["pinion"] else order[0]["xy"]
    heading = math.atan2(tail[1] - head[1], tail[0] - head[0]) if math.hypot(tail[0] - head[0], tail[1] - head[1]) > 0.01 else math.pi / 2
    plate_z = reading["plate_z"]
    if not np.isfinite(plate_z):
        plate_z = reading["bench_z"] + 0.006
    return {
        "stations": stations,
        "chain_yaw": float(heading),
        "plate_top_z": float(plate_z),
        "button": {"xy": reading["button"]["xy"], "cap_top_z": reading["button"]["top_z"], "travel": BUTTON_TRAVEL} if reading["button"] else None,
        "bench_z": reading["bench_z"],
    }


def board_geometry(ctx, reading=None) -> dict[str, Any]:
    """The episode's board geometry: locked from the first frame that shows every peg, the pinion and the button.

    The board never moves within an episode, and a later frame can only be
    worse -- the hand over the chain hides pegs, and a gear that landed tilted
    fragments into small high blobs that look like pegs. The lock is released
    by ``unlock_geometry`` when the cursor sees a fresh episode.
    """
    cached = _CACHE.get("geometry")
    if cached is not None:
        return cached
    reading = reading or read(ctx)
    geom = geometry(reading)
    if geom is None or geom["button"] is None or not reading["pinion"]:
        raise LookupError("the overhead camera does not show the whole board (pegs, pinion and button); park the arm and look again")
    _CACHE["geometry"] = geom
    return geom


def unlock_geometry():
    _CACHE.pop("geometry", None)
    _CACHE.pop("plate_z", None)


def assign(reading, geom):
    """Which loose gear belongs to each station: the one whose tooth count the spacing asks for, nearest first."""
    gears = list(reading["gears"])
    out = {}
    for st in geom["stations"]:
        near = [g for g in gears if math.hypot(g["xy"][0] - st["peg_xy"][0], g["xy"][1] - st["peg_xy"][1]) < SEAT_XY_TOL + 0.02]
        on_peg = near[0] if near else None
        if on_peg is not None:
            out[st["index"]] = on_peg
            gears.remove(on_peg)
    for st in geom["stations"]:
        if st["index"] in out:
            continue
        same = [g for g in gears if g["teeth"] == st["teeth"]]
        if not same:
            # The radius disagrees with every spacing; let the hue speak.
            same = [g for g in gears if g.get("hue_teeth") == st["teeth"]]
        if same:
            pick = min(same, key=lambda g: abs(g["r_tip"] - r_tip(st["teeth"])))
            out[st["index"]] = pick
            gears.remove(pick)
    # As many loose gears left as empty stations: the spacing that named a class may be off by one class
    # (the last station's count is read off one peg pair), and the gear that is left is the gear that fits.
    empty = [st for st in geom["stations"] if st["index"] not in out]
    if gears and len(gears) == len(empty):
        for st, g in zip(sorted(empty, key=lambda st: st["index"]), sorted(gears, key=lambda g: g["r_tip"]), strict=True):
            out[st["index"]] = g
    return out
