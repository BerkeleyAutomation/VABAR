"""The same weave, from a camera.

The non-privileged body of ``perceive_weave``: the three numbers the oracle
reads out of the scene, earned instead from one cable-camera RGB-D frame through a
text-prompted segmenter and a depth back-projection.

**Measured on ``weave3``, 2026-08-26**, against the layout the oracle returns:

===============================  ==================  ==========
Quantity                         Perceived           Error
===============================  ==================  ==========
station 0 (x, y)                 (0.5200, -0.1298)   0.2 mm
station 1 (x, y)                 (0.5200, -0.0600)   0.0 mm
station 2 (x, y)                 (0.5200, +0.0098)   0.2 mm
spool radius                     12.2 - 12.4 mm      ~0.3 mm
rod length along y               510 mm              +10 mm
===============================  ==================  ==========

Sub-millimetre from 16x16-pixel blobs because a centroid over ~189 mask pixels
is a sub-pixel estimate; the 1.75 mm/px of this rig is the *sampling* pitch, not
the error floor. The rod's +10 mm is mask dilation at the two tips, which is why
:func:`run` measures the rod's line and radius rather than trusting its extent.

**Where the two bodies genuinely differ.** One key has an honest camera answer,
one is language, and one cannot be earned at all:

``stations``
    Honest. The spools' own masks, back-projected and reduced to a centre.

``sides``
    **Language, not vision.** Which side of a post the rod must finish on is the
    instruction's ("alternating sides"), not a fact about the bench: both sides
    of a spool look identical to a camera, and at the opening frame the rod has
    not been anywhere. Derived here from the alternation the language states and
    a starting side, and that starting side is a convention this file cannot
    measure -- see :data:`FIRST_SIDE`.

``seats_x``
    **Cannot be earned, and this is the one that matters.** The oracle returns
    the *evaluator's* box centre, 24 mm from the station. What a camera can
    derive is the *physical* seat -- shaft radius plus rod radius, 9.75 mm --
    which is where the rod actually rests against the spool. Those are different
    places, and ``plan_station_stages`` documents the difference as
    load-bearing: aiming at the box's inner edge "was worth ~20 mm of margin and
    the rod spent all of it". 9.75 mm is inside the scored box but near that
    inner edge. This body returns the physical seat, because that is what is
    true; whether the margin survives is exactly what running it measures.

The pair must agree on the *keys*, never on where the values came from.
"""

import json
import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

# The runtime loads each node script standalone, so a sibling is not importable
# by package path. Put this script's own directory on the path and import it by
# name -- the same thing the runtime does for the entry module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixtures_cv  # noqa: E402
from grasp_vision import visible_cable_mask

REQUIRED_TOOLS = ("sam3.segment_text", "curve.fit_centerline")
"""What this body needs beyond the runtime's own.

``robot.get_observation`` is the runtime's and is always present. The
back-projection is done here in numpy rather than through
``geometry.mask_to_world_points`` on purpose: it is eight lines of intrinsics
arithmetic, and depending on a second bundle to do it would make this body
un-runnable wherever that bundle is missing for a reason that has nothing to do
with perception.
"""

WHY_ABSENT = (
    "the open-robot-skills sam3 bundle is not resolvable here, so the vision body of "
    "perceive_weave is declared and skipped. It is not a task failure and it is not a "
    "perception result: run the oracle body (--perception oracle, the default), or install "
    "the bundle and put it on GAP_SKILLS_PATH."
)

CAMERA = "cable"
"""The fixed third-person RGB-D view used for every routing observation."""

POST_QUERY = "orange spool"
ROD_QUERY = "thin white cable"
"""What SAM3 is asked for.

**Was "white rod", and that word stopped fitting when the cable got thinner.**
At 5 mm radius a rod is what it looked like; at 3.5 mm it is a cable, and the
detector agrees. Measured on a frame the run was failing on, same pixels:

    "thin white cable"   0.96      "white cable"   0.95
    "white wire"         0.95      "white rod"     0.72
    "rod"                0.06

0.72 still finds it in the easy case, which is why this survived so long. What
it does not survive is a board draw where the rod is partly behind a spool: the
score falls under the 0.20 floor, SAM3 returns nothing, and the episode aborts
at ~150 steps having perceived its three stations perfectly. The prompt that
describes the object gets 0.96 and has margin to spare.
"""
"""The prompts, and the colour word in each is load-bearing.

Measured on this render: ``spool`` and ``peg`` return **zero** detections,
``orange spool`` returns exactly three at 0.906; ``cable``, ``wire`` and
``stick`` return zero, ``white rod`` returns one at 0.836. The detector is
keying on the palette rather than on what the object is, which is a property of
a flat-shaded synthetic render rather than of SAM3 -- the same brittleness the
tool_hanging port hit, where a hook went undetected until the scene was
repainted. Stated here rather than buried because it is the first thing that
breaks when the bench is re-lit or re-coloured.
"""

POST_SCORE = 0.30

POST_AREA_MIN = 100
"""Smallest mask, in pixels, a detection may have and still be a station.

**A floor, not a ceiling, and getting that backwards cost two runs.** The first
version of this filter rejected detections that were too BIG, on a measurement
of 2,450 px per spool taken from a different camera pose. From where the cable
camera now stands the spools mask 172 to 200 px, and what the terminated layouts
add is not a big blob but a small one: a sliver of a terminal block, 27 to 44 px,
which sailed under a ceiling and was planned as a station at (0.406, -0.410) --
250 mm off the end of the board.

The populations are still cleanly separated, just the other way round. 100 px
sits between them with room on both sides, and it is a floor on a *detection*
rather than a bound on a known fixture, so a scene with different hardware still
works.
"""

POST_RADIUS_CEIL = 0.018
"""**Retired as a gate; kept as a reading.** Largest in-plane radius a detection
had to have to be a station [m].

It was the discriminator, sized between an F1 spool's 12.4 mm flange and a
terminal block's 20-plus. That worked for exactly as long as this suite staged
one fixture. An industrial cleat's cap measures 19 mm and is a station, so the
number now sits between two populations it can no longer separate.

What replaced it is height, which separates both boards: 23 mm for a spool
flange, 43 for a cleat cap, 10 for a terminal plate. The constant stays because
the measurement is still worth having in the log -- it is how the terminals were
first told apart, and a detection whose radius is wildly outside either
population is still worth a look."""

ROD_SCORE = 0.20
"""Detection floors. Low, deliberately: on this render the true detections score
0.9 and 0.84 while the false ones do not appear at all, so a high floor buys
nothing and a low one keeps a dimmer re-lit scene working."""

FIRST_SIDE = -1
"""Which side the first station in +y order wants.

A convention, and the honest limit of what language plus a camera gives you. The
instruction says the crossings alternate; it does not say which side the first
one takes, and at the opening frame nothing in the scene does either -- the rod
lies straight on the robot's side of every station. Wrong here and every
crossing is mirrored, which the verify would catch and no amount of looking
would.
"""

ROD_RADIUS_FLOOR = 0.002
ROD_RADIUS_CEIL = 0.020
"""Bounds on the fitted rod radius [m]. A half-thickness outside these is a
mask that caught the bench or the arm, not a rod, and is refused rather than
propagated into a seat offset."""


class Output(TypedDict):
    route: str
    scene: str
    stations: int
    body: str


def _missing(ctx: NodeContext) -> list[str]:
    """Which required tools this registry does not carry.

    ``tape_peel``'s probe, deliberately identical: a body that cannot run must
    fail as a missing dependency and never as a perception result, and the two
    bodies of a perception slot should fail the same way.
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
    body gets to make, and pinning one spelling here would make the body fail
    on a checkout whose only sin is a newer gap.
    """
    frames = observation["cameras"] if isinstance(observation, dict) else observation.cameras
    if isinstance(frames, dict):
        if name in frames:
            return frames[name]
        raise ValueError(f"no camera {name!r} in the observation; have {sorted(frames)}")
    for frame in frames or ():
        if str(frame.get("name", "")) == name:
            return frame
    have = [str(f.get("name", "?")) for f in (frames or ())]
    raise ValueError(f"no camera {name!r} in the observation; have {have}")


def _frame(block: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            f"camera {CAMERA!r} carries no depth. A weave is measured in metres off the bench, "
            "so a colour-only rig cannot run this body."
        )
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(block["intrinsics"], dtype=np.float64).reshape(3, 3)
    # The runtime hands a camera pose as a typed Se3Pose -- ``position`` and
    # ``rotation`` dicts, wxyz -- not as the flat 7-vector the env publishes
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
    return rgb, depth, K, T


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


def _detections(ctx: NodeContext, rgb: np.ndarray, query: str, floor: float) -> list[np.ndarray]:
    """Masks for one text prompt, best first, above ``floor``."""
    result = ctx.tool("sam3.segment_text", image=rgb, query=query, max_results=0)
    masks = result.get("masks") or []
    scores = [float(s) for s in (result.get("scores") or [])]
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [np.asarray(masks[i]) for i in order if scores[i] >= floor]


def run(
    ctx: NodeContext,
    instruction: str = "",
    camera: str = CAMERA,
    post_query: str = POST_QUERY,
    rod_query: str = ROD_QUERY,
    first_side: int = FIRST_SIDE,
    curve_nodes: int = 42,
) -> Output:
    absent = _missing(ctx)
    if absent:
        raise RuntimeError(f"perceive_weave vision body needs {list(absent)}: {WHY_ABSENT}")

    observation: Any = ctx.tool("robot.get_observation")
    # The instruction comes off the OBSERVATION, not from a subgraph input: a
    # subgraph's declared inputs bind to an upstream subgraph's outputs, and the
    # perception slot has nothing upstream of it. The facade already publishes
    # the episode's instruction beside the camera blocks, which is the same
    # place a real cell would carry the command it was given.
    if not str(instruction).strip():
        instruction = str(observation.get("instruction") or "")
    rgb, depth, K, T = _frame(_camera(observation, camera))

    # ---------------------------------------------------------------- posts
    #
    # CLASSICAL FIRST, the network only if it fails. A spool is rigid, one known
    # colour and one known size, bolted to the bench: everything a text-prompted
    # detector is good at solving is already known about it, and re-running a
    # network every frame to rediscover it costs 13-35 s and can return four
    # spools on one frame and three on the next. Scored on the same frames
    # against the peg bodies' own positions:
    #
    #     SAM3, as it stood          3 found   2.8 mm from truth
    #     OpenCV + height gate       3 found   2.9 mm from truth
    #     ... + the bias correction  3 found   0.9 mm from truth
    #
    # The classical path is not a compromise here -- it is more accurate, once
    # the surface-to-axis bias both paths share is taken out, and it is
    # deterministic. See ``fixtures_cv`` for why height, not colour or area, is
    # what separates a spool from a terminal plate.
    posts = fixtures_cv.find_spools(rgb, depth, K, T)
    for px, py, pr in posts:
        print(f"[posts]   opencv  r={pr * 1000:5.1f} mm  at ({px:.3f}, {py:+.3f})", flush=True)
    if not posts:
        # The colour gate found nothing at all -- a re-lit bench, a repaint, a
        # fixture this suite has not seen. Fall back rather than abort: the
        # prompt is slower and noisier but it is not colour-bound.
        print("[posts]   opencv found nothing; falling back to the text detector", flush=True)
        for mask in _detections(ctx, rgb, post_query, POST_SCORE):
            # AREA, because it is the strongest separator and it is measured in
            # the image rather than through depth -- a back-projected radius
            # shrinks when a fragment is seen edge-on, which is how a terminal
            # block slipped past the radius test at 9.6 mm.
            area_px = int(np.count_nonzero(np.asarray(mask) > 0))
            if area_px < POST_AREA_MIN:
                print(f"[posts]   ignoring a {area_px} px detection -- too small for a spool",
                      flush=True)
                continue
            pts = _to_world(mask, depth, K, T)
            if len(pts) < 8:
                continue
            centre = np.median(pts, axis=0)
            # The 90th percentile rather than the max: a flange disc's mask
            # carries a few pixels of the bench at its rim, and a max reads
            # those as radius.
            radius = float(np.percentile(np.linalg.norm(pts[:, :2] - centre[:2], axis=1), 90))
            # NOT EVERY ORANGE THING IS A STATION. The terminated layouts put a
            # block at each end of the run, in the same accent colour as the
            # stations -- on ``port3`` the prompt returned FIVE detections for
            # three pegs, and the planner tried to weave around the terminals.
            #
            # SEPARATED BY HEIGHT, not by radius, and that changed when the
            # suite grew a second fixture. The ceiling here was 18 mm, sized
            # between an F1 spool's 12.4 mm flange and a terminal's 20-plus.
            # An industrial cleat's cap measures **19 mm** -- so this fallback
            # would have rejected every station on ``panel_l2`` as "too big for
            # a spool" and aborted the episode, while the classical path in
            # front of it found all five.
            #
            # Height is what actually separates the two populations and it does
            # so for both boards: a spool's flange stands 23 mm off the bench, a
            # cleat's cap 43, and a terminal plate 10. It is the same gate
            # ``fixtures_cv`` uses, against a bench plane measured from the same
            # depth image, so the two paths now disagree about nothing.
            height = float(centre[2]) - fixtures_cv.bench_z(depth, K, T)
            if height < fixtures_cv.MIN_HEIGHT:
                print(f"[posts]   ignoring a detection {height * 1000:.0f} mm off the bench "
                      f"at ({centre[0]:.3f}, {centre[1]:+.3f}) -- a terminal, not a station",
                      flush=True)
                continue
            print(f"[posts]   accepted {area_px:6d} px  r={radius * 1000:5.1f} mm  at "
                  f"({centre[0]:.3f}, {centre[1]:+.3f})", flush=True)
            posts.append((float(centre[0]), float(centre[1]), radius))
    if not posts:
        return {"route": "not_found", "scene": "{}", "stations": 0, "body": "vision"}
    # Ordered along +y, which is the order the layout table is written in and the
    # order the pass ordering below re-derives. Not the detector's score order:
    # that is arbitrary and would permute the sides.
    posts.sort(key=lambda p: p[1])
    stations = [[p[0], p[1]] for p in posts]

    # ------------------------------------------------------------------ rod
    #
    # An ORDERED CENTRELINE, not a bounding box and not a principal axis. The
    # rod is the one thing in this scene that bends, and the axis of a curve
    # through four posts is a chord passing through none of the places a hand
    # goes. ``curve.fit_centerline`` thins the mask, recovers a traversal order
    # through breaks and crossings (TrackDLO's assignment), back-projects and
    # fits a spline in arc length -- the same s in [0, 1] the rod's own asset
    # record already publishes its features against.
    # THE THINNEST CANDIDATE, NOT THE FIRST. This fit seeds the tracker and every
    # plan for the rest of the episode, so taking the top-scoring mask on faith
    # is the single most expensive guess in the graph.
    #
    # The bench is grey, the rod is white, and so are both arms. A prompt for a
    # white rod will sometimes return a blob that has merged the cable with an
    # arm beside it -- confidently scored, and wrong in a way nothing downstream
    # can detect. Measured on ``weave3`` with a 700 mm rod: four episodes seeded
    # at 714, 707, 718 and 694 mm, and two at **963 and 983 mm**. Both inflated
    # ones failed; three of the four honest ones solved.
    #
    # Score cannot separate them and shape can. The fit reports mean radius as
    # mask area over centreline length, which is precisely the ratio that tells
    # a 3.5 mm cable from a cable welded to a gripper, so every candidate is
    # fitted and the thinnest one that is still a curve wins.
    def L_of(p):
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))

    fit, curve, best_r = None, np.zeros((0, 3)), None   # best_r holds arclength
    fallback = None    # the longest over-thick candidate, if nothing else fits
    _cands = _detections(ctx, rgb, rod_query, ROD_SCORE)
    # At the opening survey the cable rests on the bench. White robot links
    # and background-depth edge pixels can otherwise create a longer, thin
    # candidate that wins over the actual cable under RTX lighting.
    surface = fixtures_cv.bench_z(depth, K, T)
    v, u = np.indices(depth.shape)
    world_z = (T[2, 0] * (u-K[0, 2]) / K[0, 0]
               + T[2, 1] * (v-K[1, 2]) / K[1, 1] + T[2, 2]) * depth + T[2, 3]
    on_bench = np.isfinite(depth) & (depth > 0.) & (world_z > surface+.0005) & (world_z < surface+.035)
    _cands = [np.asarray(mask, dtype=bool) & on_bench for mask in _cands]
    # The opening view sees the full cable, but a text segmenter can return
    # only its free portion. Keep a height-gated jacket candidate as well.
    # It goes through the same ordered-curve fit and radius checks below.
    jacket = visible_cable_mask(rgb, depth, K, T, surface)
    if int(jacket.sum()) >= 64:
        _cands.append(jacket)
    print(f"[posts]   rod: {len(_cands)} candidate mask(s) above {ROD_SCORE}", flush=True)
    for mask in _cands:
        if int(np.count_nonzero(mask)) < 64:
            continue
        trial = ctx.tool(
            "curve.fit_centerline",
            mask=mask,
            depth=depth,
            intrinsics=K,
            camera_pose=T,
            nodes=int(curve_nodes),
        )
        pts = np.asarray(trial.get("points") or [], dtype=np.float64).reshape(-1, 3)
        r = float(trial.get("radius_m") or 0.0)
        if len(pts) < 2 or r <= 0.0:
            continue
        # The opening cable lies on the measured bench. A spline through a
        # background depth sample can be thin and very long while extending
        # below the table. Such a fit must not seed every subsequent grasp.
        if not np.all(np.isfinite(pts)) or np.any(pts[:, 2] < surface - .005) or np.any(pts[:, 2] > surface + .035):
            print("[posts]   refusing opening curve outside the observed bench band", flush=True)
            continue
        if r > ROD_RADIUS_CEIL:
            # Too thick to be the cable on its own merits, but remembered in
            # case nothing thinner turns up -- see the fallback note below.
            if fallback is None or L_of(pts) > fallback[2]:
                fallback = (trial, pts, L_of(pts))
            continue
        # LONGEST plausible, not thinnest. Radius alone ranks backwards and
        # fails in both directions: a mask merged with an arm traces a longer
        # path, so area-over-length reads SMALLER, and a mask that is only a
        # fragment reads smaller still. Thinness therefore prefers exactly the
        # two things it was meant to reject -- measured on port3, it chose a
        # 148 mm fragment of a 700 mm rod. The radius ceiling already rejects
        # anything too thick; among what survives, the longest saw most of it.
        L = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if best_r is None or L > best_r:
            fit, curve, best_r = trial, pts, L
        # A FALLBACK, because this node's failure is the whole episode's.
        #
        # `not_found` here routes straight to `abort`, so a filter with no way
        # through is not a filter -- it is an episode killer. Measured: with the
        # thinner cable some draws had every candidate over the radius ceiling,
        # the rod fit returned nothing, and the run ended at ~150 steps having
        # perceived its three spools perfectly. Keeping the longest candidate
        # regardless means a marginal fit is used and reported rather than the
        # episode being thrown away; the ceiling still decides the RANKING, it
        # just no longer decides whether there is an answer at all.
    if fit is None and fallback is not None:
        fit, curve, best_r = fallback
        print(f"[posts]   rod: no candidate under the radius ceiling; taking the "
              f"longest anyway ({best_r * 1000:.0f} mm)", flush=True)
    if fit is None or len(curve) < 2:
        print("[posts]   rod: no usable candidate at all", flush=True)
        return {"route": "not_found", "scene": "{}", "stations": len(stations), "body": "vision"}
    print(f"[posts]   rod: longest cable-shaped candidate, "
          f"arclength {best_r * 1000:.0f} mm", flush=True)
    # The fitted radius, measured as mask area over centreline length -- both in
    # metres, and neither assuming which way the rod points. The axis-aligned
    # estimate this replaces read 4.14 mm for a 5 mm rod on the arc layout,
    # because there the spread it halved was mostly the curve.
    rod_radius = float(np.clip(float(fit.get("radius_m") or 0.0), ROD_RADIUS_FLOOR, ROD_RADIUS_CEIL))

    # ---------------------------------------------------------------- sides
    # The alternation is the instruction's; the phase is FIRST_SIDE's.
    alternating = "alternat" in str(instruction).lower()
    sides = [int(first_side) * (1 if k % 2 == 0 else -1) for k in range(len(stations))]

    # ---------------------------------------------------------------- seats
    # The PHYSICAL seat: the post's own radius plus the rod's, on the named
    # side. The oracle answers the evaluator's box centre instead -- see the
    # module docstring on why these are different places and which is honest.
    seats_x = [float(posts[k][0] + sides[k] * (posts[k][2] + rod_radius)) for k in range(len(posts))]

    scene = {
        "stations": stations,
        "sides": sides,
        "seats_x": seats_x,
        "rod": [[round(float(v), 5) for v in p] for p in curve],
        # Carried for the trace, not for the planner: a run that scores badly
        # should be answerable from what the body saw without re-running it.
        "measured": {
            "post_radius_m": [round(p[2], 5) for p in posts],
            "rod_radius_m": round(rod_radius, 5),
            "rod_arclength_m": round(float(fit.get("arclength_m") or 0.0), 4),
            "rod_nodes": int(fit.get("nodes") or 0),
            "rod_ordered": bool(fit.get("ordered")),
            "alternation_in_language": bool(alternating),
        },
    }
    print(f"[posts] {len(stations)} spools  sides={sides}  "
          f"seats_x={[round(v, 4) for v in seats_x]}  rod_nodes={len(curve)}", flush=True)
    return {
        "route": "found",
        "scene": json.dumps(scene),
        "stations": len(stations),
        "body": "vision",
    }
