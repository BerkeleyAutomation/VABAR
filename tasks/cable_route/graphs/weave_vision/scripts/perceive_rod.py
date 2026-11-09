"""Read the cable's centreline, from a third-person camera pitched at the bench.

**Fit fresh, track only when the fit is broken.** A cold fit carries no history,
so when the whole rod is in frame it is simply the better estimate; the tracker's
value is entirely in surviving frames where the rod is NOT all there. Which of
the two is right is therefore a property of the view, not a fixed choice, and it
changed when the camera moved -- see :data:`WHOLE_ROD` for the eight poses that
measured it and :data:`~tasks/cable_route/staging.py::WEAVE_CABLE_ELEV` for the move.

**Discrete, not continuous.** The cable's shape only changes when a crossing
moves it, so it is read once per station rather than every frame. Continuous
tracking would spend a detector call per control step to re-derive a shape that
did not change between them.

**The failure this guards is a broken skeleton, not drift.** When something does
stand between the lens and the rod, the mask arrives in two pieces and the chain
merge resolves them to one -- a fit that is confidently wrong and *short*. Length
against the prior is what detects that, and a short fit hands the frame to the
tracker, which asks each of the 42 known nodes what the frame says about it and
moves the unanswered ones by the displacement field their visible neighbours
generate. That is Motion Coherence Theory doing the job it was written for, in
the case it was written for.

**Two gates, for two different questions.** :data:`WHOLE_ROD` asks whether a
fresh fit saw the whole object. :data:`MIN_VISIBLE` asks whether a tracked update
saw enough of the model to be believed; below it the prior simply stands, because
a tracker that accepted every frame would walk the model onto whatever happened
to be visible.

There is no oracle body here and no router above one. A privileged twin would be
a second way for this node to succeed, and a graph that can fall back to the
simulator is a graph whose perception is never load-bearing.
"""

import json
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

#: Where the centreline is born AND maintained: a fixed third-person camera
#: standing 500 mm off the cable's own run and pitched 55 degrees above the
#: bench, 1280x960.
#:
#: **Not the hand camera, and that was measured rather than assumed.** At any
#: pose this arm can hold, an eye-in-hand frames a fifth of the cable, or none
#: of it at the wide tilt that fixes the framing. An A/B over three episodes
#: came out identical either way -- 0.667 mean -- so the move it costs buys
#: nothing. A camera that can simply be *placed* where the object is gets the
#: whole rod at 0.50 mm/px.
#:
#: **And placed off the vertical, which mattered far more than the placement.**
#: Overhead, this camera looked down the axis the gripper descends, so the hand
#: covered the rod exactly when the rod was moving: 28 of 42 nodes visible, and
#: a centreline fitted at 60 mm RMS. Pitched to 55 degrees it sees 41 of 42 and
#: fits at 6.7 mm.
INIT_CAMERA = "cable"

#: The same camera. Init and update differ in the *operation* -- a cold fit
#: needs the whole object, an update needs a prior -- not in where they look.
TRACK_CAMERA = "cable"

ROD_QUERY = "thin white cable"
ROD_SCORE = 0.20
CURVE_NODES = 42

#: Nodes that must have real correspondence before a TRACKED update is believed.
#:
#: **Six, and the arithmetic matters.** 42 nodes over a 500 mm rod is 12 mm of
#: model per node, so the wrist's 155 mm window holds about thirteen of them --
#: and the hand occupies a third of its own frame, so an honest close view
#: delivers eight to ten. Ten was therefore right at the boundary and rejected
#: real wrist updates: measured, a pass that reached the viewpoint to 3 mm fell
#: back to the bench view anyway. Six is ~70 mm of rod, which is still a
#: meaningful anchor and comfortably inside what the window gives.
#:
#: Deliberately NOT raised when the camera improved. This gate now guards only
#: the fallback, which by construction runs on the frames where the view is bad;
#: tightening it against the 41/42 the good frames deliver would be tuning a
#: threshold on a population that no longer reaches it.
MIN_VISIBLE = 6

#: Correspondence mass below which a node counts as unobserved. Mirrors the
#: tracker's own floor; stated here because this node routes on it.
VIS_FLOOR = 0.05


#: Fraction of the prior's arclength a fresh fit must reach to be taken over a
#: tracked update.
#:
#: **This node used to track unconditionally, and that is now the wrong default.**
#: The tracker exists to survive occlusion, and it was paying for itself against
#: an overhead camera that lost a third of the model whenever the arm was on the
#: rod. Pitching that camera off the vertical (:data:`WEAVE_CABLE_ELEV`) removed
#: the occlusion, and with it the reason. Measured over eight arm poses from the
#: new viewpoint, scoring both against the rod's own 42 bodies:
#:
#:   cold fit per frame   median 6.7 mm   p90 8.4 mm   max  9.3 mm
#:   tracked              median 9.0 mm   p90 9.7 mm   max  9.9 mm
#:   nodes visible        median 41/42    min 40/42
#:
#: The tracker is now the WORSE estimate, and monotonically so across the eight
#: passes (7.6, 8.6, 9.2, 9.6, 9.6, 9.9): with nothing hidden, every pass is the
#: prior being dragged along rather than a hidden stretch being carried, and the
#: error only compounds. A prior is a liability when the object is fully visible.
#:
#: It is still the right fallback, because the cold fit's own failure is sharp
#: rather than gradual: a skeleton broken by the arm merges to one fragment, and
#: that measured 338 mm of a 488 mm rod. 0.80 sits between the two populations
#: with room on both sides -- honest fits ran 481-518 mm, or 0.99-1.06 of the
#: truth, and the fragment was 0.69.
WHOLE_ROD = 0.80
MAX_ROD = 1.15
"""Upper length ratio: an arm merged into the mask is not extra cable."""


def plausible_length(points, reference):
    length = _length(points)
    return len(points) >= 2 and WHOLE_ROD * reference <= length <= MAX_ROD * reference


class Output(TypedDict):
    route: str
    rod: str
    arclength_m: float
    source: str
    camera: str
    visible: int


def _camera(observation: Any, name: str):
    frames = observation["cameras"] if isinstance(observation, dict) else observation.cameras
    if isinstance(frames, dict):
        return frames.get(name)
    for frame in frames or ():
        if str(frame.get("name", "")) == name:
            return frame
    return None


def _frame(block):
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
        return None
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


#: Candidate masks considered before one is chosen, and the widest an accepted
#: one may be.
#:
#: **The top-scoring mask is not always the cable.** The bench is grey, the rod
#: is white and so are the two arms, and a text prompt that grounds "white rod"
#: will sometimes return a blob that has merged the rod with an arm beside it.
#: That blob is confidently scored and quietly wrong: measured on ``weave3``
#: with a 700 mm rod, the fits came back 714, 707, 694 mm on four episodes and
#: **963 and 983 mm** on the other two -- and every episode with an inflated
#: seed failed, while three of four with an honest one solved.
#:
#: A cable is separable from an arm by SHAPE, not by score. The fit already
#: measures mean radius as mask area over centreline length, which is exactly
#: the ratio that distinguishes them: this rod is 3.5 mm, and anything merged
#: with a gripper reads several times that. So several candidates are fitted and
#: the thinnest plausible one is taken, rather than the first.
MASK_CANDIDATES = 4
MAX_ROD_RADIUS = 0.012


def _masks(ctx: NodeContext, rgb) -> list[np.ndarray]:
    found = ctx.tool("sam3.segment_text", image=rgb, query=ROD_QUERY,
                     max_results=MASK_CANDIDATES)
    masks, scores = found.get("masks") or [], found.get("scores") or []
    return [np.asarray(m) for m, sc in zip(masks, scores) if float(sc) >= ROD_SCORE]


def _mask(ctx: NodeContext, rgb) -> np.ndarray | None:
    got = _masks(ctx, rgb)
    return got[0] if got else None


def _length(curve) -> float:
    pts = np.asarray(curve, dtype=np.float64).reshape(-1, 3)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0


def _out(route, curve, source, camera, visible) -> Output:
    # Printed, not only returned. ``node_data/<node>/output.json`` holds the
    # LAST write per node, so a per-pass fact -- which camera carried this
    # update, how much of the model was in frame -- is invisible in a trace of a
    # loop. Three claims about this node were wrong before it said so out loud.
    print(f"[track] source={source} camera={camera or '-'} visible={visible} "
          f"len={_length(curve) * 1000:.0f}mm", flush=True)
    return {
        "route": route,
        "rod": json.dumps([[round(float(v), 5) for v in p] for p in (curve or [])]) if curve else "null",
        "arclength_m": round(_length(curve), 4) if curve else 0.0,
        "source": source,
        "camera": camera,
        "visible": int(visible),
    }


def _overhead_refine(curve, observation):
    from grasp_vision import refine_grasp_curve
    refined, info = refine_grasp_curve(curve, observation)
    print(f"[track] overhead refined {info['refined']} nodes, "
          f"max XY adjustment {info['max_shift_mm']:.1f} mm", flush=True)
    return refined


def run(ctx: NodeContext, scene: str = "", camera: str = "", index: int = 0) -> Output:
    del index
    initial = json.loads(scene or '{}')
    prior = json.loads(ctx.tool("cable.curve_memory")["rod"] or "null")
    observation = ctx.tool("robot.get_observation")

    # ---------------------------------------------------------- cold start
    #
    # SEEDED FROM THE OPENING SURVEY, not re-fitted here. ``posts`` already fits
    # the whole rod from the bench view with both arms at home; by the time this
    # node first runs, ``look`` has parked the hand over station 0 and the arm
    # is *in the bench camera's way*. Measured: re-fitting at this point read
    # 381 mm of a 500 mm rod, because the hand was standing on the rest of it.
    #
    # The model has to be born from an unobstructed view of the whole object.
    # That view exists exactly once per episode, before the first move.
    if not prior:
        seeded = (json.loads(scene or "{}") or {}).get("rod")
        if seeded and len(seeded) >= 2:
            curve = [[float(v) for v in p] for p in seeded]
            curve = _overhead_refine(curve, observation)
            ctx.tool("cable.curve_memory", rod=json.dumps(curve))
            return _out("found", curve, "seeded", INIT_CAMERA, len(curve))
        block = _camera(observation, INIT_CAMERA)
        got = _frame(block) if block is not None else None
        if got is None:
            return _out("lost", None, "no-camera", INIT_CAMERA, 0)
        rgb, depth, K, T = got
        candidates = _masks(ctx, rgb)
        if not candidates:
            return _out("lost", None, "no-mask", INIT_CAMERA, 0)
        # Fit every candidate and keep the thinnest one that is still a curve.
        # This is the model the whole episode is built on, so it is worth four
        # fits once rather than a wrong prior for the rest of the run.
        best, best_r, pts = None, None, []
        for cand in candidates:
            trial = ctx.tool(
                "curve.fit_centerline", mask=cand, depth=depth, intrinsics=K,
                camera_pose=T, nodes=CURVE_NODES,
            )
            tp = trial.get("points") or []
            r = float(trial.get("radius_m") or 0.0)
            if len(tp) < 2 or r <= 0.0 or r > MAX_ROD_RADIUS:
                continue
            if best_r is None or r < best_r:
                best, best_r, pts = trial, r, tp
        if best is None:
            return _out("lost", None, "no-fit", INIT_CAMERA, 0)
        print(f"[track]   seed: {len(candidates)} candidate mask(s), took "
              f"radius {best_r * 1000:.1f} mm", flush=True)
        fit = best
        curve = [[float(v) for v in p] for p in pts]
        curve = _overhead_refine(curve, observation)
        ctx.tool("cable.curve_memory", rod=json.dumps(curve))
        return _out("found", curve, "initialised", INIT_CAMERA, len(curve))

    # ------------------------------------------------------------- update
    # The look chose a camera; prefer it, and fall back to the bench view when
    # the hand did not reach its viewpoint.
    #
    # A COLD FIT IS TRIED FIRST and taken when it returns a whole rod -- see
    # WHOLE_ROD. The tracker is the fallback, not the default, which is the
    # opposite of what this node used to do and is a consequence of the camera
    # moving off the vertical.
    order = [c for c in ((str(camera),) if camera else ()) if c]
    order += [c for c in (TRACK_CAMERA, INIT_CAMERA) if c not in order]
    frames = observation["cameras"] if isinstance(observation, dict) else observation.cameras
    have = sorted(frames) if isinstance(frames, dict) else [str(f.get("name", "?")) for f in (frames or ())]
    print(f"[track]   camera_in={camera!r} order={order} have={have}", flush=True)
    for name in order:
        block = _camera(observation, name)
        got = _frame(block) if block is not None else None
        if got is None:
            print(f"[track]   {name}: not in observation (or no depth)", flush=True)
            continue
        rgb, depth, K, T = got
        mask = _mask(ctx, rgb)
        if mask is None:
            print(f"[track]   {name}: no rod mask", flush=True)
            continue

        # A fresh fit, and its length is the test of whether to believe it. The
        # cold fit's failure mode is not drift -- it is a skeleton broken in two
        # by the arm, which the chain merge resolves to ONE fragment. A fragment
        # is short, and short is measurable against a rod whose length is known
        # from the prior. Anything full-length is a fit of the whole object with
        # no accumulated history in it, which is strictly the better estimate.
        # Anchor the bound to the opening observation, so successive fitting
        # errors cannot gradually increase the apparent material length.
        want = float((initial.get('measured') or {}).get('rod_arclength_m') or _length(prior))
        fresh = ctx.tool(
            "curve.fit_centerline", mask=mask, depth=depth, intrinsics=K,
            camera_pose=T, nodes=CURVE_NODES,
        )
        fpts = fresh.get("points") or []
        if not plausible_length(fpts, want):
            from grasp_vision import visible_cable_mask
            from fixtures_cv import bench_z
            jacket = visible_cable_mask(rgb, depth, K, T, bench_z(depth, K, T))
            if int(jacket.sum()) >= 64:
                alternative = ctx.tool('curve.fit_centerline', mask=jacket, depth=depth,
                                       intrinsics=K, camera_pose=T, nodes=CURVE_NODES)
                alternate_points = alternative.get('points') or []
                if plausible_length(alternate_points, want):
                    fpts = alternate_points
                    print(f'[track] using height-gated jacket fit: {_length(fpts)*1000:.0f} mm', flush=True)
        if plausible_length(fpts, want):
            curve = [[float(v) for v in p] for p in fpts]
            curve = _overhead_refine(curve, observation)
            if plausible_length(curve, want):
                ctx.tool("cable.curve_memory", rod=json.dumps(curve))
                return _out("found", curve, "refit", name, len(curve))
        print(f"[track]   {name}: cold fit {_length(fpts) * 1000:.0f}mm of "
              f"{want * 1000:.0f}mm expected; tracking instead", flush=True)

        upd = ctx.tool(
            "curve.track_centerline", prior=prior, mask=mask, depth=depth,
            intrinsics=K, camera_pose=T,
        )
        vis = np.asarray(upd.get("visibility") or [], dtype=np.float64)
        seen = int((vis > VIS_FLOOR).sum())
        pts = upd.get("points") or []
        if len(pts) < 2 or seen < MIN_VISIBLE:
            print(f"[track]   {name}: refused, {seen}/{len(vis)} nodes visible "
                  f"(need {MIN_VISIBLE})", flush=True)
            # Too little of the model in frame to trust the update. The prior is
            # still the best estimate -- a tracker that accepted every frame
            # would walk the model onto whatever happened to be visible.
            continue
        curve = [[float(v) for v in p] for p in pts]
        curve = _overhead_refine(curve, observation)
        if not plausible_length(curve, want):
            print(f'[track] refusing tracked length {_length(curve)*1000:.0f} mm against {want*1000:.0f} mm opening observation', flush=True)
            continue
        ctx.tool("cable.curve_memory", rod=json.dumps(curve))
        return _out("found", curve, "tracked", name, seen)

    return _out("found", prior, "held", "", 0)
