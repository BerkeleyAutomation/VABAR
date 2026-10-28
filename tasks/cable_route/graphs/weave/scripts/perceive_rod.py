"""Read the cable's centreline, from a third-person camera pitched at the bench.

**Fit fresh, track only when the fit is broken.** A cold fit carries no history,
so when the whole rod is in frame it is simply the better estimate; the tracker's
value is entirely in surviving frames where the rod is NOT all there. Which of
the two is right is therefore a property of the view, not a fixed choice, and it
changed when the camera moved -- see :data:`WHOLE_ROD` for the eight poses that
measured it and :data:`~tasks.cable_route.task.WEAVE_CABLE_ELEV` for the move.

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
MASK_CANDIDATES = 12
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


def near_bench(curve, surface):
    """A released cable fit cannot run through the table or up a robot link."""
    points = np.asarray(curve, dtype=float).reshape(-1, 3)
    return (len(points) >= 2 and np.all(np.isfinite(points))
            and np.all(points[:, 2] >= surface-.008)
            and np.all(points[:, 2] <= surface+.060))


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


#: How far behind the rod's near edge the hand is parked for a second look [m],
#: longest first.
#:
#: **A ladder, because the longest step is not always reachable.** 120 mm is
#: ``plan_station_stages``'s own retreat, which takes the hand to
#: ``WEAVE_CABLE_X - 0.12`` and is documented there as "out of the frame", and
#: the bench camera stands off the +x side of the run, so a hand behind the rod
#: projects clear of it. But that retreat is measured from the cable's nominal
#: abscissa while this one is measured from wherever the rod has been dragged
#: to, which on a routed run is further in -- and the arm's base sits at
#: x ~ 0.25, so the target lands in front of it and the solver refuses.
#: Measured on the first pilot: of two step-backs, x=0.283 arrived and x=0.288
#: did not, and the refusal cost the read.
#:
#: The rungs are tried longest first and the first arrival wins. A short step
#: is worth attempting because the hand only has to leave the lens's line to
#: the rod, not the bench; and every rung runs in the case where the previous
#: behaviour was to give up and hand back a curve one crossing out of date, so
#: no rung can be worse than not trying it.
CLEAR_BACK = (0.12, 0.08, 0.05)

#: Steps allowed per rung.
#:
#: Kept generous rather than trimmed. A refusal here has two possible causes --
#: a pose outside the envelope, and a servo that ran out of steps -- and a
#: shorter allowance only makes the second more likely while doing nothing
#: about the first. The crossing template moves 120 mm in 0.8 s, about 50 steps
#: at this cell's rate, so this is roughly five times the nominal; three
#: refused rungs still cost under a tenth of the episode's spare budget.
CLEAR_STEPS = 240

#: What counts as a hand standing in the way: near the rod across the bench, and
#: low enough over it to be between the lens and the cable [m].
#:
#: **Both, because either alone picks the wrong arm.** The support hand parks at
#: 212-230 mm from the run in xy, which a generous lateral gate admits, and it
#: sits 330 mm above the bench where it occludes nothing. The work hand at the
#: end of a crossing is 5-11 mm from the rod and 129 mm above it. Measured on
#: the three reads that fell through to the prior in the parent batch; the two
#: populations are far apart on the height alone.
NEAR_ROD = 0.25
LOW_OVER_ROD = 0.25

#: Steps allowed for the rod to stop swinging after the hand steps away.
CLEAR_SETTLE = 45


def _step_aside(ctx: NodeContext, prior) -> bool:
    """Park the hand off the rod and look again. True when it actually moved.

    **The crossing template retreats only after the LAST station.** That is
    deliberate -- see ``is_final`` in ``plan_station_stages``, which removed the
    per-pass retreat because it cost 0.8 s of travel between crossings -- but it
    leaves the hand standing over the span it has just placed for every read
    except the final one. The cold fit then comes back as a broken skeleton and
    the tracker comes back off the bench, and this node falls through to the
    prior: a curve one crossing out of date, returned as ``found``.

    Measured on the thirty-trial RTX batch of ``port_medium``: every read taken
    after the final retreat fitted the whole rod (42/42 nodes, ``refit``), while
    three reads taken with the hand parked fell through to ``held``. Each of
    those three planned its crossing against the stale shape and put the grasp
    49, 61 and 65 mm off the cable -- the hand closed on air, flew the whole
    carry empty, and the station was never seated.

    So this is not a second estimator, it is the same one given the view it
    needs.

    **Precondition: the hand this moves is not holding anything.** Every caller
    of ``run`` satisfies it today and the reason differs per caller, so it is
    worth stating rather than assuming. ``rod_sg`` runs between ``release`` and
    ``take``, and the one edge that re-enters it mid-crossing -- ``mate``'s
    ``infeasible`` back to ``station`` -- is also before ``take``.
    ``verify_weave_vision`` and ``restore_seats`` both run with the jaws open.
    ``plug_port`` carries the connector and does NOT call this: it borrows
    ``_camera`` and ``_frame`` only. A caller that holds the cable here would
    drag it, so give this function a way to be told so rather than adding one.
    """
    points = np.asarray(prior or [], dtype=float).reshape(-1, 3)
    if len(points) < 2:
        return False
    best, best_gap, best_pose = None, None, None
    for arm_id in (0, 1):
        try:
            answer = ctx.tool("robot.get_ee_pose", arm_id=arm_id)
        except Exception:  # noqa: BLE001 -- an arm that cannot be read is not in the way
            continue
        pose = answer.get("pose", answer) if isinstance(answer, dict) else answer
        position = pose.get("position") if isinstance(pose, dict) else None
        if not position:
            continue
        here = np.array([float(position[c]) for c in ("x", "y", "z")])
        gap = float(np.min(np.linalg.norm(points[:, :2] - here[:2], axis=1)))
        if gap > NEAR_ROD or here[2] - float(points[:, 2].max()) > LOW_OVER_ROD:
            continue
        if best_gap is None or gap < best_gap:
            best, best_gap, best_pose = arm_id, gap, (pose, here)
    if best is None:
        print(f"[track]   relook: no hand within {NEAR_ROD * 1000:.0f} mm of the rod and "
              f"{LOW_OVER_ROD * 1000:.0f} mm over it", flush=True)
        return False
    pose, here = best_pose
    near_edge = float(points[:, 0].min())
    # The hand keeps its height: it is already at the carry clearance the
    # crossing left it at, and the step is away from the board, so there is
    # nothing between here and there to lift over. An added lift only moves
    # the target further up the envelope, which is where the solver refuses.
    for back in CLEAR_BACK:
        target_x = near_edge - back
        if here[0] <= target_x:
            continue
        print(f"[track]   relook: arm {best} is {best_gap * 1000:.0f} mm off the rod; "
              f"stepping back {back * 1000:.0f} mm to x={target_x:.3f} and looking again",
              flush=True)
        try:
            answer = ctx.tool("robot.go_to_pose", arm_id=int(best), max_steps=CLEAR_STEPS, pose={
                "position": {"x": target_x, "y": float(here[1]), "z": float(here[2])},
                "rotation": dict(pose["rotation"])})
        except Exception as exc:  # noqa: BLE001 -- a refused step leaves the prior standing
            print(f"[track]   relook: refused ({type(exc).__name__})", flush=True)
            return False
        if isinstance(answer, dict) and answer.get("reached") is False:
            # WHICH refusal, because the two want opposite remedies: an
            # unreachable pose wants a shorter rung, a timeout wants a longer
            # allowance. The first pilot could not tell them apart.
            print(f"[track]   relook: the {back * 1000:.0f} mm step did not arrive "
                  f"({answer.get('outcome') or 'no outcome'}: {answer.get('reason') or '-'})",
                  flush=True)
            continue
        ctx.tool("cable.wait_steps", steps=CLEAR_SETTLE)
        return True
    print("[track]   relook: no step back arrived; the prior stands", flush=True)
    return False


def _read_once(ctx: NodeContext, observation, prior, initial, merge_options) -> Output | None:
    """One pass of cold-fit-then-track over the bench frame. None when neither took."""
    order = [INIT_CAMERA]
    for name in order:
        block = _camera(observation, name)
        got = _frame(block) if block is not None else None
        if got is None:
            print(f"[track]   {name}: not in observation (or no depth)", flush=True)
            continue
        rgb, depth, K, T = got
        from fixtures_cv import bench_z
        from grasp_vision import visible_cable_mask
        surface = bench_z(depth, K, T)
        jacket = visible_cable_mask(rgb, depth, K, T, surface)
        v, u = np.indices(depth.shape)
        world_z = (T[2, 0]*(u-K[0, 2])/K[0, 0]
                   + T[2, 1]*(v-K[1, 2])/K[1, 1] + T[2, 2])*depth + T[2, 3]
        on_bench = (np.isfinite(depth) & (depth > 0.)
                    & (world_z > surface+.0005) & (world_z < surface+.060))
        masks = []
        for candidate in [*_masks(ctx, rgb), jacket]:
            candidate = np.asarray(candidate, dtype=bool) & on_bench
            if int(candidate.sum()) >= 64 and not any(np.array_equal(candidate, old) for old in masks):
                masks.append(candidate)
        if not masks:
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
        # Score alone can prefer a tiny specular fragment to the cable. Use
        # the length measured at initialization to select among RGB-D fits.
        fits = []
        for candidate_mask in masks:
            candidate = ctx.tool("curve.fit_centerline", mask=candidate_mask, depth=depth,
                                 intrinsics=K, camera_pose=T, nodes=CURVE_NODES, **merge_options)
            points = candidate.get("points") or []
            radius = float(candidate.get("radius_m") or 0.)
            if not near_bench(points, surface) or not 0. < radius <= MAX_ROD_RADIUS:
                continue
            fits.append((abs(_length(points) - want), candidate_mask, candidate))
        if not fits:
            continue
        _, mask, fresh = min(fits, key=lambda candidate: candidate[0])
        fpts = fresh.get("points") or []
        if not plausible_length(fpts, want):
            from grasp_vision import visible_cable_mask
            from fixtures_cv import bench_z
            jacket = visible_cable_mask(rgb, depth, K, T, bench_z(depth, K, T))
            if int(jacket.sum()) >= 64:
                alternative = ctx.tool('curve.fit_centerline', mask=jacket, depth=depth,
                                       intrinsics=K, camera_pose=T, nodes=CURVE_NODES, **merge_options)
                alternate_points = alternative.get('points') or []
                if plausible_length(alternate_points, want) and near_bench(alternate_points, surface):
                    fpts = alternate_points
                    print(f'[track] using height-gated jacket fit: {_length(fpts)*1000:.0f} mm', flush=True)
        if plausible_length(fpts, want):
            curve = [[float(v) for v in p] for p in fpts]
            if plausible_length(curve, want):
                ctx.tool("cable.curve_memory", rod=json.dumps(curve))
                return _out("found", curve, "refit", name, len(curve))
        print(f"[track]   {name}: cold fit {_length(fpts) * 1000:.0f}mm of "
              f"{want * 1000:.0f}mm expected; tracking instead", flush=True)

        # THE OPENING SURVEY'S RADIUS, not this frame's. The tracker corrects its
        # observation from the rod's skin onto its axis, and the size of that
        # correction is the rod's radius -- a constant of the scene, measured
        # once in the opening RGB-D survey beside the arclength this node already
        # trusts from there. Re-measured per frame it is area over length, and a
        # carry frame corrupts both terms at once: the mask gains a fingertip or
        # a highlight while occlusion takes away the length that area is divided
        # by. Measured on a saved frame that ratio read 23.15 mm against the
        # opening camera's 3.47 mm, and the tracker pushed the whole cable that
        # far along the view ray -- through the bench it is lying on.
        known_radius = float((initial.get('measured') or {}).get('rod_radius_m') or 0.0)
        upd = ctx.tool(
            "curve.track_centerline", prior=prior,
            mask=jacket if int(jacket.sum()) >= 64 else mask, depth=depth,
            intrinsics=K, camera_pose=T, radius=known_radius or None,
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
        if not plausible_length(curve, want) or not near_bench(curve, surface):
            print(f'[track] refusing tracked length {_length(curve)*1000:.0f} mm against {want*1000:.0f} mm opening observation', flush=True)
            continue
        ctx.tool("cable.curve_memory", rod=json.dumps(curve))
        return _out("found", curve, "tracked", name, seen)

    return None


def run(ctx: NodeContext, scene: str = "", camera: str = "", index: int = 0) -> Output:
    del index
    initial = json.loads(scene or '{}')
    merge_options = {'merge_gap_m': .040} if len(initial.get('stations') or []) == 4 else {}
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
                camera_pose=T, nodes=CURVE_NODES, **merge_options,
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
    if camera and str(camera) != INIT_CAMERA:
        raise ValueError("Routing supports only the cable RGB-D camera")
    order = [INIT_CAMERA]
    frames = observation["cameras"] if isinstance(observation, dict) else observation.cameras
    have = sorted(frames) if isinstance(frames, dict) else [str(f.get("name", "?")) for f in (frames or ())]
    print(f"[track]   camera_in={camera!r} order={order} have={have}", flush=True)
    got = _read_once(ctx, observation, prior, initial, merge_options)
    if got is not None:
        return got

    # Neither the cold fit nor the tracker could speak for this frame. Before
    # handing back a curve that predates the last crossing, get the hand out of
    # the way and ask the same question of a clean view.
    if _step_aside(ctx, prior):
        got = _read_once(ctx, ctx.tool("robot.get_observation"), prior, initial, merge_options)
        if got is not None:
            return got
        return _out("found", prior, "held-after-relook", "", 0)
    return _out("found", prior, "held", "", 0)
