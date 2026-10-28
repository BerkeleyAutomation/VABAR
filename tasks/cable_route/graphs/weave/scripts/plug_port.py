"""Carry the free cable end toward the perceived terminal with the support hand.

The terminal follows the final crossing. RGB-D provides the cable centreline
and terminal position; robot proprioception provides the support TCP. A
fresh terminal grasp addresses a cable that has slid through the support fingers.
The physical start anchor remains pinned throughout.

Carry waypoints go around the peg line, with lower-height retries for poses
outside the support arm's envelope. The mouth pose must be reachable before
release. The node reports the arm motion and regrasp attempt; only the final
task evaluator establishes whether the free endpoint and peg seats succeeded.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terminal_vision import observe_terminal_tail

#: What SAM3 is asked for. The blocks are the same accent colour as the spools
#: and a different shape, and "block" is what separates them -- measured, the
#: spool prompt returns 2 on ``port2`` and does not pick up either terminal.
BLOCK_QUERY = "orange block"
BLOCK_SCORE = 0.50
"""Detection floor. Measured on ``port2``'s own frame, the prompt separates
cleanly and there is no reason to sit near the noise:

    "orange block"   0.97, 0.96   then 0.26 and below
    "orange spool"   0.85, 0.84   -- the SPOOLS, at a fifth the area

Both terminals come back in the nineties and the third candidate is a quarter of
that, so half is a floor with nothing near it."""

BLOCK_MIN_PX = 3400
"""Smallest mask, in pixels, that can be a terminal.

The prompt is not perfectly exclusive -- ``"orange block"`` also returns the
spools further down its ranking -- and shape words are a weak separator between
two objects of the same colour. Area is a strong one here, and the populations
moved when the terminals stopped being slabs. Measured on ``port3``'s opening
frame, before and after:

    slabs      blocks 8,100 and 11,600 px   spools 2,450 and 2,480   floor 4,000
    fittings   port 4,766 px                spools 2,479 and 2,779   floor 3,400

Two things changed and only one of them matters. The port is a channel with a
10 mm slot down the middle, so it masks about 40 per cent less than the slab it
replaced -- 4,766 against a 4,000 floor is 19 per cent of headroom, which is not
enough of a floor to be one. And the ANCHOR has stopped appearing at all: its
rails are the mounting foot and are painted metal
(:data:`~robosimstudio.scene.entities.rod.TERMINAL_METAL`), so "orange block"
no longer grounds it.

That second one is fine and worth saying why: this node wants the PORT, and it
finds it as the block furthest from the run's anchored end. One candidate makes
that trivially right rather than wrong. The disambiguation stays because a board
seen from another angle may still return both.

3,400 sits between 2,779 and 4,766 with 22 per cent of margin below and 40 above,
which is a floor with something on either side of it."""

PORT_INSET = 0.004
"""How far short of the port's centre the carry aims [m].

**Re-derived off the socket, because the block it was derived off is gone.** It
was 12 mm, and the reason given was: "The block is a 40 mm cube and the rod is
10 mm: aiming at the centre drives the end into the far wall and the contact
pushes it back out." Neither number is this scene's any more -- the terminal was
a 32 mm plate before it became a socket, and the rod has been 7 mm since the
weave family was thinned.

The port is now an open channel with a stop at the back
(:class:`~robosimstudio.scene.entities.rod.EntityCablePort`). In its own frame
the channel runs from the mouth at ``+16 mm`` to the back wall at ``-12.8 mm``,
and the block is turned so that axis lies along world ``-x``. So aiming 12 mm
short of centre put the end 4 mm inside a 29 mm channel -- technically in the
mouth, and lying almost entirely outside the thing it was supposed to be plugged
into, which is most of why the port clause read as "nearly" for so long.

4 mm short of centre puts it 12 mm in with 17 mm still ahead of it before the
stop. Deep enough to be in the socket, clear enough that an overshoot is
absorbed by the channel rather than by the wall.
"""

#: Height above the bench the carry runs at [m].
#:
#: **It was 30 mm, and a spool flange stands 23 mm.** So the carry ran the rod's
#: tail seven millimetres over the tops of the very fixtures it had just been
#: seated against, and dragged it across them on the way to the port.
#:
#: That is the whole of why the last crossing never held. Measured over three
#: episodes with the carry skipped entirely, every seat reads **0.088** against a
#: 0.047 threshold -- placement was never the problem. Run the carry and seat2
#: reads 0.000, 0.015, 0.029: the crossing nearest the port, and the one the
#: tail sweeps over on its way there.
#:
#: Low was chosen because the arm cannot hold a vertical wrist high and far out
#: at once (see the module docstring), and that constraint is real -- but it
#: binds on the MOUTH pose, which is at bench level, not on the three staging
#: poses. Those are allowed to be refused and skipped; the mouth is not.
CARRY_Z = 0.070

#: Seconds of settle after the jaws open, so the rod is measured at rest.
SETTLE = 30


class Output(TypedDict):
    exit: str
    detail: str
    port: str
    # The node contract contains only keys present on every exit. Additional
    # diagnostics are returned when observed; GaP treats NotRequired fields
    # as mandatory script outputs.


def endpoint_relative_target(port_xy, endpoint_xy, tcp_xy):
    """Translate a held endpoint onto its target without changing wrist roll."""
    return np.asarray(tcp_xy, dtype=float) + np.asarray(port_xy, dtype=float) - np.asarray(endpoint_xy, dtype=float)


def _blocks(ctx: NodeContext, camera: str) -> list[list[float]]:
    """Every orange block the camera can see, as world centroids."""
    obs = ctx.tool("robot.get_observation")
    frames = obs["cameras"] if isinstance(obs, dict) else obs.cameras
    # BOTH shapes, as ``perceive_rod._camera`` does. An observation carries its
    # frames as a dict on some paths and a list of named blocks on others, and a
    # lookup that only handles the dict returns None on the list -- silently,
    # which is how this node reported "no block in view" for three runs while
    # SAM3 was never being called at all.
    block = None
    if isinstance(frames, dict):
        block = frames.get(camera)
    else:
        for fr in frames or ():
            if str(fr.get("name", "")) == camera:
                block = fr
                break
    if block is None:
        have = sorted(frames) if isinstance(frames, dict) else [
            str(f.get("name", "?")) for f in (frames or ())]
        print(f"[plug] no {camera!r} camera in the observation; have {have}", flush=True)
        return []
    # The same two-dict, two-key lookup ``perceive_rod`` does. A frame carries
    # its channels under ``images`` on some paths and inline on others, and the
    # depth key is ``depth`` or ``depth_data`` -- measured, reading only
    # ``images["depth"]`` returned None on the live observation and this node
    # reported "no block in view" while SAM3 was grounding both blocks at 0.97.
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
        print("[plug] camera frame carries no depth", flush=True)
        return []
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
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
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, (px, py, pz)

    found = ctx.tool("sam3.segment_text", image=rgb, query=BLOCK_QUERY, max_results=0)
    _sc = [round(float(v), 2) for v in (found.get("scores") or [])]
    _ar = [int(np.asarray(m).sum()) for m in (found.get("masks") or [])[:6]]
    print(f"[plug] sam3 {BLOCK_QUERY!r}: {len(_sc)} masks, top scores {_sc[:6]}, "
          f"top areas {_ar}  (need score>={BLOCK_SCORE}, area>={BLOCK_MIN_PX})", flush=True)
    out: list[list[float]] = []
    for mask, score in zip(found.get("masks") or [], found.get("scores") or []):
        if float(score) < BLOCK_SCORE:
            continue
        v, u = np.nonzero(np.asarray(mask) > 0)
        if len(u) < BLOCK_MIN_PX:
            continue
        d = depth[v, u]
        ok = np.isfinite(d) & (d > 1e-4)
        if ok.sum() < 8:
            continue
        u, v, d = u[ok], v[ok], d[ok]
        cam = np.stack([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d,
                        np.ones_like(d)], axis=1)
        world = (T @ cam.T).T[:, :3]
        out.append([float(c) for c in np.median(world, axis=0)])
    return out


def _regrasp_free_end(ctx, support, free, here, curve, port_xy):
    """Clear the hand and establish a fresh, directly observed terminal grasp."""
    diagnostic = {"regrasp_attempted": False,
                  "tracked_end_to_hand_m": float(np.linalg.norm(np.asarray(free)[:2]-here[:2]))}
    seen = observe_terminal_tail(ctx, port_xy)
    diagnostic["tail_before"] = seen
    error = (float(np.linalg.norm(np.asarray(seen["endpoint"])[:2]-here[:2]))
             if seen.get("found") else None)
    if error is not None:
        diagnostic["end_to_hand_before_m"] = error
    # Tip proximity does not establish a pinch. Clear the hand and take a
    # newly observed terminal grasp before every carry.
    arm, arm_id = str(support["arm_name"]), int(support["arm_id"])
    diagnostic["regrasp_attempted"] = True
    try:
        ee = ctx.tool("robot.get_ee_pose", arm_id=arm_id)
        pose = dict(ee.get("pose", ee))
        pose["position"] = dict(x=float(here[0]), y=float(here[1]), z=float(here[2]+.055))
        ctx.tool("cable.set_gripper", width=-1.0, arm=arm)
        ctx.tool("robot.go_to_pose", arm_id=arm_id, pose=pose, max_steps=160)
        fresh = observe_terminal_tail(ctx, port_xy)
        diagnostic["tail_after_clear"] = fresh
        candidates = [dict(c, _source='visible tail')
                      for c in (fresh.get('grasps', []) if fresh.get('found') else [])]
        # The whole centreline was itself fit from the current RTX RGB-D view.
        # Append its inboard points even when exposed-tip candidates exist:
        # occlusion can leave a visible candidate that is real but outside the
        # support arm's reach, and that must not suppress a reachable point a
        # little farther along the same perceived cable.
        pts = np.asarray(curve, dtype=float).reshape(-1, 3)
        if np.linalg.norm(pts[-1, :2] - np.asarray(free)[:2]) < np.linalg.norm(
            pts[0, :2] - np.asarray(free)[:2]
        ):
            pts = pts[::-1]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        for distance in (.015, .025, .040, .060, .080, .100, .120, .150):
            if arc[-1] < distance + .004:
                continue
            xyz = np.array([np.interp(distance, arc, pts[:, k]) for k in range(3)])
            before = np.array([np.interp(max(0.0, distance - .004), arc, pts[:, k]) for k in range(2)])
            after = np.array([np.interp(distance + .004, arc, pts[:, k]) for k in range(2)])
            tangent = after - before
            tangent /= max(float(np.linalg.norm(tangent)), 1.0e-9)
            candidates.append({'xy': xyz[:2].tolist(), 'axis_z': float(xyz[2]),
                               'inboard_m': distance, 'tangent': tangent.tolist(),
                               '_source': 'RTX centreline endpoint'})
        if not candidates:
            raise ValueError("the inboard grasp is not visible after clearing the hand")
        retry = choice = None
        source = ''
        for candidate in candidates:
            # Candidate identity is more important than arm identity. Try both
            # hands on the true exposed tip before accepting an interior-loop
            # point merely because the support arm can reach it.
            for role in ('support', 'work'):
                try:
                    plan = ctx.tool(
                        'cable.plan_support', rod=curve.tolist(), grasp_xy=candidate['xy'],
                        grasp_axis_z=candidate['axis_z'], grasp_tangent=candidate.get('tangent'),
                        arm=role,
                    )
                except Exception:
                    continue
                if plan.get('stages') and float(plan.get('worst_grasp_error', float('inf'))) <= .010:
                    retry, choice = plan, candidate
                    source = str(candidate.get('_source', 'visible tail'))
                    break
            if retry is not None:
                break
        if retry is None:
            raise ValueError("the visible tail grasp is not reachable within 10 mm")
        choice = {k: v for k, v in choice.items() if not str(k).startswith('_')}
        diagnostic['terminal_initial_grasp'] = choice
        arm = str(retry.get('arm_name', arm))
        arm_id = int(retry.get('arm_id', arm_id))
        diagnostic['arm_name'] = arm
        diagnostic['arm_id'] = arm_id
        if retry.get('grasp_m'):
            diagnostic['grasp_m'] = retry['grasp_m']
        diagnostic['grasp_endpoint'] = (
            [float(v) for v in fresh['endpoint']]
            if source == 'visible tail'
            else [float(v) for v in np.asarray(free).reshape(-1)[:3]]
        )
        for stage in retry['stages']:
            ctx.tool("robot.execute_trajectory", trajectory=stage["trajectory"],
                     arm_id=arm_id, max_steps_per_waypoint=15)
        ctx.tool("cable.set_gripper", width=float(retry.get("close", .0045)), arm=arm)
        ctx.tool("cable.wait_steps", steps=24)
        diagnostic["regrasp_reason"] = (f"retook the {source} {choice['inboard_m']*1000:.0f} mm "
                                         "along the cable from its tip")
        print(f"[plug] fresh {source} grasp: target={choice['xy']}", flush=True)
    except Exception as exc:
        diagnostic["regrasp_failed"] = True
        diagnostic["regrasp_reason"] = f"{type(exc).__name__}: {exc}"
        print(f"[plug] terminal regrasp failed: {diagnostic['regrasp_reason']}", flush=True)
    return diagnostic


def terminal_release_height(surface, port_top_z, grasp_tcp_z, cable_axis_z):
    """Place the observed cable axis 2 mm above the observed terminal rim."""
    offset = max(0.0, float(grasp_tcp_z) - float(cable_axis_z))
    return max(float(surface), float(port_top_z) + offset + 0.002)


def _correct_terminal(ctx, support, curve, port, surface, clear_y):
    """One measured correction after release, with an endpoint-relative target."""
    result={"terminal_correction_attempted": False}
    seen=observe_terminal_tail(ctx,port[:2]);result['terminal_tail_after_release']=seen
    if not seen.get('found'):
        result['terminal_correction_reason']='free endpoint is not unambiguously visible'
        return result
    delta=np.asarray(seen['endpoint'])[:2]-np.asarray(port)[:2]
    if np.max(np.abs(delta))<=.035:
        result['terminal_correction_reason']='visible endpoint is within 35 mm of the port centre'
        return result
    arm=str(support['arm_name']);arm_id=int(support['arm_id']);retry=None;choice=None
    # A tip beyond the arm envelope may have a reachable point farther inboard.
    for candidate in seen['grasps']:
        try:
            plan=ctx.tool('cable.plan_support',rod=curve.tolist(),grasp_xy=candidate['xy'],grasp_axis_z=candidate['axis_z'],grasp_tangent=candidate.get('tangent'))
        except Exception as exc:
            result.setdefault('terminal_correction_plan_errors',[]).append(type(exc).__name__)
            continue
        if plan.get('stages') and float(plan.get('worst_grasp_error',float('inf')))<=.008:
            retry=plan;choice=candidate;break
    if retry is None:
        result['terminal_correction_reason']='no visible inboard grasp is reachable within 8 mm'
        return result
    result['terminal_correction_attempted']=True;result['terminal_correction_grasp']=choice
    target=np.asarray(port[:2])-(np.asarray(seen['endpoint'])[:2]-np.asarray(choice['xy']))
    result['terminal_correction_hand_target_xy']=target.tolist()
    try:
        ctx.tool('cable.set_gripper',width=-1.0,arm=arm)
        for stage in retry['stages']:
            ctx.tool('robot.execute_trajectory',trajectory=stage['trajectory'],arm_id=arm_id,max_steps_per_waypoint=15)
        ctx.tool('cable.set_gripper',width=float(retry.get('close',.003)),arm=arm)
        ctx.tool('cable.wait_steps',steps=24)
        ee=ctx.tool('robot.get_ee_pose',arm_id=arm_id);pose=ee.get('pose',ee)
        rotation=dict(pose['rotation']);position=pose['position']
        def move(x,y,z):
            goal=np.array([x,y,z],dtype=float)
            ctx.tool('robot.go_to_pose',arm_id=arm_id,max_steps=240,pose={
                'position':dict(zip(('x','y','z'),map(float,goal))),'rotation':rotation})
            observed=ctx.tool('robot.get_ee_pose',arm_id=arm_id);at=observed.get('pose',observed)['position']
            error=float(np.linalg.norm(goal-np.array([at[k] for k in ('x','y','z')])))
            if error>.008:raise ValueError(f'correction waypoint missed by {error*1000:.1f} mm')
        # The port rim is above the bench. Preserve the observed TCP-to-cable
        # offset and release above that rim instead of driving the pads into it.
        release_z = terminal_release_height(surface, port[2], position['z'], choice['axis_z'])
        result['terminal_correction_release_z'] = release_z
        lift_z=max(float(position['z'])+.025,release_z+.025)
        move(float(position['x']),float(position['y']),lift_z)
        move(float(target[0]),float(target[1]),lift_z)
        move(float(target[0]),float(target[1]),release_z)
        result['terminal_correction_motion_completed']=True
        print(f'[plug] endpoint-relative correction motion: hand target={target.tolist()}',flush=True)
    except Exception as exc:
        result['terminal_correction_failed']=True
        result['terminal_correction_reason']=f'{type(exc).__name__}: {exc}'
    finally:
        ctx.tool('cable.set_gripper',width=.0475,arm=arm)
        ctx.tool('cable.wait_steps',steps=30)
        try:
            ctx.tool('robot.go_to_pose',arm_id=arm_id,max_steps=200,pose={
                'position':dict(x=float(port[0]-.10),y=float(clear_y),z=float(surface+.12)),
                'rotation':dict(w=0.,x=1.,y=0.,z=0.)})
        except Exception as exc:
            result['terminal_correction_retreat_error']=type(exc).__name__
    return result


def run(ctx: NodeContext, scene: str = "", arm_id: int = 0,
        grip_open_m: float = 0.0475, grip_close_m: float = 0.003) -> Output:
    seen = json.loads(scene or "{}")
    stations = [(float(a), float(b)) for a, b in (seen.get("stations") or [])]
    rod = json.loads(ctx.tool("cable.curve_memory").get("rod") or "null")
    # ENTRY, unconditionally. Every other exit below prints too, because the
    # first version of this node printed only on the happy path and a silent
    # early return is indistinguishable in a log from a node that never ran.
    print(f"[plug] entered: {len(stations)} stations, "
          f"{0 if not rod else len(rod)} rod nodes", flush=True)
    if not rod or len(rod) < 2 or not stations:
        print("[plug] no centreline or no stations; nothing to terminate", flush=True)
        return {"exit": "unplugged", "detail": "no centreline or no stations", "port": "null"}
    curve = np.asarray(rod, dtype=np.float64).reshape(-1, 3)

    blocks = _blocks(ctx, "cable")
    if len(blocks) < 1:
        # The weave layouts have no terminals, so this is the normal path there
        # and not a failure -- see the routing note in workflow.json.
        print("[plug] no terminal detected; cannot attempt endpoint insertion", flush=True)
        return {"exit": "unplugged", "detail": "no block in view", "port": "null"}

    # WHICH BLOCK IS THE PORT: the one furthest from the rod's anchored end.
    # Both terminals are the same part and the same colour, so nothing about
    # their appearance separates them -- what does is that one of them is where
    # the run starts, and the run starts at the end the support hand is holding.
    anchored = curve[0] if curve[0][1] < curve[-1][1] else curve[-1]
    free = curve[-1] if curve[0][1] < curve[-1][1] else curve[0]
    port = max(blocks, key=lambda b: float(np.hypot(b[0] - anchored[0], b[1] - anchored[1])))
    print(f"[plug] {len(blocks)} block(s); port at "
          f"({port[0]:.3f}, {port[1]:.3f}) free end at ({free[0]:.3f}, {free[1]:.3f})",
          flush=True)

    # ONE CALL, READ TWICE. ``cable.plan_support`` is asked here rather than
    # further down because two separate things below need it: which hand is the
    # support hand (so it can be told to let go) and what height this cell's
    # jaws have to be at to straddle a rod lying on the bench. Both are
    # arithmetic over the hand's own colliders; neither reads where the rod is.
    support = ctx.tool("cable.plan_support", rod=[[float(v) for v in pt] for pt in curve])
    support_arm = str(support.get("arm_name", ""))

    # The support arm can reach the held tail. Refuse an absent arm identity
    # instead of falling back to the work arm.
    if "arm_id" not in support:
        print("[plug] no support hand from cable.plan_support; refusing to plug with "
              "the work arm -- it cannot reach the run's free end", flush=True)
        return {"exit": "unplugged", "detail": "no support arm",
                "port": json.dumps([round(float(v), 4) for v in port])}
    arm_id = int(support["arm_id"])
    print(f"[plug] {support_arm or f'arm {arm_id}'} handles terminal carry", flush=True)
    ee = ctx.tool("robot.get_ee_pose", arm_id=int(arm_id))
    p = (ee.get("pose", ee)).get("position", {})
    here = np.array([float(p.get(k, 0.0)) for k in ("x", "y", "z")])
    end_grasp = _regrasp_free_end(ctx, support, free, here, curve, port[:2])
    if end_grasp.get("regrasp_failed"):
        return {"exit": "unplugged", "detail": end_grasp["regrasp_reason"],
                "port": json.dumps(port), **end_grasp}
    if "arm_id" in end_grasp:
        support = {**support, "arm_id": int(end_grasp["arm_id"]),
                   "arm_name": str(end_grasp.get("arm_name", support_arm))}
        arm_id = int(support["arm_id"])
        support_arm = str(support["arm_name"])
    if end_grasp["regrasp_attempted"]:
        ee = ctx.tool("robot.get_ee_pose", arm_id=int(arm_id))
        p = (ee.get("pose", ee)).get("position", {})
        here = np.array([float(p.get(k, 0.0)) for k in ("x", "y", "z")])

    # THE HEIGHT COMES FROM THE HAND, NOT FROM THE FITTED CURVE, and this was
    # the whole of why the port clause read 0.000 on every episode.
    #
    # This node was the only one in the graph that took a working HEIGHT off the
    # perceived centreline -- ``surface = median(curve[:, 2])`` -- and the
    # centreline's z is not fit for that. Measured on ``port3`` episode 0: the
    # rod lies at z = 0.7535 and the plug commanded its grasp at ~0.840, so the
    # jaws closed **87 mm above the cable**, travelled to the mouth holding air
    # and reported a delivery. Across the whole episode the work hand's pads
    # never came within 78 mm of the rod, and when they were over it in xy they
    # sat 80 to 257 mm above.
    #
    # Every other node survives the same curve because it takes only x and y
    # from it: ``plan_station`` computes its own ``rest_z`` from the bench and
    # the hand's measured pad drop, which is why all three crossings seat at
    # 0.073 to 0.088 off the same fit that defeats this node.
    #
    # So do what they do. ``cable.plan_support``'s ``grasp_m`` carries exactly
    # that height -- ``max(surface_z + rod radius, surface_z + pad_tip_drop +
    # PAD_FLOOR_CLEAR)``, 0.7607 on this cell -- and it is arithmetic over the
    # hand's colliders and the bench, not a reading of where the rod is. The
    # median of the curve stays as the fallback, because a node that cannot get
    # a height at all should still do what it used to rather than nothing.
    # Preserve the wrist orientation that established the pinch. Switching to
    # the generic grasp frame here can turn this held cable by 152 degrees.
    carry_rotation = dict(ee.get("pose", ee)["rotation"])
    grasp_ref = end_grasp.get("grasp_m") or support.get("grasp_m") or []
    if len(grasp_ref) >= 3:
        surface = float(grasp_ref[2])
        print(f"[plug] grasp height {surface:.4f} m from the hand's own pad drop "
              f"(the fitted curve's median z was {float(np.median(curve[:, 2])):.4f})", flush=True)
    else:
        surface = float(np.median(curve[:, 2])) + 0.004
        print(f"[plug] no support plan; falling back to the curve's median z {surface:.4f}", flush=True)

    # Into the mouth ALONG the bench, from the station side.
    toward = np.array([port[0], port[1], surface])
    inset = toward - np.array([PORT_INSET, 0.0, 0.0]) * np.sign(port[0] - here[0] or 1.0)
    observed_tail = end_grasp.get("tail_after_clear", {})
    held_endpoint = end_grasp.get("grasp_endpoint")
    if held_endpoint is not None:
        # Use the endpoint belonging to the grasp we actually selected. A
        # local terminal view may expose a different truncated white fragment;
        # mixing that endpoint with a full-centreline grasp produces a hand-tip
        # offset that can be hundreds of millimetres wrong.
        endpoint = np.asarray(held_endpoint, dtype=float)[:2]
        inset[:2] = endpoint_relative_target(inset[:2], endpoint, here[:2])
    elif observed_tail.get("found"):
        # The jaws grasp inboard of the tip. Put the tip in the terminal while
        # leaving the jaws clear of its hood.
        endpoint = np.asarray(observed_tail["endpoint"], dtype=float)[:2]
        inset[:2] = endpoint_relative_target(inset[:2], endpoint, here[:2])
    if len(stations) == 4:
        axis_z = float(np.median(curve[:, 2]))
        inset[2] = terminal_release_height(surface, float(port[2]), surface, axis_z)

    # ROUTE AROUND THE SEATS, NOT ACROSS THEM.
    #
    # The port is further out in +x than every station, so the direct path from
    # the free end to the port sweeps over the crossings that were just seated
    # and drags them out from under their spools -- measured, both seats went
    # from held to wrong-side and the score fell 0.67 -> 0.33 while the end
    # itself landed correctly in the mouth.
    #
    # There is no shortage of cable: with the anchor at one end and two crossings
    # taken, ~225 mm of rod remains for a 96 mm span to the port. What fails is
    # the PATH, so the fix is the path. The carry goes out past the free end's
    # own y first, then across in x well clear of the station line, and only then
    # back down into the mouth -- three sides of a rectangle rather than the
    # diagonal that cuts the weave.
    # Staged off the hand's OWN pose rather than off the fitted end, because the
    # hand is what is moving and the fit is only an estimate of where it is.
    clear_y = max(float(here[1]), float(port[1]) + 0.06)
    lift = np.array([here[0], here[1], surface + CARRY_Z])
    across = np.array([inset[0], clear_y, surface + CARRY_Z])
    over = np.array([inset[0], inset[1], surface + CARRY_Z])

    def _go(target, what: str) -> bool:
        """One waypoint. A refused pose is reported and skipped, not fatal.

        The carry is four poses and they are not equally load-bearing: the
        staging heights exist to keep the rod off the fixtures, and losing one of
        them costs tidiness. Losing the MOUTH pose costs the task. Aborting the
        whole leg because a staging pose missed threw away the ones that solved
        -- measured, the first version refused the entire carry on one
        ``RuntimeError`` and never reached the port at all.

        Staging refusals remain visible in the node output. The caller treats
        a failed mouth waypoint as an unsuccessful terminal motion.
        """
        # RETRY LOWER BEFORE GIVING UP, because height is what this arm refuses
        # for. The envelope narrows as the wrist goes up and out (the module
        # docstring has the sweep), so a staging pose refused at carry height is
        # very often reachable 20 mm down -- and 20 mm down still clears a 23 mm
        # flange by more than the 30 mm this carry used to run at.
        #
        # This matters more than it did. When the carry ran at 30 mm a skipped
        # staging pose was tidiness. At 70 mm, skipping ``across`` drops the
        # hand onto the diagonal from the free end to the port, which is the
        # path that drags the cable over the crossings -- the exact failure
        # CARRY_Z was raised to stop. Silently taking it back is not an option.
        for drop in (0.0, 0.020, 0.040):
            z = float(target[2]) - drop
            try:
                ctx.tool("robot.go_to_pose", arm_id=int(arm_id), max_steps=160, pose={
                    "position": {"x": float(target[0]), "y": float(target[1]), "z": z},
                      "rotation": carry_rotation})
                if drop:
                    print(f"[plug]   {what} at {drop * 1000:.0f} mm lower", flush=True)
                return True
            except Exception as exc:  # noqa: BLE001
                last = type(exc).__name__
        print(f"[plug]   {what} refused at every height: {last}", flush=True)
        return False

    # Keep the current or renewed support pinch until the mouth waypoint.
    # The pinned start is unaffected by this terminal grasp.
    try:
        _go(lift, "up to carry height")
        # ``across`` is the one staging pose that is load-bearing: it is what
        # keeps the run's tail off the crossings. If it is lost, say so in the
        # RESULT and not only in the log, so a run's traces carry the reason its
        # seats came back thin.
        routed = _go(across, "across, clear of the stations")
        if not routed:
            print("[plug]   WARNING: carrying on the diagonal; expect the crossing "
                  "nearest the port to be dragged", flush=True)
        _go(over, "over the port")
        # The one that has to land.
        if not _go(inset, "mouth"):
            return {"exit": "unplugged", "detail": "the port mouth was unreachable",
                    "port": json.dumps([round(float(v), 4) for v in port])}
        # Only now does anything open, and only this hand.
        ctx.tool("cable.set_gripper", width=float(grip_open_m), arm=support_arm)
        ctx.tool("cable.wait_steps", steps=int(SETTLE))
        # Out of the way, so `verify` measures the rod and not the hand. Back
        # along -x rather than up: this arm is at full stretch over the port and
        # a vertical retreat from there is the pose the reach sweep refuses.
        _go(np.array([inset[0] - 0.10, clear_y, surface + 0.12]), "retreat")
    except Exception as exc:  # noqa: BLE001 -- a refused pose is a failed plug, not a crash
        print(f"[plug] refused: {type(exc).__name__}", flush=True)
        return {"exit": "unplugged", "detail": f"{type(exc).__name__} on the carry",
                "port": json.dumps([round(float(v), 4) for v in port])}

    # Observe the released tip after the retreat disturbance has settled. An
    # airborne tip can be outside the bench-height detector's accepted band.
    ctx.tool("cable.wait_steps", steps=120)
    correction = ({'terminal_correction_attempted': False} if len(stations) == 4 else
                  _correct_terminal(ctx, support, curve, port, surface, clear_y))

    # WHAT THIS LINE IS A CLAIM ABOUT. The hand, and only the hand.
    #
    # It used to read "carried the free end to (x, y)" with the MOUTH's
    # coordinates in it -- the waypoint that was commanded, not a measurement of
    # the rod. On an episode where the grasp had missed entirely it printed
    # exactly the same sentence: the free end finished at (0.461, 0.249), 178 mm
    # from the port, having never moved toward it, and the closest any body came
    # was 85 mm. The log said delivered, the score said 0.000, and the
    # disagreement took a privileged replay to see.
    #
    # The grasp is now taken off a centreline re-fitted after the last crossing
    # (the ``retrack`` node), which is the fix for the miss. This is the fix for
    # not being able to tell: the sentence says whose position it is reporting.
    # WHERE THE HAND IS, NOT WHERE IT WAS SENT. The comment above says this
    # sentence reports "whose position it is reporting" -- and it was printing
    # ``inset``, the commanded waypoint. A refused mouth pose printed the same
    # line as a delivered one, which is the whole class of bug this node keeps
    # producing: a log that describes the plan rather than the outcome.
    landed = None
    try:
        _ee = ctx.tool("robot.get_ee_pose", arm_id=int(arm_id))
        _p = (_ee.get("pose", _ee)).get("position", {})
        landed = np.array([float(_p.get(k, 0.0)) for k in ("x", "y", "z")])
    except Exception:  # noqa: BLE001 -- a pose readback is diagnostics, not the task
        pass
    if landed is None:
        print("[plug] terminal motion finished; support pose after retreat is unavailable", flush=True)
    else:
        miss = float(np.linalg.norm(landed[:2] - inset[:2]))
        print(f"[plug] support hand after retreat: ({landed[0]:.3f}, {landed[1]:.3f}); "
              "endpoint placement is evaluated separately", flush=True)
    # The exit stays ``plugged`` -- the router knows three exits and inventing a
    # fourth routes nowhere -- but the DETAIL says whether the run got there the
    # routed way or on the diagonal, because those two produce the same port
    # clause and very different seat clauses.
    return {"exit": "plugged", **end_grasp, **correction,
            "detail": ("support hand completed the port motion" if routed else
                       "support hand completed a diagonal port motion -- the "
                       "clear-of-the-stations pose was refused at every height"),
            "port": json.dumps([round(float(v), 4) for v in port])}
