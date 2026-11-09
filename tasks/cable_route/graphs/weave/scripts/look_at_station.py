"""Park the hand above the next station and look straight down.

**Active perception, and the reason it is worth a node.** A fixed overhead rig
resolves this bench at 1.75 mm/px: a 10 mm rod is under six pixels and the seat
offset that decides the score is five. The hand carries a camera, and bringing
it to 160 mm above the station it is about to work puts the same rod at
0.24 mm/px. That is the difference between measuring a rod and inferring one,
and it costs one short move the arm was going to make anyway -- the station's
own staging pose is a few centimetres from this one.

It is also the arrangement that transfers. A real cell has a wrist camera and
frequently no overhead at all; a policy that can only see from a gantry is a
policy that cannot be moved to a different bench.

**Top-down on purpose.** The seat is an offset in the bench plane, so the view
that measures it should look along the bench normal: an oblique view turns a
spool's flange disc into an ellipse whose centroid is not its centre, and turns
a lateral offset into a foreshortened one. Straight down, a pixel offset is a
metric offset through one depth sample.

**A look, not a scan.** One pose per station rather than a sweep, because the
thing being measured is local -- where this rod lies beside this post -- and a
sweep would pay for a whole-scene reconstruction the plan never reads.

Failure here is not fatal. The bench cameras still see the scene, so a refused
or unreachable viewpoint routes ``looked`` anyway with ``camera`` naming what
the next node should read. Refusing to weave because the preferred view was
unreachable would trade a better measurement for no measurement.
"""

import os
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

#: Where to drop the view this node took, when set. Debug only: a viewpoint
#: that frames the wrong thing scores exactly like one that frames the right
#: thing, because the bench camera carries the run either way -- so the only
#: way to know what the hand actually saw is to look at it.
DEBUG_DIR = os.environ.get("RSS_LOOK_DEBUG", "")

#: Set to skip the move entirely and report the bench view. The control arm of
#: "does the hand camera earn its move" -- without it, the comparison is between
#: two runs on different days rather than two arms of one experiment.
NO_WRIST = bool(os.environ.get("RSS_NO_WRIST", ""))

#: Height above the bench the hand is parked at for a look [m].
#:
#: **160 mm because 280 was unreachable, and that was measured, not guessed.**
#: Full-pose IK for a straight-down wrist over each station:
#:
#:   z = 0.28 m  ->  127-154 mm residual   (no)
#:   z = 0.22 m  ->   35-112 mm residual   (no)
#:   z = 0.16 m  ->        0 mm residual   (yes)
#:
#: The arm cannot hold a straight-down wrist high over the peg line; it can at
#: 160. Tilting 30 degrees reaches every height, but a tilt costs the property
#: the view exists for -- see the top-down note above -- so the standoff moves
#: and the orientation does not.
#:
#: At 160 mm the 640x480 / 40-degree wrist resolves 0.24 mm/px over a 155 mm
#: field: seven times the bench rig, across one station and its immediate
#: surroundings. Stated here because a node script cannot import the suite
#: without pinning itself to that layout.
LOOKDOWN_Z = 0.16

#: How far the look pose sits from the station along -x, toward the robot [m].
#: The rod lies on the robot's side of the stations at rest and is carried
#: across them, so a view centred on the post alone frames the destination and
#: not the material.
LOOKDOWN_BACK = 0.25

#: Straight down: 180 degrees about x carries a hand's own approach axis
#: (local +z) onto world -z. xyzw, as everything below newton speaks.
LOOKDOWN_TILT_DEG = 35.0
"""How far the wrist is pitched off straight-down at the viewpoint [deg].

**Not zero, and the reason changed.** A strict top-down view was chosen when
this camera measured spools: a flange disc viewed obliquely projects to an
ellipse whose centroid is not its centre. The spools are measured from the bench
now, and what the hand measures is the CABLE -- back-projected through depth,
where obliquity costs nothing.

What the tilt buys is the whole object -- but only when the camera is actually
*aimed* at it, which is the part field-of-view arithmetic does not tell you.

Counted properly, by projecting the cable's own 42 segments into the wrist frame
at each candidate pose:

===================================  ==========================
Pose                                 Cable segments in frame
===================================  ==========================
tilt  0 deg, 160 mm, 45 mm back      29 / 42   (69%)
tilt 35 deg, 160 mm, 250 mm back     37 / 42   (88%)
tilt 50 deg, 250 mm,  50 mm back      0 / 42   (0%)
===================================  ==========================

The last row was shipped, and it scored 0.50. Its ground *footprint* was the
largest of the three -- 586 mm against 155 -- which is what the arithmetic
reported and why it was chosen; at 50 degrees the camera simply looks past the
cable. Footprint size is necessary and nowhere near sufficient, and the only
honest test is to project the object and count.
"""


def _lookdown_quat():
    """Straight-down, pitched back by :data:`LOOKDOWN_TILT_DEG`. xyzw."""
    from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

    q = (R.from_euler("x", 180.0, degrees=True)
         * R.from_euler("y", LOOKDOWN_TILT_DEG, degrees=True)).as_quat()
    return tuple(float(v) for v in q)

#: Seconds the arm is given to reach the viewpoint before the look is taken
#: anyway. A pose it cannot reach is a degraded observation, not a failed task.
LOOK_STEPS = 90


class Output(TypedDict):
    route: str
    camera: str
    reached: bool
    detail: str


def run(ctx: NodeContext, index: int = 0, scene: str = "", arm_id: int = 0) -> Output:
    import json  # noqa: PLC0415

    seen = json.loads(scene or "{}")
    stations = seen.get("stations") or []
    if not stations:
        return {"route": "looked", "camera": "overhead", "reached": False,
                "detail": "no stations perceived; falling back to the bench view"}

    # The station this pass is about, in the order the planner works them.
    k = int(index) % len(stations)
    if NO_WRIST:
        print(f"[look] station={k} SKIPPED (RSS_NO_WRIST); bench view", flush=True)
        return {"route": "looked", "camera": "overhead", "reached": False,
                "detail": "hand-camera look disabled for the control arm"}
    sx, sy = (float(v) for v in stations[k])
    qx, qy, qz, qw = _lookdown_quat()
    target = {
        "position": {"x": sx - LOOKDOWN_BACK, "y": sy, "z": float(LOOKDOWN_Z) + 0.75},
        "rotation": {"w": qw, "x": qx, "y": qy, "z": qz},
    }

    reached = True
    try:
        ctx.tool("robot.go_to_pose", pose=target, arm_id=int(arm_id), max_steps=int(LOOK_STEPS))
    except Exception as exc:  # noqa: BLE001 -- an unreachable look is a degraded view, not a failure
        reached = False
        print(f"[look] station={k} refused: {type(exc).__name__}", flush=True)
        detail = f"viewpoint refused ({type(exc).__name__}); reading the bench view instead"
        return {"route": "looked", "camera": "overhead", "reached": False, "detail": detail}

    # Confirm the hand actually got there before promising the close view. A
    # servo that ran out of steps leaves the camera somewhere else entirely, and
    # a node that measured from there would report a rod that is not the rod.
    try:
        ee = ctx.tool("robot.get_ee_pose", arm_id=int(arm_id))
        pos = (ee.get("pose") or ee).get("position", {})
        here = np.array([float(pos.get(k2, 0.0)) for k2 in ("x", "y", "z")])
        want = np.array([target["position"][k2] for k2 in ("x", "y", "z")])
        miss = float(np.linalg.norm(here - want))
    except Exception:  # noqa: BLE001
        miss = 0.0
    if miss > 0.05:
        print(f"[look] station={k} missed viewpoint by {miss * 1000:.0f}mm; bench view", flush=True)
        return {"route": "looked", "camera": "overhead", "reached": False,
                "detail": f"hand stopped {miss * 1000:.0f} mm from the viewpoint; using the bench view"}
    if DEBUG_DIR:
        _dump(ctx, k)
    print(f"[look] station={k} REACHED wrist viewpoint, miss={miss * 1000:.0f}mm", flush=True)
    return {"route": "looked", "camera": "wrist", "reached": bool(reached),
            "detail": f"looking down on station {k} from {LOOKDOWN_Z * 1000:.0f} mm"}


def _dump(ctx: NodeContext, k: int) -> None:
    """Write the wrist and bench views this look produced, side by side."""
    try:
        import cv2  # noqa: PLC0415

        obs = ctx.tool("robot.get_observation")
        frames = obs["cameras"] if isinstance(obs, dict) else obs.cameras
        got = {}
        seq = frames.values() if isinstance(frames, dict) else frames
        for fr in seq:
            name = str(fr.get("name", ""))
            if name in ("wrist", "overhead"):
                got[name] = np.asarray(fr.get("rgb"), dtype=np.uint8)
        for name, rgb in got.items():
            cv2.imwrite(f"{DEBUG_DIR}/look_{k}_{name}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except Exception:  # noqa: BLE001 -- a debug dump must never affect the run
        pass
