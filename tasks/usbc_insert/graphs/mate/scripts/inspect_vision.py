"""Refine the receptacle mouth from a fresh empty-hand wrist RGB-D frame.

An aperture read must agree with the earlier camera measurement. If the local
ring detector finds a side face, fit the front plane near that earlier read
and search its footprint for a recess. Occlusion retains the earlier camera
measurement; no authored fixture pose is queried.
"""

import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

# The runtime loads each node script standalone, so a sibling is not importable
# by package path. Same move as cable_route's `fixtures_cv`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bore_cv
import relook_vision as rl

REQUIRED_TOOLS: tuple[str, ...] = ()
CAMERA = "wrist"

MAX_MOVE_MM = 6.0
"""How far this read may disagree with the bench read ACROSS THE BENCH [mm].

Horizontal only -- see :data:`MAX_RISE_MM` for the other axis, and read that
first, because the split is the whole of this gate now.

A refinement that can be arbitrarily worse than the thing it refines is not a
refinement. Between the bench read and this station the only thing that moves
the receptacle is the brace, and the shipped oracle was measured moving the
mouth by **up to 1.0 mm**; the bench read is good to about a millimetre in x and
y, because what it measures there is a real silhouette edge. So a horizontal
disagreement of six is already several times anything physical, and a
disagreement of forty is the detector having found a different surface.

That is not hypothetical. Run live, the first version of this body came back
**37 to 40 mm** from the bench read with a bore axis pitched 53 degrees below
where the bench put it, off blobs of 356-1098 px where the validated aperture is
tens -- it had locked onto a face, not the slot. Every one of those is rejected
by this gate and by AXIS_MAX_DEG below, and the schedule proceeds on the read it
already had.

**The premise is the bench read's, so it has to be kept there.** On the
documents' ``yam_station_usbc@v1`` head camera the whole-silhouette bench read
of the bolted port was 7.7-9.7 mm and 6.5-10.6 degrees off, and this gate threw
away a wrist read good to 0.2 mm on every L1 episode. The fix is in the bench
read (``ground_vision.ROOF_BAND``), not here: loosening this gate would also let
through the misdetections it exists for.
"""

MAX_RISE_MM = 45.0
"""How far this read may disagree with the bench read IN HEIGHT [mm].

**Seven times looser than the horizontal gate, because the bench view cannot
measure this axis at all.** The top-down camera looks straight down at a bore
that runs horizontally: it sees the receptacle's roof and its outline, and the
height of the hole in its face is simply not in that picture. What grounding
publishes for z is a mid-silhouette guess -- half way between what the part
stands on and its own top -- and on the block port that lands 26 mm above the
true mouth. It is honest for aiming a camera and worthless as a datum.

So the two axes are gated apart rather than through one norm. Gating them
together is what a single `MAX_MOVE_MM` did, and once the bench read stopped
measuring the aperture it meant this body rejecting its OWN correct wrist
reading -- 388 px of aperture at 305 mm, thrown away for disagreeing with a
guess. The failure the gate exists to catch is a detector locking onto a face
instead of the slot, which came back 37-40 mm off WITH a bore axis pitched 53
degrees: that is still caught, by the horizontal gate and by
:data:`AXIS_MAX_DEG`.

45 mm is sized to the receptacles this suite stages -- the tallest is 150 mm and
its bore sits within a part-height of the guess -- not to the detector.
"""

AXIS_MAX_DEG = 8.0
"""How far the axis may swing from the bench read's and still be believed [deg].

The mouth gate alone is not enough: a detection can land near the right point
with a wildly wrong normal, and the axis is what the whole stroke is aimed
along. A braced receptacle yaws by degrees -- the shipped oracle's own relook
reports 0.08 to 2.18 -- so eight is generous and still excludes the 53-degree
misdetections above."""


class Output(TypedDict):
    route: str
    inspect_body: str
    scene_mouth_point: list
    scene_bore_axis: list
    inspect_mouth_point: list
    inspect_bore_axis: list
    inspect_moved_mm: float


def run(
    ctx: NodeContext,
    scene_mouth_point: list | None = None,
    scene_bore_axis: list | None = None,
    camera: str = CAMERA,
    scene_receptacle_held: bool = False,
) -> Output:
    axis_limit = 18.0 if scene_receptacle_held else AXIS_MAX_DEG
    observation = ctx.tool("robot.get_observation")
    rgb, depth, K, T = rl._frame(rl._camera(observation, camera), camera)

    mouth, axis, info = None, None, "no prior axis"
    if scene_mouth_point and scene_bore_axis:
        import cv2  # noqa: PLC0415

        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        face = cv2.inRange(hsv, np.array(bore_cv.BLUE_LO, np.uint8),
                          np.array(bore_cv.BLUE_HI, np.uint8))
        v, u = np.indices(depth.shape)
        rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1],
                         np.ones_like(depth)], -1)
        world = (rays * depth[..., None]) @ T[:3, :3].T + T[:3, 3]
        face[np.linalg.norm(world - np.asarray(scene_mouth_point), axis=-1) > .06] = 0
        # Fit the front plane and equal-depth chamfer sections first. A ray
        # through only the deep visible wall projects above the bore centre
        # when the inspection camera looks down into the port.
        mouth, axis, info = bore_cv.recess_frame(face, depth, K, T, scene_bore_axis)
    axis, moved, rose, turned = _check(mouth, axis, scene_mouth_point, scene_bore_axis, allow_pitch=scene_receptacle_held)
    if moved > MAX_MOVE_MM or rose > MAX_RISE_MM or turned > axis_limit:
        mouth, axis, info = bore_cv.find_bore(rgb, depth, K, T)
        axis, moved, rose, turned = _check(mouth, axis, scene_mouth_point, scene_bore_axis, allow_pitch=scene_receptacle_held)
    if moved > MAX_MOVE_MM or rose > MAX_RISE_MM or turned > axis_limit:
        print(f"[inspect:vision] no consistent bore in {camera!r}: {info}; "
              "keeping the earlier camera read", flush=True)
        return _kept(scene_mouth_point, scene_bore_axis)
    print(
        f"[inspect:vision] mouth {_mm(mouth)} from {info['slot_px']} px of aperture at "
        f"{info['face_mm']:.0f} mm ({moved:.2f} mm from the bench read)  "
        f"axis ({axis[0]:+.2f}, {axis[1]:+.2f}, {axis[2]:+.2f})",
        flush=True,
    )
    return {
        "route": "found",
        "inspect_body": "vision",
        "scene_mouth_point": [float(v) for v in mouth],
        "scene_bore_axis": [float(v) for v in axis],
        # THE SAME NUMBERS AGAIN, UNDER THEIR OWN NAMES, and the duplication is
        # the point. `scene_*` is the scene frame, which every station is
        # entitled to refine and which is populated whether or not this body saw
        # anything. `inspect_*` is THIS station's own measurement and is empty
        # when there was none -- so a consumer that specifically wants "the bore
        # as read head-on through the aperture" can ask for it and can tell the
        # difference between a fresh read and a carried-through one. The re-look
        # needs exactly that distinction: see `relook_vision.run`.
        "inspect_mouth_point": [float(v) for v in mouth],
        "inspect_bore_axis": [float(v) for v in axis],
        "inspect_moved_mm": moved,
    }


def _check(mouth, axis, prior_mouth, prior_axis, *, allow_pitch=False):
    """Sign and gate a measured normal against the previous camera frame."""
    if mouth is None or axis is None:
        return None, float("inf"), float("inf"), float("inf")
    # The free bulkhead can pitch while braced. Preserve its measured normal;
    # the fixed port retains the planar heading used by its motion policy.
    axis = np.asarray(axis if allow_pitch else [axis[0], axis[1], 0.], dtype=float)
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm < 1.e-9 or not np.all(np.isfinite(mouth)):
        return None, float("inf"), float("inf"), float("inf")
    axis /= norm
    if prior_axis and axis @ np.asarray(prior_axis) < 0:
        axis = -axis
    if prior_mouth:
        delta = np.asarray(mouth, dtype=float) - np.asarray(prior_mouth, dtype=float)
        # Split, because the prior is trustworthy in one plane and not the other.
        moved = float(np.linalg.norm(delta[:2]) * 1000.)
        rose = float(abs(delta[2]) * 1000.)
    else:
        moved = rose = 0.
    return axis, moved, rose, _turn_deg(axis, prior_axis)


def _turn_deg(a, b) -> float:
    """How far the two axes disagree [deg]. Zero when the prior is unstated."""
    if b is None or len(b) == 0:
        return 0.0
    prior = np.asarray([float(v) for v in b], dtype=np.float64)
    n = float(np.linalg.norm(prior)) * float(np.linalg.norm(a))
    if n <= 0.0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(float(np.asarray(a) @ prior) / n, -1.0, 1.0))))


def _kept(mouth: Any, axis: Any) -> Output:
    return {
        "route": "found",
        "inspect_body": "vision",
        "scene_mouth_point": [float(v) for v in (mouth or [])],
        "scene_bore_axis": [float(v) for v in (axis or [])],
        "inspect_mouth_point": [],
        "inspect_bore_axis": [],
        "inspect_moved_mm": 0.0,
    }


def _mm(v) -> str:
    return "(" + ", ".join(f"{float(x) * 1e3:.1f}" for x in v) + ") mm"
