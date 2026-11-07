# SPDX-License-Identifier: Apache-2.0
"""The gear_mesh suite's registered names: one arrangement, one camera rig.

Own registrations even where numbers match a sibling's -- a task may not import
another task's, and a name several documents stage must resolve to one object.
"""

from __future__ import annotations

import vabar  # noqa: F401 -- the scripted-routine module RoboSimStudio leaves out; see vabar/__init__.py

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.specs import CollisionSpec, backend
from robosimstudio.specs.physics import ArrangementSpec
from robosimstudio_bench.taskbuild.arrangements import register_arrangement, register_camera_rig

GEAR_PHYSICS = register_arrangement(
    "gear_mesh_960hz@v1",
    # The contact-rich rigid rates (60 Hz x 16 substeps, dt = 1/960), which is
    # where polish and box_decant both ended up. What is NOT inherited is the
    # reasoning, and two of these numbers are this suite's own:
    #
    # * `enable_multiccd=True` -- the whole task. Every drive contact here is one
    #   convex tooth prism against another, a couple of millimetres across, and
    #   mujoco_warp's multi-point manifold covers mesh pairs only under this
    #   flag. polish measured what a single contact point costs on a mesh drum:
    #   visible rocking and chatter. A gear train is that failure once per tooth.
    # * `rigid_gap=5.0e-5` is the one that must not drift, and it is coupled to
    #   the tooth profile rather than free. A contact gap fattens every collider
    #   on every face, so a mesh needs `backlash > 2 * gap` or the flanks are
    #   permanently in contact and the train jams -- the same shape of error as
    #   box_decant's 2 mm seam against a 1.5 mm blade. The shipped backlash is
    #   0.15 mm at the 2.5 mm module, so 0.05 mm clears it three times over;
    #   Newton's centimetre default would not clear it at all.
    # * contact (6e4, 500) = 4 ms solref, the stiffest this substep admits
    #   (tool_sort's measurement; MuJoCo clamps below 2*dt).
    # * `impratio=10` with the elliptic cone, for polish's reason restated: the
    #   drive torque a tooth carries is the same order as the friction budget at
    #   the contact carrying it, which is the regime MuJoCo's guidance names.
    # * `contact_matching="latest"` -- CollisionSpec's own note says it is load
    #   bearing for a rod that rolls, and a tooth entering a space is the same
    #   situation: the pair that matters is renewed every step, not persistent.
    # * The buffers are raised well over polish's. A four-gear board is five
    #   toothed rings in mesh at once, each pair contacting on several teeth
    #   simultaneously, plus a gripper and whatever is still on the bench.
    #   Measured with `tasks/gear_mesh/rigs/gear_bench.py contacts` on the four-gear board,
    #   driven rather than resting, because a settled board is not the busy
    #   frame and a turning one is: median 400 simultaneous contacts and a peak
    #   of 844, which nconmax clears 19x over before the arm is even in the
    #   scene. Sized that way on purpose -- an overflow does not warn, it
    #   escalates into an illegal memory access.
    ArrangementSpec(
        control_hz=60.0,
        substeps=16,
        backends=(
            (
                "*",
                backend(
                    "mujoco",
                    use_mujoco_contacts=True,
                    nconmax=16384,
                    njmax=32768,
                    iterations=100,
                    ls_iterations=50,
                    cone="elliptic",
                    impratio=10.0,
                    enable_multiccd=True,
                    jacobian="sparse",
                ),
            ),
        ),
        collision=CollisionSpec(
            contact_ke=6.0e4,
            contact_kd=500.0,
            rigid_contact_max=32768,
            rigid_gap=5.0e-5,
            contact_matching="latest",
        ),
    ),
)

GEAR_RIG = register_camera_rig(
    "gear_mesh_views@v2",
    # Coordinates are world; the bench top's centre is (0.6, 0, 0.75). The
    # tiers' board runs its chain along world y at x = 0.62, from the button at
    # y = -0.11 to the last station at y = +0.08; a drawn variant's plate may
    # stand 3 cm from that on either axis and be turned 10 degrees, and its
    # loose gears wait on the mat at x = 0.395..0.545.
    #
    # v2 (2026-09-08). v1's overhead stood over the board at 1.18 m, 640x480:
    # the board sat in the left three fifths of the frame with the mat cut
    # off, at 0.8 mm per pixel, and the arm filled the whole frame whenever
    # the hand was over the chain. This one is centred over mat AND board,
    # half a metre up, at twice the pixels: 0.5 mm per pixel over a 0.66 x
    # 0.49 m footprint, so every draw stays inside with margin and a tooth
    # class (5 mm of tip radius apart) is ten pixels rather than five. A
    # presentation view, `iso`, is new: the rig's cameras are what a policy
    # sees and what a recording shows, and none of the three working views
    # showed the whole bench with the gears legible.
    (
        CameraSpec(
            name="overhead",
            width=1280,
            height=960,
            fov_y=50.0,
            eye=(0.55, 0.0, 1.30),
            look_at=(0.55, 0.0, 0.77),
            # Straight down is degenerate against a +z up hint. +y puts the
            # direction the chain runs at the top of the frame, so a gear
            # arriving at its peg travels UP the image rather than across it.
            up=(0.0, 1.0, 0.0),
            near=0.01,
            far=4.0,
            channels=("rgb", "depth"),
        ),
        CameraSpec(
            name="iso",
            width=1280,
            height=720,
            fov_y=40.0,
            # Three-quarter view from the side of the bench opposite the arm
            # and off the button end, so the whole mat, the plate, every peg
            # and the button are in frame at rest and the arm enters from the
            # far side rather than across the lens.
            eye=(1.05, -0.50, 1.10),
            look_at=(0.53, 0.0, 0.77),
            near=0.01,
            far=4.0,
            channels=("rgb",),
        ),
        CameraSpec(
            name="chain",
            width=640,
            height=480,
            fov_y=40.0,
            # Down the line of the chain from beyond the last station, low
            # enough that the gears overlap in the frame: whether a tooth is in
            # a space or on top of one is a thing you can see from here and
            # cannot see from overhead.
            eye=(0.62, 0.34, 0.83),
            look_at=(0.62, -0.02, 0.77),
            near=0.01,
            far=4.0,
            channels=("rgb", "depth"),
        ),
        CameraSpec(
            name="front",
            width=640,
            height=480,
            fov_y=45.0,
            eye=(0.18, -0.05, 1.02),
            look_at=(0.62, -0.01, 0.78),
            near=0.01,
            far=4.0,
            channels=("rgb", "depth"),
        ),
    ),
)


# ---------------------------------------------------------------------------
# The `fast` config's pair: what `vabar/configs/fast.toml` stages in place of
# the two above. NEW names, so every document and record that names
# gear_mesh_960hz@v1 / gear_mesh_views@v2 keeps meaning exactly what it did.
# Chosen by measurement, not by argument: tools/fast_tune/gear_mesh.py is the
# sweep.
# ---------------------------------------------------------------------------

GEAR_PHYSICS_FAST = register_arrangement(
    "gear_mesh_fast@v1",
    # The shipped arrangement with ONE number changed, so a diff of the two
    # registrations is the whole story: `substeps` 16 -> 8 (dt 1/960 -> 1/480),
    # 1.9x the idle steps/s (69.8 -> 133.6 on gearmesh-medium-train-001).
    #
    # * The substep count is the only knob that buys speed. Iterations 100 -> 20,
    #   line search 50 -> 10, tolerance 1e-8 -> 1e-6, ccd_iterations 35 -> 10
    #   each moved steps/s by at most 7%, never upward: MuJoCo-Warp's solver
    #   converges in a few iterations and exits early inside the captured graph,
    #   so the caps never bind. nconmax/njmax trimmed to 1.5x the measured peak
    #   (1408/4096) bought nothing, so the buffers keep their headroom.
    #   collide_every is inert under use_mujoco_contacts (no Newton collision
    #   pipeline exists). The CG solver and a dense Jacobian are 2.5-3x SLOWER.
    # * Why not fewer. At 6 substeps the three-gear gate still drives (0.974)
    #   but the four-gear board slips to 0.80 of its ratio (5th percentile
    #   0.56); impratio 1 rescues the gate (0.982-0.986 on five boards) but the
    #   graph then solved 3 of 5 probe instances, failing twice to seat a board
    #   that this arrangement and the shipped one both completed. This one
    #   solved 5/5, the shipped one 4/5 (same protocol, one episode each).
    #   At 4 substeps nothing drives at any impratio tried (0.11-0.69).
    # * At 8 substeps every board measured runs within 0.002 of the shipped
    #   arrangement's ratio (0.963-0.970 against 0.961-0.968), the gear-b-missing
    #   control still stops gear_c dead, and the peak contact count is the same.
    #   MuJoCo clamps the 4 ms solref to 2*dt = 4.17 ms here -- 4% softer.
    # * multiccd and the elliptic cone are NOT optional: without multiccd the
    #   train runs at 0.205 of its ratio, with a pyramidal cone at 0.924.
    GEAR_PHYSICS.with_(substeps=8),
)

GEAR_RIG_FAST = register_camera_rig(
    "gear_mesh_views_fast@v1",
    # gear_mesh_views@v2 at half the pixels per side (CameraSpec.scaled keeps the
    # field of view and scales the intrinsics): the overhead is 640x480, about
    # 1 mm per pixel at the bench, and a tooth class (5 mm of tip radius) is five
    # pixels rather than ten. gap_perception measures its own metres per pixel
    # and sizes every threshold in metres, and it read every board at this
    # resolution in the graph probes. Same poses, same names, same channels.
    tuple(camera.scaled(camera.width // 2, camera.height // 2) for camera in GEAR_RIG),
)
