# SPDX-License-Identifier: Apache-2.0
"""The connector fit: its rates, its solvers, its two camera rigs and its draw.

A rigid part going into a hole a fraction of a millimetre wider than it is, at
60 Hz x 16 substeps, transcribed from a reference rather than designed here.
MuJoCo end to end -- which is what lets the MJCF's own ``<position>`` actuators
and ``gravcomp`` keep driving the arm through a contact-rich mate.

**Two arrangements, and the difference between them is two integers and a
capture.** ``usbc_960hz@v1`` is the bolted-down fit ``port`` stages;
``usbc_bulkhead_960hz@v1`` is the same arrangement with the constraint ceilings
doubled for a scene that carries cables, and with CUDA graph capture off.
Separate names rather than a parameter, because two documents staging one name
must get one object, and a task that could pass ``nconmax=`` is a task that can
silently change what its own physics means.

This file held a second suite's coupled MuJoCo+VBD pair until the RJ45 task was
retired. What is left is deliberately not generalised into its shape: a
one-suite arrangements module that reads like one is easier to be right about
than a two-suite one with a hole in it.

**Two camera rigs, and the difference between them is what a policy may look
through.** ``usbc_multiview@v1`` is the reference's three placed views, every
eye and look-at a constant of the cell -- nothing per-variant to aim, which is
what makes it registrable at all, and the contrast the weaves' aimed rig draws.
``usbc_perception@v1`` is that rig plus the two views a graph that has to *find*
the parts needs: a true top-down over the bench and an eye-in-hand on the work
arm. Both documents stage the second one, because a rig is staged before anybody
knows which policy will run and a graph that cannot see is a worse failure than a
render that was not needed.

**And one draw**, :data:`USBC_BENCH`, which is what makes thirty episodes thirty
layouts rather than one layout run thirty times.
"""

from __future__ import annotations

import vabar  # noqa: F401 -- the scripted-routine module RoboSimStudio leaves out; see vabar/__init__.py

from dataclasses import replace

import numpy as np

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.core.math import transform_matrix, xform_from_matrix
from robosimstudio.specs import ArrangementSpec, CollisionSpec, backend
from robosimstudio_bench.taskbuild import register_arrangement, register_camera_rig

__all__ = [
    "USBC_BENCH",
    "USBC_BULKHEAD_PHYSICS",
    "USBC_EASY",
    "USBC_MEDIUM",
    "USBC_PERCEPTION_RIG",
    "USBC_PHYSICS",
    "USBC_RIG",
]


def _cam(name: str, eye, pitch_deg: float, yaw_deg: float, distance: float) -> CameraSpec:
    """One reference view, from its ``(pos, pitch, yaw)`` and a look distance.

    Both suites state their views the way the references do -- an eye and two
    angles -- and both need a look-at, so the conversion is written once here
    rather than twice as a closure inside two loaders.
    """
    pitch, yaw = np.radians(pitch_deg), np.radians(yaw_deg)
    direction = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    look = np.asarray(eye) + distance * direction
    return CameraSpec(
        name=name,
        width=640,
        height=480,
        fov_y=45.0,
        eye=tuple(float(v) for v in eye),
        look_at=tuple(float(v) for v in look),
        near=0.01,
        far=4.0,
        channels=("rgb", "depth"),
    )


USBC_PHYSICS = register_arrangement(
    "usbc_960hz@v1",
    # The reference's solver settings, restated. 60 Hz x 16 substeps (dt =
    # 1/960). `solref_time` 4 ms against that dt satisfies MuJoCo's `> 2*dt`
    # stability floor with margin -- the rig asserts the pairing. `impratio` is
    # MuJoCo's own default 1, not this package's usual 10: the reference never
    # set it, and the fit was tuned without it.
    ArrangementSpec(
        control_hz=60.0,
        substeps=16,
        backends=(
            (
                "*",
                backend(
                    "mujoco",
                    use_mujoco_contacts=False,
                    nconmax=1024,
                    njmax=2048,
                    iterations=100,
                    ls_iterations=50,
                    cone="elliptic",
                    impratio=1.0,
                ),
            ),
        ),
        collision=CollisionSpec(rigid_contact_max=8192, rigid_gap=2.0e-4),
    ),
)
"""The bolted-down fit, registered again because ``port`` stages it.

It spent one release as an unregistered constant -- "a name nothing stages is a
name that answers a typo instead of failing" -- which was right for as long as
the bulkhead was the only variant. The stationary receptacle is back, it is a
scene with no cables in it, and giving it the doubled ceilings a cable scene
needs would be sizing for bodies that are not there.
"""

USBC_BULKHEAD_PHYSICS = register_arrangement(
    "usbc_bulkhead_960hz@v1",
    # The same arrangement as `usbc_960hz@v1` with the two constraint ceilings
    # doubled, and it is a separate registration rather than a parameter for the
    # reason every arrangement is: two documents staging one name must get one
    # object, and a task that could pass `nconmax=` is a task that can silently
    # change what its own physics means.
    #
    # Doubled because of the CABLES. The bimanual scene roots a chain on each
    # connector -- 32 capsules that spend most of an episode lying on the bench
    # and the table -- and ceilings sized for the busiest frame of a cable-less
    # scene overflow against them. An overflow here is an illegal memory access,
    # not a degradation, which is why this is sized rather than tuned.
    #
    # Full finger contacts need the usual friction/normal impedance ratio of
    # 10 to hold the barrel's attitude. At 1, the stationary support grasp
    # creeps several degrees before the work arm reaches the aperture.
    #
    # `use_cuda_graph=False`, and hydroelastic is why.
    #
    # Newton's hydroelastic narrow phase reads its hash-table overflow counter
    # back to the host every launch (`sdf_hydroelastic.launch` calls `.numpy()`
    # on `ht_insert_failures`). A device-to-host copy inside a captured block is
    # illegal, and what it produces is not a clean refusal: CUDA error 906,
    # "operation would make the legacy stream depend on a capturing blocking
    # stream", then a failed re-capture, then error 101 on an unrelated scan.
    # The studio survives it by falling back to eager; a batch run does not.
    #
    # The cost is real and is the smaller one: this arrangement gives up graph
    # capture for the whole scene -- `Schedule.warmup_and_capture` and `_try_recapture`
    # both return before `capturing()` when the flag is off, which is the eager
    # path and a supported way to run. Restore it the day the overflow check
    # moves device-side.
    USBC_PHYSICS.with_backend("*", USBC_PHYSICS.backend_for("*").with_(nconmax=2048, njmax=4096, impratio=10.0)).with_(
        use_cuda_graph=False
    ),
)

USBC_RIG = register_camera_rig(
    "usbc_multiview@v1",
    # The reference multiview rig's aim. Overview frames the cell and the arm,
    # mating sits close on the tab entering the bore, side reads insertion depth
    # in profile.
    (
        _cam("overview", (0.74, -0.38, 1.04), -25.0, 137.0, 0.55),
        _cam("mating", (0.585, -0.085, 0.872), -16.0, 145.0, 0.20),
        _cam("side", (0.30, -0.05, 0.850), -6.0, 4.0, 0.16),
    ),
)


# --------------------------------------------------------- the perception rig
#
# Everything below exists because a policy that has to FIND the connectors
# cannot use the reference's three views. They were aimed by somebody who
# already knew where the parts were -- `mating` frames 200 mm of the mate line
# and nothing else -- which is exactly the knowledge a perception body is not
# allowed to have.

TOPDOWN_HEIGHT = 0.40
"""How far above the bench top the top-down eye stands [m].

Close enough to resolve a 22 mm plug and far enough to hold the whole working
span: the plug spawns at y = -0.070 and the receptacle's boot tip reaches
y = +0.076, so the view has to carry 146 mm of bench with margin for a
receptacle that has been picked up and put back somewhere else.
"""

TOPDOWN_FOV_Y = 35.0
"""Vertical field of view [deg]. Narrower than the reference views' 45.

At :data:`TOPDOWN_HEIGHT` this frames 252 mm across the short image axis and
336 mm across the long one, which is 0.35 mm/px on a 960x720 frame. The number
that matters: the plug's 10 mm width lands on ~29 pixels and its 22 mm one on
~63, so a centroid over the mask is a sub-pixel estimate of where it is rather
than a guess between two blobs. The reference `overview` resolves the same part
at 1.7 mm/px, where the whole plug is thirteen pixels across.
"""

TOPDOWN_XY = (0.44, 0.0)
"""Where the top-down looks, on the bench [m].

The mate line's own x, and a y midway between the plug's spawn and the
receptacle's rear -- not the centre of the cell. A view centred on the cell puts
the two things being measured in one corner of it.
"""


D405_WRIST_XFORM = (-0.0017, 0.079729, 0.066021, 0.0, -0.42304971, 0.90610647, 0.0)
"""Standard D405 optical pose in the arena YAM's link-6 frame.

The calibration the RoboSimStudio this suite was written against kept beside the
``yam_linear_arena`` hand (``robosimstudio/robots/grippers/yam_linear_arena.py``
at KeplerC/RoboSimStudio@eeb2aa88). Upstream's copy of that hand is otherwise
identical but no longer carries the constant, so the suite states it here.
"""


def _wrist_mount() -> tuple[float, ...]:
    """Tool-hanging's standard D405 view, expressed in this hand's body frame.

    The arena hand points along body +z and the i2rt hand along -z. Compose
    through their calibrated TCP frames to preserve the optical pose relative
    to the jaws rather than aim the camera behind the i2rt wrist.
    """
    from robosimstudio.robots.registry import get_gripper  # noqa: PLC0415

    hand = get_gripper("linear_4310")
    arena = get_gripper("yam_linear_arena")
    body_tcp = transform_matrix((*hand.tcp_offset, *hand.tcp_offset_rot))
    arena_tcp = transform_matrix((*arena.tcp_offset, *arena.tcp_offset_rot))
    return tuple(xform_from_matrix(body_tcp @ np.linalg.inv(arena_tcp) @ transform_matrix(D405_WRIST_XFORM)))


TOPDOWN = CameraSpec(
    name="topdown",
    # Finer than every other view in this rig, and the only one that is: it is
    # the view a perception body grounds the scene from, and 640x480 at this
    # standoff puts the plug's short axis on nineteen pixels.
    width=960,
    height=720,
    fov_y=TOPDOWN_FOV_Y,
    eye=(TOPDOWN_XY[0], TOPDOWN_XY[1], 0.80 + TOPDOWN_HEIGHT),
    look_at=(TOPDOWN_XY[0], TOPDOWN_XY[1], 0.80),
    # +x is "toward the robot" on this bench, so image-up along +x puts the
    # MATE LINE -- which runs along y -- across the long image axis. With the
    # default z-up hint the look-at basis is degenerate here anyway: the view
    # direction *is* -z.
    up=(1.0, 0.0, 0.0),
    near=0.01,
    far=2.0,
    channels=("rgb", "depth"),
)
"""Straight down over the bench, and straight down is the point.

The reference's ``overview`` sees the same parts at a three-quarter angle, which
is worse for this job in a way worth stating: a receptacle is measured by where
its *mouth face* is, and a face viewed obliquely projects to a quadrilateral
whose near edge is its top rim rather than its plane. Looking down the bench
normal keeps the barrel's silhouette a rectangle whose long axis is the bore
axis and whose leading edge is the mouth -- both of which a mask plus depth can
be reduced to without knowing which part it is looking at.

What it cannot give is the bore's *height*. A top-down depth reads the barrel's
upper surface, and the axis is one radius below it -- the same surface-versus-
axis bias every top-down cylinder measurement carries. The perception body
subtracts the radius the catalogue states rather than one it measured off the
silhouette, because a cylinder's silhouette is its tangent lines seen in
perspective and reads 19.0 mm for a 21.0 mm barrel.
"""

WRIST = CameraSpec(
    name="wrist",
    width=320,
    height=240,
    fov_y=58.0,
    # The ROLE, not a link. `work` resolves to whichever arm is doing the work
    # and rides that arm's tool body, so the rig is not pinned to one robot's
    # link names -- or to one importer's fixed-joint collapse.
    parent_frame="work",
    xform=_wrist_mount(),
    near=0.005,
    far=1.5,
    channels=("rgb", "depth"),
)
"""Eye in hand, riding the work arm.

The view no placed camera in this cell can stand in for, and the bulkhead
variant is why. That receptacle spends the middle of an episode **120 mm off the
table**, held in the support hand, and every fixed eye here was aimed at a bench
that no longer has it on. A hand camera is aimed by the arm, so the graph can
take a look at the receptacle where it currently is rather than where it
spawned -- which is the whole difference between a plan and a re-plan.
"""

WRIST_SUPPORT = replace(WRIST, name="wrist_support", parent_frame="support")
"""The same eye-in-hand, on the other arm."""

USBC_PERCEPTION_RIG = register_camera_rig(
    "usbc_perception@v1",
    (
        TOPDOWN,
        WRIST,
        WRIST_SUPPORT,
    ),
)
"""Three views: top-down and one eye-in-hand per arm.

Both documents stage this rather than ``usbc_multiview@v1``, and the cost is
honest -- five cameras render where three did, whenever a renderer is attached
at all. It is staged anyway because a rig is chosen before anybody knows which
policy will run, and the failure modes are not symmetric: a run that did not need
the top-down paid some milliseconds, where a perception run that found no
top-down would have had to re-stage the task to see anything.

``--renderer ""`` renders nothing and publishes no camera block, which is right
for a privileged graph and fatal for a vision one. That is not this rig's
business to prevent; the perception body says so by name when it looks.
"""


# ----------------------------------------------------------- the layout draw
#
# Everything above this line describes a cell. This describes what is *not* the
# same twice in it, which is the difference between a demo and a measurement.

PLUG_JITTER_XY = 0.018
"""How far the plug may start from its nominal spot on the bench [m].

**A hand put it there, and not carefully.** 12 mm is what somebody laying a
connector down *deliberately* holds to; 18 is what it looks like when they are
not thinking about it, and both are honest descriptions of the same bench. This
is the number that decides whether a policy is finding the part or remembering
where it was: the plug's own body is 10 mm across, so a draw at the edge of this
band starts nearly two part-widths from where the last episode's grasp went.

Widened from 12 mm once the mate started working. At 12/8 the camera path seats
17 of 30, and a benchmark a policy clears three times in five is measuring the
top of its range rather than its behaviour.

Kept inside the bench's own footprint on purpose. The bench box is 120 mm in x
and 142 in y and the plug is 47 long, so this never walks it off the edge -- a
spawn that falls is a scene the task never meant to pose, and it would be scored
as a perception failure it had nothing to do with.
"""

PLUG_JITTER_YAW = 14.0
"""How far the plug may start off the mate line's heading [deg].

Rotation about the vertical only. A connector laid down by hand is roughly
square to the bench and never upside down -- it is keyed, and a person feels
that -- so this is the honest axis to vary and roll is not.

Fourteen degrees across the plug's 47 mm length swings the tab tip 11.4 mm,
which is more than a bore's width. It is enough that the grasp yaw has to be
*measured* rather than assumed -- and, since ``ToolsUsbc._mate`` learned to turn
the plug onto the measured bore axis, enough that the TURN has to be right too.

**This is the axis the remaining failures live on**, which is why it is the one
that was widened. Recorded with ground truth at 8 degrees: a seated mate arrives
0.77-0.96 degrees off the bore and latches; a lost one arrives at 4.28 and does
not, even when it reached 11.80 mm of depth -- past the detent's own click. The
draw's yaw is what sets how much residual there is to arrive with.
"""

RECEPTACLE_JITTER_XY = 0.004
RECEPTACLE_JITTER_YAW = 3.0
"""How far a *free* receptacle may start from nominal [m, deg].

Much smaller than the plug's, and for a physical reason rather than a cautious
one: the bulkhead rests in a two-step cradle machined to its own barrel, so the
cradle is what sets its pose and a hand can only put it in one way. What is left
is the slop in that fit. A bolted receptacle does not move at all and this never
touches it.
"""


PORT_JITTER_ACROSS = 0.005
"""How far a *bolted* receptacle may stand from its drawing, ACROSS the mate line [m].

World x, perpendicular to the bore. **A bolted fixture is not a machined one.**
The port is through-bolted to the bench, and a receptacle that has been unbolted
and put back lands within the slop of its own clearance slots -- millimetres,
not the tenths a cradle holds the bulkhead to (:data:`RECEPTACLE_JITTER_XY`),
and not the centimetres a hand leaves a loose plug in (:data:`PLUG_JITTER_XY`).

What it buys the measurement is the thing a fixed port quietly withheld. Welded
at one pose forever, a policy that had merely *memorised* where the bore was
scored the same as one that found it, and Easy sat at 10/10 -- a rate that
measures the top of the range rather than the behaviour. The bore must now be
located from images every episode, because it is not where the last one left it.

This axis is free to take the full band: the port's own block already overlaps
the bench in x and z (measured AABB x 0.420-0.460 inside the bench's 0.380-0.500),
so moving along x changes no clearance that was not already zero.
"""

PORT_JITTER_ALONG = 0.0015
"""How far the same receptacle may stand from its drawing, ALONG the bore [m].

World y, and **a third of the across-axis band, because the bench ends here.**
The bench is authored "from the plug's rear face to 2 mm short of the mouth"
(:data:`~.assets.scenery.BENCH`), and the measurement agrees: the port block's
world AABB starts at y = 0.0103 against the bench's y-max of 0.0083, so the
whole clearance along this axis is 2.0 mm. The block reaches down to z = 0.748,
which is 52 mm BELOW the bench top -- the two boxes already share x and z, and
the only thing keeping them apart is those 2 mm.

So a draw that walked the port 5 mm back would bury up to 3 mm of a welded
fixture inside another welded fixture on roughly a third of all episodes. Two
statics interpenetrating produce no dynamics, which is exactly what makes it
dangerous: nothing refuses, the depth camera simply renders one box growing out
of another and the table reports a rate measured on a scene the task never
posed. 1.5 mm keeps every draw clear of that with 0.5 mm to spare.
"""

PORT_JITTER_YAW = 5.0
"""How far the bolted receptacle may be turned about the vertical [deg].

**Three degrees is offered and the geometry vetoes the corners that do not fit.**
What faces the bench is not the block's big dimension: its 150 mm extent
(:attr:`half_x` 0.075) maps to world **+z** and is the flange's HEIGHT, while
the horizontal extent across the mate line is 40 mm (:attr:`half_y` 0.020).
A yaw of theta therefore swings a mouth corner ``20 mm * sin(theta)`` along the
bore -- 1.74 mm at five degrees, against the 2.0 mm the bench leaves in front of
the mouth. Five degrees alone is legal with 0.26 mm to spare; five degrees on top
of a draw already 1.5 mm back is not, and about half of the widest turns are
refused for that reason rather than clamped.

So the two are drawn together and tested together (:func:`_port_pose_legal`),
rather than each being clamped to what is safe when the other is at its worst.
Clamping would have cost the large turns that sit far enough forward to be
perfectly legal; rejection keeps them and drops only the genuine collisions.
Measured over 400 draws: every emitted pose legal, no draw exhausting
:data:`PORT_TRIES`, and the accepted turning still spanning the full band.

The turning matters because it is the bore's AXIS, not merely its position. The
suite's own measurement is that a seated mate arrives 0.77-0.96 degrees off the
bore and a lost one arrives at 4.28 (:data:`PLUG_JITTER_YAW`), so a receptacle
that is itself three degrees off is asking the policy to measure a direction it
cannot assume rather than only a point.
"""


def _turned(spec, rng, span_xy: float, span_yaw: float) -> tuple[float, ...]:
    """One part, nudged on the bench and turned about the world vertical.

    The yaw is *composed onto* the part's declared rotation rather than
    replacing it, which is what "the same part, turned a little on the bench"
    means -- see :func:`_usbc_bench` for why that distinction is load-bearing
    here.
    """
    half = np.radians(float(rng.uniform(-span_yaw, span_yaw))) * 0.5
    x1, y1, z1, w1 = 0.0, 0.0, float(np.sin(half)), float(np.cos(half))
    x2, y2, z2, w2 = (float(v) for v in spec.quat)
    # Hamilton product yaw * q, xyzw.
    composed = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )
    pos = np.asarray(spec.pose, dtype=np.float64) + np.array(
        [
            rng.uniform(-span_xy, span_xy),
            rng.uniform(-span_xy, span_xy),
            0.0,
        ]
    )
    return tuple(float(v) for v in np.concatenate([pos, composed]))


def _placed_static(spec, dx: float, dy: float, yaw_deg: float) -> tuple[float, ...]:
    """One fixture, moved by a GIVEN offset and turned by a GIVEN angle.

    Deterministic where :func:`_turned` is not, because the port's pose is
    rejection-sampled: the draw has to be proposed, tested against the bench and
    only then committed, so the randomness lives in the caller and this applies
    what the caller settled on.

    Returned in THE SPEC'S OWN FRAME, which is what
    :func:`~robosimstudio.domains.static.entity.episode_xform` reads and what
    :meth:`~robosimstudio.domains.static.entity.EntityStatic.handle` lifts again
    on the way to the handle a feature resolves against. A port displaced here
    therefore carries its own ``bore`` with it -- rotation included, which is the
    part the region resolver's translate-only handles would not have given.

    The yaw is *composed onto* the part's declared rotation rather than replacing
    it. This port is declared at :data:`~.assets.parts.PORT_QUAT`, a non-trivial
    rotation laying the bore onto the mate line; an override would stand the
    receptacle on a different face and call it a layout.
    """
    pos = np.asarray(spec.xform[:3], dtype=np.float64) + np.array([dx, dy, 0.0])
    half = np.radians(float(yaw_deg)) * 0.5
    x1, y1, z1, w1 = 0.0, 0.0, float(np.sin(half)), float(np.cos(half))
    x2, y2, z2, w2 = (float(v) for v in spec.xform[3:7])
    composed = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )
    return tuple(float(v) for v in np.concatenate([pos, composed]))


PORT_TRIES = 64
"""How many times a port pose is redrawn before the draw gives up and stands still.

Generous, because the rejection is cheap (eight corners and a box test) and the
acceptance rate is high -- measured 97% over 400 draws, so 64 tries fails about
once in every 10**48 episodes. The fallback is the declared pose rather than an
exception for the one reason a fixture differs from a board: a weave whose pegs
silently fell back to the declared layout would be an unvaried episode counted
as varied, whereas a port that did not move is the scene this suite shipped for
its whole life and is honestly scored.
"""

PORT_BENCH_MARGIN = 2.0e-4
"""Clearance kept between the drawn port and the bench [m].

The collision gap the arrangement already uses (:data:`~.assets.rig.CONTACT_GAP`),
restated here because a pose that merely touches is a pose whose contact status
depends on floating-point luck.
"""


def _port_corners(dx: float, dy: float, yaw: float) -> np.ndarray:
    """The port block's eight world corners under one drawn offset and turn.

    Built from the block's own half-extents and the declared rotation rather
    than from anything the scene has already assembled, because this runs at
    DRAW time -- before there is a model to measure.
    """
    from robosimstudio.core.math import quat_to_mat  # noqa: PLC0415

    from .assets.stage import PORT_QUAT, reference_assembly  # noqa: PLC0415

    block = reference_assembly().port
    origin = np.asarray(reference_assembly().port_origin, dtype=np.float64) + np.array([dx, dy, 0.0])
    half = 0.5 * float(np.radians(yaw))
    turn = quat_to_mat(np.array([0.0, 0.0, np.sin(half), np.cos(half)]))
    rest = quat_to_mat(np.asarray(PORT_QUAT, dtype=np.float64))
    basis = turn @ rest
    return np.array(
        [
            origin + basis @ np.array([sx, sy, sz])
            for sx in (-block.half_x, block.half_x)
            for sy in (-block.half_y, block.half_y)
            for sz in (block.block_z0, block.mouth_z)
        ]
    )


def _port_pose_legal(dx: float, dy: float, yaw: float) -> bool:
    """Whether a drawn port pose keeps its block clear of the bench.

    The two boxes already share x and z -- the port's flange reaches 52 mm BELOW
    the bench top and sits inside its x-span -- so the whole separation is the
    2.0 mm the bench leaves in front of the mouth, and this is the test that
    2 mm is still there. Two welded statics that interpenetrate produce no
    dynamics, which is exactly why this has to be checked rather than left to
    fail loudly: nothing would refuse, the depth camera would simply render one
    box growing out of another, and the table would report a rate measured on a
    scene the task never posed.
    """
    from .assets.scenery import BENCH_CENTER, BENCH_HALF  # noqa: PLC0415

    centre = np.asarray(BENCH_CENTER, dtype=np.float64)
    half = np.asarray(BENCH_HALF, dtype=np.float64)
    corners = _port_corners(dx, dy, yaw)
    return bool(corners.min(axis=0)[1] >= centre[1] + half[1] + PORT_BENCH_MARGIN)


def _jittered_port(task, rng) -> dict[str, tuple[float, ...]]:
    """The bolted receptacle's ``static_xforms`` entry, or nothing.

    Returns an empty dict on a variant that has no welded ``port`` -- ``bulkhead``
    carries its receptacle as a free object and draws it through
    :data:`RECEPTACLE_JITTER_XY` instead, so this is a lookup rather than a
    branch on the variant's name.
    """
    for spec in task.statics:
        if spec.name != "port":
            continue
        for _ in range(PORT_TRIES):
            dx = float(rng.uniform(-PORT_JITTER_ACROSS, PORT_JITTER_ACROSS))
            dy = float(rng.uniform(-PORT_JITTER_ALONG, PORT_JITTER_ALONG))
            yaw = float(rng.uniform(-PORT_JITTER_YAW, PORT_JITTER_YAW))
            if _port_pose_legal(dx, dy, yaw):
                return {"port": _placed_static(spec, dx, dy, yaw)}
        return {"port": tuple(float(v) for v in spec.xform)}
    return {}


def _usbc_bench(task, episode_id: int, rng) -> dict[str, dict[str, tuple[float, ...]]]:
    """Draw one bench layout: where the parts were left before the robot looked.

    Emits ``object_xforms`` rather than ``object_poses``, and that is not a
    style choice. The pose form writes the quaternion as a **pure yaw about z**,
    discarding whatever the object declared -- and both parts here are declared
    at a non-trivial rotation (the plug's local +z is the *insertion* axis, laid
    onto world +y). A yaw override would stand the connector on a different face
    and call it a layout.

    A bolted receptacle is a static and is never in ``task.objects``, so it is
    not something this has to exclude by name: ``port`` draws the plug and
    nothing else, and ``bulkhead`` draws both.
    """
    del episode_id
    xforms = {}
    for spec in task.objects:
        if spec.static:
            continue
        if spec.name == "plug":
            xforms[spec.name] = _turned(spec, rng, PLUG_JITTER_XY, PLUG_JITTER_YAW)
        else:
            xforms[spec.name] = _turned(spec, rng, RECEPTACLE_JITTER_XY, RECEPTACLE_JITTER_YAW)
    return {"object_xforms": xforms, "static_xforms": _jittered_port(task, rng)}


_usbc_bench.model_key_fields = ("object_xforms", "static_xforms")
"""The draw changes the MODEL on ``bulkhead``, so every draw gets its own.

A re-pose would be enough for the two rigid bodies themselves -- ``EntityRigid``
reads ``object_xforms`` on reset as well as on build. It is not enough for the
bulkhead's rear cable: :meth:`EntityUsbcCables.build` computes that chain's
joint anchor as ``inverse(body_q[bulkhead]) @ boot_tip_world``, off the
receptacle's **drawn** pose, and nothing recomputes it afterwards. Re-posing
into episode 0's model would hang episode 7's cable off episode 0's boot tip.

Declared on the sampler rather than per variant because a sampler is the only
thing that knows which of its keys are model-changing, and ``port`` -- which
carries no cable at all -- pays one rebuild per episode for a fact about the
suite it shares a draw with. That is the trade the plumbing offers; the
alternative is two samplers that differ in an attribute.
"""

USBC_BENCH = _usbc_bench
"""One bench layout per episode: the parts where a hand would have left them.

Named here rather than written into a document, because the *cell* is what has
this property -- both variants stand on the same bench and are laid out by the
same imaginary person, and two copies of that would drift. The documents reach
it as ``(:sampler staging:USBC_BENCH)``.
"""


def _usbc_evaluation(task, episode_id: int, rng) -> dict[str, dict[str, tuple[float, ...]]]:
    """Medium evaluation draw: ±18 mm and ±50 degrees on the plug.

    A free bulkhead keeps the cradle-limited ±4 mm/±3 degree perturbation. The
    fixed port is scenery and therefore remains bolted at its declared pose.
    """
    del episode_id
    xforms = {}
    for spec in task.objects:
        if spec.static:
            continue
        if spec.name == "plug":
            xforms[spec.name] = _turned(spec, rng, PLUG_JITTER_XY, 50.0)
        elif spec.name == "bulkhead":
            xforms[spec.name] = _turned(spec, rng, RECEPTACLE_JITTER_XY, RECEPTACLE_JITTER_YAW)
    return {"object_xforms": xforms, "static_xforms": _jittered_port(task, rng)}


_usbc_evaluation.model_key_fields = ("object_xforms", "static_xforms")
USBC_MEDIUM = _usbc_evaluation


def _usbc_easy(task, episode_id: int, rng) -> dict[str, dict[str, tuple[float, ...]]]:
    """Easy fixed-port draw: ±12 mm translation and ±8 degrees of yaw.

    This is the measured acquisition band for the tight connector: the camera
    policy seated 17/30 at these bounds before the harder layout was widened.
    Keeping the port fixed separates perception/grasp variation from the
    medium tier's additional bimanual bulkhead motion.
    """
    del episode_id
    xforms = {}
    for spec in task.objects:
        if not spec.static and spec.name == "plug":
            xforms[spec.name] = _turned(spec, rng, 0.012, 8.0)
    return {"object_xforms": xforms, "static_xforms": _jittered_port(task, rng)}


_usbc_easy.model_key_fields = ("object_xforms", "static_xforms")
USBC_EASY = _usbc_easy


# The YAM station: one head camera, a left-shoulder camera and a D405 on each wrist, every one 1280x720.
YAM_STATION_RIG = register_camera_rig(
    "yam_station_usbc@v1",
    (
        CameraSpec(
            name="topdown",
            width=1280,
            height=720,
            fov_y=55.0,
            eye=(0.2525, 0.0, 1.69),
            look_at=(0.7725, 0.0, 0.89),
            up=(1.0, 0.0, 0.0),
            near=0.01,
            far=4.0,
            channels=("rgb", "depth"),
        ),
        CameraSpec(
            name="side",
            width=1280,
            height=720,
            fov_y=50.0,
            eye=(0.2125, 0.82, 1.89),
            look_at=(0.7825, 0.0, 0.97),
            near=0.01,
            far=4.0,
            channels=("rgb", "depth"),
        ),
        CameraSpec(
            name="wrist",
            width=1280,
            height=720,
            fov_y=58.0,
            parent_frame="work",
            xform=(-0.123729, 0.0017, -0.070721, 0.640714032, 0.640714032, 0.29914132, 0.29914132),
            near=0.005,
            far=1.5,
            channels=("rgb", "depth"),
        ),
    ),
)
