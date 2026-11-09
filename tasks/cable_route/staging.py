# SPDX-License-Identifier: Apache-2.0
"""The cable cell: ``cable_coupled_960hz@v1``, the rigs, and the two board samplers.

Every cable document stages the arrangement by name, and every one stages the
*same* arrangement -- fourteen routes over one bench, one rod, one solver. That
is the case the registry exists for: it used to be built once per variant by the
loader that named it, so two documents agreeing about their physics agreed by
running the same constructor twice.

The middle of this file is **the rigs**: the wrap's fixed pair and one four-view
rig per weave, each aimed at its own layout's centroid. The layouts themselves --
the world coordinates BDDL has no place for, and the placed records a document
names -- are ``assets/layout.py``'s; a rig tracking computed coordinates is
registered here under the name its document uses, reading the computation it
tracks from there.

The end of it is **the samplers**, and they are this suite's rather than the
library's for the reason the arrangement is: every rule they draw under is a fact
about this bench. ``CABLE_BOW`` varies the rod's own shape; ``WEAVE_BOARD`` draws
a whole board to a difficulty LEVEL, rejecting anything the measure does not put
in the band; ``WEAVE_RIGID`` puts the declared board down somewhere else;
``PORT_BOARD`` draws a terminated tier's pegs inside the corridor its own anchor
and its own drawn port leave, measured against that tier's declared board rather
than against a band. All four emit model-changing keys, so an episode of any of them is a
rebuild rather than a re-pose -- see ``model_key_fields`` beside each.

**The proxy coupling works on this build, and the note that said otherwise was
wrong (re-measured 2026-08-10, newton 1.5.0.dev0 pinned, RTX PRO 6000 /
sm_120).** A closed hand swept through the rod at rod height pushes it 53 mm
through ~200 proxy contacts, and a drag wrench written into ``body_f`` on a rod
segment lifts it against the whole chain's weight -- both measured on
``cable_route/weave3``. The earlier probe drove the *open* jaws down over the
rod: an 11 mm rod between pads ~40 mm apart touches neither pad, its ~200
contacts were the pad tips against the *table*, and "0.1 mm of rod motion" was
the rod not being touched. What actually failed end-to-end was above this
file: the wrap planned its grasp before the settle (and the anchor's closing
pads rotate the rod during it), and the unclamped hold latch could fight a
contact and eject the rod -- both fixed where they live, in
this task's own directory: its documents, its ``assets/rod.py``.

**The coupling works and the weave still never touched the rod (measured
2026-08-11).** Run with its latch disabled, the shipped ``RoutineWeave``
produced *zero* gripper-vs-rod proxy contacts across a whole ``weave3`` episode
and moved the cable 0.1 mm -- the same reading as the retracted note, arrived
at down a different road, which is the hazard of a number that means "nothing
happened". Two defects were half of it, and both were of the same kind:
something declared and then not delivered.

* An explicit pair list re-admitted 392 blade-vs-table pairs that
  ``EntityCable._decompose_working_tips`` had filtered, so the arm bottomed out
  on its own fingertips ~10 mm above grasp depth. The composer now cuts every
  pair list from the model's own table, which honours the filters -- see
  :func:`robosimstudio.scene.physics.routing.filtered_pairs`.
* The VBD entry took the default 64-row per-body contact buffer, and a pinch on
  a 42-capsule rod by a decomposed fingertip overflows it at 232 -- silently,
  as dropped contacts. ``rigid_body_contact_buffer_size=256`` below.

The other half was the routine's, and the whole of it is written down where
each piece lives. Placed by hand at the depth that puts the pads around the
rod, a plain friction pinch on this cable makes 12 contacts, drags it 46 mm and
lifts it 109 mm; driven by the fixed routine it holds 266 contacts at peak and
weaves without a latch at all.

**The solver is part of what the name means, which is why it is here -- as
data.** Two islands over one model, partitioned by what each entity declared:
MuJoCo owns the arms, rigid VBD owns the cable, and the hands are mirrored into
the VBD side as mass-scaled proxies -- that is how the fingers push the cable
without MuJoCo ever seeing a cable joint (it has no such joint type) and without
VBD solving an arm. The composer derives the partition from the cable's
declaration (``domain="rod"``, contacts with the surface, the fixtures and the
hands) and the arm's, routes arm-vs-scenery to the MuJoCo island through the
main pipeline and cable-vs-everything through the channel's own, and records
what it built in the manifest a repro bundle carries. Nothing here is a
factory any more; every number is a field a reader can diff.

Arm-vs-*scenery* is routed, not assumed: the MuJoCo island lists no shapes of
its own, so its view keeps the world-fixed shapes (table, pegs, ground)
alongside the robot's, and the main pipeline carries exactly the robot-vs-static
pairs. Without them the arm sank 38 mm below the tabletop, because a pair
neither island could see both sides of was a dropped contact rather than an
error. ``contact_matching="latest"`` is the reference's contact-history
setting, load-bearing for a rod that rolls.

Every number the reference states is its default -- proxy mass_scale 20,
staggered coupling at every substep, VBD 40 iterations with
``rigid_avbd_beta 1e2`` and ``rigid_contact_k_start 1e3``, MuJoCo
newton/implicitfast/elliptic at 100/20 -- with one measured exception. **The
proxy coupling runs 2 relaxation passes, not the reference's 4.** The pass
count multiplies both solvers and their restart scaffolding, and it is where
the step's time went: 16 substeps x 4 passes was ~25,600 kernel launches of
microseconds each per control step, a graph the GPU replays at ~404 ms while
drawing 104 W of its 600 W budget -- scheduling, not math. At 2 passes the same
step replays in 125 ms (3.2x), and the physics-sensitive member of this suite
got *better*, not worse: the wrap retained 1.00 turns after release against
0.73 for the 4-pass baseline in the same sweep (gate 0.5, healthy band
0.55-0.96). Measured 2026-08-11 on cable_route/weave3 and /wrap, one knob at a
time; the other multipliers stay because the same sweep showed why -- halving
VBD iterations buys ~2% once the passes are 2, and halving substeps to 480 Hz
kills the wrap outright (0.00 turns retained). The MuJoCo constraint ceilings
are this file's own: the reference's 256/128 were sized for a view with no
scenery in it, and the arm-vs-table rows joined the budget when the table did.

``cable_overhead_corner@v1`` -- the wrap's rig -- is fixed rather than aimed:
both views look at a point on the cell's own work surface, which is a constant
of the bench. The thirteen weave rigs are aimed, each at its own layout's centroid,
and are registered here under the names the documents use, so a document's own
rig name is one ``get_camera_rig`` can resolve.
"""

from __future__ import annotations

import vabar  # noqa: F401 -- the scripted-routine module RoboSimStudio leaves out; see vabar/__init__.py

import dataclasses
import itertools
import math
import warnings
from typing import Any

import numpy as np

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.scene.builtin.benches import BENCH_075
from robosimstudio.scene.frames import in_surface, point_from_surface
from robosimstudio.specs import ArrangementSpec, CollisionSpec, ReproHooksSpec, backend, coupler
from robosimstudio_bench.taskbuild import register_arrangement, register_camera_rig

from . import difficulty
from .assets.layout import (
    CELL,
    WEAVE_AXIS_Y,
    WEAVE_CABLE_LENGTH,
    WEAVE_CABLE_X,
    WEAVE_LAYOUTS,
    WEAVE_MIN_SLACK,
    WEAVE_SIDES,
    WEAVE_TAIL,
    WEAVE_TERMINALS,
    seat_offset,
    weave_anchor_y,
    weave_terminals,
)

__all__ = [
    "CABLE",
    "CABLE_BOW",
    "CABLE_FAST",
    "CABLE_REPRO_HOOKS",
    "CABLE_RIG",
    "PORT_BOARD",
    "WEAVE_BOARD",
    "WEAVE_RIGID",
    "WEAVE_RIGS",
    "cable_repro_params",
    "cable_repro_taps",
    "spawn_excursion",
]

_CELL = BENCH_075
_LOOK = (0.5, 0.0, _CELL.surface_z)
"""Where both views point: the middle of the working area, at bench height."""


def _latched_cable(scene):
    """The entity whose latch this bundle tapes, found by *having* one.

    It was ``scene.by_name[scene.request.extras[0].name]`` -- an index into a
    view. The cable sat in ``extras`` only because ``RodSpec`` declared no build
    stage; the moment it declared ``"deformables"`` like every other deformable,
    both readers here raised ``IndexError`` from inside a reset. Asking which
    entity has a latch is the question that was always meant, and it does not
    care which bin the spec is filed under.
    """
    for entity in scene.by_name.values():
        if hasattr(entity, "hold_gains") and getattr(entity, "_hold_flag", None) is not None:
            return entity
    raise LookupError("this scene has no latched cable; a repro bundle for it would tape a latch that is not there")


def cable_repro_params(scene) -> dict[str, Any]:
    """What ``repro_physics.build_hooks`` needs, read off the compiled scene.

    Only the latch: the solvers are rebuilt from the composer's manifest, which
    the exporter freezes generically. What a manifest cannot carry is a kernel
    and its device state, and the latch is both -- so its gains travel here and
    its per-step engagement in :func:`cable_repro_taps`.
    """
    cable = _latched_cable(scene)
    kp, kd = cable.hold_gains()
    return {"hold_gains": [float(kp), float(kd), float(cable.hold_force_max())]}


def cable_repro_taps(scene) -> dict[str, Any]:
    """Per-step readers of the latch state a replay must re-upload.

    The latch is driven mid-episode by whoever is running the task -- a slot
    engages on a segment, in a hand, at a taken relative pose, under gains the
    studio can retune live. All of it is device state the tape snapshots each
    control step, and the standalone module's aux sinks receive on replay.
    """
    cable = _latched_cable(scene)
    return {
        "hold_flag": cable._hold_flag.numpy,
        "hold_seg": cable._hold_seg.numpy,
        "hold_grip": cable._hold_grip.numpy,
        "hold_rel": cable._hold_rel.numpy,
        "hold_gains": cable._hold_gains.numpy,
    }


CABLE_REPRO_HOOKS = ReproHooksSpec(
    module="robosimstudio.solvers.grasp.latch",
    params=cable_repro_params,
    taps=cable_repro_taps,
)
"""The latch, standalone, for a repro bundle. The suite file names it --
``(:repro-hooks "staging:CABLE_REPRO_HOOKS")`` -- so every variant carries it.

Named rather than pointed at a file beside this one. The suite used to ship its
own copy of the kernel for the bundle to carry, byte-identical to the library's
in everything but its docstring; the exporter resolves an importable name to a
file and copies it verbatim, so one module now serves both the live scene and
the bundle. The copy it replaces is the reason this is worth saying: two kernels
kept in step by hand stay in step until somebody tunes one."""


CABLE = register_arrangement(
    "cable_coupled_960hz@v1",
    # 60 Hz x 16 substeps.
    ArrangementSpec(
        control_hz=60.0,
        substeps=16,
        backends=(
            (
                "articulated",
                backend(
                    "mujoco",
                    solver="newton",
                    integrator="implicitfast",
                    cone="elliptic",
                    iterations=100,
                    ls_iterations=20,
                    # Newton detects the arm's contacts with the scenery and the
                    # composer routes them to this island; MuJoCo only solves them.
                    use_mujoco_contacts=False,
                    njmax=1024,
                    nconmax=512,
                ),
            ),
            (
                "rod",
                backend(
                    "vbd",
                    iterations=40,
                    rigid_avbd_beta=1.0e2,
                    rigid_contact_k_start=1.0e3,
                    rigid_contact_history=True,
                    # A closed decomposed fingertip can generate more than
                    # Newton's default 64 contacts on one cable body; overflow
                    # drops contacts.
                    rigid_body_contact_buffer_size=256,
                ),
            ),
        ),
        coupling=coupler(
            "proxy",
            mode="staggered",
            # 2, not the reference's 4: the one deviation this arrangement
            # makes, bought with a measurement -- see the module docstring.
            iterations=2,
            mass_scale=20.0,
            collide_interval=1,
            contact_matching="latest",
        ),
        collision=CollisionSpec(
            # 2 mm, not the reference's 10. This default reaches the robot and
            # the scenery -- the rod pins its own gap=0.01 per shape -- and 10 mm
            # of detection margin was sized for a cell where no arm-vs-scenery
            # pair existed to feel it; with those contacts owned, the margin
            # belongs at the scale of the cell's real clearances (the weave works
            # 2 mm over the tabletop). Not a fix for the hand settling ~14 mm
            # high at grasp depth: that was measured unchanged across both
            # values, and its mechanism is fingertip-first contact carrying the
            # arm on the position servo's compliance -- 25 pad contacts, five
            # joints each 0.01-0.03 rad short -- while IK solves the same target
            # to 0.0 mm.
            rigid_gap=0.002,
            contact_matching="latest",
        ),
    ),
)

CABLE_FAST = register_arrangement(
    "cable_route_fast@v1",
    # 60 Hz x 16 substeps as the reference, one coupler pass, VBD at 20 iterations.
    dataclasses.replace(
        CABLE,
        backends=CABLE.backends_replacing("rod", CABLE.backend_for("rod").with_(iterations=20)),
        coupling=CABLE.coupling.with_(iterations=1),
    ),
)
"""The ``fast`` config's cable arrangement: :data:`CABLE` with one coupler pass and 20 VBD iterations.

**Not what any document stages.** Every cable document still names
``cable_coupled_960hz@v1``; this is selected by ``vabar/configs/fast.toml``
(``[suites.cable_route] physics = "cable_route_fast@v1"``) for a grad student
with one GPU. Re-run the numbers below with
``tools/fast_tune/cable_route.py``.

**Two multipliers cut, and the rate kept.** The step is launch-bound, so what
pays is fewer graph nodes: one coupler pass halves both solvers' restarts per
substep, and once there is one pass VBD's iterations are most of what is left --
halving them buys ~40 %, not the ~2 % :data:`CABLE`'s note measured at two
passes. Idle on ``port_easy`` 10009, no renderer, alone on an RTX PRO 6000:
15.2 and 15.1 steps/s against 7.2 for the reference in the same block (2.1x; the
reference's mean over four runs is 7.17). Still 960 Hz: 480 Hz bought the same
speed (8 substeps, one pass: 12.9-16.6) but the same rate at two passes diverged
on weave3, and :data:`CABLE`'s note records 480 Hz killing the wrap.

**Measured against the reference's own spread, not a single run of it**, because
the routine does not reproduce itself: the reference's weave3 closed success
(2,737 steps) and then 0.67 on the same command, and four of five port_easy
instances closed at a different reward in its second run. Here: weave3 0.67 and
then success (2,733), port_easy 9 of 10 seats held on those five instances (the
reference: 8, then 6), and the cell graph protocol exactly the reference's --
success on 10009 (6,297 steps against its 6,374) and 0.50 on 10062.
"""

CABLE_RIG = register_camera_rig(
    "cable_overhead_corner@v1",
    # Overhead and three-quarter views over the working area.
    (
        CameraSpec(
            name="overhead",
            width=640,
            height=480,
            fov_y=50.0,
            eye=(_LOOK[0], 0.0, _CELL.surface_z + 0.9),
            look_at=_LOOK,
            up=(1.0, 0.0, 0.0),
            near=0.01,
            far=4.0,
            channels=("rgb", "depth"),
        ),
        CameraSpec(
            name="corner",
            width=640,
            height=480,
            fov_y=50.0,
            eye=(1.35, -0.85, 1.35),
            look_at=_LOOK,
            near=0.01,
            far=4.0,
            channels=("rgb", "depth"),
        ),
    ),
)


# ---------------------------------------------------------------------------
# THE WEAVE RIGS, aimed at layouts that live in assets/layout.py.
# ---------------------------------------------------------------------------

WEAVE_CABLE_ELEV = 40.0
"""Elevation of the ``cable`` camera above the bench plane [deg], measured from +x.

**Not 90, and that cost most of a working tracker.** An overhead camera looks
down the same axis the gripper descends, so the hand stands between the lens and
the rod exactly when the rod is being moved -- which is the only time its shape
changes and the only time anyone needs to see it. Measured over eight arm poses
(each station, at grasp height and mid-lift), fitting the centreline through
SAM3 and TrackDLO from each candidate and scoring against the rod's own body
positions:

=====================  =========  =========  =========  ===============
Camera                 RMS med    RMS p90    RMS max    peg clearance
=====================  =========  =========  =========  ===============
top-down (was)          60.5 mm    113.1 mm   168.0 mm   n/a
front, 70 deg            6.4 mm     17.0 mm    41.3 mm   +163 mm
front, 65 deg            6.7 mm      9.5 mm    12.6 mm   +120 mm
front, 55 deg            6.7 mm      8.4 mm     9.3 mm    +65 mm
front, 50 deg            7.3 mm      9.9 mm    10.6 mm    +46 mm
front, 40 deg            7.1 mm      8.5 mm     8.7 mm    +16 mm
=====================  =========  =========  =========  ===============

**And then further forward and tilted in again, which 55 degrees was still
leaving on the table.** At 55 the camera looks down close to the axis the
gripper descends, so the arm crosses the cable's line of sight on the way in.
Standing the camera further off in x and dropping the pitch turns that column
side-on. Re-measured the same way, now also counting how many of the rod's 42
bodies survive a depth test rather than only how well the fit lands:

=====================  =========  =========  =========  ==========  ==========
Camera                 RMS med    RMS p90    RMS max    visible     peg clr
=====================  =========  =========  =========  ==========  ==========
55 deg, 0.50 m           4.8 mm     7.2 mm     8.3 mm    38 (35)      +65 mm
45 deg, 0.70 m           4.3 mm     5.7 mm     8.2 mm    40 (37)      +33 mm
**40 deg, 0.60 m**       4.5 mm     5.5 mm     5.9 mm    41 (39)      +17 mm
35 deg, 0.75 m           4.7 mm     5.1 mm     5.3 mm    41 (39)       +6 mm
30 deg, 0.85 m           3.9 mm     4.8 mm     5.1 mm    41 (39)       -5 mm
=====================  =========  =========  =========  ==========  ==========

``visible`` is the median of 42 with the minimum in brackets. The floor rises
from 35 to 39 as soon as the pitch comes off 55, which is the arm getting out of
the way, and it is what this move was for.

The lower bound is the PEGS, not the arm: ``peg clr`` is how far the sight line
to the resting cable passes above the 60 mm peg tops, and it goes negative below
about 35 degrees -- at 30 the fit is numerically the best of the set and the
camera is looking through the fixtures to get it, which will not survive a
taller board. 40 degrees keeps 17 mm of real clearance and has the best
worst-case fit of any candidate that does.

Overhead is an order of magnitude worse, and not by drifting: with the arm
across the rod the skeleton breaks in two and the chain merge returns one
fragment, so a 488 mm cable is fitted as 338 mm and the tracker is handed a
model of half a rod.

**Every row above was measured on a 42-capsule 0.50 m rod at the 70 mm pitch**,
which is the board this camera was chosen against. The stations stand 155 mm
apart now and the rods are 0.70 to 1.15 m, so the framing argument carries (the
lens still holds the whole run at this standoff) and the per-fit millimetres do
not: nothing has re-measured them on the wider board."""

WEAVE_CABLE_DIST = 0.60
"""How far the ``cable`` camera stands off the point it looks at [m].

Sets resolution, and barely: 0.60 m gives about 0.60 mm/px over the cable's own
run, against the 0.43 the overhead had at 0.45 m -- still three times finer than
the cell overhead.

Its real job is to put the camera FORWARD. Pitch alone cannot do that: dropping
the elevation at a fixed standoff swings the eye down toward the bench rather
than out along it, and the sight line clips the pegs before the arm is clear.
Standoff and pitch together move the eye from (0.75, 1.16) to (0.92, 1.14) --
170 mm further off the robot's end of the bench at the same height.

Measured across 0.60 to 0.85 m the standoff moves RMS by well under a
millimetre, so it is chosen for geometry rather than for accuracy."""

WEAVE_WRIST_SIDE = 0.070
WEAVE_WRIST_AHEAD = 0.030
WEAVE_WRIST_BEYOND = 0.060
"""Where the eye-in-hand camera sits on the work hand [m].

Sideways clearance, how far back along the approach it sits, and how far past
the TCP it looks. The *directions* come from the gripper's own measured axes
(see :func:`_weave_wrist_mount`); these three are the standoffs, chosen so the
camera clears a hand whose ``tip_reach`` is 35 mm and still frames the rod
rather than the fingertips."""


def _weave_wrist_mount() -> tuple[float, ...]:
    """The eye-in-hand pose on the work gripper, built from THIS hand's axes.

    **Not astro's three numbers.** They were tried and they put the camera
    inside the hand: that rig's gripper extends along +z with its jaws closing
    across x, and this one extends along **-z** -- ``tcp_offset`` is
    ``(0, 0, -0.1347)`` -- with ``close_axis`` near -y. So astro's "50 mm past
    the TCP" aimed 50 mm *back toward the mount*, and its sideways offset along
    +y lay along this hand's closing axis rather than perpendicular to it.
    Rendered, the camera saw the gripper body at a median depth of 63 mm and
    nothing else.

    Built from the spec's own axes instead: the camera stands off perpendicular
    to BOTH the approach and the closing axis (so neither the jaws nor the
    payload crosses the view), sits slightly back along the approach, and looks
    at a point past the TCP in the direction the hand actually points.

    **The HAND is still named, and that is a real limit rather than an
    oversight.** ``get_gripper("linear_4310")`` is the YAM's, and a rig is
    registered under a name a document stages -- so the mount is computed once,
    at import, before any binding exists to read a hand off. Every document that
    names one of these rigs is bound to ``yam_i2rt`` (weave3's second host row
    is ``yam_bimanual``, which carries the same hand). A pairing that brought a
    different gripper would get a camera aimed with this one's axes, and the
    fix is a rig resolved per binding rather than a constant in the mount.
    """
    import numpy as np  # noqa: PLC0415

    from robosimstudio.core.cameras import look_at_matrix  # noqa: PLC0415
    from robosimstudio.core.math import xform_from_matrix  # noqa: PLC0415
    from robosimstudio.robots.registry import get_gripper  # noqa: PLC0415

    spec = get_gripper("linear_4310")
    approach = np.asarray(spec.approach_axis, dtype=float)
    approach /= max(float(np.linalg.norm(approach)), 1.0e-12)
    close = np.asarray(spec.close_axis, dtype=float)
    close /= max(float(np.linalg.norm(close)), 1.0e-12)
    # Perpendicular to both: the one direction that watches the grasp without
    # looking through a jaw or along the line the payload occupies.
    side = np.cross(approach, close)
    side /= max(float(np.linalg.norm(side)), 1.0e-12)

    tcp = np.asarray(spec.tcp_offset, dtype=float)
    eye = side * WEAVE_WRIST_SIDE - approach * WEAVE_WRIST_AHEAD
    target = tcp + approach * WEAVE_WRIST_BEYOND
    return tuple(float(v) for v in xform_from_matrix(look_at_matrix(eye, target, close)))


def _weave_rig(variant: str) -> tuple[CameraSpec, ...]:
    """The rig a weave document names, aimed at that weave's own stations.

    ``cable_overhead_corner@v1`` looks at the wrap's peg line, which is 80 mm
    away in x and 60 mm in y from these -- close enough to look almost right in
    a still and wrong enough that the outermost station leaves the corner view.

    Per variant rather than once, because the layouts do not share a centre: the
    arc's stations average 15 mm further from the robot than the line's, and an
    overhead framed on the line crops it.

    **Four views, and the last two are what a policy is meant to read.** The
    overhead and the corner frame the whole cell and are what a fixed rig gives
    you. ``cable`` is a placed third-person camera on the ROD -- 1280x960 at
    0.60 m and 40 degrees of elevation, which is three and a half times finer
    than the cell overhead over the cable's own extent and, more to the point,
    a line of sight the working arm does not cross (see
    :data:`WEAVE_CABLE_ELEV`). ``wrist`` is the eye-in-hand, at tool_hanging's
    own lens widened to 640x480 because this one is asked to *measure* a cable
    rather than confirm a grasp.
    """
    stations = WEAVE_LAYOUTS[variant]
    look = (
        sum(x for x, _ in stations) / len(stations),
        sum(y for _, y in stations) / len(stations),
        CELL.surface_z,
    )
    # These boards are intentionally shifted inside their anchored runs. Keep
    # the overview centred on the whole cable, or the medium free tail is
    # cropped before perception can fit its centreline.
    if variant in {"port_easy", "port_medium"}:
        look = (look[0], WEAVE_AXIS_Y, CELL.surface_z)
    cameras = (
        CameraSpec(
            name="overhead",
            width=640,
            height=480,
            fov_y=50.0,
            eye=(look[0], look[1], CELL.surface_z + 0.9),
            look_at=look,
            # +x is "toward the robot" on this bench, and pointing the
            # overhead's up that way puts the peg line across the frame
            # rather than end-on, so the alternation reads as a zig-zag.
            up=(1.0, 0.0, 0.0),
            near=0.01,
            far=4.0,
            channels=("rgb", "depth"),
        ),
        CameraSpec(
            name="corner",
            width=640,
            height=480,
            fov_y=50.0,
            eye=(1.30, -0.80, 1.25),
            look_at=look,
            near=0.01,
            far=4.0,
            channels=("rgb", "depth"),
        ),
        CameraSpec(
            # A THIRD-PERSON CAMERA ON THE CABLE, not on the hand.
            #
            # The eye-in-hand was tried as the primary sensor and does not work
            # here: at any pose the arm can hold, it frames either a fifth of
            # the cable or -- at the wide tilt that fixes that -- none of it,
            # because a large ground footprint says nothing about where the
            # camera points. And an A/B against the bench view came out
            # identical (0.667 mean either way), so the move it costs buys
            # nothing.
            #
            # What the cable actually needs is resolution over its own extent
            # AND a line of sight the working arm does not cross, which a
            # *fixed* camera can simply be placed to give:
            #
            #   cell overhead   640x480  @50 deg, 900 mm -> 1119 mm  1.75 mm/px
            #   this one       1280x960  @50 deg, 600 mm ->  768 mm  0.60 mm/px
            #
            # Read ONCE PER STATION rather than continuously -- the shape only
            # changes when a crossing moves it, so a per-pass look is the
            # honest sampling rate.
            name="cable",
            width=1280,
            height=960,
            fov_y=50.0,
            # TILTED, NOT OVERHEAD -- see WEAVE_CABLE_ELEV. Standing off in +x
            # puts the camera on the far side of the peg line, looking back
            # down the bench at the cable, so the arm that is working the rod is
            # BEHIND it rather than on top of it.
            eye=(
                WEAVE_CABLE_X + 0.04 + WEAVE_CABLE_DIST * math.cos(math.radians(WEAVE_CABLE_ELEV)),
                WEAVE_AXIS_Y,
                CELL.surface_z + WEAVE_CABLE_DIST * math.sin(math.radians(WEAVE_CABLE_ELEV)),
            ),
            look_at=(WEAVE_CABLE_X + 0.04, WEAVE_AXIS_Y, CELL.surface_z),
            # World up, now that the camera is not looking down its own axis:
            # the rod runs along y and a z-up basis puts y across the long image
            # axis, which is the framing the overhead gets from up=+x. Keeping
            # +x here would be degenerate at this pitch.
            up=(0.0, 0.0, 1.0),
            near=0.01,
            far=2.0,
            channels=("rgb", "depth"),
        ),
        CameraSpec(
            name="wrist",
            # tool_hanging's own eye-in-hand lens, deliberately: 320x240 at
            # 58 degrees there, 640x480 at 58 here because this camera is asked
            # to *measure* a cable rather than confirm a grasp.
            #
            # Two earlier rigs were wrong in opposite directions. 40 degrees at
            # 160 mm resolves 0.24 mm/px and frames 155 mm -- a fifth of the
            # cable, so a tracker reading it can only ever update the stretch
            # beside the hand. 105 degrees frames the whole cable but is a
            # fisheye no real cell carries.
            #
            # A normal lens reaches the whole cable by STANDOFF AND TILT
            # instead. Measured over these stations, full-pose IK residual and
            # the resulting ground field:
            #
            #   tilt 50 deg, 250 mm  ->  586 mm field, 0.92 mm/px,  0 mm residual
            #   tilt 35 deg, 160 mm  ->  536 mm field, 0.84 mm/px,  0 mm residual
            #   tilt  0 deg, 160 mm  ->  155 mm field, 0.24 mm/px,  0 mm residual
            #
            # The tilt costs the strict top-down property, and that is
            # acceptable *here* because this view measures the CABLE --
            # back-projected through depth, where obliquity is handled -- while
            # the spools are still measured from the bench, where a flange disc
            # stays a circle.
            width=640,
            height=480,
            fov_y=58.0,
            # The ROLE, not a link. ``work`` resolves to whichever arm is doing
            # the work and rides that arm's tool body, so the rig is not pinned
            # to one robot's link names -- or to one importer's fixed-joint
            # collapse.
            parent_frame="work",
            xform=_weave_wrist_mount(),
            near=0.005,
            far=1.5,
            channels=("rgb", "depth"),
        ),
    )

    # The two fixed-end evaluation tasks expose exactly one RGB-D sensor.
    if variant in {"port_easy", "port_medium"}:
        return tuple(camera for camera in cameras if camera.name == "cable")
    return cameras


WEAVE_RIGS: dict[str, str] = {v: f"cable_{v}@v1" for v in WEAVE_LAYOUTS}
"""Which rig each weave document names: one registered name per *distinct* rig
rather than one shared name aimed four ways.

``cable_weave_corner@v1`` was the shared spelling, and it never reached
:data:`~robosimstudio_bench.taskbuild.CAMERA_RIGS` at all -- the rig went in
through ``lower(cameras=...)``, so the four documents named a rig that
``get_camera_rig`` could not resolve and any reader outside the loader saw a
dangling reference. Registering is what makes the document's own name true.
"""

for _v, _name in WEAVE_RIGS.items():
    register_camera_rig(_name, _weave_rig(_v))
del _v, _name


# ---------------------------------------------------------------------------
# THE SAMPLERS. Three, and a document names one of them by artifact reference --
# ``(:sampler staging:WEAVE_BOARD)`` -- because every rule they draw under is
# this suite's rather than the library's: the reach band, the seat pitch, the
# rod's length, the level. What they emit is ``static_poses`` (where the
# fixtures stand, in the CELL'S OWN SURFACE FRAME) and the four unit fractions
# the rod's spawn ranges scale -- and all of those change the MODEL rather than
# a pose, so each carries ``model_key_fields`` and the env rebuilds per episode.
# ---------------------------------------------------------------------------


def _cable_bow(task, episode_id: int, rng) -> dict[str, Any]:
    """A per-episode shape for the spawned rod, as four signed unit fractions.

    Emits ``cable_bow``, ``cable_shift``, ``cable_yaw`` and ``cable_wave`` in
    ``[-1, 1]``; the *amplitudes* they scale live with the cable, in
    :data:`~.assets.cable.WEAVE_CABLE_BOW` and its three siblings. Splitting them
    that way keeps the one number a document should not restate -- how bent a
    particular rod may be, which is a property of that rod and its bench -- out
    of the episode layer, while the randomness stays where every other sampled
    quantity's does.

    **Why a cable needs this and a rigid object does not.** A box sampled onto a
    grid still arrives as the same box; a cable spawned straight arrives as a rod
    whose shape is known in advance, and a policy can be right about the first
    grasp by remembering an x rather than by seeing one. Nothing in the score
    distinguishes those two until the rod bends.

    **Four fractions, not one.** The bow bends the rod's middle and leaves both
    tips where they were, so a policy could still be right about the first grasp
    by remembering a coordinate; the shift slides the whole rod, the yaw turns it
    and the second harmonic makes it meander. Each is scaled by the CABLE's own
    range, and a spec that leaves a range zero-width ignores its fraction
    entirely -- so this sampler changes nothing for a rod that has not opted in,
    which is what lets it be the whole of the wrap's sampler too if that variant
    ever wants one.

    **This lives here rather than in the library's episode samplers**, which is
    where it was first written. Nothing outside this suite has a rod whose spawn
    it scales, and a library sampler emitting four keys that exactly one task's
    spec reads is a task's sampler standing in the library.

    **THE ANCHOR BRACKET FOLLOWS THE ROD, on a terminated run.** The shift and
    the yaw both move the end that is welded to the world, and the bracket drawn
    around that weld is a static placed once -- so a sampler that moved the rod
    and not the fitting is one that puts a plate 32 mm across, with a 10 mm
    channel down it, up to 35 mm from the cable it is meant to hold the end of. Measured on ``port2`` before this
    followed: the rod's first body landed at x = 0.402 and 0.385 on two episodes
    against a bracket at 0.420.

    It goes through :func:`~.assets.layout.weave_terminals`, which is also what
    places the declared pose and what :func:`_weave_rigid` calls when it moves a
    whole board -- one arithmetic over different inputs rather than three.
    """
    del episode_id
    draw = {
        "cable_bow": float(rng.uniform(-1.0, 1.0)),
        "cable_shift": float(rng.uniform(-1.0, 1.0)),
        "cable_yaw": float(rng.uniform(-1.0, 1.0)),
        "cable_wave": float(rng.uniform(-1.0, 1.0)),
    }
    if getattr(task, "name", "") not in WEAVE_TERMINALS:
        return draw
    # Only the anchor: the port belongs to the BOARD, and this sampler does not
    # move the board.
    start, heading = _cable_start_frame(task, draw)
    anchor = weave_terminals(task.name, start_xy=start, anchor_yaw=heading)["anchor"]
    return {**draw, "static_poses": {"anchor": _surface_pose(*anchor)}}


_cable_bow.model_key_fields = ("cable_bow", "cable_shift", "cable_yaw", "cable_wave", "static_poses")
"""Sampled keys that change the model rather than only the state: the four
fractions are in the rod's spawn geometry and the bracket is welded scenery, so
two episodes with different draws need different models. Read by
``taskbuild.lower._episode_spec``; a sampler without it varies nothing the
compiler would rebuild for -- and every episode would silently be episode 0's
rod. ``static_poses`` is listed for every caller and emitted only by the
terminated ones, which costs the others nothing: a key nothing emits contributes
nothing to the key."""

CABLE_BOW = _cable_bow
"""``(:sampler staging:CABLE_BOW)``: the rod's shape, and nothing else."""


# ---------------------------------------------------------------------------
# weave_board: a task board drawn at random, to a difficulty level.
# ---------------------------------------------------------------------------

BOARD_PITCH = (0.072, 0.092)
"""Along-axis gap between adjacent stations, drawn per gap [m].

The floor is not taste. A seat is a box reaching +/- 35 mm along the line in its
own station's frame, and two stations closer than 70 mm have overlapping seats --
a segment of rod counted once for each, which is the one property every weave
document asserts and none of them re-derives. 72 mm is that floor plus the 2 mm
the rotation below can eat.

The ceiling is reach: six stations at 92 mm span 460 mm, and the working arm's
measured band at the carry height is 550 mm wide. The draw is per gap rather than
per board, so a board's spacing is uneven -- which is most of what makes one look
arranged by hand rather than by arithmetic.

**A drawn board is tighter than the fixed layouts', which stand on a 155 mm
pitch.** That is a difference in kind rather than an inconsistency: a fixed rung
spends its pitch on being unambiguous and a draw spends it on fitting six
stations inside a measured envelope, and what is held constant across the two is
the TURNING rather than the spacing."""

BOARD_DEPTH_SPREAD = 0.55
"""How far each station's stagger may stray from the board's mean, as a fraction.

The mean is *solved for* (see :func:`_board_depth`) and this is what makes the
board irregular around it. Wide enough that no two crossings of a board are the
same shape; narrow enough that the accumulated turning still lands in the band
the mean was solved for often enough to be worth drawing."""

BOARD_WOBBLE = 0.009
"""Extra cross-axis wander per station, signed and independent of its side [m].

Applied on top of the stagger, so a board is not two clean rows. Kept under the
8.25 mm seat offset: a wobble that could carry a station across the centre line
would put it on the wrong side of its own required side, which is not a harder
board but a different task."""

BOARD_YAW = 0.26
"""Half-range of the whole board's rotation about its centroid [rad].

15 degrees either way. The stations rotate; the **seats do not** -- a seat is a
box in its station's frame and the stations are placed unrotated, so a seat's
offset stays along world x however the board is turned. That is not an oversight
to be corrected: the rod still spawns along +y, so turning the board genuinely
changes each crossing's obliquity against the direction the rod arrives in, and
:func:`~tasks/cable_route/difficulty.py::profile` measures the result rather than
predicting it. The rotation is a third knob on the turning angle, and the
rejection loop is what keeps it honest."""

BOARD_CENTRE_X = (0.518, 0.552)
BOARD_CENTRE_Y = (-0.118, -0.020)
"""Where a board's centroid may sit [m]. The board then extends from there.

The ``x`` pair is the **widest** the centroid may roam, not the range it is drawn
from: a deep board has to sit near the middle of the envelope or its own stagger
falls out of it, so the draw is the narrower of this and what the stagger left
over. See ``drift`` in :func:`_weave_board`. The ``y`` pair is drawn from
directly -- nothing along the board competes for that budget except its own
length, which the pitch already answers for."""

BOARD_X = (0.495, 0.575)
BOARD_Y = (-0.336, 0.088)
"""Where any single station may stand [m] -- the hard envelope.

Measured, not assumed, and measured **at the grasp this task actually makes**.
``tools/reach_bench.py`` sweeps a top-down pose with the wrist roll free, which
is a different question and a wider answer; ``rigs/weave_reach.py`` sweeps the
weave's own yaw-0 pose against its own ``PLAN_ERROR_LIMIT``, at both the rest
and carry heights, and that is what these come from. It puts the working arm's
band at ``y`` in ``[-0.36, +0.10]`` at the tightest ``x`` a board may use and
wider everywhere else, out past ``x = 0.58``.

**The far edge is set by that measurement and the near edge is not.** The near
one is the idle floor's, below; the far one is simply where the arm stops
planning, and it is worth having all of it, because a board reaches a level by
being oblique and how oblique it can be is exactly how far across the bench its
two rows may stand.

These are that band brought in a little further, which leaves the ``y = 0.17``
support anchor 82 mm of clearance over the outermost station a board can put
down -- more than ``weave6``, whose own last station stands at +0.328 and is
past the band entirely.

**A station inside this envelope is not a station that always plans.** The
routine chains its solves, cut point to cut point, and a chained solve can land
in a basin a fresh one would not; ``rigs/weave_reach.py`` seeds every probe from
home and says so. Refusals on drawn boards are real and are the routine's -- the
envelope is drawn where boards *may* stand, not where a particular policy is
guaranteed to succeed, and a benchmark that shrank it until one baseline stopped
complaining would be a benchmark fitted to that baseline.

The ``x`` floor is ``weave6``'s own near row rather than anything wider, and the
reason is the **idle floor** rather than reach: a near-side seat reaches 45 mm
back from its station and the rod spawns 80 mm back from the peg line and
*bowed*, so a station much nearer the robot than this would put its own seat
inside the envelope the rod can arrive in."""

BOARD_TURN_FLOOR = 9.0
"""The least any station may be wrapped by [deg].

A board is drawn to a *total* turning angle, and a total can be met by a board
with one station the route barely touches -- which is what ``weave_arc`` is, at
29 degrees on its weakest station against 51 on its strongest, and it is a
deliberate control rather than something to draw by accident. A drawn board holds
every station."""

BOARD_MARGIN = 0.10
"""How far inside its band's edges a drawn board's turning must land.

Ten per cent, so a board is unambiguously at its level rather than a rounding
error from the next one -- the same standard ``tests/test_cable_difficulty.py``
holds the fixed rungs to."""

BOARD_TRIES = 6000
"""Draws before a board is given up on. See :func:`_weave_board`."""


def _in_world(xform) -> tuple[float, float]:
    """A static spec's ``(x, y)``, lifted out of the work-surface frame."""
    placed = in_surface(CELL, xform)
    return (float(placed[0]), float(placed[1]))


def _surface_yaw(xform) -> float:
    """A static spec's own yaw about z, in the frame its xform is stated in [rad].

    ``static_poses`` replaces a fixture's whole rotation with one z turn (see
    :func:`~robosimstudio.domains.static.entity.episode_xform`), so a sampler that
    perturbs a *turned* fixture has to know what it was turned by. The pegs never
    needed this -- they are declared square -- and the port is declared facing
    back down the run at ``pi`` plus its board's angle, which a draw that emitted
    a bare offset would silently discard.
    """
    x, y, z, qx, qy, qz, qw = (float(v) for v in tuple(xform)[:7])
    del x, y, z, qx, qy
    return 2.0 * math.atan2(qz, qw)


def _surface_pose(x: float, y: float, yaw: float) -> tuple[float, float, float, float]:
    """One world ``(x, y, yaw)`` as the ``static_poses`` override reads it.

    THROUGH THE SURFACE FRAME, not raw world. ``static_poses`` is read as a pose
    in the cell's work-surface frame, which is what the placed records build
    their declared xforms with -- emitting world coordinates put the pegs where
    they belonged and the SCORED SEAT REGIONS 580 mm away in x, so a graph wove
    correctly against the fixtures it could see and the evaluator read 0.00.
    Measured: peg at x = 0.534, seat centre at 1.110.
    """
    px, py, pz = point_from_surface(CELL, (float(x), float(y), CELL.surface_z))
    return (float(px), float(py), float(pz), float(yaw))


def _board_window(level: int, bands) -> tuple[float, float]:
    """The turning angle a board at ``level`` must land in [deg], with margin."""
    lower = 0.0
    for lvl, upper, _, _ in bands:
        if lvl == level:
            # The top band is open above; a board is drawn into the octave over
            # its floor rather than into infinity, because the rod runs out
            # before the arithmetic does.
            top = upper if upper != float("inf") else lower * 1.6
            return (max(lower, 45.0) * (1.0 + BOARD_MARGIN), top * (1.0 - BOARD_MARGIN))
        lower = upper
    raise ValueError(f"no band for level {level}")


def _board_depth(mean_pitch: float, turning: float, stations: int, seat: float) -> float:
    """The mean stagger a board of this shape needs to reach ``turning`` [m].

    The closed form inverted: a periodic board's accumulated turning is
    ``2 * (n - 1) * atan(excursion / pitch)`` where the excursion between
    adjacent seats is ``2 * (depth + seat)``. Solving it for the depth turns
    rejection sampling from a search into a nudge -- the draw lands in the band
    most of the time and the loop only has to reject the boards that the wobble,
    the uneven pitch and the rotation carried out of it.

    Clamped at zero: the low band at three stations needs *negative* stagger to
    hit its centre, because two seats 8.25 mm either side of a straight line on
    an 80 mm pitch already turn the route 47 degrees. Such a board is a straight
    line with uneven spacing, which is exactly what an L1 board should look like.
    """
    alpha = math.radians(turning / (2 * max(stations - 1, 1)))
    return max(0.0, 0.5 * (mean_pitch * math.tan(alpha) - 2 * seat))


def _weave_board(task, episode_id: int, rng) -> dict[str, Any]:
    """One randomly arranged board, at the difficulty level of the declared one.

    **The level is read off the task, not off the variant's name.** Whatever
    layout the loader put down is measured, and the board this draws is drawn
    into the same band -- so the sampler has no table of its own to keep in sync
    and any weave variant could take it. What varies is everything else: the gap
    between each pair of stations, how far each is staggered off the line, a
    per-station wobble on top of that, and a rotation of the whole board.

    **Drawing to a level is what makes this a sampler and not a jitter.** The
    difficulty measure runs on every draw and the draw is thrown away unless it
    lands in the band with margin, every station is inside the arm's measured
    envelope, no two seats overlap, every station is genuinely wrapped, and the
    route still fits the rod with slack to spare. A jitter would have produced
    boards that look varied and are not comparable; this produces boards that
    are a *sample from a level*, which is the thing a benchmark can average over.

    Emits ``static_poses`` for every station, plus the rod's own ``cable_bow``,
    so a board and the cable laid in front of it vary together. Both are
    model-changing.

    Raises:
        RuntimeError: If no draw satisfied every constraint. Loud rather than
            silent: a sampler that quietly fell back to the declared board would
            report a whole eval's worth of episodes as varied when one layout ran
            every time, and the scores would look like variance.
    """
    names = [s.name for s in task.statics]
    n = len(names)
    if n < 3:
        raise RuntimeError(f"{task.full_name} has {n} stations; a board needs at least three")
    seat = seat_offset(task.name)
    sides = [WEAVE_SIDES[i % len(WEAVE_SIDES)] for i in range(n)]
    rod = WEAVE_CABLE_LENGTH[task.name]

    # World coordinates, because every constraint below is stated in them: the
    # reach band was measured in world, the seats' offset runs along world x, and
    # the rod spawns at a world x. The specs hold the *surface* frame, so the
    # declared board is lifted out of it here and the drawn one is pushed back
    # into it before it is emitted -- see `_surface_pose`.
    declared = [_in_world(s.xform) for s in task.statics]
    level = difficulty.profile(declared, sides, seat=seat, tail=WEAVE_TAIL).level
    low, high = _board_window(level, difficulty.BANDS)
    target = math.sqrt(low * high)

    # The along-axis budget is the envelope's, so the pitch a board of THIS many
    # stations may be drawn at is narrower than the family's range whenever the
    # stations would not otherwise fit. Adapted rather than rejected: six
    # stations at the family's 92 mm ceiling span 460 mm against a 414 mm
    # envelope, so every draw would have been thrown away and the loop would
    # have reported "no board" for a board that exists.
    reach_pitch = (BOARD_Y[1] - BOARD_Y[0] - 0.020) / (n - 1)
    # And the ACROSS-axis budget bounds it too, which is the constraint that
    # actually binds at the top of the ladder. A board reaches its level by
    # being oblique, and at a fixed obliquity a wider pitch needs a deeper
    # stagger -- 2 * (depth + seat) = pitch * tan(alpha). So the widest pitch
    # this envelope can carry is the one whose stagger just fills it. Without
    # this, every deep draw asked for stations 5 mm outside the band and the loop
    # rejected all of them: a level that exists, reported as a level that does
    # not.
    half_x = 0.5 * (BOARD_X[1] - BOARD_X[0])
    mid_x = 0.5 * (BOARD_X[0] + BOARD_X[1])
    tan_target = math.tan(math.radians(target / (2 * (n - 1))))
    across_pitch = 2.0 * (half_x + seat) / tan_target if tan_target > 1e-6 else float("inf")
    pitch_hi = min(BOARD_PITCH[1], reach_pitch, across_pitch)
    if pitch_hi < BOARD_PITCH[0]:
        raise RuntimeError(
            f"{task.full_name}: {n} stations do not fit the board envelope at the 70 mm "
            f"seat pitch the goal needs ({pitch_hi * 1000:.0f} mm available). "
            "Widen BOARD_Y, or this variant has more stations than this bench can hold."
        )

    for _ in range(BOARD_TRIES):
        gaps = rng.uniform(BOARD_PITCH[0], pitch_hi, size=n - 1)
        along = np.concatenate([[0.0], np.cumsum(gaps)])
        along -= along.mean()
        depth = _board_depth(float(gaps.mean()), target, n, seat)
        # What the envelope has left once the stagger has taken its share, split
        # between the three things that also move a station across the board:
        # the per-station wobble, the spread of the stagger itself, and where the
        # whole board's centroid sits. A shallow board has room for all three at
        # their full range; a deep one has almost none, and dividing what is
        # there is what lets the top of the ladder be drawn at all instead of
        # rejected six hundred times. The shares are chosen, not derived -- the
        # rejection loop is still what guarantees the result.
        slack = max(0.0, half_x - depth)
        wobble = min(BOARD_WOBBLE, 0.50 * slack)
        spread = min(BOARD_DEPTH_SPREAD * depth, 0.35 * slack)
        drift = min(0.5 * (BOARD_CENTRE_X[1] - BOARD_CENTRE_X[0]), 0.15 * slack)
        across = np.array(
            [
                sides[i] * max(0.0, rng.uniform(depth - spread, depth + spread)) + rng.uniform(-wobble, wobble)
                for i in range(n)
            ]
        )
        # HOW FAR A BOARD MAY BE TURNED IS BOUNDED BY HOW DEEP IT IS, and this
        # is the constraint that decides whether the deepest boards can be drawn
        # at all. Turning the board tilts the stagger into the along-axis
        # direction: two adjacent stations staggered `depth` either side of the
        # line come `2 * depth * sin(yaw)` closer together in y, and their seats
        # overlap the moment that eats the pitch down past 70 mm. A shallow
        # board barely notices; a 27 mm serpentine at 15 degrees loses 14 mm of
        # a 75 mm gap and every draw is thrown away. Solved rather than tried:
        # `p * cos(yaw) - 2 * depth * sin(yaw) >= 0.070`, taken at the tightest
        # gap drawn, is a bound on yaw and this is it.
        tightest = float(gaps.min())
        reach_yaw = math.atan2(max(0.0, tightest - 0.070), max(2.0 * depth, 1e-6))
        yaw_limit = min(BOARD_YAW, reach_yaw)
        yaw = rng.uniform(-yaw_limit, yaw_limit)
        cos, sin = math.cos(yaw), math.sin(yaw)
        cx = mid_x + rng.uniform(-drift, drift)
        cy = rng.uniform(*BOARD_CENTRE_Y)
        # `across` runs in +x and `along` in +y before the rotation, so a board
        # at yaw 0 is the fixed layouts' own arrangement.
        xs = cx + cos * across - sin * along
        ys = cy + sin * across + cos * along
        stations = list(zip(xs.tolist(), ys.tolist(), strict=True))

        if not (BOARD_X[0] <= xs.min() and xs.max() <= BOARD_X[1]):
            continue
        if not (BOARD_Y[0] <= ys.min() and ys.max() <= BOARD_Y[1]):
            continue
        # Seats are boxes +/- 35 mm along world y in their own station's frame,
        # so disjointness is a statement about the stations' y and nothing else.
        if float(np.diff(ys).min()) < 0.070:
            continue

        p = difficulty.profile(stations, sides, seat=seat, tail=WEAVE_TAIL)
        if not (low <= p.turning <= high):
            continue
        if p.weakest < BOARD_TURN_FLOOR:
            continue
        if rod - p.length < WEAVE_MIN_SLACK:
            continue

        placed = {name: _surface_pose(x, y, 0.0) for name, (x, y) in zip(names, stations, strict=True)}
        return {"static_poses": placed, "cable_bow": float(rng.uniform(-1.0, 1.0))}

    raise RuntimeError(
        f"{task.full_name} episode {episode_id}: no board satisfied every constraint in "
        f"{BOARD_TRIES} draws at level L{level} (turning in [{low:.0f}, {high:.0f}] deg, "
        f"{n} stations, a {rod:.2f} m rod). Widen BOARD_PITCH, BOARD_DEPTH_SPREAD or the "
        f"envelope in tasks/cable_route/staging.py, or check that the declared layout is "
        f"at the level its document claims."
    )


_weave_board.model_key_fields = ("static_poses", "cable_bow")
"""Both change the *model*: the stations are welded geometry and the bow is in
the rod's spawn. Without this the environment re-poses instead of rebuilding and
every episode of a board variant would silently be episode 0's board."""

WEAVE_BOARD = _weave_board
"""``(:sampler staging:WEAVE_BOARD)``: a board drawn fresh, to its own level."""


# ---------------------------------------------------------------------------
# weave_rigid: the declared board, moved and lightly jittered.
# ---------------------------------------------------------------------------

RIGID_YAW = (-0.26, 0.26)
"""How far the whole board may be turned about its own centroid [rad].

Fifteen degrees swings the outer station of a three-peg board about 20 mm, which
is well outside anything the perception can be lucky about and well inside the
arm's envelope at the standoffs these layouts use."""

RIGID_SHIFT_X = (-0.035, 0.020)
RIGID_SHIFT_Y = (-0.035, 0.035)
"""How far the board may be set down from its declared pose [m].

**The x window slid 25 mm toward the robot; it did not narrow.** It was
``(-0.010, +0.045)`` and is ``(-0.035, +0.020)`` -- the same 55 mm of travel, so
a policy is asked to generalise over exactly as much placement as before. What
changed is where that travel is centred.

The reason is measured rather than chosen. Over 30 vision trials, ``seat0``
failed 20 times while every other clause failed at most twice, and what predicts
it is the drag its own station asks for -- how far station 0 sits from where the
rod spawns. A logistic fit over those 30 episodes::

    P(seat0 holds) = sigmoid(9.354 - 0.0706 * drag_mm),  half at 133 mm

and when ``seat0`` holds, every other clause held in 30 of 30. So the success
rate of the whole task is this one distance, and the old window put its median at
149 mm -- the wrong side of the halfway point. The fit predicted 9.8 of 30
against 10 observed, which is what makes it worth setting a bound by.

Station 0 is the sensitive one because the board yaws about its own centroid and
station 0 sits furthest from it, so it swings widest in x.

Measured on the three-peg board at a 70 mm pitch. The stations stand 155 mm apart
now, so the drags are longer and the fit's absolute midpoint is not this board's
any more; what carries is that ``seat0`` is the clause that fails and that
bringing the board toward the robot is what helps it."""

RIGID_JITTER = (0.012, 0.060)
"""Per-station jitter on top of the rigid transform [m], as ``(x, y)`` half-widths.

**The rigid transform alone cannot vary the difficulty, by construction.**
``difficulty.accumulated_turning`` is invariant to rigid motion -- one of the two
properties it was chosen for -- so a board that only moves as one piece poses the
same weave every episode, however far it travels. Measured over 200 draws of the
rigid-only sampler, the pairwise station distances had a standard deviation of
**0.000 mm**: what varied was where to reach, never what to do once you got
there.

**12 mm across the run is the figure that used to be unusable, and the guard is
why it is not any more.** The required side of a station is 8.25 mm off its axis,
so x jitter competes directly with the alternation that makes a weave a weave --
scattered by 12 mm once before, the turning collapsed to 2.2 degrees and every
episode scored itself half-done at spawn. What was missing was not a smaller
number but a test: :func:`_weaves` now checks each draw's own route and refuses
the ones that straighten.

Along ``y`` the budget is five times larger because nothing there competes with
the alternation: jitter changes the gap between neighbouring stations, and so the
sharpness of each crossing, which is the quantity the difficulty axis measures.
:data:`MIN_STATION_GAP` is what bounds it."""

THETA_FLOOR = 8.0
"""Smallest accumulated turning a jittered board may have and still be used [deg].

A floor alone is not enough and the sign test in :func:`_weaves` is the real
guard: a route whose turns all share a sign is a hook, not a weave, however much
total turning it accumulates. Both are checked because they fail differently -- a
hook fails the sign test at high theta, and a board jittered flat fails the floor
with the signs still alternating."""

MIN_STATION_GAP = 0.060
"""Closest two stations may come to each other after jitter [m].

A seat box reaches 45 mm out along its station's required side, so two stations
closer than this share scored volume and one rod can satisfy both clauses at once
-- the episode scores itself part-done for a crossing it never made. 60 mm is the
figure :data:`~.assets.layout.SCATTER_OFFSETS` was already written to respect;
this makes it a check rather than a property of one hand-chosen layout, which is
what jitter requires."""

MIN_HAND_GAP = 0.230
"""Smallest gap allowed between the two hands' targets [m].

The grippers are about 100 mm across, so 230 mm leaves better than a hand's width
of air at the closest approach of the whole episode. Checked against the LAST
station, which is always the tightest -- the support hand anchors near the run's
free tip and stays there while the work hand advances toward it, so the gap closes
monotonically.

**IT REJECTS EVERY DRAW FOR THE PLAIN WEAVES, and that is the guard working
rather than failing.** Measured over 400 draws apiece at the 70 mm pitch, with
the estimate reading the same function the routine aims with::

    variant     too near the rod   out of reach   hands crowd   accepted
    weave3            233                0            167           0
    weave4            302               65             33           0
    weave5            367               33              0           0
    weave6            389               11              0           0
    port3             233                0              7         160

The reason is :func:`~.assets.layout.weave_anchor_y`. On a terminated run the
support hand takes the rod's own far tip and clears the board. On a plain weave
it still takes the family's original constant, +0.17, which is a hand's width
from where the work hand finishes -- so the guard is right that those boards
crowd, and it is right about every one of them.

Left standing on purpose, and only ``weave3`` among the plain weaves names this
sampler. Giving the weaves the same treatment would move where their support hand
grasps, and every score this suite has recorded for them was measured with it
where it is. That is a decision to take deliberately, with a before and after,
and not a thing to change while fixing a port.

**What it costs is stated rather than hidden: ``weave3``'s 30 episodes are 30
copies of the declared scene.** Two guards refuse every draw of it -- the hand
gap above, and the clearance floor, because its near stations stand at x = 0.517
against a floor of 0.530. :func:`_weave_rigid` says so through ``warnings.warn``
rather than silently, and the document says so too."""

SPAWN_MARGIN = 0.005
"""Extra margin between the rod's worst spawn and the nearest seat box [m]."""

SEAT_REACH = 0.045
"""How far a near-side seat box reaches back from its station [m]. The documents'
own, restated here because the clearance guard is arithmetic over it."""

RIGID_CLEAR = 0.105
"""Closest a station may come to the rod's spawn line [m], as a floor.

The near-side seat box reaches 45 mm back from its station, and the rod spawns
with a bow, a shift AND a second harmonic on top. 105 mm puts the nearest box
edge at 0.480 against a 0.42 spawn. It is a FLOOR rather than the answer:
:func:`_weave_rigid` takes the larger of this and what
:func:`spawn_excursion` measures off the spec that will build the rod."""

REACH_CEILING = ((0.42, 0.210), (0.50, 0.180), (0.52, 0.210), (0.59, 0.160))
"""The working arm's +y ceiling, as measured, against x [m].

**The ceiling falls as the arm goes out, and a flat envelope could not see it.**
This was a single ``y < 0.19`` over the whole footprint, which is true at the
station line and false at the port -- and the port is the one thing out there.
``tools/reach_bench.py --task cable_route/port3 --height 0.8035``, right arm,
10 mm tolerance::

    x      0.42    0.45    0.50    0.52    0.55    0.59
    y <=  +0.210  +0.200  +0.180  +0.210  +0.190  +0.160

Linearly interpolated between the ends of that sweep, with the measured minimum
used past it. A draw whose port lands at (0.59, +0.19) is one the flat envelope
accepted and the arm refuses, which is exactly the plug carry losing waypoints to
IK."""

REACH_MARGIN = 0.020
"""Margin held below a measured band [m]. The sweep steps y in 10 mm and the
solve tolerance is 10 mm, so the last reachable row is not a cliff edge.

**Spent on the SUPPORT band and not on the working one**, and deliberately so
rather than by symmetry. This margin was introduced for one pose -- the port,
which sits at the far corner of the bench and is checked against the support
arm's band. Subtracting it from :data:`REACH_CEILING` too would tighten the
working ceiling at the station line from +0.19 to +0.172, which is a change to
where every weave's stations may stand, made as a side effect of fixing something
else. The ceiling is used bare, so :func:`_reachable` is exactly the sweep."""

SUPPORT_FLOOR = ((0.42, -0.220), (0.50, -0.190), (0.55, -0.160), (0.59, -0.160))
SUPPORT_CEIL = 0.50
"""The SUPPORT arm's band over this bench, from the same sweep [m]::

    x      0.42    0.45    0.50    0.52    0.55    0.59
    y >=  -0.220  -0.210  -0.190  -0.180  -0.160  -0.160
    y <=  +0.550  (flat, and past anything this suite asks of it)

Mirror-imaged from the working arm's, which is what two arms on one bench look
like: the support hand owns +y and gives up -y at the same rate the work hand
gives up +y. Only the FLOOR binds anything here -- the port sits at y = +0.18 --
so the ceiling is held at +0.50 as a sanity bound rather than as a measurement
being leaned on."""


def spawn_excursion(variant: str) -> float:
    """How far the rod's spawn can stray toward the stations [m].

    **All four modes, not just the bow, and that is the correction.** The
    clearance floor was a hand-set constant checked against the bow alone. The
    rod is displaced laterally by ``bow*sin(pi t) + wave*sin(2 pi t)`` plus a
    constant ``shift``, and those do not simply add up to the largest of them:

    * The two harmonics peak at different ``t`` and reinforce in between.
      Measured for the shipped (20 mm, 30 mm): the peak is **44.9 mm**, not 30.
    * ``shift`` is constant along the rod, so it adds to that peak in full.

    So the honest bound is computed, once, from the spec that will build the rod,
    rather than asserted from one of its terms.
    """
    from .assets.layout import PLACEMENTS  # noqa: PLC0415

    spec = PLACEMENTS[variant]["cable"].build()

    def half(rng) -> float:
        lo, hi = (float(v) for v in rng)
        return max(abs(lo), abs(hi))

    t = np.linspace(0.0, 1.0, 2001)
    peak = float(
        np.max(
            np.abs(
                half(getattr(spec, "bow_range", (0.0, 0.0))) * np.sin(np.pi * t)
                + half(getattr(spec, "wave_range", (0.0, 0.0))) * np.sin(2.0 * np.pi * t)
            )
        )
    )
    return peak + half(getattr(spec, "shift_range", (0.0, 0.0)))


def _ceiling(x: float) -> float:
    """The working arm's +y limit at this x [m]; see :data:`REACH_CEILING`."""
    return float(np.interp(float(x), [c[0] for c in REACH_CEILING], [c[1] for c in REACH_CEILING]))


def _reachable(points) -> bool:
    """Is every point inside the working arm's measured envelope?

    The envelope is the one this bench was characterised against rather than a
    solver call -- see :data:`REACH_CEILING`, which is where the sweep is written
    down. The -y side is limited by the table rather than by the arm.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if not ((p[:, 0] > 0.44).all() and (p[:, 0] < 0.62).all() and (p[:, 1] > -0.30).all()):
        return False
    return bool(all(float(y) < _ceiling(float(x)) for x, y in p))


def _reachable_support(points) -> bool:
    """Is every point inside the SUPPORT arm's measured envelope?"""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if not ((p[:, 0] > 0.40).all() and (p[:, 0] < 0.62).all() and (p[:, 1] < SUPPORT_CEIL).all()):
        return False
    xs = [c[0] for c in SUPPORT_FLOOR]
    ys = [c[1] for c in SUPPORT_FLOOR]
    return bool(all(float(y) > float(np.interp(float(x), xs, ys)) + REACH_MARGIN for x, y in p))


def _anchor_xy_est(task) -> tuple[float, float]:
    """Where the support hand ends up [m], read off the thing that puts it there.

    **Planar, not just y.** The two hands are separated in x as well: the support
    holds the cable on its spawn line while the work hand is out at the station
    line, which is about 100 mm further out. Comparing y alone throws that away
    and reads 261 mm of real separation as 240, which is tight enough to reject
    draws that are in fact fine -- and a sampler that rejects everything falls
    back to the declared board.

    **This was a written-down pair and the pair was fiction.** It said
    ``(0.42, 0.30)`` -- "the anchor grasp is near the run's free tip, which
    settles around (0.42, +0.30) with the current rod" -- while
    ``plan_support_stages`` was aiming the hand at ``WEAVE_ANCHOR_Y`` = +0.17,
    130 mm short of it. So :data:`MIN_HAND_GAP` measured every draw against a
    hand that was never there, and the guarantee in its own note was never once
    enforced. A constant that no code computes is a constant nothing can keep
    honest.

    An estimate rather than a solve: this runs inside a rejection loop that may
    execute dozens of times per episode. But it is now the estimate of the SAME
    number the routine aims at, from the same function, so the two cannot drift
    apart again without the drift being a code change.
    """
    return (float(WEAVE_CABLE_X), float(weave_anchor_y(task.name)))


_LEVELS: dict[str, int | None] = {}
"""Declared level per variant, memoised. See :func:`_declared_level`."""


def _declared_level(task) -> int | None:
    """The level this variant's DECLARED board measures, or ``None`` if unknown.

    Read from the layout rather than stated, for the same reason
    :func:`~.assets.layout.seat_offset` is a function: a level restated beside
    the thing it describes goes stale the first time the geometry moves.

    **Memoised, because it is asked inside the draw loop.** ``measured()``
    profiles *every* layout in the suite on each call, and this is reached up to
    2048 times per episode -- so uncached it turned a sampler that was arithmetic
    on five points into one that re-measured thirteen boards a thousand times.
    The declared layouts are module constants; nothing about the answer can
    change within a process.
    """
    name = getattr(task, "name", "")
    if name not in _LEVELS:
        profiles = difficulty.measured()
        _LEVELS[name] = profiles[name].level if name in profiles else None
    return _LEVELS[name]


def _weaves(stations, task, *, floor: float = THETA_FLOOR) -> bool:
    """Whether a jittered board still poses a weave rather than a hook.

    ``floor`` is the least accumulated turning a draw may have, defaulting to
    :data:`THETA_FLOOR`. It is a parameter because that constant is a floor for
    *jittered* boards, and one tier's declared board turns less than it:
    ``port_easy`` measures 7.9 degrees, so a sampler drawing beside it has to be
    allowed to draw the board it stands next to. A tier's own floor is stated
    relative to what it declares -- see :func:`_port_board`.

    Asked of the SCORED ROUTE, not of the station centres: what the policy has to
    install is the polyline through each station's required side, and that is
    what ``difficulty.route`` builds. Four ways a draw can stop being the task:

    * **Two stations come too close**, and their seat boxes overlap -- see
      :data:`MIN_STATION_GAP`. One rod then satisfies two clauses at once.
    * **The turns go flat.** Jitter that happens to line the stations up with
      their own seat offsets straightens the route. A spool station's seat is
      8.25 mm off its axis and :data:`RIGID_JITTER` allows 12 mm in x, so a draw
      can cancel the alternation outright, and every episode then scores itself
      half-done at spawn.
    * **The draw is not the level its document is.** Jitter does not respect the
      band on its own: measured over 200 draws of ``panel_l2``, the accumulated
      turning ranged 50 to 247 and 47 of the 200 episodes were not L2 at all --
      31 at L1 and 16 at L3. A benchmark reporting "L2: 4 of 30" over that
      mixture is reporting the sampler. The band is checked against the level the
      DECLARED layout measures rather than a number written here.
    * **The turns stop alternating.** A route bending one way throughout is a
      hook. It can carry plenty of accumulated turning while asking none of the
      over-and-under the task exists to pose, so a magnitude test alone lets it
      through -- the sign test is the one that catches it.

    Rejecting a draw costs a line of arithmetic; discovering any of these in a
    run costs the episode and reads like a policy failure.
    """
    st = np.asarray(stations, dtype=float)
    for a in range(len(st)):
        for b in range(a + 1, len(st)):
            if float(np.linalg.norm(st[a] - st[b])) < MIN_STATION_GAP:
                return False

    seat = seat_offset(task.name)
    poly = difficulty.route(
        [(float(x), float(y)) for x, y in stations],
        WEAVE_SIDES,
        seat=seat,
        tail=WEAVE_TAIL,
    )
    theta = difficulty.turns(poly)
    total = sum(abs(v) for v in theta)
    if total < floor:
        return False
    declared = _declared_level(task)
    if declared is not None and difficulty.level(total) != declared:
        return False
    # The interior turns are the crossings; the two tail turns are the run
    # arriving and leaving, and they are not part of the alternation.
    inner = [v for v in theta[1:-1] if abs(v) > 1.0e-6]
    return all(a * b < 0.0 for a, b in itertools.pairwise(inner))


def _cable_start_frame(task, draw: dict[str, float]) -> tuple[tuple[float, float], float]:
    """Where the first rod point and tangent land after spawn variation.

    Mirrors :func:`~robosimstudio.domains.rod.entity._bow_cable` for the one
    point and direction the anchor bracket has to follow. Bow and wave vanish
    at ``t = 0``, but their derivatives do not, so the endpoint tangent is
    reconstructed from the first two nodes too.

    **Every range is read off the spec that will actually build the rod.** This
    read the module-level constants directly, and a variant that narrowed one of
    them got a bracket placed for a rod it did not build. Asking the spec is the
    only version that cannot say something the builder will not do.

    **It returns the TIP, not the first body**, and an older docstring said
    otherwise while the arithmetic returned the tip all along.
    :func:`~.assets.layout.weave_terminals` is what turns a tip into a bracket
    pose, through ``WEAVE_MOUTH_INSET``, and it needs the tip to do it.

    **In WORLD, because that is the frame ``weave_terminals`` speaks.**
    ``RodSpec.start`` is in the cell's work-surface frame -- ``weave_cable``
    puts the reference's world numbers through ``point_from_surface`` at the one
    point they become a spec -- so reading it raw and handing it on lands the
    bracket a bench origin away from its cable. Measured on ``port3``: the
    anchor came out at x = -0.169 against a rod tip at 0.433, which is exactly
    the 0.6 m the surface frame is offset by. Lifted here, once, so the two ends
    reach ``_surface_pose`` in the same frame the stations do.
    """
    from .assets.layout import PLACEMENTS  # noqa: PLC0415

    spec = PLACEMENTS[task.name]["cable"].build()
    world = in_surface(CELL, (*(float(v) for v in spec.start[:3]), 0.0, 0.0, 0.0, 1.0))
    sx, sy = (float(world[0]), float(world[1]))
    lo, hi = (float(v) for v in spec.shift_range)
    off = 0.5 * (lo + hi) + 0.5 * (hi - lo) * float(draw["cable_shift"]) if abs(hi - lo) > 0.0 else 0.0

    # THE LATERAL AXIS IS DERIVED, not asserted, and that was a 30 mm bug.
    #
    # This used to read "it displaces along the LATERAL axis -- +x, for a rod
    # laid along +y" and add ``+off`` to x. ``_bow_cable`` builds the axis as
    # ``cross(up, direction)``, which for a rod along +y is **(-1, 0, 0)**. So
    # the prediction moved the tip one way and the builder moved it the other,
    # and the bracket landed at twice the shift from the rod it holds -- 29.6 mm
    # at the range's edge, off a plate 16 mm wide. Computing the axis the same
    # way the builder does is what stops the two drifting apart again.
    axis = np.asarray([float(v) for v in spec.direction], dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    lateral = np.cross([0.0, 0.0, 1.0], axis)
    if float(np.linalg.norm(lateral)) < 1.0e-9:
        lateral = np.cross([1.0, 0.0, 0.0], axis)
    lateral /= max(float(np.linalg.norm(lateral)), 1.0e-12)

    blo, bhi = (float(v) for v in getattr(spec, "bow_range", (0.0, 0.0)))
    amp = 0.5 * (blo + bhi) + 0.5 * (bhi - blo) * float(draw["cable_bow"]) if abs(bhi - blo) > 0.0 else 0.0
    wlo, whi = (float(v) for v in getattr(spec, "wave_range", (0.0, 0.0)))
    wave = 0.5 * (wlo + whi) + 0.5 * (whi - wlo) * float(draw["cable_wave"]) if abs(whi - wlo) > 0.0 else 0.0
    t = np.linspace(0.0, 1.0, int(spec.segments) + 1)
    offsets = off + amp * np.sin(np.pi * t) + wave * np.sin(2.0 * np.pi * t)
    points = (
        np.array([sx, sy], dtype=float)[None, :]
        + float(spec.length) * t[:, None] * axis[:2][None, :]
        + offsets[:, None] * lateral[:2][None, :]
    )
    ylo, yhi = (float(v) for v in spec.yaw_range)
    if abs(yhi - ylo) <= 0.0:
        tangent = points[1] - points[0]
        heading = math.atan2(float(tangent[1]), float(tangent[0])) - math.pi / 2.0
        return (float(points[0, 0]), float(points[0, 1])), heading
    ang = math.radians(0.5 * (ylo + yhi) + 0.5 * (yhi - ylo) * float(draw["cable_yaw"]))

    # THE PIVOT IS THE MEAN OF THE BOWED POINTS, which is not the mean of the
    # straight ones. ``_bow_cable`` turns about ``bowed.mean(axis=0)``, and the
    # bow's own profile ``sin(pi t)`` has mean ``2/pi`` -- so a bowed rod's
    # centroid sits about 0.64 of the amplitude off the spawn line, and pivoting
    # about the straight midpoint instead leaves a residual that grows with the
    # bow. The second harmonic is antisymmetric and contributes nothing here.
    cx, cy = (float(v) for v in points.mean(axis=0))
    c, s_ = math.cos(ang), math.sin(ang)
    rel = points - np.array([cx, cy])
    turned = np.stack(
        [cx + rel[:, 0] * c - rel[:, 1] * s_, cy + rel[:, 0] * s_ + rel[:, 1] * c],
        axis=1,
    )
    tangent = turned[1] - turned[0]
    heading = math.atan2(float(tangent[1]), float(tangent[0])) - math.pi / 2.0
    return (float(turned[0, 0]), float(turned[0, 1])), heading


def _cable_start(task, draw: dict[str, float]) -> tuple[float, float]:
    """The sampled first node, for point-only callers."""
    return _cable_start_frame(task, draw)[0]


def _weave_rigid(task, episode_id: int, rng) -> dict[str, Any]:
    """The DECLARED board, moved as one piece and lightly jittered.

    **The difference from** :data:`WEAVE_BOARD` **is the whole point.** That
    sampler draws a new arrangement every episode -- new gaps, new staggers, new
    wobble -- so a score over it is a score on a *level*. This one keeps the
    board as the document declares it, relative poses and all, and moves the
    whole thing: a yaw about its own centroid, a translation, and a small
    per-station jitter on top. What varies is mostly where the board IS.

    That is the generalisation a real cell asks for first. The fixture is bolted
    together and does not change; where it gets put down on the bench does. A
    policy that has to re-perceive the board's pose every episode, and cannot
    carry a station coordinate over from the last one, is being asked the
    question this samples.

    The rod's own shape varies alongside, through the same four fractions
    :data:`CABLE_BOW` emits, so the cable and the board move independently. On a
    terminated run both fittings follow what they terminate: the anchor bracket
    follows the CABLE (its tip moves with the shift and the yaw) and the port
    follows the BOARD, both through
    :func:`~.assets.layout.weave_terminals`, which is also what places the
    declared poses -- so the moved scene and the declared one are the same
    arithmetic over different inputs rather than two arithmetics.

    Every draw is checked before it is accepted: still a weave, every station
    inside the working arm's envelope, no near-side seat box dragged back onto
    the rod's own spawn, the two hands not crowding, and -- on a terminated run
    -- the port inside the SUPPORT arm's envelope, because the support hand is
    what carries the end into it.

    Returns ``{}`` when no draw satisfies everything, which stages the DECLARED
    board. That is a legitimate thing to fall back to and an illegitimate thing
    to do quietly, so it warns once per variant; see :data:`MIN_HAND_GAP`, which
    is the guard that does it and the variant it does it to.
    """
    stations = list(WEAVE_LAYOUTS.get(task.name) or ())
    if not stations:
        return {}
    base = np.asarray(stations, dtype=float)
    centre = base.mean(axis=0)
    # DERIVED FROM THE ROD THAT WILL BE BUILT, not from a constant. The seat
    # box's 45 mm reach plus everything the spawn can do, or the floor,
    # whichever is larger -- see `spawn_excursion` and `RIGID_CLEAR`.
    clear = max(RIGID_CLEAR, SEAT_REACH + spawn_excursion(task.name) + SPAWN_MARGIN)

    # ATTEMPTS, and why there are so many of them.
    #
    # An exhausted budget returns ``{}``, which the loader reads as "no
    # overrides" and stages the DECLARED board. So the cost of a tight guard set
    # is not a slow sampler, it is a run in which some episodes are not the
    # distribution at all but the same fixed layout.
    #
    # Measured while tuning the placement bounds: a set that rejected ~3% of
    # draws per attempt left 12% of EPISODES on the declared board, because
    # ``0.97 ** 64`` is 0.12. The guard set has since grown a level-band test
    # that refuses about a quarter of ``panel_l2`` draws, and at 512 attempts
    # that still left 1 episode in 200 on the declared board. 2048 takes it to
    # 4e-10. An attempt is arithmetic on five points; there is no reason to be
    # stingy with them, and every reason not to be.
    for _ in range(2048):
        yaw = float(rng.uniform(*RIGID_YAW))
        dx = float(rng.uniform(*RIGID_SHIFT_X))
        dy = float(rng.uniform(*RIGID_SHIFT_Y))
        draw = {
            "cable_bow": float(rng.uniform(-1.0, 1.0)),
            "cable_shift": float(rng.uniform(-1.0, 1.0)),
            "cable_yaw": float(rng.uniform(-1.0, 1.0)),
            "cable_wave": float(rng.uniform(-1.0, 1.0)),
        }
        c, s_ = math.cos(yaw), math.sin(yaw)
        rel = base - centre
        moved = np.stack(
            [
                centre[0] + dx + rel[:, 0] * c - rel[:, 1] * s_,
                centre[1] + dy + rel[:, 0] * s_ + rel[:, 1] * c,
            ],
            axis=1,
        )
        # PER-STATION JITTER, on top of the rigid move and before every guard
        # below, so a jittered board is checked for reach, clearance and hand gap
        # exactly like a rigidly moved one. See RIGID_JITTER for the budgets.
        jx, jy = RIGID_JITTER
        moved = moved + np.stack(
            [rng.uniform(-jx, jx, len(moved)), rng.uniform(-jy, jy, len(moved))],
            axis=1,
        )
        # STILL A WEAVE? The jitter is small in x on purpose, but "small" has to
        # be demonstrated per draw rather than argued once: a board is only a
        # weave while the route's turns keep alternating.
        if not _weaves(moved, task):
            continue
        # A station the arm cannot reach is not a harder episode, it is a broken
        # one; and a near-side seat that has come back to the rod's own spawn
        # scores before the robot moves.
        if moved[:, 0].min() < WEAVE_CABLE_X + clear:
            continue
        if not _reachable(moved):
            continue
        # THE TWO HANDS MUST NEVER CROWD EACH OTHER, and that is a property of
        # the DRAW rather than something to fix at run time. The support hand
        # anchors near the run's free tip and stays there; the work hand advances
        # station by station toward it, so the LAST station is always the worst
        # case.
        far = moved[int(np.argmax(moved[:, 1]))]
        anchor_est = _anchor_xy_est(task)
        if float(math.hypot(far[0] - anchor_est[0], far[1] - anchor_est[1])) < MIN_HAND_GAP:
            continue
        placed = {f"peg{i}": _surface_pose(float(x), float(y), yaw) for i, (x, y) in enumerate(moved)}
        if task.name in WEAVE_TERMINALS:
            # THE PORT HAS TO BE REACHABLE TOO, and it is not a station.
            #
            # A terminated layout puts its port beyond the LAST station, so a
            # board shifted away from the robot carries the port further still.
            # Measured: with the board out at its +45 mm limit the port sat at
            # x = 0.63 and the plug carry lost four of its five waypoints to IK
            # refusals. ``_reachable`` cannot see this because the port is not in
            # ``moved``, and the band it would check is the wrong arm's anyway:
            # the SUPPORT hand carries the end into the port.
            #
            # ``weave_terminals`` ITSELF, not the same offset written out again.
            # These two had drifted apart once already, were re-synced by hand,
            # and drifted again the next time the standoff moved.
            cable_start, cable_heading = _cable_start_frame(task, draw)
            terminals = weave_terminals(
                task.name,
                start_xy=cable_start,
                stations=moved,
                yaw=yaw,
                anchor_yaw=cable_heading,
            )
            if not _reachable_support([terminals["port"][:2]]):
                continue
            for name, (tx, ty, tyaw) in terminals.items():
                placed[name] = _surface_pose(tx, ty, tyaw)
        return {"static_poses": placed, "board_yaw": yaw, "board_dx": dx, "board_dy": dy, **draw}

    # SAY SO, once. A warning rather than a print: this is library code inside a
    # reset, and a line on stdout per episode is a line nothing can filter,
    # capture or count. ``warnings`` dedupes per location by default, so a
    # variant whose every draw is rejected says it once and the run stays
    # readable.
    warnings.warn(
        f"{task.name}: 2048 draws all rejected, so this episode stages the DECLARED board and is "
        f"not a sample of the distribution. See MIN_HAND_GAP in tasks/cable_route/staging.py.",
        stacklevel=2,
    )
    return {}


_weave_rigid.model_key_fields = (
    "static_poses",
    "cable_bow",
    "cable_shift",
    "cable_yaw",
    "cable_wave",
)
"""Every key here changes the model: the stations and the terminals are welded
geometry, and the four fractions are in the rod's spawn."""

WEAVE_RIGID = _weave_rigid
"""``(:sampler staging:WEAVE_RIGID)``: the declared board, put down somewhere else."""


# ---------------------------------------------------------------------------
# port_board: the terminated tiers' pegs, drawn between their own two terminals.
# ---------------------------------------------------------------------------

PORT_END_GAP = 0.070
"""The least a station may stand from either terminal, along the run [m].

**Both terminals hold their station and the pegs move between them**, which is
the whole shape of this sampler. The anchor follows the cable's own draw and the
port is drawn in its own bolt slots (:data:`PORT_SLOT_XY`), so the run's ends
move by millimetres while the crossings move by centimetres -- and this is how
much of the corridor between them is not available, at each end. It is measured
against the DRAWN terminals, so a port that shifted 6 mm toward the board takes
its clearance with it rather than lending it to a peg.

Seventy millimetres is the seat's own number rather than a taste: a seat box
reaches +/- 35 mm along the run in its station's frame, so a station closer than
this to the anchor bracket has its scored volume overlapping the fitting the rod
is welded into, and the rod satisfies the crossing at spawn. The declared boards
stand 92 mm from the anchor and 100 mm from the port, so this is inside what the
tiers already ship rather than a widening of them.

What the DECLARED terminals leave is 292 mm of corridor on Easy (anchor at
y = -0.352, port at +0.080) and 286 mm on Medium (-0.412 and +0.020, clipped by
the arm's own envelope at -0.336), against declared spans of 240 mm. A drawn
port moves either figure by at most :data:`PORT_SLOT_XY`."""

PORT_SLOT_XY = 0.006
"""How far the port may be drawn from its declared position, per axis [m].

**The port is bolted, not machined in, and this is the slop in its own clearance
slots.** It is the routing tier's version of what ``usbc_insert/insert_easy``
already does with its receptacle: a fitting that stands exactly where it stood
last episode is one a policy can be right about without ever having localized it,
and on a fixed-camera bench that is the difference between a perception result
and a replayed constant. The terminal plate is found by colour and by standing
10 mm off the bench (``graphs/weave/scripts/fixtures_cv.py``), so it is already
measured every episode -- this is what makes the measurement load-bearing.

Six millimetres rather than the peg wobble's five or the receptacle's own five,
and small on purpose: the point is to move the port off its remembered
coordinate, not to make the reach into it a different task. The goal region rides
the fixture (``portbox`` is ``(:target port)``), so a drawn port moves the scored
volume with it and the tolerance the route is judged at is unchanged.

Independently uniform per axis, composed with the declared pose. Drawn from the
episode's own stream before the board is, so the corridor the pegs are drawn into
is the one this port actually leaves."""

PORT_SLOT_YAW = 0.052
"""How far the port may be turned in those slots [rad].

Three degrees, the same figure ``insert_easy`` draws its receptacle's yaw in and
for the same reason -- a bolt slot lets a plate rotate as well as slide. Against
:data:`PORT_YAW`'s six for the whole board: a board turns about its own centroid
and its outer station swings 12.5 mm, where a plate 90 mm across turns about
itself and its mouth swings under 3 mm.

It is composed with the port's DECLARED yaw rather than replacing it: the channel
faces back down the run at ``pi`` plus the board's own angle
(:func:`~.assets.layout.weave_terminals`), and a draw that dropped that would
turn the mouth to face the bench."""

PORT_PITCH_BAND = (0.70, 1.30)
"""How far a drawn gap may sit from the tier's own declared pitch [fraction].

Per gap rather than per board, so a four-peg Medium draw is unevenly spaced
rather than a rescaled copy of its declared board -- which is the property that
stops a policy carrying one standoff across the whole route.

The band is stated against the DECLARED pitch instead of the family's
:data:`BOARD_PITCH` because the two tiers are 240 mm and 80 mm apart, and one
absolute range cannot be both. Its floor still runs into the seats: 70% of
Medium's 80 mm pitch is 56 mm, under the 70 mm two seats need, so the gap draw
is clamped there and the corridor caps the top."""

PORT_WOBBLE = 0.005
"""Per-station cross-run wobble [m].

Five millimetres, and the reason is :data:`~.assets.layout.SCATTER_OFFSETS`'
reason: a seat sits 8.25 mm off its post and alternates side by side, so an x
offset approaching that number can cancel the alternation and quietly straighten
the route. The turning band below would catch such a draw; keeping the wobble
well under the seat is what stops the loop spending its budget generating them.

Smaller than :data:`BOARD_WOBBLE` on purpose. A drawn board reaches its level by
being oblique and can afford to wobble inside that; the Easy tier's entire
turning is two seat widths over a 240 mm pitch, and 9 mm either way is most of
it."""

PORT_DEPTH_SPREAD = 0.35
"""How far each station's stagger may vary from the tier's declared depth
[fraction].

Zero for Easy, whose two pegs stand at one x -- a fraction of nothing is
nothing, and Easy's cross-run variation is the wobble and the board yaw. Medium
declares 30 mm of stagger and draws it in [19.5, 40.5], which changes each
crossing's obliquity without changing which side of its post the rod is
required to pass."""

PORT_YAW = 0.10
"""How far the whole drawn board may be turned about its centroid [rad].

Six degrees, against :data:`RIGID_YAW`'s fifteen, and the difference is the
corridor. These boards span 240 mm between two fixed fittings; at 15 degrees the
outer station of an Easy draw swings 31 mm across the bench, which walks the
board out of :data:`BOARD_X` and is rejected anyway. Six degrees swings it 12.5
mm -- enough that the two crossings are approached from different directions,
inside what the envelope will take.

The stations rotate and the seats do not: a station is placed unrotated (see
:func:`_surface_pose`), so a seat's offset stays along world x however the board
is turned, and the turning that costs is measured per draw rather than
predicted."""

PORT_TURN_RATIO = 2.0
"""How far a draw's accumulated turning may sit from the declared board's
[factor].

**This is what keeps a tier the tier it was calibrated as.** The suite's other
samplers draw into a difficulty LEVEL, which is right when the level is the
thing being reported; these two documents are the Easy and Medium rows of a
results table whose rates were tuned against their declared boards, and L1 spans
0 to 90 degrees -- a board drawn anywhere in it would be a different experiment
from the one the numbers came from.

So a draw is measured against the DECLARED board rather than against the band:
Easy turns 7.9 degrees and draws in [3.9, 15.8]; Medium turns 262.3 and draws in
[131.1, 524.6], which its level test then cuts to L3's own 180-360. Both stay at
their declared level -- ``_weaves`` checks that on every draw -- and neither
wanders to the far side of a band it shares with nothing.

Two, rather than something tighter, because the corridor is what limits the
variation here and a narrow ratio would leave the pegs able to slide and not much
else. Measured over 400 draws per tier: the accepted turning runs 4.6-14.6
degrees on Easy and 198-321 on Medium."""

PORT_WRAP_FRACTION = 0.5
"""The least a station may be wrapped by, as a fraction of the weakest crossing
the declared board has.

:data:`BOARD_TURN_FLOOR` is 9 degrees and Easy's declared board wraps its
stations by 3.9, so the family's absolute floor would refuse the very board this
draws beside. Relative to the tier, it says the thing that floor exists to say:
no draw may meet its total with a station the route effectively walks past."""

PORT_SLACK_ALLOWANCE = 0.020
"""How much of the tier's own spare rod a draw may spend [m].

The rod is 600 mm on Easy against a declared anchor-to-port route of 487, and
720 against 574 on Medium -- so each tier ships with a spare length its rates
were measured at, and a draw that eats it is a harder episode for a reason
nothing in the tier's description mentions. Twenty millimetres is
:data:`WEAVE_MIN_SLACK`, spent here as an allowance on the declared route rather
than as an absolute floor, because what matters is the change from the board the
tier was calibrated on.

Measured through the SAME terminals on both sides of the comparison -- the
episode's own drawn ones. Both fittings move a little every episode already, the
anchor with the cable's draw and the port in its bolt slots, and letting either
motion into only one side of this subtraction would make it a statement about the
rod's spawn or the port's slop rather than about the board."""

PORT_TRIES = 4096
"""Draws before a board is given up on. See :func:`_port_board`."""


def _port_route_length(stations, task, terminals) -> float:
    """The run's length from the anchor, through every seat, into the port [m].

    :func:`~tasks.cable_route.difficulty.profile` measures the route with two
    free tails, which is the right thing for a board that stands on its own and
    the wrong one here: on a terminated tier the rod does not trail off past the
    outermost station, it is welded at one end and has to reach a fitting at the
    other. The tails would say a board that slid 40 mm toward the port costs the
    same rod as one that did not, and it does not.
    """
    seat = seat_offset(task.name)
    seated = [(x + seat * WEAVE_SIDES[i % len(WEAVE_SIDES)], y) for i, (x, y) in enumerate(stations)]
    poly = [terminals["anchor"], *seated, terminals["port"]]
    return float(sum(math.dist(a, b) for a, b in itertools.pairwise(poly)))


def _port_board(task, episode_id: int, rng, *, port_reach_inset: float = 0.0, stagger_scale: float = 1.0) -> dict[str, Any]:
    """The pegs of a terminated tier, drawn fresh between its two fixed terminals.

    **What varies is where the crossings are, and what does not is the journey
    they are in.** The anchor bracket is where the cable's own draw puts it and
    the port is drawn in the slop of its own bolt slots, so every episode poses
    the same journey -- from this weld, into that receptacle -- over a board that
    is not the same twice: the gap between each pair of pegs, each peg's stagger
    off the line, a wobble on top of that, the whole board's rotation, and where
    it sits along the corridor between the two fittings.

    **Why the terminals move by millimetres and the board by centimetres.** A
    drawn board that carried its port with it (:data:`WEAVE_RIGID` does exactly
    that, and correctly) varies the whole cell, and what a fixed-camera
    evaluation then reports is partly how far the port went. But a port that
    never moves at all is worse in the other direction: it is a coordinate a
    policy can be right about without having looked, and every rate measured
    against it is partly a replay. :data:`PORT_SLOT_XY` is the middle -- 6 mm and
    3 degrees, which is a bolted plate's own slop, enough that the mouth has to be
    found from the image every episode and far too little to make the reach into
    it a different task. The anchor is the same bargain, already: it follows the
    rod rather than standing still.

    **The corridor and the run are measured through the DRAWN port**, not the
    declared one, so a port that shifted toward the board hands the pegs a
    shorter corridor rather than a free 6 mm. The slack test compares a drawn
    board against the declared board through those same drawn terminals: what it
    is asking is whether the BOARD spent the tier's spare rod, and putting the
    port on both sides of that subtraction is what keeps it asking only that.

    **Every draw is measured before it is accepted**, against the tier's own
    declared board rather than against a table written here: the turning within
    :data:`PORT_TURN_RATIO` of it and at its level, no station wrapped less than
    :data:`PORT_WRAP_FRACTION` of its weakest crossing, the seats disjoint, every
    station inside the working arm's measured envelope and clear of the rod's own
    bowed spawn, the two hands not crowding, every peg inside the corridor
    :data:`PORT_END_GAP` leaves between the fittings, and the anchor-to-port run
    no longer than the declared one plus :data:`PORT_SLACK_ALLOWANCE`.

    Emits ``static_poses`` for the pegs on top of everything :data:`CABLE_BOW`
    emits -- the four spawn fractions and the anchor that follows them -- so the
    board and the cable in front of it vary independently. All of it is
    model-changing.

    Raises:
        RuntimeError: If no draw satisfied every constraint. Loud rather than
            silent, for :func:`_weave_board`'s reason: an episode quietly staged
            on the declared board is one the results table counts as varied.
    """
    if task.name not in WEAVE_TERMINALS:
        raise RuntimeError(f"{task.full_name} has no terminals; a terminated board needs an anchor and a port")
    fixtures = {spec.name: _in_world(spec.xform) for spec in task.statics}
    if port_reach_inset:
        x, y = fixtures["port"]
        fixtures["port"] = (x - float(port_reach_inset), y)
    pegs = [name for name in fixtures if name.startswith("peg")]
    n = len(pegs)
    if n < 2:
        raise RuntimeError(f"{task.full_name} has {n} pegs; a terminated board needs at least two")
    sides = [WEAVE_SIDES[i % len(WEAVE_SIDES)] for i in range(n)]
    seat = seat_offset(task.name)
    rod = WEAVE_CABLE_LENGTH[task.name]

    # THE CABLE FIRST AND THE PORT SECOND, so both are the same function of the
    # seed whether the board loop below accepts on its first attempt or its four
    # hundredth. The cable draw also places the anchor, which follows the rod.
    cable = _cable_bow(task, episode_id, rng)
    port_xy = (
        fixtures["port"][0] + float(rng.uniform(-PORT_SLOT_XY, PORT_SLOT_XY)),
        fixtures["port"][1] + float(rng.uniform(-PORT_SLOT_XY, PORT_SLOT_XY)),
    )
    port_yaw = _surface_yaw(next(spec.xform for spec in task.statics if spec.name == "port")) + float(
        rng.uniform(-PORT_SLOT_YAW, PORT_SLOT_YAW)
    )
    terminals = {"anchor": fixtures["anchor"], "port": port_xy}

    # THE DECLARED BOARD IS THE REFERENCE FOR EVERY BAND BELOW, read off the task
    # rather than restated here: its pitch, its stagger, its turning, its weakest
    # crossing and the length of its run. A tier that is re-pitched or re-staggered
    # then moves this sampler with it, which is the property a number written down
    # in this file would not have.
    declared = [fixtures[name] for name in pegs]
    reference = difficulty.profile(declared, sides, seat=seat, tail=WEAVE_TAIL)
    declared_run = _port_route_length(declared, task, terminals)
    ys = [y for _, y in declared]
    pitch = float(np.mean(np.diff(ys))) if n > 1 else 0.0
    mid_x = float(np.mean([x for x, _ in declared]))
    depth = float(np.mean([abs(x - mid_x) for x, _ in declared])) * float(stagger_scale)
    low = reference.turning / PORT_TURN_RATIO
    high = reference.turning * PORT_TURN_RATIO

    # The corridor: what is left of the run once both fittings have taken their
    # clearance, and never wider than the arm's own envelope.
    corridor = (
        max(BOARD_Y[0], terminals["anchor"][1] + PORT_END_GAP),
        min(BOARD_Y[1], terminals["port"][1] - PORT_END_GAP),
    )
    span_limit = corridor[1] - corridor[0]
    gap_lo = max(0.070, PORT_PITCH_BAND[0] * pitch)
    gap_hi = min(PORT_PITCH_BAND[1] * pitch, span_limit / (n - 1))
    if gap_hi < gap_lo:
        raise RuntimeError(
            f"{task.full_name}: {n} pegs do not fit the {span_limit * 1000:.0f} mm corridor between "
            f"its terminals at the 70 mm seat pitch the goal needs. Narrow PORT_END_GAP, or this "
            "tier has more pegs than its run can hold."
        )
    # THE CROSS-RUN WINDOW IS THE TIER'S OWN NEAR ROW AND THE ARM'S OWN FAR EDGE.
    #
    # Not `RIGID_CLEAR` and not `_reachable`, and that is measured rather than
    # preferred: both are the bare weaves' constants and neither shipped tier
    # satisfies them. The spawn clearance puts a floor at x = 0.530 where Easy
    # declares 0.525 and Medium 0.505; `_reachable` wants every station at
    # y > -0.30 where Medium declares a peg at -0.32. A guard that refuses the
    # board it is drawing beside does not protect the draw, it empties it -- 4096
    # rejections per episode, measured.
    #
    # So the near edge is where this tier already stands: a draw may not bring a
    # station -- or the near-side seat box that reaches 45 mm back from it --
    # closer to the rod's spawn than the document itself does. The far edge is
    # BOARD_X's, which is the working arm's measured envelope and is the one
    # constant here both tiers are comfortably inside.
    x_window = (max(BOARD_X[0], min(x for x, _ in declared)), BOARD_X[1])

    for _ in range(PORT_TRIES):
        gaps = rng.uniform(gap_lo, gap_hi, size=n - 1)
        along = np.concatenate([[0.0], np.cumsum(gaps)])
        along -= along.mean()
        across = np.array(
            [
                sides[i] * rng.uniform(depth * (1.0 - PORT_DEPTH_SPREAD), depth * (1.0 + PORT_DEPTH_SPREAD))
                + rng.uniform(-PORT_WOBBLE, PORT_WOBBLE)
                for i in range(n)
            ]
        )
        yaw = float(rng.uniform(-PORT_YAW, PORT_YAW))
        cos, sin = math.cos(yaw), math.sin(yaw)
        # `across` runs in +x and `along` in +y before the rotation, so a draw at
        # yaw 0 is the declared board's own arrangement.
        shape_x = cos * across - sin * along
        shape_y = sin * across + cos * along
        # THE CENTROID IS DRAWN FROM WHAT THE BOARD'S OWN WIDTH LEAVES, rather
        # than drawn from the middle and rejected when it does not fit. The two
        # differ in the distribution as well as the cost: a wide board rejected
        # at the edges is a board whose accepted draws cluster in the centre, and
        # a corridor 292 mm long holding a 240 mm board has little enough travel
        # already without half of it being lost to that.
        room_x = (x_window[0] - float(shape_x.min()), x_window[1] - float(shape_x.max()))
        room_y = (corridor[0] - float(shape_y.min()), corridor[1] - float(shape_y.max()))
        if room_x[1] < room_x[0] or room_y[1] < room_y[0]:
            continue
        xs = float(rng.uniform(*room_x)) + shape_x
        ys_drawn = float(rng.uniform(*room_y)) + shape_y
        stations = list(zip(xs.tolist(), ys_drawn.tolist(), strict=True))

        # Seats are boxes +/- 35 mm along world y in their own station's frame,
        # so disjointness is a statement about the stations' y and nothing else.
        if float(np.diff(ys_drawn).min()) < 0.070:
            continue
        if not _weaves(stations, task, floor=low):
            continue
        p = difficulty.profile(stations, sides, seat=seat, tail=WEAVE_TAIL)
        if p.turning > high or p.weakest < PORT_WRAP_FRACTION * reference.weakest:
            continue
        # The support hand holds the run near its free tip while the work hand
        # advances station by station toward it, so the LAST station is the worst
        # case for the two crowding.
        far = stations[int(np.argmax(ys_drawn))]
        anchor_est = _anchor_xy_est(task)
        if math.hypot(far[0] - anchor_est[0], far[1] - anchor_est[1]) < MIN_HAND_GAP:
            continue
        if _port_route_length(stations, task, terminals) > declared_run + PORT_SLACK_ALLOWANCE:
            continue

        placed = {name: _surface_pose(x, y, 0.0) for name, (x, y) in zip(pegs, stations, strict=True)}
        placed["port"] = _surface_pose(*port_xy, port_yaw)
        # The anchor the cable draw placed stays in: it follows the rod, and the
        # rod is not what this loop moved.
        return {**cable, "static_poses": {**cable.get("static_poses", {}), **placed}}

    raise RuntimeError(
        f"{task.full_name} episode {episode_id}: no board satisfied every constraint in {PORT_TRIES} "
        f"draws ({n} pegs, turning in [{low:.1f}, {high:.1f}] deg, a {span_limit * 1000:.0f} mm "
        f"corridor, a {rod:.2f} m rod). Widen PORT_PITCH_BAND or PORT_TURN_RATIO in "
        "tasks/cable_route/staging.py, or check that the declared board is the tier it claims."
    )


_port_board.model_key_fields = (
    "static_poses",
    "cable_bow",
    "cable_shift",
    "cable_yaw",
    "cable_wave",
)
"""Every key changes the model rather than only the state: the pegs, the anchor
bracket and the port are welded geometry and the four fractions are in the rod's
spawn. Without this the environment re-poses instead of rebuilding and every
episode of a tier would silently be episode 0's board."""

PORT_BOARD = _port_board
"""``(:sampler staging:PORT_BOARD)``: a terminated tier's pegs, drawn between an
anchor that follows its cable and a port drawn in its own bolt slots."""


def _port_board_reachable(task, episode_id: int, rng) -> dict[str, Any]:
    """Development revision: terminal 50 mm closer in x; original scoring and layout guards."""
    return _port_board(task, episode_id, rng, port_reach_inset=0.050, stagger_scale=0.5)


_port_board_reachable.model_key_fields = _port_board.model_key_fields
PORT_BOARD_REACHABLE = _port_board_reachable


# The YAM station: one head camera, a front camera and a D405 on each wrist, every one 1280x720.
YAM_STATION_RIG = register_camera_rig(
    "yam_station_cable@v1",
    (
        CameraSpec(
            name="cable",
            width=1280,
            height=720,
            fov_y=50.0,
            eye=(0.9196, -0.06, 1.1357),
            look_at=(0.46, -0.06, 0.75),
            up=(0.0, 0.0, 1.0),
            near=0.01,
            far=2.0,
            channels=("rgb", "depth"),
        ),
        CameraSpec(
            name="overhead",
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
