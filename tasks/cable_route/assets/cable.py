# SPDX-License-Identifier: Apache-2.0
"""The rod cable: a 0.91 m chain of capsules, and what it takes to push one.

The **easiest** asset in the corpus to annotate, and instructively so. A chain
substrate ships its own coordinate system, so ``cable.point(0.5)`` is meaningful
the moment the thing exists -- no selector, no baked index, no resolver. Every
"feature" a task ever wants of a cable is a normalised arc length, and arc length
is free.

That is worth dwelling on, because the intuition runs the other way. A deformable
body *sounds* like the hard case for declarative annotation and a rigid box like
the easy one. In practice the ordering is: chain (nothing to author) < parametric
sheet (derived recipes) < rigid mesh (measured constants) < garment (all three
tiers). What makes annotation expensive is not deformability, it is the absence
of a parameterisation.

Where this asset earns its keep instead is **physics**. The cable needs a
coupled-proxy arrangement -- MuJoCo owns the arms, rigid VBD owns the rod, and
the gripper bodies are mirrored into the VBD side as mass-scaled kinematic
proxies -- because MuJoCo has no rod joint and VBD should not solve an arm. None
of that is a fact about routing a cable; it is a fact about *what a simulable rod
cable is*, and it lived in a task only because no asset layer existed to own it.

It is declared here by **pinning the existing profile**, not by describing the
arrangement. That is deliberate: the profile is today's factory registered
verbatim under a versioned key, so equivalence is true by construction rather
than by testing, and a composer has to earn this asset rather than being trusted
with it on day one. Numbers are the reference's own.
"""

from __future__ import annotations

from robosimstudio_materials.rod.declare import Rod

from robosimstudio.assetlib.registry import register_asset
from robosimstudio.assetlib.semantics.features import FeatureSpec
from robosimstudio.assetlib.semantics.published import PublishedState
from robosimstudio.assetlib.semantics.states import StateSpec
from robosimstudio.scene.builtin.benches import BENCH_075
from robosimstudio.scene.frames import point_from_surface
from robosimstudio.specs import CellSpec

# Registers ``cable_rod`` and ``cable_peg`` as entity classes. Imported here
# rather than left to the corpus walk so that holding this record is enough to
# build one: a declaration whose implementation may or may not have been
# imported is a declaration that fails at scene-build time, on a different
# machine, for a reason that reads as a missing entity.
from . import rod as _rod  # noqa: F401 -- registers the entity classes
from .rod import SpecCable

__all__ = [
    "CABLE",
    "LENGTH",
    "RADIUS",
    "SEGMENTS",
    "WEAVE_CABLE_RADIUS",
    "WEAVE_CABLE_SEGMENTS",
    "weave_bend",
    "weave_cable",
    "wrap_cable",
]

LENGTH = 0.9144
"""36 inches [m] -- the reference's own rod length."""
RADIUS = 0.0110
SEGMENTS = 28
"""Capsule count. Segment spacing must stay above ``2 * radius``."""

CABLE = register_asset(
    Rod.from_length(
        name="cable_rod",
        length=LENGTH,
        radius=RADIUS,
        summary="A 0.91 m rod cable of 28 capsules: bendable, draggable, windable.",
        tags=("cable", "wire", "rod"),
        # Tier 0, in effect: the substrate's parameterisation *is* the feature
        # space, so these name landmarks rather than define them. ``point``
        # takes its ``s`` at reference time -- cable.point(0.63) -- which is why
        # the declaration carries only the convention.
        points={
            "end_a": 0.0,
            "end_b": 1.0,
            "midpoint": (0.5, "Named because the pick-place variant grasps here."),
        },
        touches=("surface", "fixtures", "hands"),
        features=(
            # The wrap variant's second hand, as a declared init mechanism rather
            # than a world-space tuple buried in a spec. What used to be
            # ``anchor_grasp=(0.58, 0.12, 0.761)`` -- three numbers that only
            # made sense against one table height -- becomes "hold me here",
            # with the arc length doing the work the coordinates were doing.
            FeatureSpec(
                name="anchor",
                kind="anchor",
                payload={"at": {"s": 0.6361}, "grip": 0.010, "release_on": "never"},
                provenance=(
                    "The reference's anchored wrap, at its own y = 0.12 -- which on a 28-capsule rod "
                    "laid from -0.4572 to +0.4572 is 0.6361 of the way along, not the 0.63 this said "
                    "first. Five millimetres, and the rounder number was written before anything read "
                    "it; a declaration nobody has checked against the thing it describes drifts in "
                    "exactly this direction. Grip is 10 mm -- about 1.5 mm of interference a side. The "
                    "working arm squeezes at 3 mm because it must drag the rod around a peg; commanding "
                    "that here loads the anchor wrist with the full squeeze reaction through the 20x "
                    "proxy and deflects the pinch 30 mm off the rod. At mu 60 a light pinch pins its "
                    "spot, which is all the anchor has to do."
                ),
            ),
        ),
        states=(
            StateSpec(
                name="straight",
                measure="straightness",
                op=">=",
                value=0.95,
                summary="Laid out along its own axis rather than coiled or kinked.",
                provenance="Designed here; the reference ships no shape predicate.",
            ),
        ),
        published=(
            # Both latches, because neither is measurable from outside. A hold
            # here is a kinematic coupling written into a warp array -- not a
            # contact, not a pose -- so a document asking "is the anchor holding"
            # has nothing to measure and everything to read. Which is the case
            # published states exist for: internal state, declared instead of
            # reached for.
            PublishedState(
                name="held",
                type="bool",
                summary="The working hand's latch is engaged on a segment.",
                tier="measurable",
            ),
            PublishedState(
                name="anchor_held",
                type="bool",
                summary="The second hand's latch is engaged, pinning the cable where the anchor says.",
                tier="measurable",
            ),
        ),
        notes=(
            "Pinned to the reference arrangement rather than described: MuJoCo owns the arms, "
            "rigid VBD owns the rod, gripper bodies mirror into the VBD side as mass-scaled "
            "proxies at 20x. The main collision pipeline is narrowed to cable-versus-static on "
            "purpose -- arm-versus-cable contact comes exclusively from the proxy pipeline, and "
            "letting the main one see those pairs too would double-count them. The factory that "
            "assembles it is tasks/cable_route/staging.py's registered cable_coupled_960hz@v1, the "
            "arrangement every cable document stages by name. "
            "WHICH ARMS MIRROR is no longer stated here. It was ``params={'proxy_arms': "
            "('right',)}`` with a note that a two-handed task overrides it and that the composer "
            "would one day derive it from what the task declares touching -- which is what the "
            "port did, in two steps: first off the wrap's anchor (an anchor IS a second hand on "
            "the rod), and now off the binding's own roles, because the weaves' second hand "
            "arrives mid-episode rather than in the initial pose and every document names it "
            "(proxy_arms_for carries the measurements). A default that every two-handed caller "
            "has to remember to override is a default that gets forgotten once, silently, as a "
            "hand passing through the cable."
        ),
        provenance=(
            "Ported from newton_tasks/sim/yam_cable_sim.py at its defaults: 28-capsule add_rod "
            "cable, SolverCoupledProxy(mjc+vbd), the reference's own keyframe scripts."
        ),
    )
)


# ---------------------------------------------------------------------------
# PLACING ONE. The record above names no ``build``, deliberately: a ``build`` is
# a zero-arg factory answering for the *type*, and the rod is a different rod
# in every document that stages one -- the reference's 0.91 m at 11 mm for the
# wrap, anchored by the second hand; 0.70 to 1.15 m at 3.5 mm for the thirteen
# weaves, at four different capsule counts. What is shared is the material and
# the section, and that is what lives here. Where each starts is
# ``layout.py``'s, which mints one record per document -- ``wrap_cable``,
# ``weave3_cable`` -- with one of these, partially applied to that variant's
# coordinates, as its ``build``.
# ---------------------------------------------------------------------------

WEAVE_CABLE_RADIUS = 0.0035
"""The weave rod's section [m]: 7 mm across, against the reference's 22 mm.

Thinned twice, and both times by the fixture. The route has to bend around the
F1's 4.75 mm shaft several times, and the reference's 28 capsules at 22 mm cannot
take that radius without the capsules interpenetrating. 10 mm held the turn on a
70 mm pitch; at the 155 mm pitch this family stands on, the crossings are longer
and the rod has to lie flatter through them, and 7 mm is what a bench-scale
harness looks like beside a 19 mm cleat standoff anyway.

It also sets the seat: a cable resting against the spool's shaft sits
``ROUND_SHAFT_RADIUS + WEAVE_CABLE_RADIUS`` = 8.25 mm from the station's axis,
which is the number every seat box and every difficulty profile is measured
about (:func:`~.layout.seat_offset`)."""

WEAVE_CABLE_SEGMENTS = 42
"""Capsules on the four members that predate :data:`WEAVE_SEGMENT_PITCH`.

Kept, and kept for one reason: the four shipped weaves score
``containment >= 0.047``, and 0.047 is *two capsules of forty-two*. Re-counting
their rods would leave the threshold saying a different thing than the number
their documents were recorded against. Everything else in the family is
discretised at the pitch instead -- see :func:`weave_segments`."""

WEAVE_SEGMENT_PITCH = WEAVE_CABLE_SEGMENTS / 0.50
"""Capsules per metre, so a longer rod is a longer rod and not a coarser one.

The turn a station asks of the rod is the same 8.25 mm radius at every length,
and a capsule pitch is what decides whether that turn is resolved or is a corner
between two straight sticks. ``weave6``'s route wants 1.15 m; at the constant 42
its pitch would be 27.4 mm against 11.9, and a 90 degree station turn spans about
13 mm of arc -- so the whole turn would land inside one capsule.

84 per metre is 42 over the 0.50 m the family's original rod was, which is an
11.9 mm pitch."""


def weave_segments(variant: str, length: float) -> int:
    """How many capsules this variant's rod is built from.

    The four members that predate :data:`WEAVE_SEGMENT_PITCH` keep the count
    they were measured at; everything else is discretised at the family's pitch
    and rounded to even, so a rod has a midpoint capsule boundary rather than a
    midpoint capsule and ``cable.midpoint`` interpolates the same way at every
    length.

    **The four kept counts are now coarse, and that is the cost of the
    threshold.** Their rods grew with the 155 mm pitch -- 0.70 to 0.90 m against
    the 0.50 they were written at -- so 42 capsules is a 16.7 to 21.4 mm pitch
    rather than 11.9. Re-discretising them would resolve their turns better and
    would also change what ``0.047`` means, which is the one number four recorded
    documents share. The share was chosen; the pitch was not.

    Args:
        variant: The weave being staged.
        length: Its rod length [m].
    """
    if variant in ("weave3", "weave4", "weave_arc", "weave_stagger"):
        return WEAVE_CABLE_SEGMENTS
    return 2 * round(0.5 * WEAVE_SEGMENT_PITCH * length)


def weave_bend(length: float, segments: int = WEAVE_CABLE_SEGMENTS) -> tuple[float, float]:
    """A weave rod's bend stiffness and damping [N*m, N*m*s], at material parity.

    **The number that decides whether a weave can hold a shape**, and the one
    thing about this rod that was never rescaled when it was thinned.

    ``SpecCable`` ships the reference's 0.04 N*m, measured on a 22 mm rod at a
    32.7 mm capsule pitch. A cable joint's bend stiffness is the section's
    ``EI`` over the pitch it spans: ``EI`` goes as the fourth power of the
    radius, so the same *material* at 7 mm is ``(3.5/11)^4 = 0.01025`` of it,
    and the shorter pitch multiplies back by ``32.7/pitch``. On ``weave5``'s
    0.95 m at 80 capsules that is 0.0282 of the reference, or 0.00113 N*m.

    What eight and a half times too stiff looks like is not a rod that refuses
    to bend. It is a rod that **translates**: a hand that takes hold at one
    station and pulls carries the whole rod with it rather than putting a
    bend between two pegs, so each crossing undoes the one before it. It shows
    up in the score as an end state that is *straight* -- every recorded
    weave3 episode, latched or not, finished at ``straightness`` 0.988 to 0.999
    against a route that is nothing but three reversals -- and the seats it
    does score are the ones a straight rod can lie across.

    Scaled from the reference rather than written down, because the family
    varies the length and the count independently: ``weave_stagger`` is 0.90 m
    at 42 capsules and ``panel_l2`` is 0.90 m at 76, so the two have the same
    rod and different joints, and a constant would be right for one of them.

    Args:
        length: This variant's rod length [m].
        segments: How many capsules that length is built from -- the pitch is
            the two together, and the pitch is what the stiffness scales on.
    """
    reference = SpecCable(scene_friction=True)
    reference_pitch = reference.length / reference.segments
    pitch = length / segments
    scale = (WEAVE_CABLE_RADIUS / reference.radius) ** 4 * (reference_pitch / pitch)
    return reference.bend_ke * scale, reference.bend_kd * scale


# ---------------------------------------------------------------------------
# WHAT THE SPAWN VARIES, per episode. Four ranges the rod declares and a sampler
# fills in as unit fractions -- see ``RodSpec.bow_range`` and friends, and
# ``staging.CABLE_BOW``, which is what draws them. A spec that leaves a range
# zero-width ignores its fraction entirely, so the wrap is untouched by all four.
# ---------------------------------------------------------------------------

WEAVE_CABLE_BOW = (-0.020, 0.020)
"""Lateral bow drawn per episode for the weave spawns [m].

Trimmed from 25 mm, and the 5 mm is half of the idle-floor fix written up on
:data:`~.layout.WEAVE_CABLE_X`. Splitting the correction between the two numbers
is what lets the spawn land on an x the arm demonstrably plans to: 20 mm of bow
off a 0.42 spawn puts the rod's worst capsule at 0.440, and the nearest seat box
any layout declares begins at 0.450.

What it costs is small and the reason for the bow survives it. 20 mm of
half-cosine on a 700 mm rod is still well inside what one laid down by hand
takes, and still far more than enough that a policy cannot be right about the
first grasp by remembering an x."""

WEAVE_CABLE_WAVE = (-0.030, 0.030)
"""Second-harmonic amplitude drawn per episode for the weave spawns [m].

**What makes the spawn look laid down instead of pulled taut.** The bow is one
arc over the whole rod, so at 20 mm on a 700 mm cable it reads as a straight rod
with a slight bend. The second harmonic crosses the axis once, so the rod
meanders -- which is what a cable does when someone puts it down.

**It spends more of the idle-floor budget than the bow does, and the two do not
simply take the larger.** ``bow*sin(pi t)`` and ``wave*sin(2 pi t)`` peak at
different ``t`` and reinforce between them: over the shipped (20, 30) the peak
of the sum is 44.9 mm, not 30, and the constant 15 mm shift adds to that in
full. Computed rather than argued -- :func:`~.staging.spawn_excursion` is the
function, and it reads the ranges off the spec that will build the rod.

**That is 59.9 mm off a 0.42 spawn, and most of the family's seat boxes are
inside it.** A near-side box reaches 45 mm back from its station, so its edge is
the near row's x less 45 mm; the rod's worst capsule reaches x = 0.480. Per
layout, at the worst corner of the draw::

    near row x   box edge   room against the bow   against the full spawn
    weave_arc      0.5335     0.4885     +48.5 mm            +8.6 mm
    panel_l2       0.5350     0.4900     +50.0 mm           +10.1 mm
    weave3/port3   0.5240     0.4790     +39.0 mm            -0.9 mm
    weave4, port2, board_l1, board_l2
                   0.5200     0.4750     +35.0 mm            -4.9 mm
    weave5, board_l3
                   0.5075     0.4625     +22.5 mm           -17.4 mm
    weave_stagger, weave6, board_l4
                   0.4950     0.4500     +10.0 mm           -29.9 mm

**What that bound is and is not.** It is the ENVELOPE: bow, harmonic and shift
all at an extreme and all aligned, at the one arc position where the two
harmonics reinforce. A seat also asks for a share of the rod within +/- 35 mm of
its own station in y, and the peak is a few capsules wide, so a draw reaching
the box is not the same as a draw scoring it. No rollout has measured an idle
score at these constants -- the 0.500-standing-still reading
:data:`~.layout.WEAVE_CABLE_X` records was taken at a 0.44 spawn and a 25 mm
bow, before either was trimmed.

It is recorded rather than fixed. Moving the spawn again or trimming the
harmonic changes what every episode of every weave looks like, which is a
before-and-after of its own; ``tests/test_cable_difficulty`` gates the BOW,
which is the term the 0.42 was sized against, and says why in its own docstring.
"""

WEAVE_CABLE_SHIFT = (-0.015, 0.015)
"""Lateral offset drawn per episode for the weave spawns [m].

**Sized by the idle floor, not by taste.** The rod spawns at
:data:`~.layout.WEAVE_CABLE_X` = 0.42 and the nearest seat box the three- and
four-peg line layouts declare begins at 0.475, so the rod's worst capsule has
55 mm of room. The bow already spends 20 mm of it; 15 mm of shift takes the
worst case to 0.455, leaving 20 mm. A spawn that reached a seat box would score
before the robot moved -- which is the failure :data:`~.layout.WEAVE_CABLE_X`
documents at length, and the reason this number is small.

That argument is the shift's own and predates the second harmonic, which spends
the 20 mm it left and more; :data:`WEAVE_CABLE_WAVE` carries the sum."""

WEAVE_CABLE_YAW = (-4.0, 4.0)
"""In-plane rotation drawn per episode for the weave spawns [deg].

Four degrees over a 700 mm rod swings each tip about 24 mm, which is the same
order as the bow and lands inside the same margin. What it buys that a shift
cannot: the rod is no longer PARALLEL to the station line, so the standoff
differs from station to station within one episode and a plan cannot carry one
offset across all of them."""


def weave_cable(
    *,
    variant: str,
    length: float,
    xy: tuple[float, float],
    anchored: bool = False,
    cell: CellSpec = BENCH_075,
) -> SpecCable:
    """A weave's rod at t = 0: nearly straight, along +y from the world ``xy``, resting on the surface.

    Spawned from a start and a direction rather than from a route -- the field
    that spawns a cable already routed is the exact opposite of what these
    variants ask. Getting it anywhere else is the robot's job.

    Straight-*ish* for every layout, including the bent ones. A rod pre-curved to
    match an arc would be a rod that had already been half-routed, and the arc's
    whole content is that the carry direction changes station to station. What
    varies per episode is the four spawn modes above, none of which is a route.

    **No anchor grasp, and both arms start at home.** ``wrap`` opens with its
    second hand already closed on the rod -- a start state that is a robot
    posture -- and the weaves did too, inheriting it. They do not now: an episode
    begins with the YAM in the pose the embodiment declares, touching nothing,
    which is what makes two variants comparable by eye and what a policy can
    assume without reading the task. Taking the far end is a thing a *policy*
    does, on its own first move, rather than a condition the scene arrives in.

    Args:
        variant: The weave being staged; :func:`weave_segments` reads it.
        length: This variant's rod length [m].
        xy: Where the rod's first segment starts, in the reference's world
            coordinates; the rod runs +y from here.
        anchored: Whether the run's first body is welded to the world where it
            was built -- true on the terminated variants, whose anchor bracket
            is drawn around that weld. ``anchors=("start",)`` fixes the capsule
            at ``builder.body_q``, which is its own COM pose after the spawn
            modes have moved it, and that is the frame
            :func:`~.layout.weave_terminals` places the bracket in.
        cell: The bench the world numbers were authored on.
    """
    x, y = xy
    segments = weave_segments(variant, length)
    bend_ke, bend_kd = weave_bend(length, segments)
    contact_only = variant in {"port_easy", "port_medium"}
    return SpecCable(
        scene_friction=True,  # this cell is the cable's; see RodSpec.scene_friction
        start=point_from_surface(cell, (x, y, cell.surface_z + WEAVE_CABLE_RADIUS)),
        direction=(0.0, 1.0, 0.0),
        length=length,
        radius=WEAVE_CABLE_RADIUS,
        segments=segments,
        # The rod was thinned from the reference's 22 mm and its bend was not
        # rescaled with it; see :func:`weave_bend` for the factor and for what
        # a rod eight times too stiff does to a route made of reversals.
        bend_ke=bend_ke,
        bend_kd=bend_kd,
        # NOT STRAIGHT, and not by much. A rod that arrives dead straight every
        # episode lets a policy be right about the first grasp for the wrong
        # reason -- an axis fit, a midpoint, a remembered x -- and none of those
        # announce themselves when the rod bends. All four ranges are signed and
        # all four are inert until a sampler supplies their fractions, so a
        # ``(:sampler fixed)`` episode of any of these builds the straight rod
        # byte for byte.
        bow_range=WEAVE_CABLE_BOW,
        wave_range=WEAVE_CABLE_WAVE,
        shift_range=WEAVE_CABLE_SHIFT,
        yaw_range=WEAVE_CABLE_YAW,
        # THE RUN IS BOLTED AT ITS FIRST END, on the terminated variants only.
        #
        # The branch this came from carried a ``pin_start`` flag of its own.
        # ``anchors`` is the field that already means this, and it means it
        # better: it welds the first capsule at ``builder.body_q[body]`` -- the
        # capsule's own COM pose, after the bow, the shift, the yaw and the
        # harmonic have all been applied -- which is the exact frame the branch's
        # comment measured its bracket offset in.
        anchors=("start",) if anchored else (),
        # Evaluated tiers use physical fingertip friction, never the synthetic
        # segment-to-hand force latch, and retain hand/fixture collisions.
        allow_grasp_latch=not contact_only,
        filter_fingertip_scenery=not contact_only,
    )


def wrap_cable(*, x: float, anchor_y: float, anchor_arm: str, cell: CellSpec = BENCH_075) -> SpecCable:
    """The reference's rod, laid along +y and centred on it, with ``wrap``'s anchor grasp.

    The record's own rod -- ``LENGTH`` at ``RADIUS`` in ``SEGMENTS`` capsules
    are ``SpecCable``'s defaults, and this asserts they still agree rather than
    restating them. It runs from y = -0.4572 to +0.4572, which is what makes the
    ``anchor`` feature's s = 0.6361 and the reference's y = 0.12 the same point.

    The anchor is a coordinate here and an arc length in the record, and the
    two say the same thing: a coordinate is what the IK can be aimed at, and
    coordinates are what a document does not have -- so the conversion lives
    here, once. The reference's world numbers go through ``point_from_surface``
    like the pegs', because the spec's ``start`` and ``anchor_grasp`` are both
    in the work-surface frame.

    Args:
        x: The rod's line [m], world; the reference's own 0.58.
        anchor_y: Where the second hand pre-grips it [m], world.
        anchor_arm: Which arm holds it -- the binding's own support arm, stated
            rather than left to the spec's default, which is the embodiment's
            last declared arm.
        cell: The bench the world numbers were authored on.
    """
    reference = SpecCable()
    if (reference.length, reference.radius, reference.segments) != (LENGTH, RADIUS, SEGMENTS):
        raise ValueError(
            f"cable_rod declares {LENGTH} m x {RADIUS} m in {SEGMENTS} capsules, but SpecCable's defaults are "
            f"{reference.length} x {reference.radius} in {reference.segments}: the record and the rod it builds drifted"
        )
    z = cell.surface_z + RADIUS
    return SpecCable(
        scene_friction=True,  # this cell is the cable's; see RodSpec.scene_friction
        start=point_from_surface(cell, (x, -0.5 * LENGTH, z)),
        anchor_grasp=point_from_surface(cell, (x, anchor_y, z)),
        anchor_arm=anchor_arm,
    )
