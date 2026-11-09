# SPDX-License-Identifier: Apache-2.0
"""Where everything stands, and one placed record per name a document uses.

The world coordinates BDDL has no place for -- where each variant's stations
stand, where its rod starts, where the anchor hand takes it -- and the records
that carry them. A document names ``peg0 - weave3_peg0``; this is what says
where ``peg0`` is.

**A record per document-visible name, minted from the type.** A record's
``build`` is a zero-arg factory answering for a *type*, and the type is not
enough here: ``peg0`` is a ``cable_peg_round`` standing somewhere different in
eleven documents and a ``cable_cleat`` in a twelfth, and ``cable`` is a rod that
is 0.70 m in two documents, 1.15 m in two others and 0.91 m with a second hand
on it in the wrap. So each document names a record of its own --
``weave3_peg0``, ``wrap_cable`` -- minted here from the type's declaration with
the one fact the type cannot state: where it stands. The features, the states,
the mesh are the type's, so a seat measured about ``peg0.axis`` reads
identically in every variant, and a document that swaps a spool for a cleat
changes the physics and not the question.

The slack guard runs here, at the point a layout and its rod meet: a layout
whose goal path is longer than its cable is an impossible task that builds,
steps and scores zero for ever while looking exactly like a hard one -- which is
the failure a family of thirteen arrangements is most likely to introduce by
arithmetic. A route that does not fit refuses to register, so the suite fails
to load rather than ships a weave nobody could make.

Every factory converts the reference's own world ``xy`` through
``point_from_surface`` at the one point it becomes a spec, so the numbers stay
legible as the reference's and the spec is in the work-surface frame, where it
survives a different table. The rigs aimed at these layouts are ``staging.py``'s,
registered under the names the documents use, and so are the two samplers that
redraw a board every episode.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from functools import partial

from robosimstudio_materials.rod.declare import Rod

from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.registry import register_asset
from robosimstudio.scene.builtin.benches import BENCH_075

from .cable import (
    CABLE,
    WEAVE_CABLE_RADIUS,
    weave_cable,
    weave_segments,
    wrap_cable,
)
from .fixtures import (
    BLOCK_HALF,
    CABLE_BLOCK,
    CABLE_CLEAT,
    CABLE_PEG,
    CABLE_PEG_ROUND,
    CLEAT_POST_RADIUS,
    ROUND_SHAFT_RADIUS,
    cleat,
    post,
    spool,
    terminal,
)

__all__ = [
    "ANCHOR_Y",
    "CABLE_X",
    "CELL",
    "CLEAT_VARIANTS",
    "PEG_XY",
    "PLACEMENTS",
    "WEAVE_ANCHOR_INSET",
    "WEAVE_ANCHOR_Y",
    "WEAVE_AXIS_Y",
    "WEAVE_CABLE_LENGTH",
    "WEAVE_CABLE_X",
    "WEAVE_LAYOUTS",
    "WEAVE_MIN_SLACK",
    "WEAVE_MOUTH_INSET",
    "WEAVE_PEG_X",
    "WEAVE_PITCH",
    "WEAVE_PORT_DEPTH",
    "WEAVE_PORT_STANDOFF",
    "WEAVE_SIDES",
    "WEAVE_TAIL",
    "WEAVE_TERMINAL_GAP",
    "WEAVE_TERMINALS",
    "WEAVE_TERMINAL_ENTITY",
    "seat_offset",
    "weave_anchor_y",
    "weave_terminals",
]

CELL = BENCH_075
"""The cell these variants were authored in. Named, not assumed: the constants
below are world coordinates in a sim whose table top was 0.750, and this is what
converts them into offsets that survive a different table -- for the pegs, the
rod and the anchor grasp alike, each through ``point_from_surface`` at the one
point it becomes a spec, inside the factory that places it. The reference's
numbers stay legible as the reference's; the specs they turn into are in the
work-surface frame."""

CABLE_X = 0.58
"""The wrap's rod line [m]: the reference's own."""
PEG_XY = ((0.44, -0.06), (0.44, 0.10), (0.44, 0.22))
"""The wrap's three posts [m], transcribed from the reference scene. The task
winds onto the first; the other two are there because they are there."""
ANCHOR_Y = 0.12
"""Where ``wrap``'s second hand pre-grips the rod [m]. The reference's own, and
0.6361 of the way along a rod running from y = -0.4572 to y = +0.4572 -- which
is what ``cable.anchor`` declares and what this converts back into a coordinate
the IK can be aimed at."""

WEAVE_PEG_X = 0.52
"""The weave's peg line [m]. The ManipulationNet board's own staging offset on
this bench, reused deliberately: it is the one x where a reach solve over the
whole footprint came back with nothing out of reach."""

WEAVE_AXIS_Y = -0.06
"""The weave's centre line [m], and the one number here a reach check moved.

``wrap`` is bimanual and this is not -- one hand weaves while the other pins the
cable -- so every waypoint has to sit inside the **right** arm's envelope alone.
Centred on the cell, a four-peg line put its last peg at y = +0.105 and the plan
failed there by 47 mm while the other 33 waypoints solved to 2 mm. Shifting the
whole layout 60 mm toward the working arm puts every crossing a full grid step
inside the wall. The layout moved rather than the carry height because the peg's
60 mm is what sets the height, and going under a peg top is not a trade."""

WEAVE_PITCH = 0.155
"""Peg spacing along the line [m].

**It was 70 mm and it is 155, and the reason is the SEATS rather than the
cable.** A seat is a box reaching +/- 35 mm along the line in its own station's
frame, so two stations on a 70 mm pitch have boxes that meet exactly -- a rod
lying between them counts once for each, and "a segment counted for one station
is not also counted for its neighbour" is the property every weave document
asserts and none of them re-derives. 155 mm is that floor with more than a
seat's width of air either side, which is also what leaves room for a board the
sampler has yawed and jittered.

What it costs is the difficulty. Accumulated turning is
``2 * (N - 1) * atan(excursion / pitch)`` and the pitch is downstairs, so
doubling it more than halves the turning of every layout that predates it: the
four-peg line went from 93 degrees to 36.5. The family bought that back with
station count and stagger depth rather than by bringing the pitch in --
see :mod:`~tasks/cable_route/difficulty.py`, whose ladder moved with it.

The cable's own bound is comfortably clear at this pitch. The turn radius is
``ROUND_SHAFT_RADIUS + WEAVE_CABLE_RADIUS`` = 8.25 mm and the rod needs room to
come out of one turn before entering the opposite one; 155 mm is about nine turn
diameters between reversals, where 70 mm was four."""

WEAVE_CABLE_X = 0.42
"""Where a weave's cable starts [m]: 100 mm nearer the robot than the peg line.

Far enough that no peg is seated at t = 0 -- the documents' seats reach 45 mm
back from their station -- so no weave can be passed by its own layout. Near
enough that the whole carry is one short move rather than a reach across the
bench. Stated once and shared, because "the score starts at zero" is a property
of the family and not of any one arrangement.

**It was 0.44, and at 0.44 the property had stopped being true.** Two changes
arrived separately and only interact at the staggered layouts. The rod gained a
per-episode BOW (see :data:`~.cable.WEAVE_CABLE_BOW`), which carries its middle
toward the stations; and the family gained layouts whose near row stands at
x = 0.495 rather than 0.52, whose near-side seat boxes therefore reach back to
x = 0.450. A rod bowed to 0.465 sits *inside* those boxes, with about six
capsules inside the +/- 35 mm window of each near station -- a containment of
0.115 against a threshold of 0.047.

Measured, not reasoned about: ``weave_stagger`` and ``weave6`` both scored
**0.500 standing still**, which is every near-side seat of each. ``weave3``,
``weave4`` and ``weave_arc`` stood at x = 0.52 and read 0.000.

The reading only appears after the rod has SETTLED. ``worst_of`` scores the
weaker of two groups, and at the reset frame the rod is still falling, so the
release group's ``speed`` clause reads zero and hides whatever the seats say.
The first pass at this measurement read 0.000 everywhere for that reason.

0.42 rather than something lower because that is where the arm still plans:
``rigs/weave_reach.py --x 0.38,0.40,0.41,0.42,0.44`` comes back clean at 0.42 at
both the rest and carry heights, and holed at 0.41 and below. The bow was
trimmed to 20 mm to match; what the second harmonic then spent of the remaining
margin is written up on :data:`~.cable.WEAVE_CABLE_WAVE`, including the three
layouts it does not clear."""

WEAVE_SIDES = (-1, +1, -1, +1)
"""Which side of each peg the cable must finish on, in x: -1 is the robot's side.

Alternating, which is what makes any of these a weave rather than a row of
independent drags -- the cable crosses the peg line between every adjacent pair,
and each crossing is held only by the two pegs either side of it."""

CLEAT_VARIANTS = ("panel_l2",)
"""Variants whose stations are industrial cleats rather than F1 spools.

A tuple rather than a name test, because the fixture is a property of the
BOARD and more panels are the obvious next thing -- and because
:func:`seat_offset` has to read the same list to know how thick a station is."""


def seat_offset(variant: str) -> float:
    """How far off a station's axis a seated cable rests, on this board [m].

    The station's own radius plus the rod's. **One function rather than a
    constant**, because the suite now stages two fixtures of different
    thickness and everything that measures a weave -- the difficulty axis, the
    sampler's guard that a jittered board is still a weave, the slack guard
    below -- has to ask the board rather than assume the spool.

    It read ``ROUND_SHAFT_RADIUS + WEAVE_CABLE_RADIUS`` in three places. Left
    alone, an industrial panel would have been measured as though its stations
    were 4.75 mm F1 spools and reported as L1 at 90 degrees, when the cleats it
    actually carries put it at 143 and L2. A difficulty axis that describes a
    bench nobody staged is the one failure this suite has already had.
    """
    post_radius = CLEAT_POST_RADIUS if variant in CLEAT_VARIANTS else ROUND_SHAFT_RADIUS
    return post_radius + WEAVE_CABLE_RADIUS


def _line(
    pegs: int,
    *,
    pitch: float = WEAVE_PITCH,
    x: float = WEAVE_PEG_X,
    centre: float = WEAVE_AXIS_Y,
):
    """``pegs`` stations on a straight line, centred on :data:`WEAVE_AXIS_Y`.

    Centred rather than grown from one end, so a three-peg weave is the four-peg
    one with its ends brought in rather than a different place on the bench.
    """
    span = (pegs - 1) * pitch
    return tuple((x, centre - 0.5 * span + i * pitch) for i in range(pegs))


def _arc(pegs: int, *, radius: float = 0.22, pitch: float = WEAVE_PITCH):
    """The same stations bent onto a circle of ``radius``, bulging away from the
    robot at the ends.

    The centre sits behind the peg line, so the middle stations stay on
    :data:`WEAVE_PEG_X` and the outer ones move away in x. What that costs a
    policy is the thing worth having: the direction the cable must be carried to
    reach a peg is no longer the same at every peg, so one carry vector cannot
    serve the whole route and an open-loop stroke has to be re-aimed per station.
    """
    centre_x = WEAVE_PEG_X + radius
    span = (pegs - 1) * pitch
    out = []
    for i in range(pegs):
        # Arc length from the centre station, converted to an angle, so the
        # spacing measured ALONG the arc is the pitch -- bending a line must not
        # quietly change how far apart the stations are.
        # `pi -` rather than `pi +`: both put the stations on the same arc, and
        # only this one walks them in +y like every other layout. Reversed, peg0
        # would be the far station here and the near one everywhere else, which
        # would make a per-station comparison across the family read backwards.
        theta = math.pi - (-0.5 * span + i * pitch) / radius
        out.append((centre_x + radius * math.cos(theta), WEAVE_AXIS_Y + radius * math.sin(theta)))
    return tuple(out)


def _stagger(
    pegs: int,
    *,
    depth: float = 0.025,
    pitch: float = WEAVE_PITCH,
    x: float = WEAVE_PEG_X,
    centre: float = WEAVE_AXIS_Y,
):
    """A line split into two rows, ``depth`` either side of :data:`WEAVE_PEG_X`.

    Phased **against** the required sides: a station the cable must pass on the
    robot's side is the near one, and its neighbour on the far side is the far
    one. That is what makes this the demanding shape in the family -- the cable
    swings 2*(depth + seat) in x between adjacent stations where the straight
    weave asks for two seat widths, so the route has depth as well as width and
    the rod is carried toward and away from the robot rather than only across.
    """
    return tuple(
        (x + depth * WEAVE_SIDES[i % len(WEAVE_SIDES)], y)
        for i, (_, y) in enumerate(_line(pegs, pitch=pitch, centre=centre))
    )


WEAVE_TERMINAL_GAP: dict[str, float] = {
    "port_easy": 0.100,
    "port_medium": 0.100,
}
"""Anchor-to-board and board-to-port clearance for the requested tiers [m]."""

TERMINATED_SPAN = {"port_easy": 0.60, "port_medium": 0.72}
"""The span the board is laid out against [m] -- NOT the rod's length.

These were one number until the rod had to grow. The board is centred between
the anchored start and the port, and the anchor sits at ``WEAVE_AXIS_Y`` minus
half of this; the port then follows the last station. So the anchor-to-port run
is ``2 * WEAVE_TERMINAL_GAP + (pegs - 1) * pitch`` and does NOT depend on the
rod at all -- rod beyond this span is pure slack, which is the only reason the
two can be separated.

**Separated rather than raised together, on purpose.** Raising the shared
constant to lengthen the rod also walks the whole assembly ``-y`` by half the
increase: at +120 mm of rod the anchor, the board and the port all move 60 mm,
which puts ``port_medium``'s first station at y = -0.380, behind the work arm's
own base at -0.310. That is a different bench, and the 18/30 measured on this
one would not describe it. Holding the datum keeps every fixture where it was
measured and lets :data:`WEAVE_CABLE_LENGTH` be the one thing that moved."""


def terminated_spawn_y(variant: str) -> float:
    """Where a terminated run's rod starts [m]. The anchor is welded here."""
    return WEAVE_AXIS_Y - 0.5 * TERMINATED_SPAN[variant]



def _terminated_board(pegs: int, *, variant: str, pitch: float, depth: float, x: float):
    """Place a board symmetrically between the anchored start and the port."""
    anchor_y = terminated_spawn_y(variant)
    centre = anchor_y + WEAVE_TERMINAL_GAP[variant] + 0.5 * (pegs - 1) * pitch
    return _stagger(pegs, depth=depth, pitch=pitch, x=x, centre=centre)


SCATTER_OFFSETS = ((+4.0, -8.0), (-3.0, +0.0), (+5.0, +10.0))
"""Per-station offsets for :func:`_scatter`, in millimetres: ``(dx, dy)`` about
the even line.

Written down rather than sampled, because a FIXED layout has to be the same
board every episode -- the drawn boards (``board_l*``) are where per-episode
variety lives, and a rung that moved would stop being a rung.

Chosen against three constraints rather than for looks.

**The x offsets must stay well under the seat alternation, or they cancel the
weave.** A crossing's seat sits ``ROUND_SHAFT_RADIUS + WEAVE_CABLE_RADIUS`` =
8.25 mm off its post, alternating side by side, and that alternation IS the
route's turning. The first draft of this table used +/-12 mm, which is larger
than the alternation: the three seats landed within 3 mm of one x, the route
straightened out, and the measured turning collapsed to 2.2 degrees -- an
"irregular board" that had quietly stopped being a weave. Held to 5 mm the seats
still alternate by about 9 mm and the turning survives, at 12.6 degrees on this
pitch. The rejected draft's own figures -- a 7.5 mm spread in x and 10.5 degrees
of turning at the shipped seat, pitch and tail -- are what the +/-12 mm table
measures now; the branch quoted 2.2 degrees for it, at a pitch neither tree
ships.

The -x offsets are held smaller still, because a station pushed toward the robot
brings its near-side seat box with it. The y gaps stay above 60 mm so no two
seats overlap; on a 155 mm pitch they are 163 and 165 mm."""


def _scatter(pegs: int, *, pitch: float = WEAVE_PITCH):
    """An irregular board: stations off the line in BOTH axes, unevenly spaced.

    A row of posts at one x on an even pitch is the easiest thing to perceive and
    the easiest thing to plan for -- one standoff and one gap serve every
    station, so a policy that is right about the first is right about the rest
    for reasons that have nothing to do with seeing.

    This is what a board someone actually drilled looks like: the stations sit at
    different distances from the robot AND at uneven gaps along it, so each
    crossing has its own approach and the route cannot be carried across on one
    offset. It is not the :func:`_stagger` arrangement, which phases its depth
    *against* the required sides to be as hard as possible; this one is only
    uneven, which is a different thing from adversarial.
    """
    out = []
    for i, (x, y) in enumerate(_line(pegs, pitch=pitch)):
        dx, dy = SCATTER_OFFSETS[i % len(SCATTER_OFFSETS)]
        out.append((x + dx / 1000.0, y + dy / 1000.0))
    return tuple(out)


CLEAT_GAPS = (0.088, 0.074, 0.084, 0.078)
"""The industrial panel's station gaps, front to back [m].

Uneven on purpose, and not by much. A row of cleats at one pitch is what a
catalogue drawing looks like; a backplane someone actually drilled has the clips
where the panel had room, so each crossing gets its own approach and a policy
cannot carry one standoff across the board.

The magnitudes are chosen against the difficulty band rather than for taste.
Accumulated turning is set by the seat offset over the gap, so the gaps are what
put this layout at 143 degrees -- mid-L2, clear of both edges, where a tighter
set would cross into L3 and a looser one back into L1. They are well under
:data:`WEAVE_PITCH` and can be: a cleat's seat sits 13 mm off its axis against a
spool's 8.25, so the same excursion needs less gap to be the same angle."""

CLEAT_X = 0.535
"""How far the cleat row stands from the robot [m].

Ten millimetres beyond ``port3``'s furthest station, which stands at 0.525:
close enough that the two terminated variants ask the same reach of the same
arm and differ in the weave, and far enough that the cleats' own 9.5 mm posts
do not put a seated cable nearer the robot than a spool does."""


def _cleat_row():
    """Five cleats on :data:`CLEAT_GAPS`, centred on the bench's own axis.

    IN LINE, unlike :func:`_scatter`, and that is a statement about the fixture
    rather than a relaxation. A spool board is scattered because a row of thin
    posts at one x is the easiest thing there is to perceive and to plan for. A
    cleat row is not that: the standoffs are thick enough that the required side
    of each is a real excursion, so the alternation carries the difficulty and
    the uneven gaps carry the irregularity. Adding an x stagger on top pushed the
    turning to 178 degrees, hard against the L2 ceiling, where a jittered board
    would keep crossing into L3 and the variant would stop being the level it
    says it is.
    """
    ys = [0.0]
    for gap in CLEAT_GAPS:
        ys.append(ys[-1] + gap)
    mid = 0.5 * (ys[0] + ys[-1])
    return tuple((CLEAT_X, y - mid + WEAVE_AXIS_Y) for y in ys)


WEAVE_LAYOUTS: dict[str, tuple[tuple[float, float], ...]] = {
    # ---- the difficulty ladder. The count and the stagger depth rise together,
    # and the accumulated turning rises faster than either: 12.6, 36.5, 119.9,
    # 232.2 degrees. Which LEVELS those are is difficulty.py's to say, and at
    # this pitch they are L1, L1, L2, L3 -- see LADDER there.
    "weave3": _scatter(3),  # N=3  irregular board
    "weave4": _line(4),  # N=4  d=0
    "weave5": _stagger(5, depth=0.0125),  # N=5  d=12.5 mm
    "weave6": _stagger(6, depth=0.025),  # N=6  d=25 mm
    # ---- off the ladder: each shares a level with a rung and differs in how
    # that level's turning is made.
    "weave_arc": _arc(4),  # uneven turns, one barely-wrapped station
    "weave_stagger": _stagger(4),  # obliquity rather than count
    # ---- terminated runs: the same weave, but the run now STARTS and ENDS at
    # something. Two stations on one, because what that variant is about is the
    # two ends; a longer weave between them is weave4's question already
    # answered. Three on the other, on weave3's irregular board.
    "port2": _line(2),
    "port3": _scatter(3),
    # Requested VABAR tiers. Easy keeps the outer two stations of a three-peg,
    # 120 mm-pitch line (240 mm span); Medium uses four 80 mm-pitch stations
    # split across rows 60 mm apart.
    "port_easy": _terminated_board(
        3, variant="port_easy", pitch=0.120, depth=0.008, x=0.533
    )[::2],
    # PITCH STAYS 80 mm, and the 92 mm experiment is why it is written down.
    #
    # The hand needs 30.93 mm of clearance about a station, so 80 mm leaves only
    # 80 - 2*31 = 18 mm of corridor between neighbours -- narrower than the hand.
    # Widening to 92 mm opens that to 30 mm and it made the tier WORSE, not
    # better: 2 episodes, 0 of 8 seats, both aborting in ~85 s on
    #
    #   REFUSE: the centreline's nearest point is 166 mm from the station
    #
    # A wider board spreads the stations from 240 to 276 mm, and the PERCEIVED
    # centreline is only ~52% complete (measured: median cold-fit coverage 0.52
    # to 0.56, flat across stations). Past about 80 mm the far stations fall
    # outside what the fit can see, `choose_grasp` refuses at its 120 mm gate,
    # and `fixed_anchor` turns that refusal into a raise. The board cannot be
    # spread until the fit can see the whole rod.
    "port_medium": _terminated_board(
        4, variant="port_medium", pitch=0.080, depth=0.030, x=0.535
    ),
    # ---- the industrial panel. Five bolted cleats instead of three lab spools,
    # and the point of it is the DIFFICULTY as much as the look: a cleat's
    # standoff is 9.5 mm where the F1 spool's groove is 4.75, so a cable seated
    # against one rests 13 mm off the station's axis instead of 8.25 and every
    # crossing turns further. On the spacing above that is 143 degrees of
    # accumulated turning against port3's 13, on the same bench, with the same
    # rod and the same goal semantics.
    "panel_l2": _cleat_row(),
    # ---- drawn boards: the entry here is the DECLARED layout, which is the
    # rung's own. It is what the scene builds when nothing samples -- and, more
    # to the point, what `staging:WEAVE_BOARD` measures to find out which level
    # it is drawing for. Every episode replaces it.
    "board_l1": _line(3),
    "board_l2": _line(4),
    "board_l3": _stagger(5, depth=0.0125),
    "board_l4": _stagger(6, depth=0.025),
}
"""Every weave's stations, in world coordinates, by variant.

Thirteen arrangements of one problem, and the only thing that varies between
them is this table and the fixture drawn at each entry -- same rod material,
same goal shape, same seats. That is deliberate: a family whose members differ
in one dimension is one a result can be attributed in, and thirteen tasks that
each changed three things would not be.

Every station here is inside the working arm's own envelope at the carry height,
measured rather than assumed: ``rigs/weave_reach.py`` sweeps the weave's own
yaw-0 pose at both the rest and carry heights and puts the working arm's band at
``y`` in ``[-0.36, +0.10]`` at the tightest ``x`` a board may use and wider
everywhere else. ``weave6`` -- the widest layout -- runs from ``y = -0.448`` to
``y = +0.328``, which is past that band at both ends and is the one member whose
outer stations the routine reaches by chaining rather than from home. It is a
known limit of that layout rather than a property of the family; the drawn
boards are held inside :data:`~.staging.BOARD_Y` and do not have it."""

WEAVE_CABLE_LENGTH: dict[str, float] = {
    "weave3": 0.70,
    "weave4": 0.80,
    "weave_arc": 0.80,
    "weave5": 0.95,
    "weave_stagger": 0.90,
    "weave6": 1.15,
    # The terminated runs carry the same rod as each other so a score on one and
    # a score on the other differ in the board. 0.82 leaves port3's 696 mm route
    # -- three seats and the leg into the port -- 124 mm of slack.
    "port2": 0.82,
    "port3": 0.82,
    "port_easy": TERMINATED_SPAN["port_easy"],
    # LONGER THAN ITS LAYOUT SPAN, measured rather than chosen. Over a 30-trial
    # RTX batch the executed route ate 607 mm median (706 max) getting to the
    # last station, against the 444 mm of straight lines between seat points
    # that `_weave_slack` used to assume -- the difference being the wrap at
    # each post and the slack the weave leaves lying in each span. That left a
    # 99 mm median tail, against a nominal last-seat-to-port leg of 113 mm.
    #
    # Those four figures need no port position and are what this length rests
    # on. An earlier draft of this comment also claimed the free end "clips the
    # portbox corner a median 33 mm from the port's own centre" and that 27 of
    # 30 tails were shorter than the line to the port. BOTH WERE WRONG: they
    # measured against the port's DECLARED xy, and the sampler draws the board
    # -- `weave_terminals` derives the port from the last station actually laid,
    # plus a board yaw and the port's own slot slop -- so the true port moves
    # per episode and is not recoverable from the recorded trace. The evaluator
    # is the only honest witness on reach. What it says: at 720 mm the portbox
    # fraction sat at 0.049-0.066 against a 0.049 floor on every episode, which
    # is marginal on its face; at 840 mm it roughly doubles.
    #
    # 0.84 and not 0.80, because `weave_segments` discretises at
    # WEAVE_SEGMENT_PITCH: 0.84 lands on 70 capsules at exactly 12.00 mm, which
    # is this rod's own pitch at 0.72/60, so the bend stiffness `weave_bend`
    # derives is unchanged at 0.001116 N*m. A length that changed the pitch
    # would change the material as well as the reach.
    "port_medium": 0.84,
    # The panel's seated route is 420 mm plus the run out to the port; 0.90
    # leaves the same order of slack port3 has over its own route.
    "panel_l2": 0.90,
    # A drawn board takes its rung's rod, and the sampler refuses any draw whose
    # route does not fit it with WEAVE_MIN_SLACK to spare -- so the length is a
    # constraint on the boards rather than something the boards vary.
    "board_l1": 0.70,
    "board_l2": 0.80,
    "board_l3": 0.95,
    "board_l4": 1.15,
}
"""Rod length per variant [m], long enough that the route fits with slack to
spare. :func:`_weave_slack` is what checks that rather than this claiming it.

They grew with :data:`WEAVE_PITCH`: a four-peg line spans 465 mm of bench where
it spanned 210, and the rod has to cover that plus its tails plus the floor."""

WEAVE_TAIL = 0.04
"""Rod left trailing past the outermost station [m]. Enough that the free end is
not asked to stop exactly on a peg, little enough that it is not slack the weave
has to drag around."""

WEAVE_MIN_SLACK = 0.02
"""What :func:`_weave_slack` must clear, at build."""

WEAVE_ROUTE_SLACK: dict[str, float] = {"port_easy": 0.0, "port_medium": 0.13}
"""Rod the EXECUTED route leaves lying in its spans, beyond the taut path [m].

A taut polyline through the seat points is not what a routed cable looks like.
Measured over ``port_medium``'s 30-trial RTX batch: the rod between the anchor
and the last station ran 607 mm median against 444 mm of straight lines and
~38 mm of wrap, so ~130 mm was slack the weave simply leaves behind -- bows
between stations that no hand pulls out, because nothing in the choreography
ever tensions the run.

It is charged to the guard rather than designed away because it is a property
of the choreography, not of the layout: a board sized as though the route were
taut is a board whose rod cannot finish it, which is the failure this guard
exists to catch.

**Zero for ``port_easy`` means unmeasured, not zero.** Its two-station route
has had no batch of its own, and a figure scaled off ``port_medium``'s four
stations is a guess. Asserting the guess here fails ``port_easy``'s build --
its rod is 600 mm against a 533 mm taut route, so any real slack term of 70 mm
or more condemns it -- and condemning a task that has recorded results on the
strength of an unmeasured number is worse than admitting the guard does not
cover it. Measure it on a batch before putting a number here."""

WEAVE_ANCHOR_Y = 0.17
"""Where the support hand takes a plain weave's rod [m]: the far end's hold.

**A policy's first move, not a posture the scene arrives in.** Both arms still
start at the embodiment's own home, touching nothing -- see
:func:`~.cable.weave_cable` -- and ``RoutineWeave`` opens by reaching here with
the support hand and pinching the rod by contact
(:func:`~tasks/cable_route/routine.py::plan_support_stages`), then pays
slack out through the crossings. This is the number that hold aims at.

It was ``axis + half the rod - 20 mm`` -- 20 mm in from the far tip, which is a
sound place to hold a rod and a poor thing to derive a *robot pose* from. The
rods are not all the same length, so the anchor moved with them, and the support
arm started every variant in a slightly different posture.

A constant instead, so the plain weaves differ **only** in where the stations
stand. A support arm that grasps somewhere else per variant is a second
difference, and a second difference is what makes a result impossible to
attribute.

**Which stopped meaning what it was written to mean, and see
:func:`weave_anchor_y`.** It was chosen as "20 mm in from the far tip" of a
0.50 m rod, whose tip is at y = +0.19. The rods are 0.70 to 1.15 m now, so a
constant y is a grip that walks further and further toward the middle of the run
as the cable grows -- and on the terminated variants that put it in the middle
of the board. The plain weaves keep it anyway; the terminated ones do not."""

WEAVE_ANCHOR_INSET = 0.020
"""How far in from the rod's free tip the support hand takes it [m].

What :data:`WEAVE_ANCHOR_Y` originally meant, restored as the thing it meant.
20 mm is enough rod past the pads that the pinch is on the cable rather than on
its end cap.
"""


def weave_anchor_y(variant: str) -> float:
    """Where the support hand takes the rod [m].

    **The constant was right for a 0.50 m rod and wrong for this one, and the
    failure is the one a stale constant always has: it kept working, quietly, in
    the wrong place.** :data:`WEAVE_ANCHOR_Y` = 0.17 was written as "``axis`` +
    half the rod - 20 mm" for the family's original 0.50 m cable. The terminated
    variants run 0.82 m, tip at +0.35 -- so the support hand was taking the rod
    180 mm in from its end, in the middle of the board, and parking there for
    the whole episode. Measured on ``port3`` at that constant::

        support hand         (0.420, +0.170)
        last station         (0.525, +0.105)   123 mm away
        port mouth           (0.586, +0.180)   167 mm away

    The grippers are about 100 mm across. That is not "an anchor at the far end"
    and a weave beside it; that is two hands in one place, and the cable between
    two pinches a hand's width apart cannot do anything but fight them. The
    sampler's own guard -- ``MIN_HAND_GAP`` = 230 mm -- never fired, because it
    was checking against a written-down pair no code put a hand at.

    So the inset comes back and the length goes in: 20 mm in from THIS rod's
    tip, wherever that is. On ``port3`` the hand moves to +0.330, which puts it
    248 mm from the last station and 224 mm from the port -- past the guard the
    sampler was already trying to enforce at the station, and the left arm
    reaches it with room to spare (its measured band runs to ``y <= +0.55`` at
    x = 0.42).

    **Only the terminated variants move.** The plain weaves keep the constant,
    because changing where the support hand grasps changes what their recorded
    scores mean and no result here is asking for that. ``weave6`` has the same
    defect waiting for it -- its last station stands at +0.328, past the
    constant -- and it is a separate thing to fix, on purpose, with its own
    before-and-after.
    """
    if variant not in WEAVE_TERMINALS:
        return WEAVE_ANCHOR_Y
    # 20 mm in from THIS rod's tip, and the tip is the spawn plus the rod --
    # which stopped being ``axis + half the rod`` when the two were separated.
    return terminated_spawn_y(variant) + WEAVE_CABLE_LENGTH[variant] - WEAVE_ANCHOR_INSET


# ---------------------------------------------------------------------------
# THE TWO ENDS OF A TERMINATED RUN.
# ---------------------------------------------------------------------------

WEAVE_TERMINALS = ("port2", "port3", "panel_l2", "port_easy", "port_medium")
"""Variants whose run is TERMINATED at both ends by a fitting.

A tuple rather than a flag on every weave, because adding the fittings to the
existing layouts would change what those documents score -- their goals are
three to six seat clauses and nothing else, and a scene carrying two more
fixtures that no clause mentions is a scene that is quietly harder for no
recorded reason. The terminated run is its own variant and says so.

One list, read by everything that has to know: the rod's ``anchors``, the
records minted below, ``weave_anchor_y`` above, and ``staging``'s sampler when
it moves a board and has to move the port with it. The branch this came from
kept a second copy in the sampler that had ``panel_l2`` missing, so every
episode of the panel yawed its cleats and left the socket where the document
declared it."""

WEAVE_PORT_DEPTH = 0.045
"""How far past its face the port counts a cable as landed [m].

The seat boxes' own 45 mm, deliberately: a station and a port are the same
question asked about different fixtures -- is the rod where this thing wants it
-- and a route scored against two different tolerances is a route whose score
says which clause it failed rather than how well it ran.
"""

WEAVE_PORT_STANDOFF = 0.075
"""How far past the last station the port sits, along the station line [m].

**This number is set by the LEAD-OUT ANGLE, and it went down before it went
up.** The port is beyond the last station in +y by this much and out in +x by
:data:`WEAVE_PORT_DEPTH` plus a half-block, so the angle the final leg makes
with the station line is ``atan2(dx, dy)`` -- and that angle is what decides
whether delivering the end into the port also drags the last crossing off its
spool. Measured on ``port3``, whose last seat sits 69 mm in x from the port::

    standoff   port           lead-out   what happens to the last seat
    0.025      0.586, 0.130    70 deg    all six bodies pulled out
    0.050      0.586, 0.155    54 deg
    0.075      0.586, 0.180    43 deg    the pull runs ALONG the run
    0.090      0.586, 0.195    38 deg

At 70 degrees the carry pulls the seated span sideways across its own fixture;
the whole point of an F1 spool is to hold a cable that is pressed INTO its
groove, and nothing holds one levered over the top. At 43 degrees the pull has
more along-the-run component than across it. 75 mm is the first standoff where
``port.x - seat.x`` and the standoff are the same size, which is the geometric
statement of that.

**It was 25 mm for a day, and the reason it was wrong is instructive.** The
working arm's ceiling falls as it reaches out -- see
:data:`~.staging.REACH_CEILING`, which gives ``y <= +0.21`` at x = 0.52 and
``y <= +0.16`` at x = 0.59 -- so a port beyond +0.16 is one the work arm cannot
plan to, and pulling the standoff in was the fix for that. It was the fix to the
wrong arm's problem. **The support hand carries the end into the port**, because
the end is in the support arm's half of the bench and not in the work arm's (see
``graphs/weave_vision/scripts/plug_port.py``), and the support arm reaches
``y <= +0.55`` across this whole band. The work arm's ceiling never applied to
the port at all.

**The two bounds this may not cross, from when it was ``WEAVE_PITCH``.** One
pitch beyond the board was fine at a 70 mm pitch; at 155 mm it puts the port at
y = +0.26 and breaks the task twice -- port3's route would run 763 mm of an
820 mm rod with nothing left for the bow, and the port projects outside a
1280-wide frame, so the perception could never see the thing it was routing to.
Both still bind. At 75 mm the route is 696 mm of that rod and the port is
comfortably in frame.

Deliberately NOT tied to the pitch: the port's distance from the board is a
property of the terminal, and the pitch is a property of the board.
"""

WEAVE_MOUTH_INSET = 0.008
"""How far the anchor bracket's centre sits PAST the rod's tip, along the run [m].

**The one number that decides whether the fixed end looks fixed**, and it was
wrong in the direction that reads worst. The bracket used to be placed a full
half-extent SHORT of the tip, so its near face met the rod's end cap and the
cable stopped dead against a wall it was welded nowhere near. From the camera
that is a rod lying beside a block, which is what it was.

A terminal is something a cable goes INTO.
:class:`~tasks/cable_route/assets/terminals.py::EntityCableTerminal` draws its
hood over ``local x`` in ``[-0.016, 0]``, so putting the plate's centre 8 mm
beyond the tip lands the rod's last 8 mm under the hood -- out of sight, which
is what terminating in a fitting looks like -- while the saddle straddles the
run 5 to 11 mm further out.

It is safe to bury the tip here in a way it was not safe to bury it in a box.
The old note recorded the reason for the old offset: "centred on the end, the
first two bodies are inside the box and the contact solver spends the episode
pushing them out -- the rod's start sat 14 mm inside a 20 mm half-extent." That
was a solid slab. Nothing in the bracket sits under the cable now, by
construction, so there is nothing to push out.
"""

WEAVE_TERMINAL_ENTITY = {"anchor": "cable_anchor", "port": "cable_port"}
"""Which fitting each end of the run is.

**The two ends stopped being the same part**, and this is where that is said.
They were both ``cable_block`` -- one 32 x 32 x 10 mm slab drawn twice -- which
was true of the physics and false about everything a cable does at an end: one
of these is where a run is CLAMPED and the other is where one is PUT, and a slab
is neither. The asset record is still one part with one footprint; what differs
is the fitting standing on it, which is what ``EntityCableTerminal`` and
``EntityCablePort`` draw.
"""


def weave_terminals(
    variant: str,
    *,
    start_xy: tuple[float, float] | None = None,
    stations: Sequence[tuple[float, float]] | None = None,
    yaw: float = 0.0,
    anchor_yaw: float = 0.0,
) -> dict[str, tuple[float, float, float]]:
    """Where the run's two ends terminate: ``{name: (x, y, yaw)}``.

    **Both are derived from the rod and the layout, never written down**, because
    a fitting whose position is a constant is one that stops meaning "the end of
    this run" the moment a layout changes -- and the terminated variants have
    their board moved every episode.

    **The arguments are what stops the two copies drifting.** This used to take a
    variant and nothing else, so it could only answer for the DECLARED rod and
    the DECLARED board; the sampler that moves both then restated the offsets on
    its own side, and they came apart exactly as restated numbers do -- the
    anchor went to a rod length that had been 0.70 and was now 0.82, putting the
    bracket 60 mm along a cable it was supposed to hold the end of. So the
    episode's own rod start and its own stations come in as arguments, and there
    is one function that knows where a terminal goes.

    The ANCHOR sits at the rod's start, :data:`WEAVE_MOUTH_INSET` past the tip
    along the run, with its mouth facing the way the run leaves. The PORT sits
    beyond the LAST station, out in +x, with its channel facing back down the run.

    Args:
        variant: The weave being staged.
        start_xy: Where the rod's first point actually landed, or ``None`` for
            the declared spawn.
        stations: Where the stations actually stand, or ``None`` for the declared
            layout.
        yaw: How far the board was turned [rad]. Applies to the port.
        anchor_yaw: The rod tangent's turn from its declared +y direction [rad].
    """
    laid = tuple(stations) if stations is not None else WEAVE_LAYOUTS[variant]
    # The DATUM, not the rod: a longer rod runs further past the port, it does
    # not start further back. See :data:`TERMINATED_SPAN`.
    spawn_y = terminated_spawn_y(variant) if variant in TERMINATED_SPAN else WEAVE_AXIS_Y - 0.5 * WEAVE_CABLE_LENGTH[variant]
    sx0, sy0 = start_xy if start_xy is not None else (WEAVE_CABLE_X, spawn_y)
    # The run leaves the anchor along +y -- the rod's own spawn direction, which
    # `weave_cable` states.
    run = math.pi / 2.0 + float(anchor_yaw)
    anchor = (sx0 + WEAVE_MOUTH_INSET * math.cos(run), sy0 + WEAVE_MOUTH_INSET * math.sin(run), run)
    sx, sy = laid[-1]
    # BEYOND the board, always, and never on the station's required side.
    #
    # Keying the port to the last crossing's side put it back toward the robot
    # whenever that side was -1 -- measured on the three-peg board, at x = 0.460
    # with the rod spawning at 0.412, so the run STARTED 6 mm off the block it
    # was supposed to be routed into. A port is where the run comes out, and on
    # this bench that is away from the robot whatever the last crossing did.
    #
    # Its channel faces back down the run, which on this bench is -x: the plug
    # stages over the mouth and descends, and a channel square to the approach is
    # one the end drops into rather than across.
    standoff = WEAVE_TERMINAL_GAP.get(variant, WEAVE_PORT_STANDOFF)
    port = (sx + (WEAVE_PORT_DEPTH + BLOCK_HALF[0]), sy + standoff, math.pi + yaw)
    return {"anchor": anchor, "port": port}


def _weave_seated_xy(variant: str) -> tuple[tuple[float, float], ...]:
    """Where the cable's centreline has to end up, station by station.

    The seat: this board's post radius plus the rod radius, on the station's
    required side. Used to measure the route rather than to drive anything --
    no policy reads it.
    """
    seat = seat_offset(variant)
    return tuple((x + seat * WEAVE_SIDES[i % len(WEAVE_SIDES)], y) for i, (x, y) in enumerate(WEAVE_LAYOUTS[variant]))


def _weave_wrap(variant: str, points) -> float:
    """Rod spent going AROUND the posts rather than between them [m].

    A polyline through the seat points turns at a corner; a cable turns on an
    arc of the post it is pressed against, radius :func:`seat_offset`. The
    difference is ``r * theta`` at every interior vertex, and on this board's
    262.3 degrees of total turning that is ~38 mm the straight-line measure
    never saw. Small next to the rod, not small next to the 20 mm the guard
    was clearing by.
    """
    radius = seat_offset(variant)
    total = 0.0
    for before, here, after in zip(points, points[1:], points[2:]):
        into = (here[0] - before[0], here[1] - before[1])
        away = (after[0] - here[0], after[1] - here[1])
        n_into, n_away = math.hypot(*into), math.hypot(*away)
        if n_into < 1e-9 or n_away < 1e-9:
            continue
        cosine = (into[0] * away[0] + into[1] * away[1]) / (n_into * n_away)
        total += radius * math.acos(max(-1.0, min(1.0, cosine)))
    return total


def _weave_slack(variant: str) -> float:
    """Rod left over once the route is taken up [m].

    A guard, not a statistic. A layout whose goal path is longer than its cable
    is an impossible task that builds, steps and scores zero for ever -- which
    looks exactly like a hard one, and is the failure mode a family of thirteen
    arrangements is most likely to introduce by arithmetic.

    The route measured is the SEATS and their tails. A terminated run's leg out
    to the port is not in it: the port sits 100 mm past the last seat and the
    rod has to reach it, so this is a lower bound on what those two variants
    need. Their own margin is written on :data:`WEAVE_CABLE_LENGTH`.
    """
    seated = list(_weave_seated_xy(variant))
    if variant in WEAVE_TERMINAL_GAP:
        ends = weave_terminals(variant)
        points = [ends["anchor"][:2], *seated, ends["port"][:2]]
        run = sum(math.dist(a, b) for a, b in itertools.pairwise(points))
        # The port goal requires three complete capsules beyond the route.
        terminal_length = 3 * WEAVE_CABLE_LENGTH[variant] / weave_segments(
            variant, WEAVE_CABLE_LENGTH[variant]
        )
        return WEAVE_CABLE_LENGTH[variant] - (
            run + _weave_wrap(variant, points) + terminal_length + WEAVE_ROUTE_SLACK[variant]
        )
    run = sum(math.dist(a, b) for a, b in itertools.pairwise(seated))
    # Plus the tails: the rod overhangs the outer stations at both ends, which is
    # what the weave pulls in as it takes up slack.
    return WEAVE_CABLE_LENGTH[variant] - (run + _weave_wrap(variant, seated) + 2 * WEAVE_TAIL)


# ---------------------------------------------------------------------------
# THE PLACED RECORDS. One per name each document uses, keyed by variant and
# then by the document's own entity name, so ``PLACEMENTS["weave3"]["peg0"]``
# is the record ``weave3.task`` types ``peg0`` as. Registered under
# ``<variant>_<name>`` -- the spelling the documents write.
# ---------------------------------------------------------------------------


def _placed(base: AssetRecord, *, variant: str, name: str, where: str, build) -> AssetRecord:
    """``base``'s declaration under ``<variant>_<name>``, with the factory that places it.

    Everything a document may ask of the type -- its features, its states, its
    published flags, its mesh -- is the type's and travels unchanged; only the
    name, the summary's tail and the ``build`` are this record's own.
    """
    return register_asset(
        replace(
            base,
            name=f"{variant}_{name}",
            summary=f"{base.summary.rstrip('.')} -- {variant}'s {name}, {where}.",
            build=build,
        )
    )


def _weave_records(variant: str) -> dict[str, AssetRecord]:
    """Every entity a weave document names, as the record that places it.

    Raises:
        ValueError: For a weave whose route does not fit its rod.
    """
    length = WEAVE_CABLE_LENGTH[variant]
    slack = _weave_slack(variant)
    if slack < WEAVE_MIN_SLACK:
        raise ValueError(
            f"{variant} needs {length - slack:.3f} m of cable and has {length:.3f}: the route is "
            f"longer than the rod, so no policy could make it. Lengthen WEAVE_CABLE_LENGTH["
            f"{variant!r}], bring WEAVE_PITCH in, or move WEAVE_CABLE_X nearer the stations."
        )
    # ONE FIXTURE PER STATION, and which one is the only thing that differs
    # between a spool board and the industrial panel. The stations keep the name
    # ``peg{i}`` across both: every goal clause, every sampler and every
    # perception node addresses a station by that name, and renaming the part
    # would rename the question.
    #
    # `cable_peg_round` rather than the `cable_peg` the wrap stands on. A weave
    # is the one thing on this bench that a plain post is actively bad at: a
    # cable laid against one rides up it under tension and lifts off the top,
    # and a fixture that captures the rod is what holds a route the hand has let
    # go of. The spool does it with a groove; the cleat with an overhanging cap.
    is_cleat = variant in CLEAT_VARIANTS
    station_type = CABLE_CLEAT if is_cleat else CABLE_PEG_ROUND
    station_build = cleat if is_cleat else spool
    out: dict[str, AssetRecord] = {
        f"peg{i}": _placed(
            station_type,
            variant=variant,
            name=f"peg{i}",
            where=f"station {i} at x = {xy[0]:.3f} m, y = {xy[1]:.3f} m",
            build=partial(station_build, f"peg{i}", xy, cell=CELL),
        )
        for i, xy in enumerate(WEAVE_LAYOUTS[variant])
    }
    # The rod is centred on the weave's axis, on the robot's side of the stations.
    # Declared at its own length and section rather than borrowed from
    # ``cable_rod``: the reference's record says 0.91 m at 11 mm and carries the
    # wrap's anchor, and neither is true of a weave's rod. What is the type's --
    # the landmarks, the straightness state, the latch flags -- is restated
    # from it, so the two cannot drift apart.
    segments = weave_segments(variant, length)
    terminated = variant in WEAVE_TERMINALS
    out["cable"] = register_asset(
        Rod.from_length(
            name=f"{variant}_cable",
            length=length,
            radius=WEAVE_CABLE_RADIUS,
            summary=(
                f"{variant}'s rod: {length:.2f} m of {segments} capsules at "
                f"{WEAVE_CABLE_RADIUS * 2000:.0f} mm, laid 100 mm short of its stations at t = 0"
                + (" with its first body welded to the anchor bracket." if terminated else ".")
            ),
            points={"end_a": 0.0, "end_b": 1.0},
            tags=CABLE.tags,
            states=CABLE.states,
            published=CABLE.published,
            notes=(
                "cable_rod's material at a thinner section and a shorter pitch, with the bend "
                "rescaled to match -- see weave_bend. Under the same coupled arrangement, for the "
                "same reasons; cable_rod's notes are this rod's too."
            ),
            provenance=(
                "This stack: cable_rod thinned to bend around ManipulationNet's F1 and cut to the "
                "route; see WEAVE_CABLE_LENGTH for the length and _weave_slack for the guard on it."
            ),
            build=partial(
                weave_cable,
                variant=variant,
                length=length,
                # The DATUM for a terminated run, whose anchor is welded here and
                # must not walk when the rod grows. See :data:`TERMINATED_SPAN`.
                xy=(
                    WEAVE_CABLE_X,
                    terminated_spawn_y(variant) if variant in TERMINATED_SPAN
                    else WEAVE_AXIS_Y - 0.5 * length,
                ),
                anchored=terminated,
                cell=CELL,
            ),
        )
    )
    if terminated:
        # The two ends, at the poses ``weave_terminals`` derives from this
        # variant's declared rod and declared board. An episode that moves the
        # board moves them through the SAME function rather than through a
        # second copy of these offsets.
        for name, (x, y, yaw) in weave_terminals(variant).items():
            out[name] = _placed(
                CABLE_BLOCK,
                variant=variant,
                name=name,
                where=f"at x = {x:.3f} m, y = {y:.3f} m, facing {math.degrees(yaw):.0f} deg",
                build=partial(
                    terminal,
                    name,
                    (x, y),
                    entity=WEAVE_TERMINAL_ENTITY[name],
                    yaw=yaw,
                    cell=CELL,
                ),
            )
    return out


def _wrap_records() -> dict[str, AssetRecord]:
    """Every entity the wrap document names, as the record that places it."""
    out: dict[str, AssetRecord] = {
        f"peg{i}": _placed(
            CABLE_PEG,
            variant="wrap",
            name=f"peg{i}",
            where=f"at x = {xy[0]:.2f} m, y = {xy[1]:.2f} m",
            build=partial(post, f"peg{i}", xy, cell=CELL),
        )
        for i, xy in enumerate(PEG_XY)
    }
    # ``anchor_arm`` is the binding's own support arm (the suite file's
    # ``(:support-arm left)``), stated rather than left to the spec's
    # default: the default is the embodiment's last declared arm, and
    # yam_bimanual declares left first.
    out["cable"] = _placed(
        CABLE,
        variant="wrap",
        name="cable",
        where=f"along x = {CABLE_X:.2f} m with the second hand on it at y = {ANCHOR_Y:.2f} m",
        build=partial(wrap_cable, x=CABLE_X, anchor_y=ANCHOR_Y, anchor_arm="left", cell=CELL),
    )
    return out


PLACEMENTS: dict[str, dict[str, AssetRecord]] = {
    **{variant: _weave_records(variant) for variant in WEAVE_LAYOUTS},
    "wrap": _wrap_records(),
}
"""Every variant's placed records, by the entity name its document uses.

The table ``task.py`` used to materialise, as registered records the generic
loader builds through: ``weave3.task`` types ``peg0 - weave3_peg0`` and
``build_entities`` asks this record for the spool at station 0. A document
added without a row here names records that do not exist, and the typechecker
reports the gap by name."""
