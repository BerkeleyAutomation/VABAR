# SPDX-License-Identifier: Apache-2.0
"""How hard a weave is, in two numbers and one band.

The weave family is one problem in several arrangements, and until now the only
statement of which arrangement was harder was prose -- ``weave_stagger``'s
document calls itself "the hard member of the family" and ``weave_arc``'s says an
arc "is harder", and neither is a quantity anybody could check, order or
reproduce on a layout that does not ship yet. This module is that quantity.

**Two axes, one scalar.**

``N`` -- the number of stations. It is the reward's denominator (the goal scores
the fraction of seats filled), the number of sequential subgoals a plan has, and
the horizon over which a station worked early can be undone by a station worked
late.

``Theta`` -- the **accumulated turning angle** of the scored route [deg]: the
total absolute curvature the rod's centreline must acquire, measured from the
straight rod the episode starts with. It is what the route asks of the *cable*
rather than of the *schedule*.

They are not independent, and that is the point rather than a defect. For a
layout with a constant pitch and a constant lateral excursion,

    Theta = 2 * (N - 1) * alpha,     alpha = atan(excursion / pitch)

so peg count and crossing sharpness are two ways of buying the same turning.
:func:`profile` reports all three -- ``N``, ``obliquity`` (``alpha``, the part of
the geometry that does not know how many pegs there are) and ``turning``
(``Theta``, the aggregate) -- and the LEVEL is a band in ``Theta`` alone. A level
says how much turning; ``N`` and ``alpha`` say where it came from.

**Why turning is the right thing to measure**, and not route length or peg count
on its own. Three separate mechanisms in this scene are proportional to it:

*Retention.* A station holds the route only where the route turns at it. The
per-station turn ``theta_i`` is that station's retention margin, and a station
with a small one is a seat the rod leaves as soon as the hand does -- which the
goal scores, because nothing in this family latches.

*Springback.* A Kirchhoff rod turning ``theta`` over a fillet of radius ``rho``
stores ``EI * theta / (2 * rho)`` of bending energy, so the whole route holds
``EI * Theta / (2 * rho)``. That energy is what tries to undo the weave in the
settle after release, and the settle is when the verdict is read.

*Capstan.* Tension delivered through the route falls as ``exp(-mu * Theta)``. At
``weave3``'s 12.6 deg a pull at the free end reaches the far station at 94%
strength; at ``weave6``'s 232 deg it reaches 30%. Past roughly half a turn a
policy that *drags* the rod through the route stops working and one that *pays
slack out* station by station starts to be the only thing that does -- which is
a change in kind, not in degree, and is why the bands are placed where they
are.

**Where the bands are.** At a quarter turn, a half turn and a full turn::

    L1  Theta <  90 deg     one corner
    L2   90 <= Theta < 180  a crossing
    L3  180 <= Theta < 360  a serpentine
    L4  360 <= Theta        a coil

Round fractions of a turn rather than fitted percentiles, because a band fitted
to the four arrangements that happen to ship is a band that moves when a fifth
does. A benchmark's difficulty axis has to be statable before the tasks are, and
these are: they are properties of the curve, not of this bench, this rod or this
robot, and a layout on somebody else's table is placed by the same arithmetic.

**A level is a distribution of layouts, not a layout.** That falls out of
defining difficulty on the geometry: if a board's level can be computed before
anybody stages it, boards can be *drawn* to a level rather than written down.
``staging:WEAVE_BOARD`` does exactly that -- it samples the spacing between
every pair of stations, each station's stagger and wander, the whole board's
rotation and its centroid, and keeps a draw only if :func:`profile` puts it in
the band with margin and every other constraint holds. The four ``board_l*``
variants are that sampler beside each of the four fixed weaves ``weave3`` to
``weave6``, and a score on one of them is a score on the *level* where a score
on ``weave5`` is a score on ``weave5``.

Both are wanted, which is why both ship. A regression needs a layout that does
not move; a generalisation claim needs one that does.

**The constants moved and the ladder moved with them.** Every number in this
module is a function of the layout table and the seat offset, and both changed:
``WEAVE_PITCH`` went from 70 mm to 155 and the rod was thinned from 10 mm to 7,
so the excursion over the gap -- which is the whole of the turning -- fell by
more than half at every station. Nothing in the suite reaches L4 any more,
``weave3`` and ``weave4`` share L1, and :data:`LADDER` is three rungs rather
than four. The bands did not move: they are quarter, half and whole turns of the
tangent, which is the property that made them worth stating before the tasks
were. What moved is which arrangement lands in which, and the honest response to
that is to re-read the table rather than to re-fit the bands.

Read the ladder::

    uv run python tasks/cable_route/rigs/difficulty_ladder.py
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

__all__ = [
    "BANDS",
    "LADDER",
    "Profile",
    "accumulated_turning",
    "level",
    "profile",
    "route",
    "turns",
]


BANDS: tuple[tuple[int, float, str, str], ...] = (
    (1, 90.0, "corner", "one reversal; a single crossing between two stations"),
    (2, 180.0, "crossing", "a repeated crossing; passes begin to disturb each other"),
    (3, 360.0, "serpentine", "depth: the rod is carried toward and away as much as across"),
    (4, math.inf, "coil", "capstan regime; the route must be paid out, not dragged"),
)
"""``(level, upper bound on Theta [deg], name, what the band adds)``.

Half-open below: a profile is at the first level whose bound it is under. The
bounds are a quarter, a half and a whole turn of the tangent."""


@dataclass(frozen=True)
class Profile:
    """One layout's difficulty, as measured rather than as claimed.

    Attributes:
        stations: How many pegs -- the reward's denominator.
        turning: Accumulated turning angle of the scored route [deg].
        obliquity: ``turning / (2 * (stations - 1))`` [deg] -- the mean
            half-turn per crossing, and the part of the geometry that does not
            depend on how many stations there are.
        retention: The turn at each station in order [deg], signed. The smallest
            magnitude is the weakest seat in the route, which is a sharper thing
            to know than the total.
        length: Length of the scored route including its tails [m].
        level: Which band :data:`BANDS` puts ``turning`` in.
    """

    stations: int
    turning: float
    obliquity: float
    retention: tuple[float, ...]
    length: float
    level: int

    @property
    def band(self) -> str:
        """The level's short name."""
        return next(name for lvl, _, name, _ in BANDS if lvl == self.level)

    @property
    def weakest(self) -> float:
        """The smallest station turn [deg] -- the seat the route holds worst."""
        return min(abs(v) for v in self.retention) if self.retention else 0.0

    @property
    def capstan(self) -> float:
        """Fraction of a pull at the free end that reaches the far station.

        ``exp(-mu * Theta)`` at ``mu = 0.3``, which is the coefficient the
        reference's rod-on-plastic contact is set up with. Reported rather than
        gated: it is the reading that says *why* a level is a different problem
        and not a longer one.
        """
        return math.exp(-0.3 * math.radians(self.turning))


def route(
    stations,
    sides,
    *,
    seat: float,
    tail: float,
    axis=(0.0, 1.0),
):
    """The scored route: the polyline the goal's seats describe, with its tails.

    The seats are boxes in each station's own frame reaching from 3 mm off the
    axis out to 45 mm on the required side, and a rod that satisfies one passes
    through it against the shaft -- so the point the route is measured through is
    the station's centre offset by ``seat`` on its side, which is where a seated
    rod's centreline rests.

    **The tails run along the rod's spawn direction**, and that is a fact about
    the scene rather than a convention chosen here. Every weave lays its rod
    straight along ``+y`` and 80 mm short of the stations; the route it must
    finish in still trails off along ``+y`` at both ends. So the turning measured
    between those tails is exactly the turning the episode has to *install* --
    the total change in tangent direction between the start state and the goal
    state -- and not an artefact of where somebody decided the curve begins.

    Args:
        stations: Station centres ``[(x, y), ...]`` in world coordinates [m].
        sides: ``+1``/``-1`` per station: which side of it the rod must finish
            on, along the station's own ``x``.
        seat: Distance from a station's axis to a seated centreline [m] --
            the shaft radius plus the rod radius.
        tail: How far the rod trails past the outermost station [m].
        axis: The rod's spawn direction in the table plane, normalised.

    Returns:
        The polyline as ``[(x, y), ...]``, ``len(stations) + 2`` long.
    """
    seated = [(x + seat * sides[i % len(sides)], y) for i, (x, y) in enumerate(stations)]
    ax, ay = axis
    head = (seated[0][0] - ax * tail, seated[0][1] - ay * tail)
    foot = (seated[-1][0] + ax * tail, seated[-1][1] + ay * tail)
    return [head, *seated, foot]


def turns(polyline) -> tuple[float, ...]:
    """The signed turn at every interior vertex of ``polyline`` [deg].

    Signed, because the sign is what says a weave alternates: a route whose turns
    all share a sign is a curve bending one way, which is a hook rather than a
    weave however much total turning it has. The magnitude is what
    :func:`accumulated_turning` sums; the sign is what
    :func:`~tasks.cable_route.difficulty.profile` keeps so the caller can see
    both.
    """
    out = []
    for i in range(1, len(polyline) - 1):
        ax = polyline[i][0] - polyline[i - 1][0]
        ay = polyline[i][1] - polyline[i - 1][1]
        bx = polyline[i + 1][0] - polyline[i][0]
        by = polyline[i + 1][1] - polyline[i][1]
        out.append(math.degrees(math.atan2(ax * by - ay * bx, ax * bx + ay * by)))
    return tuple(out)


def accumulated_turning(polyline) -> float:
    """Total absolute curvature of ``polyline`` [deg] -- the ``Theta`` above.

    ``sum |theta_i|``, which for a polyline is the discrete form of
    ``integral |kappa| ds``. Two properties are what make it usable as a
    benchmark axis: it is invariant to rigid motion of the whole layout, so a
    weave staged on another bench is the same difficulty; and it is
    dimensionless, so it does not change when the pitch, the rod gauge or the
    table does. Route *length* has neither property, which is why the axis is
    this and not that.
    """
    return sum(abs(v) for v in turns(polyline))


def level(turning: float) -> int:
    """Which band a turning angle falls in. See :data:`BANDS`."""
    return next(lvl for lvl, bound, _, _ in BANDS if turning < bound)


def profile(
    stations,
    sides,
    *,
    seat: float,
    tail: float,
    axis=(0.0, 1.0),
) -> Profile:
    """Measure one layout. Arguments are :func:`route`'s."""
    poly = route(stations, sides, seat=seat, tail=tail, axis=axis)
    theta = turns(poly)
    total = sum(abs(v) for v in theta)
    n = len(stations)
    return Profile(
        stations=n,
        turning=total,
        obliquity=total / (2 * (n - 1)) if n > 1 else 0.0,
        retention=theta,
        length=sum(math.dist(a, b) for a, b in itertools.pairwise(poly)),
        level=level(total),
    )


LADDER: dict[int, str] = {1: "weave3", 2: "weave5", 3: "weave6"}
"""The canonical member of each level that this suite reaches.

**Three rungs, not four, and both facts are measurements rather than choices.**
Every profile below is what :func:`measured` returns at the shipped layout table
and the shipped seat offsets, re-derived rather than remembered::

    weave3          N=3  scatter, d=0        12.6 deg   L1   <- rung
    weave4          N=4  line,    d=0        36.5 deg   L1
    weave5          N=5  stagger, d=12.5 mm  119.9 deg  L2   <- rung
    weave6          N=6  stagger, d=25 mm    232.2 deg  L3   <- rung
    weave_arc       N=4  arc r=0.22          160.1 deg  L2
    weave_stagger   N=4  stagger, d=25 mm    139.3 deg  L2
    panel_l2        N=5  cleats, uneven      142.9 deg  L2
    port2           N=2  line                12.2 deg   L1
    port3           N=3  scatter             12.6 deg   L1
    board_l1        N=3  line                24.3 deg   L1
    board_l2        N=4  line                36.5 deg   L1
    board_l3        N=5  stagger, d=12.5 mm  119.9 deg  L2
    board_l4        N=6  stagger, d=25 mm    232.2 deg  L3

``weave3`` was L1 before and still is. ``weave4`` was the L2 rung at a 70 mm
pitch and is not at 155 -- it shares L1 with ``weave3`` now, at three times the
turning and one more station, which is a real difference between two members of
one band rather than a rung. ``weave5`` and ``weave6`` each dropped a level.
Nothing in the suite is L4: reaching a full turn on this pitch needs either a
ninth station or a stagger deeper than the working arm's envelope allows (see
:data:`~.staging.BOARD_X`, which is 80 mm wide), and neither is a thing to add
to make a table look complete.

**The two axes still rise together, which is what the ladder is for.** 3, 5 and
6 stations at 0, 12.5 and 25 mm of stagger depth, and the turning goes up faster
than either -- 12.6, 119.9, 232.2 -- because the two multiply. A ladder that
raised only one axis would measure only one thing.

The other arrangements are **off the ladder on purpose**, and each pairs with a
rung it shares a level with:

``weave_arc``, ``weave_stagger`` and ``panel_l2`` all sit at L2 beside
``weave5`` and differ in where the turning comes from. ``weave_stagger`` buys
its 139 degrees with four stations at 23.2 degrees of obliquity where
``weave5`` buys 120 with five at 15.0 -- how many crossings against how sharp
each one is, which is a question the level alone cannot answer and exactly what
a benchmark wants a pair for. ``weave_arc`` buys 160 with turns that are not
equal to each other at all: 35.3, -29.1, -51.0, 44.8 against the line's even
-15.0, +30.0, -30.0, +30.0, -15.0. ``panel_l2`` buys 143 from the FIXTURE
rather than from the board -- five cleats on a 74 to 88 mm spacing, whose 9.5 mm
standoff puts a seated cable 13 mm off each axis where a spool puts it 8.25.

**The pairs are no longer matched to within 5%**, and they were at the old
pitch. ``weave_stagger`` is 16% off ``weave5`` and ``weave_arc`` 34%; both are
comfortably inside L2 and neither is the same number. A gap between two of them
is attributable to the level plus that spread, which is weaker than the claim
the old table could make and is what the geometry now supports.
"""


def measured() -> dict[str, Profile]:
    """Every shipped weave's profile, read off the loader's own layouts.

    The one entry point that does not take geometry: it goes and gets the
    layouts the tasks are actually built from, so a table printed from this
    cannot describe a bench nobody staged. ``rigs/difficulty_ladder.py`` is the
    command that prints it.
    """
    from .assets.layout import (  # noqa: PLC0415
        WEAVE_LAYOUTS,
        WEAVE_SIDES,
        WEAVE_TAIL,
        seat_offset,
    )

    # PER VARIANT, because the suite stages two fixtures of different thickness.
    # A single ``ROUND_SHAFT_RADIUS + WEAVE_CABLE_RADIUS`` measured the
    # industrial panel as though its 9.5 mm cleats were 4.75 mm spools and
    # reported L1 for a board that is L2. See ``seat_offset``.
    return {
        name: profile(layout, WEAVE_SIDES, seat=seat_offset(name), tail=WEAVE_TAIL)
        for name, layout in WEAVE_LAYOUTS.items()
    }
