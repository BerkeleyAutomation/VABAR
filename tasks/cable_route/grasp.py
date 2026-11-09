# SPDX-License-Identifier: Apache-2.0
"""Where on the cable to take hold, and whether there is enough of it to get there.

**Why this is its own module.** The choice used to be one line inside
``routine.plan_station_stages``::

    nearest = rod[int(np.argmin(np.abs(rod[:, 1] - sy)))]

-- the point of the perceived centreline closest to the station *in y*, with no
entry point, no arguments, and nothing anywhere that asked whether it was a
sensible place to grab. Two things follow from that, and both were measured
rather than guessed:

* **A truncated fit grabs its own endpoint.** ``argmin`` always returns
  something. When the fitted curve stops short of the outermost station -- the
  vision body's own notes record it choosing "a 148 mm fragment of a 700 mm
  rod" -- the winner is wherever the polyline happened to end, which can be a
  hundred millimetres from the station. Nothing downstream notices: the grasp is
  perfectly reachable, so the IK residual gate passes and the hand descends on
  cable that is not there.
* **And the same point picks the motion.** ``routine`` reads ``nearest[0]``
  again to decide drag-versus-lift-over. A grasp taken off the end of a short
  fit therefore also chooses the template, and a drag emitted where a lift-over
  was owed can never carry the rod across its post -- the station is lost for
  the rest of the episode.

**And the slack arithmetic ran too late to matter.** The one place in this task
that asks how much cable lies between two points is ``plan_support_feed``, which
runs *after* the grasp is fixed and whose only remedy is walking the support
hand, clamped at 90 mm. If the grasp left too little material there was no way
back. Here it is a precondition instead: a candidate that cannot reach the seat
without pulling the anchor is rejected before anything moves.

Everything below is pure -- a polyline and some numbers in, a choice out -- so
the oracle and the vision body get an identical rule, and it runs under
``--no-privileged`` because it reads nothing but what perception published.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

MAX_STATION_OFFSET = 0.060
"""How far from the station, along the run, a grasp may be taken [m].

The guard the bare ``argmin`` never had. On the shipped layouts the stations sit
154 mm apart, so a grasp more than 60 mm off is closer to being abreast of the
*neighbouring* station than its own, and on a healthy fit the winner is within a
few millimetres. Anything past this is the truncated-fit signature rather than a
grasp worth taking, and the caller is told so instead of being handed a point.
"""

SLACK_MARGIN = 0.015
"""Cable in hand beyond the straight-line need, before a candidate is believed [m].

The rod is held at its far tip and routed from there outwards, so the material
between the anchor and the grasp is the only cable available to reach the seat.
Ask for exactly the straight-line distance and the span is a taut string at the
moment of release; ask for a little more and it arrives slack. Fifteen
millimetres is the smallest margin that stayed positive across the shipped
layouts once the anchor's own 2.8 mm surface-to-axis bias is allowed for.
"""


class Choice(NamedTuple):
    """What was chosen, and everything needed to argue with the choice."""

    ok: bool
    point: np.ndarray
    index: int
    offset_m: float
    slack_m: float
    need_m: float
    reason: str


def _arclength(rod: np.ndarray) -> np.ndarray:
    """Cumulative arc length along the polyline [m], starting at zero."""
    steps = np.linalg.norm(np.diff(rod, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def choose_grasp(
    rod: Any,
    station_xy: Any,
    seat_xy: Any,
    anchor_xy: Any = None,
    *,
    axis: Any = (0.0, 1.0, 0.0),
    max_offset: float = MAX_STATION_OFFSET,
    slack_margin: float | None = None,
) -> Choice:
    """Pick the point on ``rod`` to take, for a station that must reach ``seat_xy``.

    The rule, in the order the two conditions actually bind:

    1. **Closest to the station along the run.** ``axis`` is the direction the
       cable runs and the posts are spaced along -- the axis the station's own
       position is meaningful in. Projecting onto it rather than taking a 3-D
       distance is deliberate: the cable's *lateral* offset is the thing the
       station is about to change, so including it would score a span for being
       already-wrong-side, and a bent rod's nearest 3-D point can sit on the
       returning limb.
    2. **With enough material behind it.** The support hand pins the far tip, so
       the cable between that anchor and the candidate is all the length there is
       to reach the seat with. A candidate whose arc to the anchor is shorter
       than the anchor-to-seat straight line plus ``slack_margin`` is skipped and
       the search walks *away* from the anchor -- more cable, at the cost of
       being further from the station -- until one qualifies or the offset gate
       stops it.

    **``slack_margin=None`` reports without moving the grasp, and that is the
    default.** The slack arithmetic then still runs -- ``slack_m``, ``need_m``
    and a ``reason`` come back either way, and the pay-out downstream is glad of
    them -- but the point returned is the nearest in-gate candidate, which is
    what the routine has always taken. Passing a margin opts into the *walk*:
    skipping starved candidates for ones with more cable behind them.

    That split is deliberate rather than timid. Measured on ``weave4`` over six
    episodes with the camera, the walk scored **0/6 against a 1/6 baseline** --
    at that sample size the two are indistinguishable, which is exactly why it
    is not on by default: an unmeasured behaviour change should not ride in on
    the back of a measured bug fix. The offset gate below is on always, because
    it only ever *reports*; it is the caller that decides what a refusal means.

    ``anchor_xy`` may be ``None``, in which case the far end of the polyline
    along ``axis`` is used. That is where the support hand is put, so the two
    agree on the shipped schedule, and it keeps this callable from a body that
    has a centreline but no arm state.

    Returns a :class:`Choice`. ``ok`` false means *do not grasp on this
    evidence* -- the caller is expected to refuse the station rather than
    descend on a point this function has already said it does not believe.
    """
    rod = np.asarray(rod, dtype=np.float64).reshape(-1, 3)
    if len(rod) < 2:
        return Choice(False, np.zeros(3), -1, float("inf"), 0.0, 0.0,
                      f"a centreline needs at least two points, got {len(rod)}")

    u = np.asarray(axis, dtype=np.float64)
    u = u / max(float(np.linalg.norm(u)), 1.0e-12)
    station = np.asarray(station_xy, dtype=np.float64)[:2]
    seat = np.asarray(seat_xy, dtype=np.float64)[:2]

    s = _arclength(rod)
    along = rod @ u
    station_along = float(np.asarray([station[0], station[1], 0.0]) @ u)

    if anchor_xy is None:
        # The far end ALONG THE RUN, which is the tip the support hand takes.
        i_anchor = int(np.argmax(along))
        anchor = rod[i_anchor][:2]
    else:
        anchor = np.asarray(anchor_xy, dtype=np.float64)[:2]
        i_anchor = int(np.argmin(np.linalg.norm(rod[:, :2] - anchor, axis=1)))

    # What reaching the seat costs, measured from the anchor: the cable has to
    # span that straight line at minimum, whatever route it takes getting there.
    margin = SLACK_MARGIN if slack_margin is None else float(slack_margin)
    need = float(np.linalg.norm(seat - anchor)) + margin
    walk = slack_margin is not None

    offsets = np.abs(along - station_along)
    # Nearest first, so the ordinary case takes the ordinary point and the walk
    # only happens when the ordinary point is too close to the anchor. STABLE,
    # so an exact tie -- a station squarely between two samples of the polyline,
    # which happens whenever the fitter's spacing divides the station pitch --
    # goes to the lower index, the way `argmin` broke it before this existed.
    # Without that the default is *not* the old behaviour on those stations.
    order = np.argsort(offsets, kind="stable")
    first: Choice | None = None
    # The most cable any in-gate candidate had, so a refusal can quote the best
    # case rather than the nearest one -- "40 mm" when a point at the edge of
    # the gate had 100 reads as a worse problem than it is, and the two call for
    # different responses (pay out slack, versus give up on the station).
    best_in_gate: Choice | None = None
    for idx in order:
        i = int(idx)
        offset = float(offsets[i])
        slack = float(abs(s[i] - s[i_anchor]))
        if first is None:
            first = Choice(False, rod[i], i, offset, slack, need, "")
        if offset > float(max_offset):
            # Ordered by offset, so everything after this is further still.
            break
        if best_in_gate is None or slack > best_in_gate.slack_m:
            best_in_gate = Choice(False, rod[i], i, offset, slack, need, "")
        if slack >= need or not walk:
            return Choice(slack >= need, rod[i], i, offset, slack, need,
                          f"{offset * 1e3:.0f} mm off the station, {slack * 1e3:.0f} mm of "
                          f"cable behind it against {need * 1e3:.0f} needed")

    # Nothing qualified. Say which of the two conditions did the refusing,
    # because they call for opposite responses: too far means the fit is short
    # of this station and the station should be skipped; too little cable means
    # the anchor has to pay some out first.
    best = first or Choice(False, rod[0], 0, float("inf"), 0.0, need, "")
    if best.offset_m > float(max_offset):
        return best._replace(
            reason=f"the centreline's nearest point is {best.offset_m * 1e3:.0f} mm from the "
                   f"station along the run, past the {max_offset * 1e3:.0f} mm gate -- the fit "
                   f"does not reach this station")
    short = best_in_gate or best
    return short._replace(
        reason=f"the most cable any point within {max_offset * 1e3:.0f} mm of the station has "
               f"behind it is {short.slack_m * 1e3:.0f} mm, against {need * 1e3:.0f} needed -- "
               f"the anchor has to pay out {(need - short.slack_m) * 1e3:.0f} mm first")
