"""Work the gear down its peg square to its link, let go, then step aside and look.

The gear is held in the jaws, so the TCP going down is the gear going down,
and a leg that does not lower the TCP is a gear that is stuck -- on the peg
(the compliant grip lets it tilt and a tilted bore binds) or on its
neighbour's teeth. Either way it is released where it stopped: on the shaft it
seats itself. One pass per visit; a gear the camera then finds perched at its
peg rather than seated is picked up again from there on the next visit (the
cursor counts visits).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gearboard as gb  # noqa: E402

LEG_STEPS = 8
STEP_M = 0.002
LEAD_M = 0.0045
"""Aim each leg this far past its target: the servo counts a target within its arrival ball as reached."""
MAX_LEGS = 48
STALL_STEPS = 5
FALL_EPS = 2.0e-4
RELEASE_SETTLE = 60
"""Steps the released gear is given on its own peg before the hand leaves [steps].

**A descent that stops short is let go, once, and the camera judges it.** The
Franka hand's version re-gripped the released gear and pushed on, up to three
times; on the 2F-85 that loop never ends well and costs about 1,500 steps a
station. Measured on gearmesh-easy-single-01 (outputs/gear_levels/rq_v3): the
gripped gear went 4.4 mm down the peg and wedged -- two pads at 10 N let it
tilt a degree or two, and a 16 mm bore on a shaft with 0.3 mm of clearance
binds past about one degree -- then, let go, it dropped 12.7 mm to its seat
within 40 steps. The re-grip that followed closed on a SEATED gear with the
tool centre 7 mm above the modelled bottom (the pads slide down the rim as
they close, so the hold is not the pick's), which the arrival band read as a
jam, and two more releases and re-grips changed nothing. On the three-gear
board the same loop spent 18,000 steps to seat two.

So a stall short of arrival is one release: the jaws open, the gear settles
under its own weight on the cone and shaft that centre it and let it turn to
find the space, and the hand steps aside so the overhead camera can say
seated or perched. A perched gear is picked up off its peg on the station's
next visit, a third of a tooth pitch further round."""
RELEASE_PHASES = (0.25, -0.25, 0.5)
"""Where the released gear is turned to, in tooth pitches from the across-the-chain heading, each turn
followed by another release.

**Centred on the heading that keeps the fingers off the neighbours.** The jaws
close across the chain, and every turn here stays within half a pitch of that
-- 11 degrees on a 16-tooth gear -- so the pads never swing round towards the
gear next door. The descent itself turns through one pitch, from half a pitch
before that heading to half a pitch past it (DESCENT_LEGS).

**Because a perched gear does not find the space by itself often enough.** The
teeth carry mu 1.4, so a gear whose tips landed on a seated neighbour's tips
stays there under its own 30 g (measured on the Franka hand: a seated
neighbour drops the seat rate from 81 % to 27 %), and the chain it has to mesh
into cannot turn -- the pinion's velocity servo holds it. What can turn is the
gear: the jaws take it where it rests, the wrist turns it a quarter of a tooth
pitch about its own peg, and the jaws open again. Three turns and the first
release cover a whole pitch at four phases, for about 200 steps each -- a
station's next visit costs 2,500. A gear that seated on an earlier release is
simply turned against the locked chain and slips in the pads."""
ARRIVE_M = 0.006
"""The hand counts as down when within this of the modelled release height. Measured on the Franka hand: bore
friction against the compliant servo stops it 4.7-5.2 mm short at a free station and the release lets gravity
finish; a gear whose teeth landed on its seated neighbour's stops 6.9 mm short. On the 2F-85 a gripped gear
wedges higher (see RELEASE_SETTLE), and the release is the same act either way."""
JAM_STALLS = 6
"""Stalled steps, short of arrival, before the gear is let go where it stopped."""

DESCENT_LEGS = 24
"""Legs over which the descent turns the gear through one whole tooth pitch, half a pitch either side of
the across-the-chain heading, so every phase is passed once and the pads stay off the neighbours."""

TURN_PER_LEG = 0.045
"""How far the gear was turned per descent leg [rad] before the wrist roll was honoured on this pairing.

Rather more than a tooth pitch over a descent that reaches the seat in about
thirty legs -- a 24-tooth gear's pitch is 0.26 rad -- so the gear passes each
clock once on its way down instead of being swept across it twice a cycle. The
oscillation this replaces was measured against it: over the six two-gear boards
tried both ways, sweeping seated 4 of 12 gears and turning steadily seated 4 of
12, while holding one fixed clock seated 2 of 6. What actually recovers a jam is
letting the gear go, not how it was turned into it."""


def _descend(ctx, station, base_q, x, y):
    """Step down square to the chain, the gear's height read off the hand's.

    A descent that stops short of the arrival band is released where it
    stopped (see :data:`RELEASE_SETTLE`); one that arrives is released at the
    bottom. Either way the node ends with the gear on its peg and the jaws
    open, and the camera says which of the two it is.
    """
    z_bottom = float(station["drop_tcp_z"])
    sweep = float(station["sweep"])
    z_top = z = gb.tcp(ctx)[2]
    stalled, stalls_total, steps = 0, 0, 0
    lowest = z
    tz = z
    heights = []
    while steps < MAX_LEGS:
        steps += 1
        # A steady turn, not a shake. The gear has to arrive at the one clock
        # its neighbour's spaces will accept, and it is held square to the chain
        # the whole way down; what changes is a slow rotation through rather
        # more than one tooth pitch over the descent, so every phase is passed
        # through once instead of being swept back and forth across.
        turn = -0.5 * sweep + min(1.0, steps / DESCENT_LEGS) * sweep
        q = gb.quat_yaw(base_q, turn)
        gb.leg(ctx, gb.pose(x, y, z, q), LEG_STEPS)
        z_next = max(z_bottom, z - STEP_M)
        gb.leg(ctx, gb.pose(x, y, z_next - LEAD_M, q), LEG_STEPS)
        tz = gb.tcp(ctx)[2]
        heights.append(round(float(tz), 4))
        if tz < lowest - FALL_EPS:
            stalled = 0
            lowest = tz
            # The schedule follows the hand, never runs ahead of it: a lead target below a hand that
            # only creeps would otherwise count the creep as progress and call the bottom reached.
            z = max(z_next, tz - 0.003)
        else:
            stalled += 1
            stalls_total += 1
        arrived = tz <= z_bottom + ARRIVE_M
        if arrived and stalled >= STALL_STEPS:
            break
        if (not arrived) and stalled >= JAM_STALLS:
            break
    arrived = bool(tz <= z_bottom + ARRIVE_M)
    # Let go here, arrived or wedged: on the shaft the gear finishes the descent itself.
    pre_open = float(station["pre_open_m"])
    ctx.tool("robot.set_grip", width_m=pre_open, ramp_steps=20, settle_steps=RELEASE_SETTLE)
    gb.let_go()
    # Arrived or not: the tool centre reaching the bottom does not say the gear did. On this hand the pads
    # slide down the rim of a gear that is held up by its neighbour's teeth, and the hand arrives with the
    # gear perched (rq_roll/pair-01: the gear rose 33 mm in the jaws while the hand went down). So the gear
    # is turned to the next phase and let go again, every time: taken lightly, turned a quarter pitch in
    # place about its own peg, released. See RELEASE_TURNS.
    turns = 0
    for phase in RELEASE_PHASES:
        turns += 1
        q = gb.quat_yaw(base_q, phase * sweep)
        ctx.tool("robot.set_grip", preset="close", ramp_steps=20, settle_steps=30)
        gb.leg(ctx, gb.pose(x, y, tz, q), 40)
        ctx.tool("robot.set_grip", width_m=pre_open, ramp_steps=20, settle_steps=RELEASE_SETTLE)
    gb.let_go()
    return {
        "release_turns": turns,
        "stalls": stalls_total,
        "steps": steps,
        "arrived": arrived,
        "released_tcp_z": round(float(tz), 4),
        "short_by_m": round(float(tz - z_bottom), 4),
        "bottom_tcp_z": round(z_bottom, 4),
        "top_tcp_z": round(z_top, 4),
        "heights": heights,
        "tip_reach_m": round(float(gb.reach_now(ctx)), 4),
    }


def run(ctx, index: int):
    geom = gb.board(ctx)
    station = gb.station(geom, index)
    name = station["gear"]
    x, y, _ = gb.tcp(ctx)
    info = _descend(ctx, station, gb.station_quat(geom, index), x, y)
    ctx.tool("robot.set_grip", preset="open", ramp_steps=20, settle_steps=30)
    gb.step_aside(ctx)
    # A released gear turns the last bit into mesh on its own; give it a moment before judging it, twice.
    ctx.tool("robot.wait_steps", steps=30)
    ok = gb.seated(ctx, index, geom)
    if not ok:
        ctx.tool("robot.wait_steps", steps=45)
        ok = gb.seated(ctx, index, geom)
    return {"route": "seated" if ok else "lost", "index": int(index), "gear": name, "descent": info}
