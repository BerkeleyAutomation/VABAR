"""Re-seat spans displaced by the terminal carry, with one measured pass.

The fixed anchor does not prevent a pull at the free end from straightening an
earlier bend. After terminating the run, observe each seat again and reuse the
ordinary contact grasp/cross/release motions for any missing span. The pass is
bounded by the number of observed pegs; the final verifier still checks the
entire route, including the terminal that a repair may itself disturb.
"""

import json
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
from gap import NodeContext

# Sibling helpers resolve inside the GaP skill package.
from .perceive_rod import run as observe_rod
from .run_legs import run as run_legs


def seat_arclength(curve, station, side):
    """Length of a measured polyline inside the declared seat footprint [m]."""
    points = np.asarray(curve, dtype=float).reshape(-1, 3)
    centre = np.asarray(station, dtype=float)[:2]
    lower = centre + ([.003, -.035] if side > 0 else [-.045, -.035])
    upper = centre + ([.045, .035] if side > 0 else [-.003, .035])
    length = 0.
    for a, b in zip(points[:-1], points[1:]):
        delta = b[:2] - a[:2]
        enter, leave = 0., 1.
        for axis in range(2):
            if abs(delta[axis]) < 1e-12:
                if not lower[axis] <= a[axis] <= upper[axis]:
                    leave = -1.
                    break
            else:
                t0, t1 = sorted(((lower[axis]-a[axis])/delta[axis],
                                 (upper[axis]-a[axis])/delta[axis]))
                enter, leave = max(enter, t0), min(leave, t1)
        length += max(0., leave-enter) * float(np.linalg.norm(b-a))
    return length


class Output(TypedDict):
    exit: str
    repairs: int
    observations: list


def run(ctx: NodeContext, scene: str = "") -> Output:
    seen = json.loads(scene or '{}')
    stations, sides = seen.get('stations') or [], seen.get('sides') or []
    if len(stations) not in (2, 4) or len(sides) != len(stations):
        return {'exit': 'checked', 'repairs': 0, 'observations': []}
    if len(stations) == 4:
        return {'exit': 'checked', 'repairs': 0, 'observations': [], 'reason': 'Medium uses its first crossing action without recovery pickups'}
    order = sorted(range(len(stations)), key=lambda j: stations[j][1])
    planner_order = ([j for j in reversed(order) if sides[j] > 0] +
                     [j for j in reversed(order) if sides[j] < 0])
    repaired, observations = 0, []

    def execute(plan):
        common = dict(plan=json.dumps(plan), arm_id=int(plan['arm_id']),
                      grip_open_m=float(plan['grips']['open']['command_m']),
                      grip_close_m=float(plan['grips']['close']['command_m']))
        run_legs(ctx, legs='enter,reach,take', open_before=True, close_after=True, **common)
        run_legs(ctx, legs='move,cross,place', **common)
        run_legs(ctx, legs='clear,retreat', open_before=True, **common)
        ctx.tool('cable.wait_steps', steps=60)

    for station in order:
        measured = observe_rod(ctx, scene=scene)
        if measured['route'] != 'found':
            return {'exit': 'unobserved', 'repairs': repaired, 'observations': observations}
        curve = json.loads(measured['rod'])
        coverage = seat_arclength(curve, stations[station], sides[station])
        observations.append({'station': station, 'visible_arclength_m': coverage})
        # These scenes ask for two 12 mm cable capsules. Use metric arc length
        # so resampling a vision curve cannot change this decision.
        if coverage >= .026:
            print(f'[restore] station {station}: {coverage*1000:.1f} mm visible; leave seated span', flush=True)
            continue
        plan = ctx.tool('cable.plan_station', index=planner_order.index(station),
                        stations=stations, sides=sides, seats_x=seen.get('seats_x'),
                        rod=curve, final_pass=True)
        if float(plan.get('worst_grasp_error', float('inf'))) > .025 or not plan.get('grasp_ok', True):
            observations[-1]['repair'] = 'infeasible'
            continue
        print(f'[restore] station {station}: {coverage*1000:.1f} mm visible; re-seat after terminal carry', flush=True)
        execute(plan)
        observations[-1]['repair'] = 'executed'
        repaired += 1

        # One point in the box is not a seat: the finalized task asks for two
        # 12 mm cable capsules. Reobserve the result, and when the visible span
        # is still short, borrow a little more interior slack and place another
        # measured point at the same scored seat. This establishes cable on
        # both sides of the bend rather than shifting the placement target.
        measured = observe_rod(ctx, scene=scene)
        if measured['route'] != 'found':
            continue
        curve = json.loads(measured['rod'])
        after = seat_arclength(curve, stations[station], sides[station])
        observations[-1]['after_first_m'] = after
        if after >= .026:
            continue
        second = ctx.tool('cable.plan_station', index=planner_order.index(station),
                          stations=stations, sides=sides, seats_x=seen.get('seats_x'),
                          rod=curve, slack_margin=.045, final_pass=True)
        if float(second.get('worst_grasp_error', float('inf'))) > .025 or not second.get('grasp_ok', True):
            observations[-1]['second_touch'] = 'infeasible'
            continue
        print(f'[restore] station {station}: only {after*1000:.1f} mm after first touch; '
              'borrow another 25 mm of interior slack', flush=True)
        execute(second)
        observations[-1]['second_touch'] = 'executed'
        repaired += 1
    return {'exit': 'checked', 'repairs': repaired, 'observations': observations}
