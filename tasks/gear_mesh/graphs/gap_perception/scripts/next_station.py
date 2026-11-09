"""Which station to work next, from the camera; or that the board is assembled as far as it will be.

The first visit of an episode (no gear on any peg, none in the hand) resets
the per-episode bookkeeping, so a process that runs several episodes does not
carry one board's visits into the next.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gearboard as gb  # noqa: E402
import sight  # noqa: E402

DEBUG_DIR = os.environ.get("GEAR_VISION_DEBUG", "")

PEEK_STEP = (0.05, 0.12)
"""How far the parked hand steps back and away from the mat for a second look [m]: 5 cm towards the arm's
base and 12 cm across, which takes the fingers off the mat's corner and to the frame's edge."""


def _dump(ctx, tag):
    if not DEBUG_DIR:
        return
    try:
        import time

        import numpy as np

        os.makedirs(DEBUG_DIR, exist_ok=True)
        rgb, depth, K, T = sight.frame(ctx)
        np.savez(os.path.join(DEBUG_DIR, f"{time.strftime('%H%M%S')}_{tag}.npz"), rgb=rgb, depth=depth, K=K, T=T)
    except Exception:  # noqa: BLE001
        pass


def _fresh(reading, geom) -> bool:
    none_seated = not any(gb.seated_in(reading, geom, s["index"]) for s in geom["stations"])
    all_on_mat = all(gb.distance_xy(g["xy"], s["seat"][:2]) > 0.03 for g in reading["gears"] for s in geom["stations"])
    return none_seated and all_on_mat and len(reading["gears"]) >= len(geom["stations"])


def run(ctx, count_attempt: bool = True):
    # PARK BEFORE THE FIRST LOOK OF AN EPISODE. The frame that builds the board has to show all of it,
    # and the arm sits whereever the episode started it -- over the mat as often as not, hiding a gear
    # that is then never assigned to a station and never picked. Measured: a two-gear board read one gear,
    # seated it, and reported itself done. Home is the one pose the arm is out of frame from.
    if sight._CACHE.get("geometry") is None:
        gb.step_aside(ctx)
    reading = sight.read(ctx)
    sight._CACHE["table_z"] = float(reading["bench_z"])
    # Where the pads are right now, from the hand; the carry height keeps them clear of the pegs.
    drop = gb.reach_now(ctx)
    x, y, z = gb.tcp(ctx)
    floor = float(reading["bench_z"]) + drop + gb.CARRY_ABOVE_BENCH
    if z < floor - 0.01:
        q = ctx.tool("robot.get_ee_pose")["pose"]["rotation"]
        gb.leg(ctx, gb.pose(x, y, floor, q), 250)
    ctx.tool("robot.set_grip", preset="open", settle_steps=10)
    reading = sight.read(ctx)
    if not reading["pegs"] or any(
        sight.station_state(reading["view"], p["xy"], 0.02, reading["plate_z"])[0] == "hidden" for p in reading["pegs"]
    ):
        gb.step_aside(ctx)
        reading = sight.read(ctx)
    if sight._CACHE.get("geometry") is None:
        # First look of the process, or after a fresh episode was recognised: the arm is at home, the board is clear.
        # Looked for up to three times before giving up: the first frame after the arm parks is the path
        # tracer's least converged, and one rtx episode (gearmesh-easy-single-03) ended on it.
        geom = None
        for attempt in range(3):
            try:
                geom = gb.board(ctx, reading)
                break
            except LookupError as exc:
                if attempt == 2:
                    raise LookupError(f"{exc}; the frame showed pegs={len(reading['pegs'])} pinion={reading['pinion'] is not None} button={reading['button'] is not None}") from exc
                gb.step_aside(ctx)
                ctx.tool("robot.wait_steps", steps=30)
                reading = sight.read(ctx)
    else:
        geom = gb.board(ctx, reading)
        # A fresh episode under a stale lock: nothing on any peg, at least one gear per station on the mat, and
        # MORE loose gears than the last look found -- a gear that was seated or held is back on the mat. A
        # missed pick leaves the board exactly as fresh as it was and used to reset the visits here, so one
        # unreachable station was tried until the budget ran out (pair-09, train-005: fourteen picks).
        loose = len(reading["gears"])
        if (
            sight._CACHE.get("visits")
            and _fresh(reading, geom)
            and len(reading["pegs"]) == len(geom["stations"])
            and loose > int(sight._CACHE.get("loose_count", loose))
        ):
            sight.unlock_geometry()
            sight._CACHE["visits"] = {}
            sight._CACHE["used_spots"] = []
            sight._CACHE.pop("flip", None)
            geom = gb.board(ctx, reading)
    sight._CACHE["loose_count"] = len(reading["gears"])
    # A station with no gear of its class in view while gears are missing: the parked hand hides the corner
    # of the mat under it (medium-train-010: the steel gear 4 cm from the tool centre, under the fingers).
    # Look once more with the hand held out of the mat's footprint, then park again.
    missing = [s for s in geom["stations"] if s["assigned"] is None and not gb.seated_in(reading, geom, s["index"])]
    if missing and len(reading["gears"]) < len(missing) + len([s for s in geom["stations"] if s["assigned"] is not None]):
        hx, hy, hz = gb.tcp(ctx)
        q = ctx.tool("robot.get_ee_pose")["pose"]["rotation"]
        gb.leg(ctx, gb.pose(hx - PEEK_STEP[0], hy - PEEK_STEP[1], hz, q), 200)
        ctx.tool("robot.wait_steps", steps=15)
        peek = sight.read(ctx)
        if len(peek["gears"]) > len(reading["gears"]):
            reading = peek
            geom = gb.board(ctx, reading)
        ctx.tool("robot.go_home")
        ctx.tool("robot.wait_steps", steps=10)
    fresh = _fresh(reading, geom)
    if fresh and sight._CACHE.get("fresh_seen") is not True:
        sight._CACHE["visits"] = {}
        sight._CACHE["used_spots"] = []
        sight._CACHE["fresh_seen"] = True
    if not fresh:
        sight._CACHE["fresh_seen"] = False
    pending = gb.open_stations(ctx, geom)
    seated = [s["index"] for s in geom["stations"] if gb.seated_in(reading, geom, s["index"])]
    seen = [
        {"teeth": g["teeth"], "found_by": g.get("source"), "xy": [round(v, 4) for v in g["xy"]], "top_z": round(g["top_z"], 4)}
        for g in reading["gears"]
    ]
    states = {s["index"]: gb.station_state(reading, geom, s["index"])[0] for s in geom["stations"]}
    pegs = {s["index"]: s["peg_source"] for s in geom["stations"]}
    if not seen and len(seated) < len(geom["stations"]):
        _dump(ctx, "no_gears")
    common = {
        "seated": seated,
        "stations_count": geom["stations_count"],
        "chain_yaw_deg": round(__import__("math").degrees(geom["chain_yaw"]), 1),
        "gears_seen": seen,
        "stations": states,
        "peg_sources": pegs,
        "station_teeth": {s["index"]: s["teeth"] for s in geom["stations"]},
        "assigned": {s["index"]: None if s["assigned"] is None else [round(v, 4) for v in s["assigned"]["xy"]] for s in geom["stations"]},
        "button_query": (reading["button"] or {}).get("query"),
        "visits": dict(sight._CACHE.get("visits", {})),
    }
    if not pending:
        return {"route": "done", "index": -1, "remaining": 0, **common}
    nxt = pending[0]
    if count_attempt:
        visits = sight._CACHE.setdefault("visits", {})
        visits[nxt["index"]] = visits.get(nxt["index"], 0) + 1
    return {
        "route": "more",
        "index": int(nxt["index"]),
        "remaining": len(pending),
        "gear": nxt["gear"],
        "gear_at": [round(v, 4) for v in nxt["assigned"]["xy"]],
        **common,
    }
