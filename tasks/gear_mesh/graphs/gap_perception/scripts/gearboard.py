"""The board as the overhead camera sees it, in the shape the choreography scripts expect.

Same names as the paper-table graphs' ``gearboard`` (``board``, ``station``,
``gear_position``, ``seated``, ``open_stations``, ``leg``...) so the legs are
the same code; every reading comes from ``sight`` (one RGB-D frame of the
``overhead`` camera, segmented by SAM3) instead of ``sim.*``. Gears are named by
their tooth class (``gear16``, ``gear20``, ``gear24``): a class is what a radius
reads as, and two gears of one class are interchangeable on this board.

What the hand is, the hand says: the top-down base orientation comes from
``robot.grasp_frame`` with the jaws set to close ACROSS the chain the camera
measured, and where the fingertips sit below the tool centre from
``robot.get_gripper``'s live ``tip_reach_m`` -- live, because on a geared hand
the pads swing on their linkage and the figure moves ten millimetres over the
stroke. What the hand holds, the pick measures: with the jaws closed on a gear
lying on the mat, the tool centre's height over the gear's centre is the one
number every carry and seat height is built on, and it is read off the hand
and the camera at that moment rather than modelled. Every height is relative
to the bench the camera measured, so a different cell or camera changes no
number here.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sight  # noqa: E402

RIM_CLEAR = 0.0035    # fingertips above whatever the gear lies on, at the pre-open width
"""Where the pad tips stop above the mat before the jaws close [m].

**Read at the pre-open width, because that is where the pads are.** On the
2F-85 the pads move down the approach axis as the fingers close -- 11.5 mm past
the tool centre at a 67 mm pre-open, 15 mm once closed on a 55 mm gear -- so a
close begun with the tips 2.5 mm off the mat ends with them 1.3 mm off it (the
gear takes the rest, and the arm gives a millimetre). 3.5 leaves 12 mm of a
16 mm face between the pads and the tips clear of the mat through the close."""
JAW_MARGIN = 0.020    # pre-open 10 mm a side beyond the gear's tip diameter
"""How much wider than the gear the jaws wait before closing [m].

Ten millimetres a side on the 2F-85, where the Franka hand had six. Its pads
swing on a four-bar and at a part-open the tips lean in past the pad centres,
and the servo counts a target within 4 mm as reached: measured, a 55 mm gear
approached 4.3 mm off centre at a 67 mm pre-open was caught on the rim by a
pad tip, flipped on edge and carried that way (pair-02, pair-09). The largest
gear, 65 mm, pre-opens to the full 85 mm stroke."""
RELEASE_GAP = 0.002   # let go this far above the seat
CARRY_ABOVE_BENCH = 0.08
"""How far above the bench the PADS travel [m] -- not the tool centre.

Clear of a peg standing 44 mm proud of the bench and of a gear hanging in the
jaws. The tool centre rides this plus whatever :func:`reach_now` reads, which
is what makes the height mean the same thing on a hand whose pads hang 11 mm
below its tool centre and on one where they hang 150 mm below it."""
VISITS = 2
"""Passes at one station before it is given up -- and, since it is per station, the loop's whole budget.

**Two, because a third pass buys almost nothing and the button has to be pressed.** A pass costs about
2,490 control steps (fitted over 56 solved episodes), so four passes at each of a two-gear board's
stations is 20,000 against a 12,000-step document: the first sweep had three boards with both gears
seated that never reached the button, which scores zero however well assembled the board is. And the
passes do not pay: over 65 documents the first pass at a station seated 74 % of the time, the second
33 %, the third 29 %, and visits three and later won two seats in the whole sweep. Two passes per
station leaves every budget room for the press."""
STATED_REACH_RESIDUAL = 0.0021
"""How much deeper the pads bite than a hand's STATED fingertip reach [m]; only for a runtime whose
``robot.get_gripper`` reports no live ``tip_reach_m``. Measured on the Franka hand (stated 9.3, touched 12.6)."""
FACE = sight.FACE


def quat_yaw(q: dict[str, float], yaw: float) -> dict[str, float]:
    """``q`` turned by ``yaw`` about world +z."""
    c, s = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    w2, x2, y2, z2 = q["w"], q["x"], q["y"], q["z"]
    return {"w": c * w2 - s * z2, "x": c * x2 - s * y2, "y": c * y2 + s * x2, "z": c * z2 + s * w2}


def pose(x: float, y: float, z: float, q: dict[str, float]) -> dict[str, Any]:
    return {"position": {"x": float(x), "y": float(y), "z": float(z)}, "rotation": dict(q)}


def distance_xy(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def gear_name(teeth: int) -> str:
    return f"gear{int(teeth)}"


def tcp(ctx) -> tuple[float, float, float]:
    p = ctx.tool("robot.get_ee_pose")["pose"]["position"]
    return (float(p["x"]), float(p["y"]), float(p["z"]))


FINE_TOLERANCE = 0.0015
"""The arrival ball of a leg that has to put a bore over a peg [m]; the servo's own is 4 mm."""


def leg(ctx, pose, steps: int, tolerance: float = 0.0):
    """One bounded servo leg: the generic move counts a target within its arrival ball as reached and has no cap of its own.

    ``tolerance`` shrinks that ball for this leg (see FINE_TOLERANCE); 0 keeps the servo's default."""
    if tolerance > 0.0:
        return ctx.tool("robot.go_to_pose", pose=pose, max_steps=int(steps), tolerance=float(tolerance))
    return ctx.tool("robot.go_to_pose", pose=pose, max_steps=int(steps))


def reachable(ctx, pose) -> bool:
    """Whether the arm can stand at ``pose`` to within the servo's own arrival ball, asked of its solver without moving.

    With the wrist roll honoured a top-down pose has one joint-7 angle left to
    it, and at the far side of the mat that angle can lie past the joint's
    stop: measured on gearmesh-easy-pair-09 and medium-train-005, the pick's
    descent was refused at every visit and the hand never left carry height.
    The jaw heading's half-turn twin is the same grasp on a gear, so a pick that
    cannot stand at one tries the other.
    """
    try:
        got = ctx.tool("robot.solve_ik", pose=pose)
    except Exception:  # noqa: BLE001
        return True
    if not isinstance(got, dict):
        return True
    if "within_servo_tolerance" in got:
        return bool(got["within_servo_tolerance"])
    return bool(got.get("success", True))


def reach_now(ctx) -> float:
    """How far this hand's pads reach below its tool centre [m], as the hand stands right now.

    **Asked of the hand each time, never measured by touch.** ``robot.get_gripper``
    reports ``tip_reach_m`` off the finger colliders at the current opening, and
    on the 2F-85 that opening decides the answer: 5.9 mm at full open, 11.5 mm
    at a 67 mm pre-open, 15-16 mm closed on a gear. The earlier calibration --
    lowering the open jaws onto the mat until the tool centre stalled -- read
    MINUS 10.8 mm on this hand, because a vertical push at the fingertips folds
    its passive four-bar (the adaptive mode) and the tool centre travels
    another 19 mm before anything stalls; every height built on that number
    drove the pads through the gears. A runtime that reports no live figure
    gets the hand's stated reach plus the one measured residual.
    """
    try:
        got = ctx.tool("robot.get_gripper")
        live = got.get("tip_reach_m") if isinstance(got, dict) else None
        if live is not None:
            return float(live)
    except Exception:  # noqa: BLE001
        pass
    try:
        finger = ctx.tool("robot.describe_gripper").get("finger") or {}
        return float(finger.get("reach_m", 0.0) or 0.0) + STATED_REACH_RESIDUAL
    except Exception:  # noqa: BLE001
        return 0.0


def take_hold(ctx, gear_centre_z: float) -> float:
    """Record what the closed jaws hold: the tool centre's height over the gear's centre [m].

    Read the moment the jaws close on a gear lying flat, from ``robot.get_ee_pose``
    and the camera's height for the gear, and kept until the next pick. It is
    the number a carry and a seat need -- where the gear's bottom is relative to
    the frame the arm is commanded in -- and measuring it here makes it true for
    whatever hand closed, however its pads moved while closing.
    """
    lift = float(tcp(ctx)[2] - gear_centre_z)
    sight._CACHE["held_lift"] = lift
    sight._CACHE["held_top_above_tcp"] = 0.5 * FACE - lift
    return lift


def let_go():
    """The jaws hold nothing the graph knows about: forget the last pick's hold."""
    sight._CACHE.pop("held_lift", None)
    sight._CACHE.pop("held_top_above_tcp", None)


def station_quat(geom: dict[str, Any], index: int) -> dict[str, float]:
    """The approach for one station: square to the link that station sits on.

    On a straight board every station shares the board's heading and this is
    the board's own grasp frame; on a bent one each gear is met square to its
    own link, which is where the clearance actually is."""
    q = dict(geom.get("station_q", {}).get(int(index)) or geom["base_q"])
    # The half-turn twin of the same jaw heading, when the first proved out of the arm's reach (see pick).
    return quat_yaw(q, math.pi) if sight._CACHE.get("flip") else q


def base_quat(ctx, chain_yaw: float) -> dict[str, float]:
    """Top-down, square to the chain the camera measured, from this hand's own axes.

    **Ninety degrees off the centreline, and that is a clearance statement.**
    The stations sit a pitch apart along the chain and a seated gear's tips
    reach most of the way to its neighbour's, so a hand that closed ALONG the
    chain would bring a finger down between two meshing gears. Square to it,
    the fingers pass either side of the chain where nothing stands.
    ``robot.grasp_frame`` turns that heading into whatever quaternion this
    particular hand needs, so the graph states the direction and the robot
    states its own wrist."""
    key = round(float(chain_yaw), 3)
    cached = sight._CACHE.get("base_quat")
    if cached is not None and cached[0] == key:
        return dict(cached[1])
    heading = math.degrees(chain_yaw) - 90.0
    q = dict(ctx.tool("robot.grasp_frame", close_heading_deg=heading)["rotation"])
    sight._CACHE["base_quat"] = (key, q)
    return dict(q)


# ------------------------------------------------------------------ readings
def board(ctx, reading: dict[str, Any] | None = None) -> dict[str, Any]:
    """Every station's world geometry plus the button and the heights the legs use, from the camera."""
    reading = reading or sight.read(ctx)
    geom = sight.board_geometry(ctx, reading)
    assigned = sight.assign(reading, geom)
    plate_top = float(geom["plate_top_z"])
    for st in geom["stations"]:
        state, top, _cov = sight.station_state(reading["view"], st["peg_xy"], sight.r_tip(int(st["teeth"])), plate_top)
        if state == "perched":
            rt = sight.r_tip(int(st["teeth"]))
            xy = sight.perched_centre(reading["view"], st["peg_xy"], float(top), rt)
            assigned[int(st["index"])] = {"xy": xy, "top_z": float(top), "r_tip": rt, "teeth": int(st["teeth"]), "px": 0, "perched": True}
    drop = reach_now(ctx)
    # What a held gear's centre sits below the tool centre: measured at the last pick when there was one,
    # else what a pick at this opening would give.
    held_lift = sight._CACHE.get("held_lift")
    lift = float(held_lift) if held_lift is not None else drop + RIM_CLEAR - 0.5 * FACE
    stations = []
    for st in geom["stations"]:
        k = int(st["index"])
        teeth = int(st["teeth"])
        seat = (float(st["peg_xy"][0]), float(st["peg_xy"][1]), plate_top + 0.5 * FACE)
        rt = sight.r_tip(teeth)
        stations.append(
            {
                "index": k,
                "gear": gear_name(teeth),
                "teeth": teeth,
                "r_tip": rt,
                "seat": seat,
                "peg_top_z": float(st["peg_top_z"]),
                "peg_source": st.get("peg_source", "?"),
                "staged_tcp_z": seat[2] + FACE + RELEASE_GAP + lift,
                "drop_tcp_z": seat[2] + RELEASE_GAP + lift,
                "sweep": 2.0 * math.pi / float(teeth),
                "link_yaw": float(st.get("link_yaw", geom["chain_yaw"])),
                "pre_open_m": 2.0 * rt + JAW_MARGIN,
                "assigned": assigned.get(k),
            }
        )
    table_z = float(reading["bench_z"])
    return {
        "stations_count": len(stations),
        "face": FACE,
        "lift": lift,
        "held_lift": held_lift,
        "tip_drop": drop,
        "table_z": table_z,
        # The PADS ride CARRY_ABOVE_BENCH over the bench; the tool centre rides
        # wherever that puts it for this hand.
        "carry_z": table_z + drop + CARRY_ABOVE_BENCH,
        "plate_top_z": plate_top,
        "mat_top_z": table_z + 0.004,
        "chain_yaw": float(geom["chain_yaw"]),
        "base_q": base_quat(ctx, float(geom["chain_yaw"])),
        "station_q": {
            int(st["index"]): base_quat(ctx, float(st["link_yaw"])) for st in stations
        },
        "stations": stations,
        "button": geom["button"],
        "reading": reading,
    }


def station(geom: dict[str, Any], index: int) -> dict[str, Any]:
    return geom["stations"][int(index)]


def held_gear(ctx, reading: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """The disc in the hand, if the camera sees one at carry height: ``{xy, top_z, r_tip, teeth}``."""
    reading = reading or sight.read(ctx)
    return reading["held"]


def gear_position(ctx, name: str, reading: dict[str, Any] | None = None) -> tuple[float, float, float]:
    """The gear's centre (x, y, z): in the hand if a disc of its class is up there, else on the bench."""
    reading = reading or sight.read(ctx)
    teeth = int(str(name).replace("gear", ""))
    h = reading["held"]
    if h is not None and int(h["teeth"]) == teeth:
        return (float(h["xy"][0]), float(h["xy"][1]), float(h["top_z"]) - 0.5 * FACE)
    same = [g for g in reading["gears"] if int(g["teeth"]) == teeth]
    if not same:
        raise LookupError(f"the camera sees no {name}")
    g = max(same, key=lambda d: d["px"])
    return (float(g["xy"][0]), float(g["xy"][1]), float(g["top_z"]) - 0.5 * FACE)


def station_state(reading: dict[str, Any], geom: dict[str, Any], index: int):
    st = station(geom, index)
    return sight.station_state(reading["view"], st["seat"][:2], st["r_tip"], geom["plate_top_z"])


def seated_in(reading: dict[str, Any], geom: dict[str, Any], index: int) -> bool:
    return station_state(reading, geom, index)[0] == "seated"


def seated(ctx, index: int, geom: dict[str, Any] | None = None) -> bool:
    """A gear-thick disc sitting at the peg, its top one face above the plate. The hand must not be over it."""
    reading = sight.read(ctx)
    geom = geom or board(ctx, reading)
    return seated_in(reading, geom, index)


def open_stations(ctx, geom: dict[str, Any]) -> list[dict[str, Any]]:
    """Stations still worth a pass: not seated, with a gear of their class in view, and not visited too often."""
    reading = geom["reading"]
    visits = sight._CACHE.setdefault("visits", {})
    out = []
    for st in geom["stations"]:
        if seated_in(reading, geom, st["index"]):
            continue
        if st["assigned"] is None:
            continue
        if visits.get(st["index"], 0) >= VISITS:
            continue
        out.append(st)
    # Least-tried first: a station that has already had its pass waits while one that has had none is
    # tried, so a stubborn gear cannot spend the board's whole budget before its neighbour is attempted.
    out.sort(key=lambda st: (visits.get(st["index"], 0), st["index"]))
    return out


def clear_patch(reading) -> tuple[float, float]:
    """A spot on the mat with nothing on it, to put a hand down on.

    Farthest from every disc and peg the camera saw, over a small grid across
    the mat -- which is where the loose gears wait, so the bench under it is
    the height :func:`sight.read` already measured.
    """
    seen = [g["xy"] for g in reading["gears"]] + [p["xy"] for p in reading["pegs"]]
    held = reading.get("held")
    if held is not None:
        seen.append(held["xy"])
    xs = [g["xy"][0] for g in reading["gears"]] or [0.47]
    x0 = min(xs)
    spots = [(x0 - dx, y) for dx in (0.0, 0.04) for y in (-0.11, -0.055, 0.0, 0.055, 0.11)]
    return max(spots, key=lambda sp: min([distance_xy(sp, o) for o in seen] or [1.0]))


def step_aside(ctx):
    """Lift to carry height, then park at home: the only pose from which the arm is out of the camera's view.

    Stepping sideways is not enough -- the links then drape across the frame
    and hide the mat (measured: a +8 cm step put the forearm over two gears).
    """
    x, y, z = tcp(ctx)
    table_z = sight._CACHE.get("table_z")
    drop = reach_now(ctx)
    if table_z is not None and z < table_z + drop + CARRY_ABOVE_BENCH - 0.01:
        q = ctx.tool("robot.get_ee_pose")["pose"]["rotation"]
        leg(ctx, pose(x, y, table_z + drop + CARRY_ABOVE_BENCH, q), 250)
    ctx.tool("robot.go_home")
    ctx.tool("robot.wait_steps", steps=10)


def gripper_fraction(ctx) -> float:
    """robot.get_gripper's position: 1.0 fully open, about 0 closed on nothing, part-way closed on a gear."""
    try:
        return float(ctx.tool("robot.get_gripper").get("position", float("nan")))
    except Exception:  # noqa: BLE001
        return float("nan")


def holding(ctx) -> tuple[bool, float]:
    """Something is between the pads: the jaws stopped part-way (neither fully open nor closed on nothing)."""
    f = gripper_fraction(ctx)
    return (f == f and 0.08 < f < 0.92), f
