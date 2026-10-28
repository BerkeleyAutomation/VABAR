# SPDX-License-Identifier: Apache-2.0
"""Where everything stands on the bench, and how an episode moves it.

The suite's placement arithmetic, kept beside the entities it places. A
document says *what* is on the bench -- a mat, a board, one gear per station --
and a record's ``build`` says how to put one of those in a scene. Three of the
placements depend on the **variant**, and each has a different home:

* **A board whose length follows its stations.** There is a record per station
  list (``records.GEAR_BOARDS``), each carrying :func:`board_for` with its
  stations bound, so a document names the board it wants -- ``gear_board_train``
  -- and the pegs fall where that board's gears mesh.
* **A gear whose spot follows how many others wait loose beside it, or which
  peg it starts seated on.** A ``gear_wheel_20t`` is one thing wherever it lies,
  and ``quad`` has two of them loose at different spots where ``quad_seeded``
  starts both on their pegs -- so no record can carry the placement, and the
  document's ``(:start-poses poses/<variant>.json)`` does: one entry per named
  gear, **written by** :func:`write_start_poses` from :func:`gears_for` and
  pinned to it by ``tests/test_gear_mesh.py``. Derived, then recorded: the
  arithmetic stays the single statement, and the file is what the document
  layer can read.
* **The episode sampler.** :func:`scatter_and_clock`, named by the suite as
  ``(:sampler assets.layout:scatter_and_clock)``.

:data:`STATIONS` and :data:`SEEDED` are the two tables that make a variant a
tier; the stations are also what each document's ``:objects`` say, in order.

Layout is in the surface frame; the bench top's centre is the origin, world
(0.6, 0, 0.75). One arm, mounted behind the bench.

**The chain runs across the arm, not away from it.** The board is yawed a
quarter turn so its stations march along world y -- left to right in front of
the robot -- rather than out along x. Four stations plus a pinion plus a button
span 270 mm, and spent along x that would put the far station 0.27 m further out
than the near one, at the end of the reach envelope, while the button sat under
the shoulder. Spent along y they are all at the same radius: the difference
between stations is a shoulder pan, which is the one joint that has the range.

**Gears start flat on the bench, at rest, and nothing is measured.** A disc lying
on a flat surface has its centre exactly half a face above it, so a spawn pose is
a settled pose by construction -- no drop, no settle pass. That is the same
discipline as the peg spacing: derive it, do not measure it. ``box_decant``'s
carton had to be measured because a carton is not a disc, and it cost that suite
a bench parking a blade in free air 42 mm above the seam. The poses file is
derived output, not a measurement, and the test that pins it is what keeps it so.

**Why the pre-seated pair is stations 1 and 3.** They are not adjacent, so they
touch nothing and their clock angles are genuinely free -- which is what makes
sampling them meaningful. Seat two neighbours instead and their relative phase is
fixed by their own teeth, so "with variation in angle" would describe one number,
not two. Free on their pegs, they turn to accept whatever the policy drops
between them, which is what a real gear does and why a gear can be assembled at
all.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robosimstudio.core.render.palette import GEAR_BRASS, GEAR_BRONZE, GEAR_STEEL, MAT_LIGHT
from robosimstudio.specs import StaticSpec, TaskSpec

from .board import SpecGearBoard, board_frame, forward_of, plate_centre
from .gear import SpecGear
from .gearing import gear_frame, mesh_phase

__all__ = [
    "BOARD_AT",
    "BOARD_DRIFT",
    "BOARD_TURN",
    "CHAIN_BEND",
    "BOARD_YAW",
    "GEAR_GAP",
    "LOOSE_SPAN",
    "LOOSE_X",
    "MAT_CENTER",
    "MAT_HALF",
    "MAT_INSET",
    "PLATE_GAP",
    "POSES_DIR",
    "SEEDED",
    "STATIONS",
    "TIERS",
    "VARIANTS",
    "VARIANTS_FILE",
    "DrawnLayout",
    "GEAR_ALLOYS",
    "Variant",
    "board_for",
    "catalogue_gear",
    "draw_layout",
    "gear_mat",
    "gears_for",
    "gears_for_variant",
    "loose_at",
    "plate_clearance",
    "scatter_and_clock",
    "seated_pose",
    "start_poses",
    "start_poses_for",
    "station_names",
    "variant_board",
    "write_start_poses",
    "yaw_quat",
]

TIERS: dict[str, tuple[int, ...]] = {
    "pair": (16, 20),
    "train": (16, 20, 24),
    "quad": (16, 20, 24, 20),
    "quad_seeded": (16, 20, 24, 20),
}
"""Tooth counts on each hand-written tier's pegs, from the pinion outward.

The tiers, and the sizes are half of what makes them tiers. ``train`` is the
three-gear board NVIDIA's own gear task uses. ``quad`` adds a fourth station and
repeats the 20-tooth gear, so two of the four loose gears are interchangeable and
the policy has to notice which peg each one belongs on -- a gear on the wrong peg
does not mesh, because the peg spacing is cut for the pair it separates.

The drawn variants (:data:`VARIANTS`) join these in :data:`STATIONS` below."""

SEEDED: dict[str, tuple[int, ...]] = {"quad_seeded": (1, 3)}
"""Which stations start with a gear already on them, per variant."""

POSES_DIR = "poses"
"""Where a variant's recorded start poses live, under the task directory: ``poses/<variant>.json``."""

BOARD_AT = (0.02, 0.0)
"""Where the board's PLATE CENTRE sits [m, surface frame] -- world (0.62, 0)."""

BOARD_YAW = math.pi / 2
"""Turns the board's local +x, the chain, onto world +y: across the arm."""

LOOSE_X = -0.13
"""Surface x the loose gears wait on -- world 0.47, in front of the board.

Between the arm and the board, so a pick is a reach in and a place is a reach
across rather than a reach through: at the board's own x a gear would sit inside
the plate's footprint, and beyond it every carry would cross the assembly."""

LOOSE_SPAN = 0.20
"""How wide the loose gears are spread in y [m]. Four 65 mm gears need 260 mm
end to end at a touch; spread over 200 mm they overlap, so they are staggered in
x as well -- see :func:`loose_at`."""

MAT_HALF = (0.075, 0.135, 0.002)
MAT_CENTER = (LOOSE_X, 0.0)
"""A mat under the loose gears. Not scenery: a gear is a rolling body on its rim
and a bench top is where it rolls to. The mat's friction is what makes a gear
set down at an angle stay where it was put."""

_CELLS = 6
"""How many spots the episode sampler scatters loose gears over."""

# ------------------------------------------------------------- drawn variants
VARIANTS_FILE = "variants.json"
"""The drawn variants' table, beside this module: what ``tools/build_gear_scenes.py`` writes.

One row per generated document -- ``gearmesh-easy-pair-07`` and its kin -- with
the draw it recorded: which stations, where the plate stands and how it is
turned, where each loose gear lies and how it is clocked. Derived output, the
way ``poses/<variant>.json`` is: :func:`draw_layout` at the row's seed is the
one way a row is made, and ``tests/test_gear_mesh.py`` refuses a table a fresh
draw disagrees with."""

BOARD_DRIFT = 0.03
"""How far a drawn variant's plate centre may stand from :data:`BOARD_AT`, each axis [m]."""

BOARD_TURN = math.radians(10.0)
"""How far a drawn variant's chain may be turned off :data:`BOARD_YAW` [rad]."""

CHAIN_BEND = (math.radians(18.0), math.radians(42.0))
"""How far a drawn chain turns at each station after the first [rad], least and most.

**A train is a polyline, not a line.** Two gears mesh at a centre distance and
that fixes the length of a link, never its direction, so a board may turn at
every station and still be assemblable by construction. Drawn boards bend
because a straight chain lets a policy find the third peg by extending the line
through the first two, which is not the same as finding it; and because the
gears then sit around a bend, so the hand has to come at each one square to its
own link rather than square to the board. The sign alternates per station, so a
three-station chain arcs one way and then the other rather than curling up on
itself and folding a gear into its neighbour-but-one."""

MAT_INSET = 0.008
"""A drawn gear's tip circle stays this far inside the mat's edge [m]."""

GEAR_GAP = 0.025
"""Least free mat between two drawn gears' tip circles [m].

The pre-open is the tip diameter plus 12 mm and the pads have width: two gears
closer than this cannot both be picked without the jaws closing on a neighbour."""

PLATE_GAP = 0.015
"""Least free bench between a drawn gear's tip circle and the plate's edge [m]."""

DRAW_RESTARTS = 200
DRAW_TRIES = 500
"""How hard :func:`draw_layout` tries: whole-layout restarts, and spots per gear within one."""


@dataclass(frozen=True)
class Variant:
    """One row of :data:`VARIANTS_FILE`."""

    name: str
    level: str
    family: str
    index: int
    seed: int
    stations: tuple[int, ...]
    board_xy: tuple[float, float]
    board_yaw: float
    gears: tuple[tuple[str, int, float, float, float], ...]
    """``(name, teeth, x, y, yaw)`` per station, in station order; x, y in the surface frame."""
    bends: tuple[float, ...] = ()
    """How far the chain turns at each station after the first [rad]."""
    tries: int = 0


def _load_variants() -> dict[str, Variant]:
    path = Path(__file__).with_name(VARIANTS_FILE)
    if not path.is_file():
        return {}
    out: dict[str, Variant] = {}
    for row in json.loads(path.read_text())["variants"]:
        out[str(row["name"])] = Variant(
            name=str(row["name"]),
            level=str(row["level"]),
            family=str(row["family"]),
            index=int(row["index"]),
            seed=int(row["seed"]),
            stations=tuple(int(z) for z in row["stations"]),
            board_xy=(float(row["board"]["xy"][0]), float(row["board"]["xy"][1])),
            board_yaw=float(row["board"]["yaw"]),
            gears=tuple(
                (str(g["name"]), int(g["teeth"]), float(g["xy"][0]), float(g["xy"][1]), float(g["yaw"]))
                for g in row["gears"]
            ),
            bends=tuple(float(b) for b in (row["board"].get("bends") or ())),
            tries=int(row.get("tries", 0)),
        )
    return out


VARIANTS: dict[str, Variant] = _load_variants()
"""The drawn variants, by document name. Empty until the generator has run."""

STATIONS: dict[str, tuple[int, ...]] = {**TIERS, **{v.name: v.stations for v in VARIANTS.values()}}
"""Tooth counts on every variant's pegs, from the pinion outward: the tiers, then the drawn table."""


def station_names(count: int) -> tuple[str, ...]:
    """``gear_a``, ``gear_b``, ... -- the object names the documents use, in station order."""
    return tuple(f"gear_{chr(ord('a') + i)}" for i in range(count))


def gear_mat() -> StaticSpec:
    """The mat the loose gears wait on -- ``gear_mat``'s ``build``.

    The bench is the cell's; the plate, the pegs and the button are the board
    entity's own geometry. This is the one thing on the bench a record can place
    unaided, because it stands in the same place in every variant.
    """
    return StaticSpec(
        name="mat",
        half_extents=MAT_HALF,
        xform=(MAT_CENTER[0], MAT_CENTER[1], MAT_HALF[2], 0.0, 0.0, 0.0, 1.0),
        friction=0.9,
        color=MAT_LIGHT,
    )


def _resting_z() -> float:
    """A gear's centre when it lies flat on the mat: the mat's top plus half a face.

    On the MAT, not the bench: a disc's centre sits exactly half a face above
    whatever it lies on, and forgetting the 4 mm of mat under it would spawn
    every gear that far inside it. A spawn pose is a settled pose here only
    because the arithmetic is right.
    """
    return 2.0 * MAT_HALF[2] + 0.5 * SpecGear.face


GEAR_ALLOYS: dict[int, tuple[float, float, float]] = {16: GEAR_STEEL, 20: GEAR_BRASS, 24: GEAR_BRONZE}
"""What each catalogue gear is cut from: steel, brass, bronze.

A real gear set mixes alloys, and this one has to be told apart by something a
camera can read -- the tip radii are 5 mm apart and three identical steel discs
on a grey mat defeated the segmenter. The palette says what the three are; the
finish table gives all of them the same brushed metal, so they differ in colour
and not in material."""


def catalogue_gear(teeth: int) -> SpecGear:
    """One catalogue gear, resting flat on the mat at its centre -- the gear records' ``build``.

    What a record can say about placing a gear is what it rests on: the mat.
    *Where* on the mat, or on which peg instead, is the variant's, and the
    document records it per named gear (``(:start-poses poses/<variant>.json)``,
    written by :func:`write_start_poses`); the loader overlays that pose and
    quaternion on what this returns. So this is also what a gear the recording
    forgot looks like -- at the mat's centre, on top of any other it forgot --
    which ``tests/test_gear_mesh.py`` refuses rather than lets two gears share.
    """
    return SpecGear(
        name=f"gear_wheel_{teeth}t",
        category=f"gear_wheel_{teeth}t",
        teeth=teeth,
        pose=(MAT_CENTER[0], MAT_CENTER[1], _resting_z()),
        color=GEAR_ALLOYS[int(teeth)],
    )


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """A turn of ``yaw`` about +z as the xyzw quaternion a pose entry records."""
    return (0.0, 0.0, math.sin(0.5 * float(yaw)), math.cos(0.5 * float(yaw)))


def board_for(
    stations: tuple[int, ...],
    *,
    at: tuple[float, float] = BOARD_AT,
    yaw: float = BOARD_YAW,
    bends: tuple[float, ...] = (),
) -> SpecGearBoard:
    """A board cut for ``stations``, its plate centred on ``at`` and its chain turned to ``yaw``.

    Stated as a plate-centre placement rather than as the pinion end, because
    the plate is centred on the span it carries and the span depends on the
    stations: four boards of three different lengths stand at the same place
    under the same camera this way. ``SpecGearBoard`` derives its own ``at``
    from the ``pose``/``quat`` pair, which is also the pair a document's poses
    file overlays -- so a drawn variant moves the board through the same two
    fields, and this factory needs no second build to find the offset.

    **Parameterised by the stations, and that is why there is a board record
    per station list.** A zero-argument factory has to pick one, so
    ``records.GEAR_BOARDS`` binds it: ``gear_board_train`` is this with
    ``(16, 20, 24)``, and a document names the board whose stations are its
    ``:objects``, in order.
    """
    names = station_names(len(stations))
    return SpecGearBoard(
        name="board",
        stations=stations,
        gears=names,
        pose=(float(at[0]), float(at[1]), 0.0),
        quat=yaw_quat(yaw),
        bends=tuple(float(b) for b in bends),
    )


def variant_board(variant: str) -> SpecGearBoard:
    """The board a variant stands: the tiers' at :data:`BOARD_AT`, a drawn one where its row says."""
    if variant in TIERS:
        return board_for(TIERS[variant])
    row = VARIANTS[variant]
    return board_for(row.stations, at=row.board_xy, yaw=row.board_yaw, bends=row.bends)


def _station_xy(board: SpecGearBoard, frame: dict[str, Any], station: int) -> tuple[float, float]:
    """Where station ``station``'s peg stands, in the surface frame.

    One arithmetic for the seated pose and for recognising a gear that starts
    seated, so the two cannot disagree. Along the board's own chain direction,
    whatever its yaw: the tiers run it along surface +y, a drawn variant may
    turn it a few degrees either way.
    """
    fx, fy = forward_of(board)
    along, across = frame["points"][station + 1]
    # Along the chain's first link and across it: on a straight board the across
    # term is zero and this is the offset it always was.
    return board.at[0] + fx * along - fy * across, board.at[1] + fy * along + fx * across


def seated_pose(board: SpecGearBoard, station: int) -> tuple[tuple[float, float, float], float]:
    """Where a gear already on station ``station`` sits, and how it is clocked.

    The clock is ``mesh_phase``'s answer against the pinion, chained outward --
    the same computation the bench assembles a whole board with. A seeded gear
    with no seated neighbour is free to be anywhere, so the sampler turns it; the
    phase here is only a starting point that is guaranteed not to be inside
    anything.
    """
    frame = board_frame(board)
    frames, links = frame["frames"], frame["links"]
    # Chained along the polyline: each pair is clocked against its own link, and
    # a link that turns carries its gear's clock round with it.
    yaw, heading = 0.0, 0.0
    bends = tuple(float(b) for b in board.bends)
    for index in range(station + 1):
        if index:
            turn = bends[index - 1] if index - 1 < len(bends) else 0.0
            heading += turn
            yaw += turn
        yaw, _ = mesh_phase(frames[index], frames[index + 1], links[index], yaw)
    x, y = _station_xy(board, frame, station)
    return (x, y, board.plate_thickness + frame["seat_z"]), float(board.yaw) + yaw


def loose_at(index: int, count: int) -> tuple[float, float, float]:
    """Where loose gear ``index`` of ``count`` waits, flat on the mat.

    Staggered in x as well as spread in y: four gears up to 65 mm across do not
    fit side by side inside the reachable band, and two rows put every one of
    them within a hand's width of the same reach.
    """
    row = index % 2
    column = index // 2
    columns = max(1, (count + 1) // 2)
    span = LOOSE_SPAN if columns > 1 else 0.0
    y = -0.5 * span + (span * column / max(1, columns - 1) if columns > 1 else 0.0)
    return (LOOSE_X + (0.055 if row else -0.055), y, _resting_z())


def gears_for(stations: tuple[int, ...], seeded: tuple[int, ...] = ()) -> tuple[SpecGear, ...]:
    """One gear per station, seeded ones on their pegs and the rest on the mat.

    **Parameterised by the variant, which is what the gear records' ``build``
    cannot carry.** A ``gear_wheel_20t`` is one thing wherever it lies, but
    where it lies depends on the variant: ``quad`` has two of them loose at
    different spots, and ``quad_seeded`` starts both on their pegs. So the
    answer is recorded per named gear -- :func:`start_poses` is this, in the
    shape ``(:start-poses ...)`` reads -- and the record's factory,
    :func:`catalogue_gear`, says only what a gear rests on.
    """
    board = board_for(stations)
    loose = [i for i in range(len(stations)) if i not in seeded]
    out: list[SpecGear] = []
    for index, teeth in enumerate(stations):
        name = station_names(len(stations))[index]
        if index in seeded:
            pose, yaw = seated_pose(board, index)
        else:
            pose = loose_at(loose.index(index), len(loose))
            yaw = 0.0
        out.append(
            SpecGear(
                name=name,
                category=f"gear_wheel_{teeth}t",
                teeth=teeth,
                pose=pose,
                quat=(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)),
            )
        )
    return tuple(out)


def gears_for_variant(variant: str) -> tuple[SpecGear, ...]:
    """One gear per station for a named variant: a tier's from :func:`gears_for`, a drawn one's from its row."""
    if variant in TIERS:
        return gears_for(TIERS[variant], SEEDED.get(variant, ()))
    row = VARIANTS[variant]
    return tuple(
        SpecGear(
            name=name,
            category=f"gear_wheel_{teeth}t",
            teeth=teeth,
            pose=(x, y, _resting_z()),
            quat=yaw_quat(yaw),
        )
        for name, teeth, x, y, yaw in row.gears
    )


# ------------------------------------------------------------------ start poses
def start_poses(stations: tuple[int, ...], seeded: tuple[int, ...] = ()) -> dict[str, Any]:
    """Where each named gear starts, in the shape ``(:start-poses FILE)`` reads.

    ``{"objects": {name: {"pose": [x, y, z], "quat": [x, y, z, w]}}}`` -- one
    entry per gear of :func:`gears_for`, and nothing else: the mat and the board
    are placed by their own records, and a body or joint name would only restate
    the entity's default.
    """
    return {
        "objects": {
            gear.name: {"pose": [float(v) for v in gear.pose], "quat": [float(v) for v in gear.quat]}
            for gear in gears_for(stations, seeded)
        }
    }


def start_poses_for(variant: str) -> dict[str, Any]:
    """Where everything a named variant records starts: its gears, and -- drawn -- its board.

    A tier's file is :func:`start_poses` of its stations, unchanged. A drawn
    variant's carries one more entry, ``board``, stating the plate centre and
    the chain's turn as the ``pose``/``quat`` pair ``SpecGearBoard`` reads, so
    the loader's overlay moves the board the way it moves a gear.
    """
    if variant in TIERS:
        return start_poses(TIERS[variant], SEEDED.get(variant, ()))
    row = VARIANTS[variant]
    objects: dict[str, Any] = {
        gear.name: {"pose": [float(v) for v in gear.pose], "quat": [float(v) for v in gear.quat]}
        for gear in gears_for_variant(variant)
    }
    objects["board"] = {
        "pose": [float(row.board_xy[0]), float(row.board_xy[1]), 0.0],
        "quat": [float(v) for v in yaw_quat(row.board_yaw)],
    }
    return {"objects": objects}


def write_start_poses(home: Path, variants: tuple[str, ...] | None = None) -> dict[str, Path]:
    """Write every variant's ``poses/<variant>.json`` under the task directory ``home``.

    The one way those files are made. Run it after changing anything a gear's
    start pose derives from -- the mat, the loose grid, a board's stations, the
    tooth profile that clocks a seeded gear, a drawn row -- and
    ``tests/test_gear_mesh.py`` will refuse a tree where the files and this
    arithmetic disagree::

        from robosimstudio_bench.taskbuild.discovery import task_module
        task_module("gear_mesh", "assets.layout").write_start_poses(home)

    Args:
        home: The task directory.
        variants: Which to write; every one in :data:`STATIONS` when ``None``.

    Returns:
        The path written, per variant.
    """
    out: dict[str, Path] = {}
    for variant in variants if variants is not None else tuple(STATIONS):
        path = Path(home) / POSES_DIR / f"{variant}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # One line per gear, so a diff reads as "gear_c moved" rather than as
        # forty lines of brackets.
        rows = ",\n".join(
            f"  {json.dumps(name)}: {json.dumps(entry)}" for name, entry in start_poses_for(variant)["objects"].items()
        )
        path.write_text('{\n "objects": {\n' + rows + "\n }\n}\n")
        out[variant] = path
    return out


# ------------------------------------------------------------------ drawing
def plate_clearance(board: SpecGearBoard, xy: tuple[float, float], radius: float) -> float:
    """Free bench between a disc of ``radius`` at ``xy`` and the plate's edge [m]; negative overlapping.

    In the board's own frame, whatever its yaw: the disc's centre is taken
    along and across the chain, and the plate is the rectangle
    ``board_frame``'s ``plate_half`` describes about :func:`plate_centre`.
    """
    cx, cy = plate_centre(board)
    fx, fy = forward_of(board)
    dx, dy = float(xy[0]) - cx, float(xy[1]) - cy
    along, across = dx * fx + dy * fy, -dx * fy + dy * fx
    hx, hy, _ = board_frame(board)["plate_half"]
    ox, oy = max(abs(along) - hx, 0.0), max(abs(across) - hy, 0.0)
    if ox == 0.0 and oy == 0.0:
        return -min(hx - abs(along), hy - abs(across)) - float(radius)
    return math.hypot(ox, oy) - float(radius)


@dataclass(frozen=True)
class DrawnLayout:
    """What :func:`draw_layout` decided: the plate's placement, the chain's bends, and each gear's spot."""

    board_xy: tuple[float, float]
    board_yaw: float
    gears: tuple[tuple[float, float, float], ...]
    """``(x, y, yaw)`` per station, in station order."""
    tries: int
    bends: tuple[float, ...] = ()
    """How far the chain turns at each station after the first [rad]."""


def draw_layout(seed: int, stations: tuple[int, ...]) -> DrawnLayout:
    """Draw where a variant's board and loose gears stand, at ``seed``, rejecting what a hand cannot work.

    The plate centre is drawn within :data:`BOARD_DRIFT` of :data:`BOARD_AT`
    and the chain within :data:`BOARD_TURN` of :data:`BOARD_YAW`; then each
    gear, in station order, is dropped uniformly on the mat -- its tip circle
    :data:`MAT_INSET` inside the edge -- and kept only if it clears the plate
    by :data:`PLATE_GAP` and every gear already down by :data:`GEAR_GAP`. A
    gear that finds no spot restarts the whole layout with a fresh board; a
    draw that never settles raises and names the knobs rather than relaxing
    one. Clocks are drawn last, uniformly over a turn.

    Deterministic in ``seed``: the table records the result, and the test that
    pins the table re-runs this.
    """
    rng = np.random.default_rng(int(seed))
    # A chain of three or more turns at every station past the first, alternating. Two stations stay
    # straight: with one link after the first there is nothing for a bend to distinguish -- a two-peg
    # board turned at its only joint is the same board stood at a different yaw, which the draw already
    # varies.
    bends = (
        tuple(float((-1.0) ** index * rng.uniform(*CHAIN_BEND)) for index in range(len(stations) - 1))
        if len(stations) >= 3
        else ()
    )
    radii = [
        float(
            gear_frame(module=SpecGear.module, teeth=int(z), face=SpecGear.face, bore_radius=SpecGear.bore_radius)[
                "r_tip"
            ]
        )
        for z in stations
    ]
    tries = 0
    for _restart in range(DRAW_RESTARTS):
        tries += 1
        bx = BOARD_AT[0] + float(rng.uniform(-BOARD_DRIFT, BOARD_DRIFT))
        by = BOARD_AT[1] + float(rng.uniform(-BOARD_DRIFT, BOARD_DRIFT))
        byaw = BOARD_YAW + float(rng.uniform(-BOARD_TURN, BOARD_TURN))
        board = board_for(tuple(stations), at=(bx, by), yaw=byaw, bends=bends)
        placed: list[tuple[float, float, float]] = []
        for r in radii:
            spot = None
            for _try in range(DRAW_TRIES):
                x = float(rng.uniform(MAT_CENTER[0] - MAT_HALF[0] + r + MAT_INSET, MAT_CENTER[0] + MAT_HALF[0] - r - MAT_INSET))
                y = float(rng.uniform(MAT_CENTER[1] - MAT_HALF[1] + r + MAT_INSET, MAT_CENTER[1] + MAT_HALF[1] - r - MAT_INSET))
                if plate_clearance(board, (x, y), r) < PLATE_GAP:
                    continue
                if any(math.hypot(x - px, y - py) < r + pr + GEAR_GAP for px, py, pr in placed):
                    continue
                spot = (x, y)
                break
            if spot is None:
                break
            placed.append((spot[0], spot[1], r))
        if len(placed) == len(radii):
            clocks = [float(rng.uniform(0.0, 2.0 * math.pi)) for _ in radii]
            return DrawnLayout(
                board_xy=(bx, by),
                board_yaw=byaw,
                gears=tuple((x, y, clock) for (x, y, _r), clock in zip(placed, clocks, strict=True)),
                tries=tries,
                bends=bends,
            )
    raise RuntimeError(
        f"no layout for stations {tuple(stations)} at seed {seed} after {DRAW_RESTARTS} restarts x {DRAW_TRIES} "
        f"spots: the mat ({2 * MAT_HALF[0]:.3f} x {2 * MAT_HALF[1]:.3f} m) cannot hold these gears at "
        f"GEAR_GAP={GEAR_GAP}, PLATE_GAP={PLATE_GAP}, MAT_INSET={MAT_INSET}; change a knob, not a check"
    )


# ------------------------------------------------------------------ episodes
_SPOTS = tuple(
    (
        LOOSE_X + (0.055 if index % 2 else -0.055),
        -0.5 * LOOSE_SPAN + LOOSE_SPAN * (index // 2) / max(1, (_CELLS - 1) // 2),
    )
    for index in range(_CELLS)
)
"""The cells a loose gear is scattered over: two rows, three columns, on the mat."""

_TURNS = tuple(index * math.pi / 6.0 for index in range(12))
"""The clocks a gear is enumerated through: one revolution in twelfths."""


def _seated_gears(task: TaskSpec) -> tuple[str, ...]:
    """Which of the board's gears start on their pegs, read off the spec itself.

    A gear is seeded when its declared pose is over its own station -- within
    the peg's radius, which is the same tolerance the board scores ``seated``
    with. Read rather than told, so the sampler takes no configuration and the
    document's objects stay the single statement of what is where.
    """
    board = next((e for e in task.extras if isinstance(e, SpecGearBoard)), None)
    if board is None:
        raise ValueError(f"task {task.full_name!r} has no gear board to read the seeded gears off")
    frame = board_frame(board)
    seated: list[str] = []
    for station, name in enumerate(board.gears):
        spec = task.object(name)
        x, y = _station_xy(board, frame, station)
        if math.hypot(spec.pose[0] - x, spec.pose[1] - y) <= frame["peg_radius"]:
            seated.append(name)
    return tuple(seated)


def scatter_and_clock(task: TaskSpec, episode_id: int, rng: Any) -> dict[str, Any]:
    """This suite's episode sampler: scatter the loose gears, turn the seeded ones.

    **The first randomised sampler in this corpus**, so it is worth saying what
    it may and may not move. ``object_poses`` carries ``(x, y, z, yaw)`` per
    object and nothing else reaches the model: a sampler cannot change a gear's
    size, its tooth count, or how many there are. Those are the variant. What it
    varies is where the loose gears wait and how they lie, and -- in
    ``quad_seeded`` -- how the two gears already on their pegs are clocked, which
    is free precisely because those two stations are not adjacent and so touch
    nothing.

    **One sampler for both groups, deliberately.** ``compose`` merges what its
    samplers return with ``dict.update``, which is a *shallow* merge -- so two
    samplers each reporting ``object_poses`` do not combine, the second silently
    discards the first, and every loose gear goes back to its declared spot while
    reading like a randomised layout. One function reporting one dict cannot have
    that bug.

    **And not ``grid_xy_quat`` for the seeded gears.** That sampler reports
    ABSOLUTE x and y, keeping only the declared z, which is right for a gear
    waiting on the mat and catastrophic for one already on its peg: a zero-width
    range does not mean "leave it alone", it means "put it at the origin", and the
    gear would start inside the board. A seated gear keeps all three coordinates
    and varies only its clock.

    Enumerated rather than drawn, ``grid_xy_quat``'s reason: episode ``i`` takes
    the ``i``-th cell of the layout product, so a run of twelve covers the space
    evenly instead of clumping the way independent draws do. The stride per
    object is coprime with the cell count so two gears do not march together.

    Takes no configuration -- which gears are loose and which are seated is read
    off the task -- so it has the plain ``(task, episode_id, rng)`` shape a
    document names: ``(:episodes (:sampler assets.layout:scatter_and_clock))``
    in ``gear_mesh.suite``.
    """
    seated = _seated_gears(task)
    loose = tuple(o.name for o in task.objects if o.name.startswith("gear_") and o.name not in seated)
    poses: dict[str, tuple[float, float, float, float]] = {}
    for index, name in enumerate(loose):
        spot = _SPOTS[(episode_id + index * 5) % len(_SPOTS)]
        poses[name] = (
            spot[0] + float(rng.uniform(-0.006, 0.006)),
            spot[1] + float(rng.uniform(-0.006, 0.006)),
            float(task.object(name).pose[2]),
            _TURNS[(episode_id + index * 7) % len(_TURNS)],
        )
    for index, name in enumerate(seated):
        # All three coordinates kept: this gear is on its peg and stays there.
        # Only the clock turns, over a whole revolution, which is every phase a
        # gear with no seated neighbour can be in.
        spec = task.object(name)
        poses[name] = (
            float(spec.pose[0]),
            float(spec.pose[1]),
            float(spec.pose[2]),
            _TURNS[(episode_id + index * 5) % len(_TURNS)],
        )
    return {"object_poses": poses}
