# SPDX-License-Identifier: Apache-2.0
"""Fixtures: the furniture tasks are staged on and against.

Pegs, hooks, a wall, a hanger rail. Small rigid things whose whole content is a
handful of measured constants -- the tier-1 case, and the reason a rigid asset
costs almost nothing to annotate.

Three pegs stand on the cable bench, and the winding measure is stated about a
peg's ``axis`` rather than about "the peg" -- which is what lets the same measure
serve a bollard, a spool or a doorknob without a line of new code.

**Four fixtures now, and the last three arrived with the terminated runs.** The
industrial ``cable_cleat`` declares its ``shaft`` about a 9.5 mm standoff where
the F1 spool declares one about a 4.75 mm groove, so a seat measured against
either reads identically and the panel is a harder weave for the one reason that
is geometry rather than tuning. ``cable_block`` is the two ends of a run, one
part with one footprint realised by two entities -- the bracket a run is clamped
under and the socket a free end is put into. All three are drawn from primitives
by :mod:`~tasks/cable_route/assets/terminals.py`; only the spool is a mesh.

A hanger rail and a wall hook used to sit here too, and are gone. They were
invented to demonstrate that hanging generalises -- annotate a ``loop`` on a mug
and it hangs from the same ``threaded`` measure, with nothing written on either
side -- and a demonstration that no shipped task uses is a thing to argue in
prose rather than to carry as content.
"""

from __future__ import annotations

import math

from robosimstudio.assetlib.lower import declared_asset
from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.registry import register_asset
from robosimstudio.assetlib.semantics.features import FeatureSpec
from robosimstudio.assetlib.sim import SimRequirements, contacts_with
from robosimstudio.core.render.palette import ACCENT_PROP
from robosimstudio.scene.builtin.benches import BENCH_075
from robosimstudio.scene.frames import point_from_surface
from robosimstudio.specs import CellSpec, StaticSpec

__all__ = [
    "CABLE_BLOCK",
    "CABLE_CLEAT",
    "CABLE_PEG",
    "CABLE_PEG_ROUND",
    "cleat",
    "post",
    "spool",
    "terminal",
]

PEG_RADIUS = 0.013
PEG_HEIGHT = 0.06
"""The cable bench's pegs, at the reference's own dimensions [m]."""

ROUND_SHAFT_RADIUS = 0.00475
ROUND_FLANGE_RADIUS = 0.0125
ROUND_HEIGHT = 0.0235
ROUND_GROOVE_Z = (0.0025, 0.019679)
"""ManipulationNet's F1, measured off the vendored mesh [m].

Imported from ``rod.py``'s constants rather than retyped
would be better still, and is not possible: that module imports newton, and a
catalogue listing must not. The entity asserts the pair against the mesh at
build, which is the check that keeps these two honest.
"""

CABLE_PEG = register_asset(
    AssetRecord(
        name="cable_peg",
        summary="A short upright post a cable can be wound around.",
        tags=("peg", "post"),
        features=(
            FeatureSpec(
                name="shaft",
                kind="rod",
                payload={
                    "axis": [[0.0, 0.0, -PEG_HEIGHT / 2], [0.0, 0.0, PEG_HEIGHT / 2]],
                    "radius": PEG_RADIUS,
                },
                provenance="The reference's peg: 13 mm radius, 60 mm tall.",
            ),
            FeatureSpec(
                name="axis",
                kind="axis",
                payload={"origin": [0.0, 0.0, 0.0], "direction": [0.0, 0.0, 1.0]},
                provenance=(
                    "The winding is measured about this rather than about 'the peg', which is what "
                    "lets the same measure serve a bollard, a spool or a doorknob."
                ),
            ),
        ),
        sim=SimRequirements(
            # World-fixed: a peg is bolted to the bench, and the entity that
            # builds it is an ``EntityStatic``. Saying ``rigid`` would claim a
            # free body, which is a claim about the physics an arrangement has
            # to serve -- and the cable cell's names a backend for the arm and
            # the rod and nothing else, correctly.
            domain="static",
            contacts=contacts_with("surface"),
        ),
        provenance="cable_route's pegs, transcribed from the reference scene.",
    )
)

CABLE_PEG_ROUND = register_asset(
    AssetRecord(
        name="cable_peg_round",
        summary="ManipulationNet's F1 spool: a 9.5 mm shaft between two 25 mm flanges.",
        tags=("peg", "post", "spool"),
        files={"mesh": "round_peg.stl"},
        features=(
            FeatureSpec(
                name="shaft",
                kind="rod",
                payload={
                    # The SHAFT, which is what a cable actually touches -- not
                    # the flange, which is only what the part measures. A goal
                    # reading this to place or check a wrap wants the surface the
                    # rod seats against.
                    "axis": [[0.0, 0.0, ROUND_GROOVE_Z[0]], [0.0, 0.0, ROUND_GROOVE_Z[1]]],
                    "radius": ROUND_SHAFT_RADIUS,
                },
                provenance="Measured off round_peg.stl: r = 4.75 mm, groove z = 2.5 to 19.68 mm.",
            ),
            FeatureSpec(
                name="axis",
                kind="axis",
                payload={"origin": [0.0, 0.0, 0.0], "direction": [0.0, 0.0, 1.0]},
                provenance=(
                    "The same declaration cable_peg makes, so a winding or a seat measured about "
                    "one fixture's axis reads identically about the other's. Swapping a post for a "
                    "spool must change the physics and not the question."
                ),
            ),
        ),
        sim=SimRequirements(
            # World-fixed: a peg is bolted to the bench, and the entity that
            # builds it is an ``EntityStatic``. Saying ``rigid`` would claim a
            # free body, which is a claim about the physics an arrangement has
            # to serve -- and the cable cell's names a backend for the arm and
            # the rod and nothing else, correctly.
            domain="static",
            contacts=contacts_with("surface"),
        ),
        provenance=(
            "ManipulationNet/mnet_client @ 2745ccc6, assets/cable_management/cad_files/round_peg.stl, "
            "byte-identical. Centimetres; the loader applies x0.01 and asserts the result. See "
            "this task's assets/PROVENANCE.md."
        ),
    )
)


# ---------------------------------------------------------------------------
# PLACING ONE. The half of each declaration a document builds through -- what
# ``drape_board()`` is to ``drape_board`` -- except that neither record here can
# name its factory as ``build=``: a ``build`` is zero-arg and answers for a
# *type*, and a peg is placed per station. ``peg0`` is a spool in four documents
# and stands somewhere different in each, so the type cannot say where it goes
# and the factory takes the station instead. ``layout.py`` holds the stations
# and mints one record per document-visible name -- ``weave3_peg0`` -- from the
# type's declaration with one of these, partially applied, as its ``build``.
#
# Both take the reference's own world ``xy`` and convert it through
# ``point_from_surface`` at the one point it becomes a spec, so the numbers
# stay legible as the reference's and the spec is in the work-surface frame,
# where it survives a different table.
# ---------------------------------------------------------------------------


def post(name: str, xy: tuple[float, float], *, cell: CellSpec = BENCH_075) -> StaticSpec:
    """A ``cable_peg`` standing on ``cell``'s work surface at the world ``xy``.

    Cylinders, which ``StaticSpec``'s procedural shape is not -- ``EntityPeg``
    reads the half-extents as ``(radius, _, half_height)`` and says so loudly,
    because a cable wound around a box is a different task. Centred half its
    height above the surface, which is the post's own fact and not the layout's.
    """
    x, y = xy
    return StaticSpec(
        name=name,
        entity="cable_peg",
        half_extents=(PEG_RADIUS, PEG_RADIUS, PEG_HEIGHT / 2),
        xform=(*point_from_surface(cell, (x, y, cell.surface_z + PEG_HEIGHT / 2)), 0.0, 0.0, 0.0, 1.0),
        color=ACCENT_PROP,
    )


def spool(name: str, xy: tuple[float, float], *, cell: CellSpec = BENCH_075) -> StaticSpec:
    """A ``cable_peg_round`` -- ManipulationNet's F1 -- on ``cell``'s surface at the world ``xy``.

    No ``half_extents``: the mesh carries its own size and the entity asserts it.
    Passing one would let a static declare a peg the fixture is not.

    The ``asset`` is this suite's, from the record that already declares it.
    ``EntityPegRound`` lives in the library and the mesh lives here, so the path
    has to travel *up* on the spec rather than be reached for from below -- which
    is what it was doing, against its own module's directory, until every weave
    stopped building. Its base sits on the surface: the mesh's origin is its
    bottom flange.
    """
    x, y = xy
    return StaticSpec(
        name=name,
        entity="cable_peg_round",
        asset=declared_asset(CABLE_PEG_ROUND),
        xform=(*point_from_surface(cell, (x, y, cell.surface_z)), 0.0, 0.0, 0.0, 1.0),
        color=ACCENT_PROP,
    )


# ---------------------------------------------------------------------------
# THE TERMINATED RUN'S HARDWARE: the two ends, and the industrial station the
# panel stands on. Neither is in the reference scene -- it stops at a peg line
# -- and both are declared here because they are this suite's fixtures.
# ---------------------------------------------------------------------------

BLOCK_HALF = (0.016, 0.016, 0.005)
"""The end terminals' declared FOOTPRINT [m] -- 32 x 32 mm, 10 mm of envelope.

Flat, because these are terminals rather than crates: a 40 mm cube stands as
tall as a spool's flange and reads as something the cable must be lifted over,
when what a run does at its end is lie against the thing it terminates at.

**It is a footprint and no longer a shape.** These were one box each, drawn
straight from these three numbers, and a slab says nothing about how a cable is
held at an end -- the welded end read as a rod lying beside a block. What stands
on the footprint now is a fitting:
:class:`~tasks/cable_route/assets/terminals.py::EntityCableTerminal` at the
anchor (rails, a hood, a saddle across the cable) and
:class:`~tasks/cable_route/assets/terminals.py::EntityCablePort` at the port (a
channel with a stop, open on top). Both stay inside 32 x 32 mm, which is what
keeps every number measured against this one true -- the port's scored region,
the standoffs ``weave_terminals`` places by, the seat boxes beside it.
"""

CABLE_BLOCK = register_asset(
    AssetRecord(
        name="cable_block",
        summary="A terminal block a cable is anchored at or routed into.",
        tags=("block", "terminal", "port", "anchor"),
        features=(
            FeatureSpec(
                name="mouth",
                kind="axis",
                payload={"origin": [0.0, 0.0, 0.0], "direction": [0.0, -1.0, 0.0]},
                provenance=(
                    "The face a cable enters by, pointing back along the run. Declared as an axis "
                    "rather than a box so the same block serves as an anchor at one end and a port "
                    "at the other -- what differs between the two is which way it faces, and a "
                    "direction is the only thing that has to."
                ),
            ),
        ),
        sim=SimRequirements(
            # World-fixed, exactly as the pegs are: a terminal is bolted to the
            # bench and the entities that realise it are ``EntityStatic``
            # subclasses. The branch this came from said ``rigid``, which claims
            # a free body -- and the cable cell's arrangement names a backend
            # for the arm and the rod and nothing else, so a rigid claim here
            # would be a domain no solver in this cell serves.
            domain="static",
            contacts=contacts_with("surface"),
        ),
        provenance=(
            "AUTHORED. There is no upstream part: the reference cable scenes end at a peg line, "
            "and the two ends of a real run terminate at something. This said 'a plain box is the "
            "honest stand-in -- it asserts a position and a face and claims no fastening it does "
            "not model', and the box stopped being honest once the rod was actually welded to it: "
            "a fastening the scene DOES model, drawn as something that could not do it. The record "
            "still declares one part with one footprint and one mouth; the two entities that "
            "realise it draw the bracket and the socket that footprint carries."
        ),
    )
)

CLEAT_POST_RADIUS = 0.0095
CLEAT_POST_BASE = 0.007
CLEAT_POST_TOP = 0.037
"""The industrial cleat's standoff [m]: r = 9.5 mm, from 7 mm off the bench to 37.

Declared here rather than reached for from the entity, so the geometry a goal is
scored against and the geometry the physics builds come from one statement. The
spool records the same two things the same way.
"""

CLEAT_HALF = (0.016, 0.020, 0.0215)
"""The cleat's overall half-extents [m]: 32 x 40 x 43 mm.

The envelope of the drawn assembly -- foot 28 x 40 x 7, a 19 mm standoff 30 mm
tall, a 32 x 32 x 6 cap on top -- declared here because a static built from
primitives has no mesh for the compiler to measure."""

CABLE_CLEAT = register_asset(
    AssetRecord(
        name="cable_cleat",
        summary="A bolted cable cleat: 28 x 40 mm foot, a 19 mm standoff, and a retaining cap.",
        tags=("cleat", "post", "standoff", "industrial"),
        features=(
            FeatureSpec(
                name="shaft",
                kind="rod",
                payload={
                    # The STANDOFF, which is what a cable actually bears
                    # against -- the same declaration the spool makes about its
                    # groove, so a seat measured against one fixture reads
                    # identically against the other.
                    "axis": [[0.0, 0.0, CLEAT_POST_BASE], [0.0, 0.0, CLEAT_POST_TOP]],
                    "radius": CLEAT_POST_RADIUS,
                },
                provenance="Declared geometry; the entity builds from the same constants.",
            ),
            FeatureSpec(
                name="axis",
                kind="axis",
                payload={"origin": [0.0, 0.0, 0.0], "direction": [0.0, 0.0, 1.0]},
                provenance=(
                    "The same declaration cable_peg and cable_peg_round make. Swapping a spool "
                    "for a cleat must change what the bench LOOKS like and how hard the weave "
                    "is; it must not change what the question is."
                ),
            ),
        ),
        sim=SimRequirements(domain="static", contacts=contacts_with("surface")),
        provenance=(
            "AUTHORED -- no upstream mesh. Drawn from primitives by "
            "tasks/cable_route/assets/terminals.py's EntityCableCleat, at the proportions of a "
            "bolted cable cleat on a machine backplane. The spool is ManipulationNet's F1 "
            "and looks like laboratory hardware; this is what a harness is dressed against "
            "in a cabinet, and it is thicker, which is most of why the weave is harder."
        ),
    )
)


def cleat(name: str, xy: tuple[float, float], *, cell: CellSpec = BENCH_075) -> StaticSpec:
    """A ``cable_cleat`` bolted to ``cell``'s work surface at the world ``xy``.

    No ``asset``: the part is drawn from primitives and declares no mesh file,
    so pointing it at one fails the suite's load with "asset 'cable_cleat'
    declares no file 'mesh'". The ``half_extents`` are the assembly's envelope
    rather than a shape the entity reads -- a drawn fixture has no mesh for the
    compiler to measure and this is what it measures instead.

    Its origin is the bench: :class:`EntityCableCleat` draws upward from there,
    the way the spool's own mesh origin is its bottom flange.
    """
    x, y = xy
    return StaticSpec(
        name=name,
        entity="cable_cleat",
        half_extents=CLEAT_HALF,
        xform=(*point_from_surface(cell, (x, y, cell.surface_z)), 0.0, 0.0, 0.0, 1.0),
        color=ACCENT_PROP,
    )


def terminal(
    name: str,
    xy: tuple[float, float],
    *,
    entity: str,
    yaw: float = 0.0,
    cell: CellSpec = BENCH_075,
) -> StaticSpec:
    """A ``cable_block`` standing on ``cell``'s surface at the world ``xy``, turned by ``yaw``.

    One record, two entities: ``cable_anchor`` is the bracket a run is clamped
    under and ``cable_port`` is the socket a free end is put into. Which one is
    the caller's, because the difference is what the fitting *does* and not what
    the part measures -- the footprint is :data:`BLOCK_HALF` either way.

    **On the surface, not half in it.** The blocks were placed at
    ``surface_z + BLOCK_HALF[2]`` because a box is centred on its own middle. A
    terminal is drawn UPWARD from its own origin, so its origin is the bench.

    Args:
        name: The document-visible fixture name -- ``anchor`` or ``port``.
        xy: Where it stands, in the reference's world coordinates.
        entity: ``"cable_anchor"`` or ``"cable_port"``.
        yaw: Which way the fitting faces [rad]; local +x is the direction the
            run leaves by.
        cell: The bench the world numbers were authored on.
    """
    x, y = xy
    return StaticSpec(
        name=name,
        entity=entity,
        half_extents=BLOCK_HALF,
        xform=(
            *point_from_surface(cell, (x, y, cell.surface_z)),
            0.0,
            0.0,
            math.sin(0.5 * yaw),
            math.cos(0.5 * yaw),
        ),
        color=ACCENT_PROP,
    )
