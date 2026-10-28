# SPDX-License-Identifier: Apache-2.0
"""What this suite declares: a gear per catalogue size, a board per station list, and the mat.

**The features are computed, not transcribed.** A gear's bore is a ``loop`` whose
clear radius has to be what the *collider* leaves rather than what the drawing
says, and its rim is a ``grasp_box`` whose half-width is the tip radius -- both
of which ``gearing.gear_frame`` already derives. Writing either as a literal here
would put the same number in two places and let a change to the module move one
of them, which is the failure ``gear_frame`` exists to prevent, restated at the
declaration layer.

**One record per tooth count, the way ``tool_hanging`` ships three wrenches.**
A record is a claim about a *thing*, and a 16-tooth gear and a 24-tooth gear are
not the same thing: they have different bores to thread, different rims to grasp
and different ratios to turn at. The catalogue is small and shared, so the
records are generated from it rather than written out.

**The board publishes everything a goal says about the gears.** ``_published``
resolves out of ``snapshot.extras`` alone and an ``ObjectSpec`` lands in the
``objects`` bin, so a gear cannot publish its own state -- a record that declared
one would score every episode zero while reading perfectly correctly. This is
``box_decant``'s arrangement, where the seam field publishes the carton's states.
"""

from __future__ import annotations

from functools import partial

from robosimstudio.assetlib import AssetRecord, register_asset
from robosimstudio.assetlib.semantics import FeatureSpec, PublishedState, StateSpec
from robosimstudio.specs import SimRequirements, contacts_with

from . import board, gear, gearing  # noqa: F401 -- registers the entity kinds
from .board import SPIN_GATE_FRACTION, SPIN_WINDOW
from .gear import GEAR_DENSITY, SpecGear
from .gearing import gear_frame, gear_volume
from .layout import STATIONS, board_for, catalogue_gear, gear_mat

CATALOGUE: tuple[int, ...] = (14, 16, 20, 24)
"""Tooth counts this suite ships, at the shared module.

Four, and the shared module is what makes them a catalogue rather than four
gears: any two of them mesh, at a centre distance ``gearing.chain_centres``
computes. 14 is the drive pinion's, kept in the catalogue because the board
builds it from the same arithmetic even though no policy ever handles it.

Size variation is which of these a variant draws. There is no continuous size
axis and this is the honest reason why: nothing an episode sampler reports
reaches ``AssetSpec.scale`` or a shape's extents, and a gear scaled alone would
stop meshing with everything else on the board."""

MAX_STATIONS = 4
"""The most pegs any board in this suite carries.

Not ``len(CATALOGUE) - 1``, which is what it was first written as and is wrong
for a reason worth stating: a board may put the SAME tooth count on two stations,
and the four-gear tier does exactly that -- two 20-tooth gears, interchangeable
with each other and with nothing else, which is most of what makes it the hard
tier. So the station count is a property of the boards, not of the catalogue.
The record declares a state per station up to this many because a record is a
claim about the board, and it is the same board whether two pegs are filled or
four."""

MODULE = SpecGear.module
FACE = SpecGear.face
BORE = SpecGear.bore_radius


def _gear_record(teeth: int) -> AssetRecord:
    """One catalogue gear, with its bore, its axis and its rim."""
    frame = gear_frame(module=MODULE, teeth=teeth, face=FACE, bore_radius=BORE)
    mass = GEAR_DENSITY * gear_volume(frame)
    return AssetRecord(
        name=f"gear_wheel_{teeth}t",
        summary=(
            f"A {teeth}-tooth involute spur gear, {2 * frame['r_tip'] * 1000:.0f} mm across the tips "
            f"and {FACE * 1000:.0f} mm thick, bored to drop over a peg and mesh with its neighbours."
        ),
        tags=("gear", "spur_gear", "workpiece", "assembly"),
        features=(
            FeatureSpec(
                name="bore",
                kind="loop",
                payload={
                    "center": [0.0, 0.0, 0.0],
                    "normal": [0.0, 0.0, 1.0],
                    # What the COLLIDER leaves, not what the drawing says. The hub
                    # is a ring of convex sectors and each one chords its inner
                    # arc, so the clear radius is smaller than the bore by a
                    # sagitta -- and a peg sized to the drawing jams.
                    "inner_r": float(frame["bore_effective"]),
                    "thickness": float(FACE),
                },
                provenance=(
                    "Derived from gearing.gear_frame, which is also what the peg is sized against, "
                    "so `encloses(gear.bore, board.peg)` and the geometry it scores are one number."
                ),
            ),
            FeatureSpec(
                name="axis",
                kind="axis",
                payload={"origin": [0.0, 0.0, 0.0], "direction": [0.0, 0.0, 1.0]},
                provenance="The gear's own spin axis: its bore's normal, and what a spin rate is signed against.",
            ),
            FeatureSpec(
                name="body",
                kind="grasp_box",
                payload={
                    "half": [float(frame["r_tip"]), float(frame["r_tip"]), float(frame["half_face"])],
                    "center_local": [0.0, 0.0, 0.0],
                },
                provenance=(
                    "Stated rather than left to the compiler's OBB for polish_strip's reason -- the "
                    "body is built from its own parts, and the grip measure should read the tip "
                    "circle the jaws actually close across rather than a hull over fifty prisms."
                ),
            ),
        ),
        sim=SimRequirements(domain="rigid", contacts=contacts_with("surface", "fixtures", "objects", "robot")),
        # The gear, resting on the mat. A gear is one thing wherever it lies,
        # but where it lies is the variant's -- loose on the mat at a spot that
        # depends on how many others are loose, or seeded on its peg -- and a
        # zero-argument factory cannot carry that. So the factory says what a
        # gear rests on, and each document records where its own named gears
        # start: ``(:start-poses poses/<variant>.json)``, written from
        # ``layout.gears_for`` and pinned to it by tests/test_gear_mesh.py.
        build=partial(catalogue_gear, teeth),
        provenance=(
            f"Authored for this corpus. Standard 20-degree involute at module {MODULE * 1000:.2f} mm, "
            f"{gearing.DEDENDUM_FACTOR} dedendum, backlash {frame['backlash'] * 1000:.2f} mm, generated by "
            f"gearing.py and decomposed into {teeth} tooth prisms plus {frame['hub_sectors']} hub sectors -- "
            f"every one convex by construction, because MuJoCo collides a mesh as its hull and a gear "
            f"imported whole is a disc. Mass {mass * 1000:.1f} g at {GEAR_DENSITY:.0f} kg/m^3 (acetal), "
            "derived from the parts rather than stated. The vendored spur_gear_20t/30t under "
            "assets/vab_tool_sort are a module-5 pair at 110 and 160 mm, too large for any gripper "
            "here and of unestablished licence; they are a cross-check, not the source."
        ),
    )


GEARS = tuple(register_asset(_gear_record(teeth)) for teeth in CATALOGUE)

_BOARD_STATES: tuple[PublishedState, ...] = (
    PublishedState(
        name="turning_fraction",
        type="float",
        summary="Share of the board's gears that are both seated and turning.",
        tier="perceivable",
        range=(0.0, 1.0),
    ),
    PublishedState(
        name="gears_seated",
        type="float",
        summary="How many gears are on their own peg, flat and at seat height.",
        tier="perceivable",
    ),
    PublishedState(
        name="started",
        type="bool",
        summary="The start button has been pressed past its latch, at any time this episode.",
        tier="perceivable",
    ),
    PublishedState(
        name="press_travel",
        type="float",
        summary="How far the plunger is currently depressed [m].",
        tier="measurable",
        unit="m",
    ),
    PublishedState(
        name="pinion_speed",
        type="float",
        summary="The motor pinion's rate about the board's up axis [rad/s].",
        tier="measurable",
        unit="rad/s",
    ),
)


def _station_gates(index: int) -> tuple[StateSpec, ...]:
    """The two things a goal says about one station, as named states.

    Both are thresholds over a *normalised* measure, which is what lets one
    record serve a two-gear board and a four-gear one. A rate in rad/s could not:
    the ideal differs per station and per variant, and the document would carry a
    different literal for each.
    """
    return (
        StateSpec(
            name=f"gear{index}_turning",
            measure=f"gear{index}_ratio",
            op=">=",
            value=SPIN_GATE_FRACTION,
            summary=f"The gear on station {index} is turning at its share of the chain's speed.",
            provenance=(
                f"Half of the station's own ideal ratio, and the margin is wide at both ends. "
                f"Measured on the assembled three-gear board at 4 rad/s with the rate windowed over "
                f"{SPIN_WINDOW:.2f} s: every gear holds 0.945, which is exactly the pinion's own servo "
                "droop of 3.78/4.00 -- the mesh itself loses nothing -- and the 5th percentile of that "
                "reading is 0.86. A gear whose upstream neighbour is missing is turned by nothing and "
                "reads 0. Windowing is what makes the gate placeable at all: sampled instantaneously "
                "the same healthy gear's 5th percentile is 0.35, 0.14 and 0.05 of ideal going down "
                "the chain, because a driven gear rattles through its own backlash. "
                "See tasks/gear_mesh/rigs/gear_bench.py."
            ),
        ),
        StateSpec(
            name=f"gear{index}_on_peg",
            measure=f"gear{index}_seated",
            op=">=",
            value=1.0,
            summary=f"The gear on station {index} is over its own peg, flat, at seat height.",
            provenance=(
                "Geometric and exact, off body poses: within the peg's own radius of the seat in the "
                "plane, within half a face in height, and within 12 degrees of the board's up axis. "
                "The lateral tolerance is the peg radius rather than a chosen number because a gear "
                "further off than that is not on the peg at all -- its bore could not contain it. "
                "`encloses(gear.bore, peg.shaft)` asks the same question in the goal language's own "
                "words and is the better form; it is not used here because a `rod` feature's endpoints "
                "would have to be one variant's station positions, and the stations are what varies."
            ),
        ),
    )


def _station_states(index: int) -> tuple[PublishedState, ...]:
    """What the board says about the gear on one station.

    Declared for every station the catalogue could fill rather than for the ones
    a given variant uses: a record is a claim about the board, and the board is
    the same board whether two gears or four are on it.
    """
    return (
        PublishedState(
            name=f"gear{index}_spin",
            type="float",
            summary=f"Rate of the gear on station {index}, unsigned [rad/s].",
            tier="measurable",
            unit="rad/s",
        ),
        PublishedState(
            name=f"gear{index}_signed",
            type="float",
            summary="The same rate, signed about the board's up axis -- neighbours counter-rotate.",
            tier="internal",
            unit="rad/s",
        ),
        PublishedState(
            name=f"gear{index}_ratio",
            type="float",
            summary=f"Station {index}'s rate as a share of what the intact chain would give it.",
            tier="perceivable",
            range=(0.0, 2.0),
        ),
        PublishedState(
            name=f"gear{index}_ideal",
            type="float",
            summary=f"What station {index} would turn at with the chain intact [rad/s].",
            tier="internal",
            unit="rad/s",
        ),
        PublishedState(
            name=f"gear{index}_seated",
            type="bool",
            summary=f"The gear on station {index} is on its peg, flat, at seat height.",
            tier="perceivable",
        ),
    )


GEAR_MAT = register_asset(
    AssetRecord(
        name="gear_mat",
        summary="The mat the loose gears wait on, in front of the board.",
        tags=("mat", "fixture"),
        sim=SimRequirements(domain="static", contacts=contacts_with("objects", "robot")),
        provenance=(
            "Its own record rather than polish's `work_mat`: the registry accepts an identical "
            "redundant copy but not a differing one, and a task may not import a sibling's assets, "
            "so sharing a name across two suites is a coupling nothing checks. A gear is a rolling "
            "body on its rim and a bench top is where it rolls to; the mat's friction is what makes "
            "one set down at an angle stay where it was put."
        ),
        # The one thing on this bench that stands in the same place in every
        # variant, so the record can place it unaided.
        build=gear_mat,
    )
)

_TIER_BOARDS: dict[tuple[int, ...], str] = {
    STATIONS["pair"]: "gear_board_pair",
    STATIONS["train"]: "gear_board_train",
    STATIONS["quad"]: "gear_board_quad",
}
"""The three boards the hand-written tiers introduced, by their stations."""


def board_record_name(stations: tuple[int, ...]) -> str:
    """The record a document names for a board cut for ``stations``.

    A tier's board keeps the name the tier gave it; any other station list --
    the drawn variants' -- is named by its teeth, ``gear_board_24_20``, so the
    name says what the pegs are cut for and two documents that draw the same
    stations share one record.
    """
    key = tuple(int(z) for z in stations)
    return _TIER_BOARDS.get(key) or "gear_board_" + "_".join(str(z) for z in key)


BOARDS: dict[str, tuple[int, ...]] = {
    board_record_name(stations): stations for stations in sorted({tuple(s) for s in STATIONS.values()}, key=lambda s: (len(s), s))
}
"""One board record per station list any variant stands, named for the tier that introduced it or for its teeth.

A board's length and its peg spacing follow its stations, and the stations are
the variant -- what a document's ``:objects`` say, in order. A record's ``build``
is a zero-argument factory, so each of these binds ``layout.board_for`` to one
station list, and a document names the board whose pegs are cut for its gears:
``board - gear_board_train``. ``quad`` and ``quad_seeded`` share the four-station
board, because it is the same board whether two of its pegs start filled or
none do; every drawn variant with the same stations shares one the same way.
The states and the published names are the same on every one: a record is a
claim about the board, and the claim does not change with its length."""


def _board_record(name: str, stations: tuple[int, ...]) -> AssetRecord:
    """The board cut for ``stations``, with everything a goal says about the gears on it."""
    return AssetRecord(
        name=name,
        summary=(
            f"The assembly board cut for {len(stations)} stations ({', '.join(f'{z}t' for z in stations)}): "
            "a plate carrying a row of pegs at the centre distances its gears mesh at, a motor pinion "
            "on the first station, and a sprung start button."
        ),
        tags=("board", "fixture", "assembly", "machine"),
        states=(
            *(gate for index in range(MAX_STATIONS) for gate in _station_gates(index)),
            StateSpec(
                name="loose",
                measure="gears_seated",
                op="<=",
                value=0.0,
                summary="No gear is on a peg: the board as an episode finds it.",
                provenance=(
                    "The init-state mirror of `assembled`. Read off body poses, which are settled and "
                    "correct at the reset instant -- unlike a spin or a contact force, both of which "
                    "polish records as init clauses that read false at t=0 and true a dozen steps later."
                ),
            ),
            StateSpec(
                name="unstarted",
                measure="started",
                op="<=",
                value=0.0,
                summary="The motor has not been started.",
                provenance=(
                    "The latch, not the plunger's depth. Geometric and exact at reset: the entity zeroes "
                    "it in `reset`, so a board can never come up already running -- which would score "
                    "every episode's spin clause before the policy touched anything."
                ),
            ),
            StateSpec(
                name="running",
                measure="turning_fraction",
                op=">=",
                value=1.0,
                summary="Every gear the board expects is seated on its peg and turning.",
                provenance=(
                    "1.0 rather than a fraction: the task is the whole train, and a chain with a hole "
                    "in it is not a partly-assembled gearbox but a broken one. The per-gear rate gate "
                    "behind this is half of each gear's own ideal ratio -- see the `spin` atom's "
                    "calibration and tasks/gear_mesh/rigs/gear_bench.py."
                ),
            ),
        ),
        published=(
            *_BOARD_STATES,
            *(state for index in range(MAX_STATIONS) for state in _station_states(index)),
        ),
        sim=SimRequirements(domain="rigid", contacts=contacts_with("objects", "robot", "surface")),
        provenance=(
            "Authored for this corpus, after nvidia-isaac/Isaac-cap's `gear_insertion` -- which places "
            "three Factory gears on a plate and scores them SEATED AND STILL, at three post offsets "
            "stated as constants. Here the peg spacing is derived from the gears' pitch radii instead, "
            "so a board that does not mesh cannot be authored, and the score is whether the train "
            "turns. The pinion is a velocity-driven revolute on polish's rotary spindle pattern; the "
            "button is this repository's first prismatic joint."
        ),
        # Placed by the layout, with this record's stations bound: the plate is
        # centred on ``layout.BOARD_AT`` whatever its length.
        build=partial(board_for, stations),
    )


GEAR_BOARDS = {name: register_asset(_board_record(name, stations)) for name, stations in BOARDS.items()}
