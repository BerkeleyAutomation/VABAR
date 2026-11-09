# SPDX-License-Identifier: Apache-2.0
"""The VAB-industrial hand tools: the boring family, and the point of it.

A hammer, a drill, a clamp, screwdrivers, pliers, a syringe, a multimeter, a
breadboard -- twenty-three types across eight problem files. Rigid meshes out of
a vendored export, and between them they declare **almost nothing** -- which is
the fact worth recording rather than a gap in the work.

A rigid asset's grasp geometry is *derived*. The compiler already fits an
oriented box to every mesh it imports, works out the graspable segment, and can
say whether a given hand fits it; ``sim.get_object_obb`` has served exactly that
for as long as the privileged tools have existed. Re-declaring it here would be
transcribing a computation, and a transcription is a thing that goes stale.

So these carry a substrate, a tag, and the sort of thing they are. Everything a
task needs to say about them -- where they start, which compartment they belong
in -- is a fact about the *task*, and lives in the task's own document.

That is the shape the porting story predicts: annotation cost tracks how
parametric an object's geometry is, not how complicated the object looks. A
claw hammer is visually the most awkward thing in this file and needs the least
said about it.

**Why the compartment labelling is not here.** Each export records which bin
quadrant each tool type is labelled for, and it is tempting to hang that off the
asset. It would be wrong: a hammer is not *intrinsically* a compartment-0 thing.
Another problem file over the same tools labels them differently, and the
tool-sort suite already has variants that do. The labelling is the task's.

**The one thing six of them do declare is a pair of USD stages**, and that is a
file reference rather than a property: ``sort_electrical`` has two restatements
that build the same pile through Newton's USD importer instead of its MJCF one,
and a document naming ``(:objects-from usd)`` needs somewhere to look up *which*
stage. :data:`USD_STAGES` is that lookup, and it lives here for the same reason
every other path does -- an asset says where its files are, and a loader that
said it again would be a second copy to keep in step.
"""

from __future__ import annotations

from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.domains.rigid import Body
from robosimstudio.assetlib.registry import register_asset

__all__ = ["INDUSTRIAL_STEMS", "TOOLS", "USD_STAGES"]

_NOTES = "Rigid mesh in a MuJoCo cell; nothing exotic. Collision geometry comes from the export."

_PROVENANCE = (
    "VAB-industrial's tool-sort export, vendored under assets/vab_tool_sort/. Meshes, collision "
    "geometry and settled poses are the export's own, recorded at seed 0; this declares what the "
    "objects ARE and leaves where they start to the task."
)

#: What each tool is, in one line, and what a perception stack would call it.
#: Tags are for grounding language and connecting to a detector -- never for
#: dispatch. Physics grounds types; perception grounds tags, and BDDL conflates
#: the two, which is why the distinction is kept explicit here.
_TOOLS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("hammer", "A claw hammer: a heavy head on a long handle.", ("hammer", "tool")),
    ("drill", "A power drill: a body with a grip and a chuck.", ("drill", "power_tool", "tool")),
    ("round_nut", "A round nut: a small annular fastener.", ("round_nut", "nut", "fastener")),
    ("clamp", "A spring clamp.", ("clamp", "tool")),
    # The five other variants' tools. These were missing for a while, and the
    # checker said so on every run -- 42 `catalog-gap` problems across four
    # documents, each naming a type nothing had declared. The gap was real: the
    # four above are exactly what `sort_all` uses, and nobody went back for the
    # rest when the other variants were ported.
    #
    # Every mesh is in the vendored export already; only the declaration was
    # missing. Nothing here is a new object.
    (
        "phillips_screwdriver",
        "A screwdriver with a cross-tip blade.",
        ("phillips_screwdriver", "screwdriver", "tool"),
    ),
    (
        "slotted_screwdriver",
        "A screwdriver with a single flat blade.",
        ("slotted_screwdriver", "screwdriver", "tool"),
    ),
    # Distinct from the slotted one in the export, and kept distinct here. The
    # two names are used interchangeably in the world and NOT in these problem
    # files: `sort_new` names both, as separate types with separate meshes, and
    # collapsing them would silently halve that variant's object count.
    (
        "flathead_screwdriver",
        "A screwdriver with a flat blade, a second flat-tip pattern in this export.",
        ("flathead_screwdriver", "screwdriver", "tool"),
    ),
    ("pliers", "Slip-joint pliers: two hinged handles closing on a serrated jaw.", ("pliers", "tool")),
    (
        "adjustable_wrench",
        "An adjustable wrench: a fixed jaw and a worm-driven sliding one.",
        ("adjustable_wrench", "wrench", "tool"),
    ),
    ("utility_knife", "A retractable-blade utility knife.", ("utility_knife", "knife", "tool")),
    ("tape_measure", "A retracting tape measure in a case.", ("tape_measure", "tool")),
    # Tagged `consumable`, not `tool`. `sort_bench_consumables_into_bins` sorts
    # on that distinction, and a tag is what grounds the instruction's own word.
    ("insulating_tape", "A roll of insulating tape.", ("insulating_tape", "tape", "consumable")),
    (
        "syringe",
        "A single-use syringe, which the sharps variant disposes of rather than sorts.",
        ("syringe", "sharps", "consumable"),
    ),
    # --- the industrial pool ------------------------------------------------
    #
    # The ten below come out of `vab_tool_sort/assets/industrial/` rather than
    # the export's YCB meshes, and `sort_electrical` and `sort_wiring` are the
    # first two variants to sort them. Nothing else changes: they are rigid
    # meshes in the same cell, declared the same way, and the export still owns
    # their geometry and mass.
    #
    # Four of them collide by name with something already here or already in the
    # export's flat mesh pool, and each is kept distinct for the reason
    # `flathead_screwdriver` above is. See the individual comments.
    ("battery", "A D-cell battery: a plain cylinder with a raised terminal.", ("battery", "cell", "consumable")),
    ("multimeter", "A handheld multimeter: a slab body with a dial and a screen.", ("multimeter", "meter", "tool")),
    (
        "flashlight",
        "A tactical flashlight: a knurled aluminium tube with a lens at one end.",
        ("flashlight", "torch", "tool"),
    ),
    (
        "safety_glasses",
        "Wraparound safety glasses: a clear lens on two thin arms.",
        ("safety_glasses", "eyewear", "ppe"),
    ),
    # A third pattern of pliers alongside `pliers` and `combination_pliers`,
    # and the one `sort_wiring` asks a policy to tell apart from the second.
    (
        "cutting_pliers",
        "Diagonal cutting pliers: two hinged handles on a shearing jaw.",
        ("cutting_pliers", "pliers", "tool"),
    ),
    (
        "combination_pliers",
        "Combination pliers: a serrated jaw with a cutter behind the pivot.",
        ("combination_pliers", "pliers", "tool"),
    ),
    ("breadboard", "A solderless breadboard: a perforated plastic slab.", ("breadboard", "electronics", "component")),
    ("wire_spool", "A small spool of hookup wire.", ("wire_spool", "spool", "consumable")),
    # A fourth screwdriver, distinct from the three above and from all of them
    # in the mesh. Upstream files it as `industrial/screwdriver` precisely
    # because the bare name was already taken; collapsing it into one of the
    # others would silently halve `sort_electrical`'s distractor count.
    ("screwdriver", "A screwdriver from the industrial pool, scanned rather than modelled.", ("screwdriver", "tool")),
    # NOT `tape_measure`, which is above and is a *different mesh by a different
    # author* -- upstream's own ATTRIBUTION.md disambiguates the two as
    # `tape_measure` and `industrial/measuring_tape`. Both are vendored, both
    # are named by a document, and merging them would put one variant's
    # distractor in another variant's goal.
    ("measuring_tape", "A retracting measuring tape in a round case.", ("measuring_tape", "tool")),
)


_INDUSTRIAL = "vab_tool_sort/assets/industrial"

INDUSTRIAL_STEMS: dict[str, str] = {
    "battery": "d_battery",
    "multimeter": "multimeter",
    "flashlight": "tactical_flashlight",
    "safety_glasses": "safety_glasses",
    "cutting_pliers": "cutting_pliers",
    "combination_pliers": "combination_pliers",
    "breadboard": "breadboard",
    "wire_spool": "wire_spool",
    "screwdriver": "screwdriver",
    "measuring_tape": "measuring_tape",
}
"""Type name to the ``industrial/`` asset stem, for the ten this file declares.

The map is not the identity, and that is the point of writing it down: the plate
in ``sort_electrical`` reads ``BATTERY`` while the mesh is ``d_battery``, and a
document that stated the mesh name would be stating something about the vendored
tree rather than about the task.

It is the whole industrial pool this corpus declares, and every entry gives its
type a ``mesh`` and a ``texture``. Those two exist for the suites that *stage
these tools themselves* rather than taking them out of a dumped export --
``astro`` is the first -- and are the one thing a per-object import needs that
an export-resident object does not: ``vab_tool_sort/assets/industrial/`` vendors
each mesh beside its own ``diffuse.png``, with nothing in the ``.obj``
connecting them.
"""

USD_STAGES: tuple[str, ...] = ("battery", "multimeter", "flashlight", "safety_glasses", "cutting_pliers", "screwdriver")
"""Which of the ten also ship a pair of USD stages.

A separate claim from :data:`INDUSTRIAL_STEMS` rather than a second copy of it:
these six are the meshes ``sort_electrical``'s ten objects are instances of,
which is the whole set the two USD variants need, and
``tools/build_industrial_usd.py`` writes them. The other four are declared and
have no USD, which is a fact about what has been built and not about the tools.
"""


def _tool(name: str, summary: str, tags: tuple[str, ...]) -> AssetRecord:
    # An asset names its files; it does not name a path twice. The two USD
    # spellings are separate keys rather than one path with a suffix rule
    # because they are different claims about the same tool -- one carries
    # physics schemas and one carries none -- and `declared_file` asking for
    # the one the staging named is what keeps the choice in the document.
    files: dict[str, str] = {}
    if name in INDUSTRIAL_STEMS:
        stem = INDUSTRIAL_STEMS[name]
        files["mesh"] = f"{_INDUSTRIAL}/{stem}/meshes/{stem}.obj"
        files["texture"] = f"{_INDUSTRIAL}/{stem}/textures/diffuse.png"
        # The per-tool MJCF, named so a suite staging this tool itself can read
        # the mass and friction its author stated rather than inventing them.
        # Not the geometry -- an AssetSpec imports a mesh, not an MJCF.
        files["mjcf"] = f"{_INDUSTRIAL}/{stem}/{stem}.xml"
    if name in USD_STAGES:
        stem = INDUSTRIAL_STEMS[name]
        files["usd"] = f"{_INDUSTRIAL}/{stem}/{stem}.usdc"
        files["usd_geom"] = f"{_INDUSTRIAL}/{stem}/{stem}_geom.usdc"
    return Body.from_export(
        "vab_tool_sort",
        name=name,
        summary=summary,
        tags=tags,
        # No features. The grasp box is derived from the mesh by the compiler,
        # and declaring one here would be a copy of a computation rather than a
        # fact about the object.
        files=files,
        notes=_NOTES,
        provenance=_PROVENANCE,
    )


TOOLS = tuple(register_asset(_tool(name, summary, tags)) for name, summary, tags in _TOOLS)
"""Every tool type the tool_sort problems name, registered on import.

Twenty-three, not the four `sort_all` uses. Five variants were ported from
Python later and their tools were never declared, which the checker reported as
42 catalog gaps and nothing acted on -- a gap that is invisible until something
asks the catalog a question, and now something does. The last ten arrived with
`sort_electrical` and `sort_wiring`, declared in the same commit as the
documents that name them rather than left for the same discovery twice.

The `industrial/` pool holds 39 tools and this declares ten. The other 29 are
vendored and undeclared, which is the right state for them: an asset record is a
claim that something is *in the corpus*, and nothing here has been asked to sort
a spur gear yet. `assets/PROVENANCE.md` is where the other 29 are accounted for.
"""
