# SPDX-License-Identifier: Apache-2.0
"""What this task's document names: a syringe, and the two bins it stands between.

Three types, and each is here rather than borrowed because **a task directory has
to be something you can hand to somebody**. This suite was carved out of
`tool_sort`, and the carving is what exposed how much it had been getting for free
from whichever sibling happened to load first: `syringe` from `tool_sort`'s own
records and `bin` from `astro`'s. Neither is an import, so neither is something a
grep would have found -- the asset registry is global and every root loads
eagerly, so a borrowed declaration looks exactly like an owned one until the
directory travels somewhere its lender does not.

**Declared, not copied wholesale.** `tool_sort`'s records module declares twenty
types; this document names two of them. Bringing the other eighteen would say
this task is about hammers and tape measures, which it is not.

**The bins are `robosuite_bin`, not `bin`, and that is the interesting part.**
`bin` is taken -- by `astro`, which declares an Isaac grocery tote under it -- and
`AssetRecord` equality compares `files` by key and basename, so declaring these
bins as `bin` would have meant either claiming their geometry comes from the
DROID's Isaac stage (it does not; it comes from the robosuite export) or being
refused outright. The registry refusing it is the registry working: two different
things may not share a name.

So `tool_sort`'s nine remaining documents still say `bin1 - bin` and still get
`astro`'s tote record for a robosuite bin. Nothing breaks, because that record
declares no features and only its summary and tags are read -- but it is a latent
collision, and the honest fix is for the VAB tasks to name a VAB bin. This suite
does; the others are left as found rather than rewritten in a commit about the
syringe.
"""

from __future__ import annotations

from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.registry import register_asset
from robosimstudio.assetlib.sim import SimRequirements, contacts_with

__all__ = ["BIN", "INSTRUMENT_TRAY", "SYRINGE"]

# `family=` and `contacts_with=(...)` in the two declarations below are the
# spellings this directory was copied with; the field is `domain` here and a
# contact is a rule rather than a role name. The runtime keeps `.family` as a
# reader and `contacts_with(...)` as the function that builds those rules, so
# this is the same declaration in the vocabulary the checkout speaks -- and the
# whole of what the copy needed changed to load.
# The three declarations below dropped one field on the way in: `substrate`,
# which stated the state layout an asset is stored in. This runtime derives it
# from `sim.domain` and `AssetRecord` no longer takes it, so stating it again
# here would be a second opinion about the same fact rather than a fact.
_SIM = SimRequirements(
    domain="rigid",
    contacts=contacts_with("surface", "fixtures", "robot"),
    # Word for word what `tool_sort` states, because `AssetRecord` equality
    # compares `sim` exactly and `syringe` is a name both suites declare. A
    # rephrasing here would read as a second opinion about the physics and be
    # refused as one.
    notes="Rigid mesh in a MuJoCo cell; nothing exotic. Collision geometry comes from the export.",
)

_EXPORT_PROVENANCE = (
    "VAB-industrial ToolSortPickPlace, robosuite 1.4.0, "
    "MuJoCo 3.3.3, seed 0, dumped by tools/dump_vab_scene.py. The export owns the geometry "
    "and the mass; this declares what the thing IS and leaves where it starts to the document."
)

SYRINGE = register_asset(
    AssetRecord(
        name="syringe",
        summary="A single-use syringe, which this task disposes of rather than sorts.",
        # Tagged `consumable` rather than `tool`, and `sharps` is the word that
        # grounds the instruction: the box is a *sharps* box, and what belongs in
        # one is a property of the object rather than of the receptacle.
        tags=("syringe", "sharps", "consumable"),
        # No features. The grasp box is derived from the mesh by the compiler, and
        # declaring one here would be a copy of a computation rather than a fact
        # about the object -- which matters more here than anywhere: this task
        # fails on *orientation*, and a declared grasp frame would look like an
        # answer to that.
        sim=_SIM,
        files={"export": "vab_tool_sort"},
        provenance=_EXPORT_PROVENANCE,
    )
)

BIN = register_asset(
    AssetRecord(
        name="robosuite_bin",
        summary="A rectangular plastic bin from the robosuite tool-sort bench, open at the top.",
        tags=("bin", "tote", "container"),
        # No features here either, and for a specific reason: what is worth
        # declaring about a bin is its cavity, and a cavity is the *task's* --
        # this document states its own quadrants over its own bins. See
        # `astro/assets/containers.py`, which argues the same point about the
        # tote it declares under the name this one could not use.
        sim=_SIM,
        files={"export": "vab_tool_sort"},
        provenance=_EXPORT_PROVENANCE,
    )
)

INSTRUMENT_TRAY = register_asset(
    AssetRecord(
        name="instrument_tray",
        summary="A shallow 200 x 400 x 12 mm stainless instrument tray, open at the top, that syringes are laid out in.",
        tags=("tray", "instrument_tray", "container"),
        # No features, for the reason the bins give: what is worth declaring
        # about a tray is its cavity, and this task's documents score nothing
        # against it. The MJCF itself carries `cavity_min`/`cavity_max` sites;
        # `task.py` reads the tray's dimensions off the generator's constants
        # to stand a floor collider inside it, since the mesh's convex hull
        # would fill the pan solid.
        sim=SimRequirements(
            domain="rigid",
            contacts=contacts_with("surface"),
            notes="Decorative mesh; the loader supplies the floor it is stood on as a primitive.",
        ),
        files={"mjcf": "instrument_tray/instrument_tray.xml"},
        provenance=(
            "VAB-industrial, add-all-syringes-sharps-box branch, "
            "assets/objects/instrument_tray as generated by assets/objects/_utils/make_tray.py; "
            "vendored beside this declaration. See PROVENANCE.md."
        ),
    )
)

# `sharps_box` is deliberately **not** declared, and that is not an oversight.
#
# The document names `syringe_box - sharps_box` so its region has a frame to hang
# off and so the typechecker can see the reference resolve. There is no body: the
# box is welded into bin2 as geometry by the export, and `task.py`'s `_finish`
# drops container fixtures from the object list for exactly that reason. A record
# would claim there is a thing to import, place and grasp, and the one fact worth
# having about it -- where its aperture sits inside bin2 -- is in the export's
# sidecar, measured, where the loader reads it.
