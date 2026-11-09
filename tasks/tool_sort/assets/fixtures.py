# SPDX-License-Identifier: Apache-2.0
"""What ``syringe``'s fixtures name: a robosuite bin, and a box that is deliberately not declared.

Two bins under the name ``robosuite_bin``, and each is here rather than borrowed
because **a task directory has to be something you can hand to somebody**. The
syringe variant was carved out of this suite for a while, and the carving is what
exposed how much it had been getting for free from whichever sibling happened to
load first: ``syringe`` from this suite's own records and ``bin`` from ``astro``'s.
Neither is an import, so neither is something a grep would have found -- the
asset registry is global and every root loads eagerly, so a borrowed declaration
looks exactly like an owned one until the directory travels somewhere its lender
does not. The variant came back; the record it wrote for its bins came with it.

**The bins are ``robosuite_bin``, not ``bin``, and that is the interesting part.**
``bin`` is taken -- by ``astro``, which declares an Isaac grocery tote under it --
and ``AssetRecord`` equality compares ``files`` by key and basename, so declaring
these bins as ``bin`` would have meant either claiming their geometry comes from
the DROID's Isaac stage (it does not; it comes from the robosuite export) or
being refused outright. The registry refusing it is the registry working: two
different things may not share a name.

So this suite's other ten documents still say ``bin1 - bin`` and still get
``astro``'s tote record for a robosuite bin. Nothing breaks, because that record
declares no features and only its summary and tags are read -- but it is a
latent collision, and the honest fix is for the VAB tasks to name a VAB bin.
``syringe`` does; the others are left as found rather than rewritten in a commit
about the merge.
"""

from __future__ import annotations

from robosimstudio.assetlib.registry import register_asset
from robosimstudio.domains.rigid import Body

__all__ = ["BIN"]

_NOTES = "Rigid mesh in a MuJoCo cell; nothing exotic. Collision geometry comes from the export."

_EXPORT_PROVENANCE = (
    "VAB-industrial ToolSortPickPlace, robosuite 1.4.0, "
    "MuJoCo 3.3.3, seed 0, dumped by tools/dump_vab_scene.py. The export owns the geometry "
    "and the mass; this declares what the thing IS and leaves where it starts to the document."
)

BIN = register_asset(
    Body.from_export(
        "vab_tool_sort",
        name="robosuite_bin",
        summary="A rectangular plastic bin from the robosuite tool-sort bench, open at the top.",
        tags=("bin", "tote", "container"),
        # No features here either, and for a specific reason: what is worth
        # declaring about a bin is its cavity, and a cavity is the *task's* --
        # the document states its own quadrants over its own bins. See
        # `astro/assets/containers.py`, which argues the same point about the
        # tote it declares under the name this one could not use.
        notes=_NOTES,
        provenance=_EXPORT_PROVENANCE,
    )
)

# `sharps_box` is deliberately **not** declared, and that is not an oversight.
#
# `syringe` and `sort_power` name `syringe_box - sharps_box` so their region has
# a frame to hang off and so the typechecker can see the reference resolve.
# There is no body: the box is welded into bin2 as geometry by the export, and
# the loader drops container fixtures from the object list for exactly that
# reason. A record would claim there is a thing to import, place and grasp, and
# the one fact worth having about it -- where its aperture sits inside bin2 --
# is in the export's sidecar, measured, where the loader reads it.
