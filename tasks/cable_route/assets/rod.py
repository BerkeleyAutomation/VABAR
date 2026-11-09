# SPDX-License-Identifier: Apache-2.0
"""What this task calls the library's rod and its fixtures: six names, bound here.

The construction half of :mod:`.cable`, which is the declaration half -- except
that the construction moved to :mod:`robosimstudio.domains.rod` when
``threaded_needle`` turned out to need the same capsule chain, and two tasks may
not reach each other. What is left here is the part that was always this task's:
**the names**.

That is not a formality. Every entity name in this tree is declared by the task
that stages it -- ``film``, ``tape``, ``garment``, ``usbc_rig``, ``vab_workcell``
-- and none by the library. The mechanism being shared does not make the
vocabulary shared: ``cable_rod`` is what *this* task's documents ask for, and a
colleague's rod task may bind the same classes to whatever its own documents say.
Keeping :func:`~robosimstudio.scene.registry.register_entity` on this side is
what keeps that true.

Three of the six classes are not the library's at all --
:mod:`~tasks/cable_route/assets/terminals.py` draws the run's two ends and the
industrial cleat here, because a saddle bracket is a prop and the capsule chain
is a mechanism. Binding them alongside the library's three keeps one list of
what this suite's documents may name.

The measurements are also still this task's, and they live where they always did:
:class:`~robosimstudio.domains.rod.SpecCable`'s defaults are the weave
reference's own rod, and ``assets/cable.py`` and ``assets/layout.py`` override
per variant.
"""

from __future__ import annotations

from robosimstudio_materials.rod import (
    CableDriveSpec,
    CableLatchSpec,
    EntityCable,
    EntityPeg,
    EntityPegRound,
    SpecCable,
)

from robosimstudio.scene.registry import register_entity

from .terminals import EntityCableCleat, EntityCablePort, EntityCableTerminal

__all__ = [
    "CableDriveSpec",
    "CableLatchSpec",
    "EntityCable",
    "EntityCableCleat",
    "EntityCablePort",
    "EntityCableTerminal",
    "EntityPeg",
    "EntityPegRound",
    "SpecCable",
]

# Called rather than decorated, because the classes are imported rather than
# defined here. `register_entity` returns the class unchanged, so binding a name
# to a library class is the same operation as decorating a local one -- and
# `ENTITIES` compares registrations by (module basename, qualname), so a second
# task binding the same class to the same name is accepted as identical rather
# than refused as a duplicate.
register_entity("cable_rod")(EntityCable)
register_entity("cable_peg")(EntityPeg)
register_entity("cable_peg_round")(EntityPegRound)
# The industrial fixture, drawn rather than imported -- see CABLE_CLEAT. Its
# class is this task's own (``assets/terminals.py``), so this one IS a
# decoration in everything but spelling; it is called for symmetry with the
# three above rather than because it has to be.
register_entity("cable_cleat")(EntityCableCleat)
# THE TWO ENDS ARE NO LONGER ONE NAME, and the reason they were is worth keeping
# because it stopped being true. This said: "The end blocks are plain scenery: a
# box with half-extents and a pose, which is exactly what `EntityStatic` builds.
# No dedicated class, because there is no geometry to assert -- `EntityPeg`
# exists to refuse a box where a cylinder was meant, and a block that IS a box
# has nothing to be wrong about."
#
# It was right about `EntityPeg` and wrong about the blocks, because a box is
# not what either end of a run is. A fixed end is *clamped* and a port is
# something an end is *put into*, and a slab drawn for both says neither: the
# welded end read as a rod lying beside a block, which is what it looked like
# because it is what the geometry said. Two names now, because the two ends
# differ in the one way that matters -- one holds a cable and the other receives
# one -- and `cable_block` stays as the ASSET both are built from, which is
# still one part with one footprint.
register_entity("cable_anchor")(EntityCableTerminal)
register_entity("cable_port")(EntityCablePort)
