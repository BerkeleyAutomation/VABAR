# SPDX-License-Identifier: Apache-2.0
"""The USB-C plug and receptacle: geometry, and the surgery a rig needs.

Two modules that are content and happen to be code.

``stage`` is the reference assembly -- a plug and a port whose every dimension
comes from the connector standard by way of the insertion reference this suite
reproduces. It builds their colliders, because a 0.15 mm clearance fit is not
something a mesh importer's convex hull will give you.

``rig`` is the build-time surgery: fingertip blades narrow enough to reach into
a 16.8 mm mouth, and the arithmetic that turns a desired grip into a finger
command.

``parts`` is the declaration a *task* names: a cavity on the port and a
grasp_box on the plug, both read out of the reference assembly rather than
retyped, so the declaration and the collider it describes cannot disagree --
and, on the two free parts, the factory that places each where the assembly
spawns it.

``entity`` is the cables and the detent, ``scenery`` the bench and the cradle.
Every one of them is a record naming its placement factory, which is what lets
the document alone stage the whole scene: ``(:objects cables - usbc_cables)``
lowers to the spec the record builds.

All of it moved here from the task that first needed it. A plug is a plug -- its
dimensions are facts about the part, not about the task of inserting it -- and
the same is true of a hand modified to grip one. What stayed behind is the
*solution*: the tools, the graph.
"""

from __future__ import annotations

from . import entity, parts, rig, scenery, stage

__all__ = ["entity", "parts", "rig", "scenery", "stage"]
