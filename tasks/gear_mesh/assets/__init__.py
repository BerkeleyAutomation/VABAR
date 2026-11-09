# SPDX-License-Identifier: Apache-2.0
"""What this task assembles, and the thing it assembles onto.

Five modules. ``gearing`` is the arithmetic -- involute profiles, the convex
decomposition, and the centre distances that make a board mesh; it imports
nothing from this stack and is testable without a GPU. ``gear`` is one free
rigid body built from that arithmetic. ``board`` is the plate, the pegs whose
spacing the arithmetic dictates, the pinion that drives the chain and the button
that starts it -- and, because an ``ObjectSpec`` may not publish a state, it is
also what says anything about the gears. ``layout`` is where all of it stands on
the bench, how an episode moves it, and the tier tables (``STATIONS``,
``SEEDED``) the recorded start poses are written from. ``records`` is what the
documents name: a gear per catalogue size, a board per station list, the mat --
each with the factory the generic loader builds it through.

Import order is build order: ``records`` pulls these in so ``@register_entity``
has run before a scene names one. There is no ``task.py``; discovery executes
this package and lowers the documents itself.
"""

from __future__ import annotations

from . import board, gear, gearing, layout
from .board import EntityGearBoard, SpecGearBoard, board_frame
from .gear import EntityGear, SpecGear
from .gearing import chain_centres, gear_frame, gear_parts, gear_volume, mesh_phase, mesh_separation

__all__ = [
    "EntityGear",
    "EntityGearBoard",
    "SpecGear",
    "SpecGearBoard",
    "board",
    "board_frame",
    "chain_centres",
    "gear",
    "gear_frame",
    "gear_parts",
    "gear_volume",
    "gearing",
    "layout",
    "mesh_phase",
    "mesh_separation",
]
