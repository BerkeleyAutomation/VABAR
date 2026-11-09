# SPDX-License-Identifier: Apache-2.0
"""How hard a gear_mesh variant is, from its stations alone.

The measure slot ``docs/task_layout.md`` names, filled the way
``cable_route/difficulty.py`` fills it: a number read off the layout, so a
suite can order its own variants and a document's provenance can say which
rung it stands on without anybody taking the name on trust.

**One axis, and it is the chain's length.** A gear on its peg is free to turn,
and turning is how it finds its neighbour's tooth spaces -- so the first gear
meets only the pinion, and every gear after it meets a neighbour that is
already seated and cannot move aside. Each added station is one more such
landing, and one more thing a knock upstream can undo. Repeated tooth counts
are reported because they are what the four-station tier is built on -- two
interchangeable gears, and a peg spacing cut for one of them -- but they do not
move the level: a three-gear board with two 20-tooth gears is still three
landings.

The bands are the tiers the documents ship: one or two stations is ``easy``,
three is ``medium``, four is ``hard``. ``tools/build_gear_scenes.py`` stamps a
drawn variant's level from :func:`profile`, and ``tests/test_gear_mesh.py``
holds the table to it, so the level in a document's name is a computed claim.
"""

from __future__ import annotations

from typing import Any

__all__ = ["LEVELS", "level", "profile"]

LEVELS: dict[int, str] = {1: "easy", 2: "easy", 3: "medium", 4: "hard"}
"""The tier a chain of this many stations belongs to."""


def level(stations: tuple[int, ...]) -> str:
    """The tier a board cut for ``stations`` stands on."""
    count = len(tuple(stations))
    if count not in LEVELS:
        raise ValueError(f"no tier for a {count}-station board; the bands cover {sorted(LEVELS)}")
    return LEVELS[count]


def profile(stations: tuple[int, ...]) -> dict[str, Any]:
    """Everything the measure reads off a station list: its length, its repeats, and the tier."""
    counts = tuple(int(z) for z in stations)
    return {
        "stations": len(counts),
        "repeats": len(counts) - len(set(counts)),
        "level": level(counts),
    }
