# SPDX-License-Identifier: Apache-2.0
"""Industrial wallpaper on a scenery room's walls.

The project page's clips put a photograph-like backdrop behind the robot: a lab
with green walls behind the syringe bench, a warehouse with racks behind the
tool-sorting bench. The images are ``assets/backdrop/`` (provenance in
``ASSET_PROVENANCE.md``). Here they are laid across whole walls of the same room
:func:`~robosimstudio.core.render.room.house_room` draws, once each, upright.

Path tracer only. A scenery builder is called by the RTX backend alone, so the
tiled renderer a perception policy is scored through still sees no backdrop.
"""

from __future__ import annotations

from robosimstudio.core.render.finish import Finish
from robosimstudio.core.render.room import ROOM_PLASTER, ROOM_WALL, WALL_HEIGHT, WALL_SETBACK, house_room
from robosimstudio.paths import resolve_path

__all__ = ["papered_room", "wallpaper"]

SIDES = ("-x", "+x", "+y", "-y")


def wallpaper(image: str, center, half, side: str) -> Finish:
    """A finish that lays ``image`` across one wall's face exactly once.

    The tracer's box UVs are in world metres along the face's two axes. So the
    scale is one over the face's size, and the offset moves this wall's own
    centre to the middle of the image.
    """
    along = 1 if side[1] == "x" else 0  # an x wall runs along y, a y wall along x
    su, sv = 1.0 / (2.0 * float(half[along])), 1.0 / (2.0 * float(half[2]))
    return Finish(
        roughness=0.9,
        texture=str(resolve_path(f"backdrop/{image}")),
        uv_scale=(su, sv),
        uv_offset=(0.5 - float(center[along]) * su, 0.5 - float(center[2]) * sv),
    )


def papered_room(
    kit,
    papers: dict[str, str],
    *,
    walls: tuple[str, ...] = SIDES,
    setback: float = WALL_SETBACK,
    wall_height: float = WALL_HEIGHT,
    wall_color=ROOM_WALL,
    wall_finish: Finish = ROOM_PLASTER,
    prefix: str = "room",
    **room,
) -> None:
    """:func:`house_room`, with the walls ``papers`` names wearing an image.

    Args:
        papers: Wall side (``"-x"``, ``"+y"``, ...) to an image in
            ``assets/backdrop/``. Each named side is built even if ``walls``
            leaves it out.
        walls, setback, wall_height, wall_color, wall_finish, prefix, room:
            As :func:`house_room` takes them; the papered walls keep its geometry.
    """
    house_room(
        kit,
        walls=tuple(w for w in walls if w not in papers),
        setback=setback,
        wall_height=wall_height,
        wall_color=wall_color,
        wall_finish=wall_finish,
        prefix=prefix,
        **room,
    )
    cell = getattr(kit, "cell", None)
    cx, cy, _ = (float(v) for v in (getattr(cell, "center", None) or (0.0, 0.0, 0.0)))
    ground_z = float(getattr(cell, "ground_z", 0.0) or 0.0)
    span = setback + 0.5  # house_room's default: closes the corners
    half_h = float(wall_height) / 2.0
    for side, image in papers.items():
        sign = 1.0 if side[0] == "+" else -1.0
        if side[1] == "x":
            center, half = (cx + sign * setback, cy, ground_z + half_h), (0.02, span, half_h)
        else:
            center, half = (cx, cy + sign * setback, ground_z + half_h), (span, 0.02, half_h)
        kit.box(
            f"{prefix}_wall_{side[1]}{'p' if sign > 0 else 'n'}",
            center,
            half,
            color=wall_color,
            finish=wallpaper(image, center, half, side),
        )
