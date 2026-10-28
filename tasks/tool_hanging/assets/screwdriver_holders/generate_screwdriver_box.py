"""Generate the pegboard-mounted screwdriver box mesh in authored metres."""

from pathlib import Path

import trimesh


def box(extents, center):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


# Local X points out from the board, local Y spans the board, and local Z is up.
# The open interior is 57 x 102 x 61 mm: a compact pegboard cup for three or
# four screwdriver handles without dominating the board.
parts = [
    box((0.065, 0.110, 0.004), (0.0325, 0.0732, 0.002)),       # floor
    box((0.004, 0.110, 0.071), (0.002, 0.0732, 0.0395)),      # pegboard-side wall
    box((0.004, 0.110, 0.048), (0.063, 0.0732, 0.028)),       # lower front wall
    box((0.065, 0.004, 0.071), (0.0325, 0.0202, 0.0395)),     # left wall
    box((0.065, 0.004, 0.071), (0.0325, 0.1262, 0.0395)),     # right wall
]

# Two visible mounting straps rise behind the container. Their short rearward
# caps read as hooks that pass through pegboard slots and bear on the back face.
for y in (0.043, 0.1034):
    parts.extend(
        [
            box((0.008, 0.018, 0.040), (0.000, y, 0.090)),
            box((0.018, 0.018, 0.008), (-0.005, y, 0.107)),
            box((0.006, 0.018, 0.018), (-0.011, y, 0.101)),
        ]
    )

mesh = trimesh.util.concatenate(parts)
mesh.remove_unreferenced_vertices()
mesh.export(Path(__file__).with_name("screwdriver_box.stl"))
