#!/usr/bin/env python3
"""Generate the package-owned USDA closure for the AUTOLab gear-mesh families.

Geometry is derived from the pinned AUTOLab gear_mesh source (``autolab_root()``).  The generated
gear keeps the RSS convex-per-tooth and convex hub-sector decomposition.
One board USD uses a layout variant set for every source station ordering and
bend while retaining the tapered pegs, driven pinion, and sprung button. A
compact manifest records the exact pair/train layouts used for build-time
variation.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import shutil

import numpy as np
from isaaclab_arena_gap.task_sources import autolab_root

ROOT = Path(__file__).resolve().parents[1]
SOURCE = autolab_root() / "tasks/gear_mesh/assets/gearing.py"
VARIANTS_SOURCE = SOURCE.with_name("variants.json")
OUT = ROOT / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets"
VARIANTS_OUT = (
    ROOT / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/tasks/gear_insertion/variants.json"
)
GEAR_COLORS = {
    16: (0.68, 0.70, 0.74),
    20: (0.78, 0.66, 0.32),
    24: (0.72, 0.50, 0.38),
}


def _source_module():
    spec = importlib.util.spec_from_file_location("autolab_gear_mesh_gearing", SOURCE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _array(values) -> str:
    return ", ".join(str(int(v)) for v in values)


def _points(values) -> str:
    return ",\n                ".join(
        f"({x:.9g}, {y:.9g}, {z:.9g})" for x, y, z in values
    )


def _display_color(srgb: tuple[float, float, float]) -> str:
    """Encode AUTOLab's display/sRGB palette as USD linear RGB."""

    def linear(value: float) -> float:
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return ", ".join(f"{linear(value):.9g}" for value in srgb)


def _mesh(
    name: str,
    vertices: np.ndarray,
    indices: np.ndarray,
    color: tuple[float, float, float],
    material_path: str,
) -> str:
    counts = np.full(len(indices) // 3, 3, dtype=np.int32)
    return f'''        def Mesh "{name}" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "NewtonCollisionAPI", "NewtonMeshCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            uniform token physics:approximation = "convexHull"
            bool physics:collisionEnabled = 1
            float physics:contactOffset = 0.00005
            float physics:restOffset = 0
            float newton:contactGap = 0.00005
            rel material:binding:physics = <{material_path}>
            int[] faceVertexCounts = [{_array(counts)}]
            int[] faceVertexIndices = [{_array(indices)}]
            point3f[] points = [
                {_points(vertices)}
            ]
            uniform token subdivisionScheme = "none"
            color3f[] primvars:displayColor = [({_display_color(color)})]
        }}
'''


def _gear_meshes(
    gearing,
    teeth: int,
    color: tuple[float, float, float],
    material_root: str,
    indent: str = "",
    uniform_material: str | None = None,
) -> str:
    frame = gearing.gear_frame(
        module=0.0025, teeth=teeth, face=0.016, bore_radius=0.006
    )
    blocks = []
    for label, verts, indices in gearing.gear_parts(frame, tip_chamfer=0.0006):
        material = uniform_material or (
            "teeth" if label.startswith("tooth") else "bore"
        )
        blocks.append(
            _mesh(label, verts, indices, color, f"{material_root}/{material}")
        )
    return "".join(blocks).replace("\n        ", "\n        " + indent)


def _material(name: str, friction: float, indent: str = "    ") -> str:
    return f'''{indent}def Material "{name}" (
{indent}    prepend apiSchemas = ["PhysicsMaterialAPI", "NewtonMaterialAPI", "MjcMaterialAPI"]
{indent})
{indent}{{
{indent}    float physics:dynamicFriction = {friction}
{indent}    float physics:staticFriction = {friction}
{indent}    float physics:restitution = 0
{indent}}}
'''


def _prism(
    radius: float,
    height: float,
    *,
    taper: float = 0.0,
    tip_scale: float = 1.0,
    sides: int = 16,
):
    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    rings = (
        [(radius, 0.0), (radius, height)]
        if not taper
        else [(radius, 0.0), (radius, height - taper), (radius * tip_scale, height)]
    )
    vertices = np.vstack(
        [np.hstack([unit * r, np.full((sides, 1), z)]) for r, z in rings]
    )
    vertices = np.vstack([vertices, [[0, 0, height]], [[0, 0, 0]]])
    top, bottom = len(vertices) - 2, len(vertices) - 1
    indices: list[int] = []
    last = sides * (len(rings) - 1)
    for i in range(sides):
        j = (i + 1) % sides
        indices += [last + i, last + j, top, j, i, bottom]
        for ring in range(len(rings) - 1):
            lo, hi = sides * ring, sides * (ring + 1)
            indices += [lo + i, hi + i, lo + j, lo + j, hi + i, hi + j]
    return vertices.astype(np.float32), np.asarray(indices, dtype=np.int32)


def generate_gear(gearing, teeth: int) -> None:
    root_name = f"GearMesh{teeth}t"
    path = OUT / f"vabar__gear_mesh_{teeth}t/vabar__gear_mesh_{teeth}t.usda"
    path.parent.mkdir(parents=True, exist_ok=True)
    meshes = _gear_meshes(
        gearing,
        teeth,
        GEAR_COLORS[teeth],
        f"/{root_name}/PhysicsMaterials",
    )
    path.write_text(f"""#usda 1.0
(
    defaultPrim = "{root_name}"
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root_name}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    float physics:density = 1400
    def Scope "PhysicsMaterials"
    {{
{_material("teeth", 1.4, "        ")}{_material("bore", 0.15, "        ")}    }}
{meshes}}}
""")


def _board_geometry(gearing, stations: tuple[int, ...], bends: tuple[float, ...]):
    counts = (14, *stations)
    frames = [
        gearing.gear_frame(module=0.0025, teeth=teeth, face=0.016, bore_radius=0.006)
        for teeth in counts
    ]
    points = [(0.0, 0.0)]
    heading = 0.0
    for index, (near, far) in enumerate(zip(counts[:-1], counts[1:], strict=True)):
        if index:
            heading += bends[index - 1] if index - 1 < len(bends) else 0.0
        distance = 0.0025 * (near + far) / 2.0
        points.append(
            (
                points[-1][0] + distance * math.cos(heading),
                points[-1][1] + distance * math.sin(heading),
            )
        )
    reach = max(
        point[0] + frame["r_tip"] for point, frame in zip(points, frames, strict=True)
    )
    back = min(
        point[0] - frame["r_tip"] for point, frame in zip(points, frames, strict=True)
    )
    left = min(
        point[1] - frame["r_tip"] for point, frame in zip(points, frames, strict=True)
    )
    right = max(
        point[1] + frame["r_tip"] for point, frame in zip(points, frames, strict=True)
    )
    button_x = back - 0.018 - 0.011
    plate_center = (0.5 * (reach + button_x), 0.5 * (right + left))
    return {
        "frames": frames,
        "points": points,
        "peg_radius": min(frame["bore_effective"] for frame in frames[1:]) - 0.0006,
        "button_at": (button_x, 0.0),
        "plate_center": plate_center,
        "plate_half": (
            0.5 * (reach - button_x) + 0.012,
            0.5 * (right - left) + 0.012,
        ),
    }


def _board_variant(asset_key: str, geometry: dict) -> str:
    plate_center_x, plate_center_y = geometry["plate_center"]
    pinion_x = geometry["points"][0][0] - plate_center_x
    pinion_y = geometry["points"][0][1] - plate_center_y
    button_x = geometry["button_at"][0] - plate_center_x
    button_y = geometry["button_at"][1] - plate_center_y
    plate_half_x, plate_half_y = geometry["plate_half"]
    peg_overrides = []
    for index in range(3):
        active = index < len(geometry["points"]) - 1
        translate = ""
        if active:
            point = geometry["points"][index + 1]
            translate = (
                f"\n                    double3 xformOp:translate = "
                f"({point[0] - plate_center_x:.9g}, "
                f"{point[1] - plate_center_y:.9g}, 0)"
            )
        peg_overrides.append(
            f"""                over "station_{index}_peg" (
                    active = {str(active).lower()}
                )
                {{{translate}
                }}
"""
        )
    return f'''        "{asset_key}" {{
            over "base"
            {{
                over "plate"
                {{
                    double3 xformOp:scale = ({2 * plate_half_x:.9g}, {2 * plate_half_y:.9g}, 0.006)
                }}
{"".join(peg_overrides)}            }}
            over "pinion"
            {{
                double3 xformOp:translate = ({pinion_x:.9g}, {pinion_y:.9g}, 0.008)
            }}
            over "pinion_joint"
            {{
                point3f physics:localPos0 = ({pinion_x:.9g}, {pinion_y:.9g}, 0.008)
            }}
            over "button"
            {{
                double3 xformOp:translate = ({button_x:.9g}, {button_y:.9g}, 0.016)
            }}
            over "button_joint"
            {{
                point3f physics:localPos0 = ({button_x:.9g}, {button_y:.9g}, 0.016)
            }}
        }}
'''


def generate_board(gearing, layouts: dict[str, dict]) -> None:
    root_name = "GearMeshBoard"
    path = OUT / "vabar__gear_mesh_board/vabar__gear_mesh_board.usda"
    path.parent.mkdir(parents=True, exist_ok=True)
    peg_radii = {round(float(layout["peg_radius"]), 12) for layout in layouts.values()}
    if len(peg_radii) != 1:
        raise ValueError(f"gear board layouts require different peg radii: {peg_radii}")
    peg, peg_i = _prism(peg_radii.pop(), 0.038, taper=0.010, tip_scale=0.22)
    button, button_i = _prism(0.011, 0.008)
    button[:, 2] -= 0.008
    pinion = _gear_meshes(
        gearing,
        14,
        # Arena's warehouse HDR shifts ACCENT_PLUG's observed chroma toward
        # brass.  This renderer-calibrated display value reproduces the RSS
        # ~0.40 (r-g)/(r-b) pinion ratio without changing the graph's gate.
        (0.90, 0.525, 0.27),
        f"/{root_name}/PhysicsMaterials",
        "    ",
        uniform_material="pinion",
    )
    peg_mesh = _mesh(
        "peg",
        peg,
        peg_i,
        (0.86, 0.82, 0.68),
        f"/{root_name}/PhysicsMaterials/peg",
    )
    button_mesh = _mesh(
        "cap",
        button,
        button_i,
        (0.85, 0.35, 0.12),
        f"/{root_name}/PhysicsMaterials/button",
    )
    peg_blocks = []
    for index in range(3):
        peg_blocks.append(
            f"""        def Xform "station_{index}_peg"
        {{
            uniform token[] xformOpOrder = ["xformOp:translate"]
{peg_mesh}        }}
"""
        )
    pegs = "".join(peg_blocks)
    variants = "".join(
        _board_variant(asset_key, geometry)
        for asset_key, geometry in sorted(layouts.items())
    )
    path.write_text(f"""#usda 1.0
(
    defaultPrim = "{root_name}"
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root_name}" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI", "NewtonArticulationRootAPI"]
    prepend variantSets = "layout"
    variants = {{
        string layout = "board_20"
    }}
)
{{
    def Scope "PhysicsMaterials"
    {{
{_material("plate", 0.8, "        ")}{_material("peg", 0.15, "        ")}{_material("pinion", 1.0, "        ")}{_material("button", 1.2, "        ")}    }}
    def Xform "base" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        float physics:mass = 2
        def PhysicsFixedJoint "world_joint"
        {{
            rel physics:body1 = </{root_name}/base>
        }}
        def Cube "plate" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            double size = 1
            double3 xformOp:translate = (0, 0, -0.003)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            rel material:binding:physics = </{root_name}/PhysicsMaterials/plate>
            color3f[] primvars:displayColor = [({_display_color((0.22, 0.22, 0.23))})]
        }}
{pegs}    }}
    def Xform "pinion" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        float physics:mass = 0.02
        uniform token[] xformOpOrder = ["xformOp:translate"]
{pinion}    }}
    def PhysicsRevoluteJoint "pinion_joint" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular", "MjcJointAPI"]
    )
    {{
        uniform token physics:axis = "Z"
        rel physics:body0 = </{root_name}/base>
        rel physics:body1 = </{root_name}/pinion>
        point3f physics:localPos1 = (0, 0, 0)
        float drive:angular:physics:damping = 0.0698131701
        float drive:angular:physics:maxForce = 1.5
        float drive:angular:physics:stiffness = 0
        float drive:angular:physics:targetVelocity = 0
        uniform double mjc:armature = 0.0003
    }}
    def Xform "button" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        float physics:mass = 0.02
        uniform token[] xformOpOrder = ["xformOp:translate"]
{button_mesh}    }}
    def PhysicsPrismaticJoint "button_joint" (
        prepend apiSchemas = ["PhysicsDriveAPI:linear", "MjcJointAPI"]
    )
    {{
        uniform token physics:axis = "Z"
        rel physics:body0 = </{root_name}/base>
        rel physics:body1 = </{root_name}/button>
        point3f physics:localPos1 = (0, 0, 0)
        float physics:lowerLimit = -0.008
        float physics:upperLimit = 0
        float drive:linear:physics:damping = 4
        float drive:linear:physics:stiffness = 400
        float drive:linear:physics:targetPosition = 0
        uniform double mjc:armature = 0.001
    }}
    variantSet "layout" = {{
{variants}    }}
}}
""")


def _target_offsets(geometry: dict) -> tuple[tuple[float, float, float], ...]:
    plate_center_x, plate_center_y = geometry["plate_center"]
    return tuple(
        (point[0] - plate_center_x, point[1] - plate_center_y, 0.008)
        for point in geometry["points"][1:]
    )


def generate_variants(gearing) -> dict[str, dict]:
    source_rows = json.loads(VARIANTS_SOURCE.read_text())["variants"]
    rows = []
    pair_assets: dict[tuple[int, ...], str] = {}
    layouts = {
        f"board_{teeth}": _board_geometry(gearing, (teeth,), ())
        for teeth in GEAR_COLORS
    }
    for source in source_rows:
        family = str(source["family"])
        if family not in {"pair", "train"}:
            continue
        stations = tuple(int(value) for value in source["stations"])
        bends = tuple(float(value) for value in source["board"].get("bends", ()))
        if family == "pair":
            asset_key = pair_assets.setdefault(
                stations, "board_pair_" + "_".join(map(str, stations))
            )
            geometry = layouts.setdefault(
                asset_key, _board_geometry(gearing, stations, bends)
            )
        else:
            asset_key = "board_train_" + str(source["index"]).zfill(3)
            geometry = _board_geometry(gearing, stations, bends)
            layouts[asset_key] = geometry
        target_offsets = _target_offsets(geometry)
        rows.append(
            {
                "name": source["name"],
                "family": family,
                "seed": int(source["seed"]),
                "stations": list(stations),
                "bends": list(bends),
                "board_variant": asset_key,
                "board_xy": [float(value) for value in source["board"]["xy"]],
                "board_yaw": float(source["board"]["yaw"]),
                "target_offsets_xyz": [list(offset) for offset in target_offsets],
                "gears": [
                    {
                        "name": gear["name"],
                        "teeth": int(gear["teeth"]),
                        "xy": [float(value) for value in gear["xy"]],
                        "yaw": float(gear["yaw"]),
                    }
                    for gear in source["gears"]
                ],
            }
        )
    VARIANTS_OUT.write_text(json.dumps({"variants": rows}, indent=2) + "\n")
    return layouts


def remove_obsolete_board_assets() -> None:
    for path in OUT.glob("vabar__gear_mesh_board_*"):
        if path.is_dir():
            shutil.rmtree(path)


def generate_mat() -> None:
    path = OUT / "vabar__gear_mesh_mat/vabar__gear_mesh_mat.usda"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""#usda 1.0
(
    defaultPrim = "GearMeshMat"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "GearMeshMat" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "MaterialBindingAPI"]
)
{{
    bool physics:kinematicEnabled = 1
    float physics:mass = 1
    rel material:binding:physics = </GearMeshMat/PhysicsMaterials/mat>
    def Scope "PhysicsMaterials"
    {{
{_material("mat", 0.9, "        ")}    }}
    def Cube "mat" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        double size = 1
        double3 xformOp:scale = (0.15, 0.27, 0.004)
        double3 xformOp:translate = (0, 0, -0.002)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [({_display_color((0.62, 0.60, 0.56))})]
    }}
}}
""")


if __name__ == "__main__":
    gearing = _source_module()
    for teeth in GEAR_COLORS:
        generate_gear(gearing, teeth)
    generate_mat()
    layouts = generate_variants(gearing)
    generate_board(gearing, layouts)
    remove_obsolete_board_assets()
