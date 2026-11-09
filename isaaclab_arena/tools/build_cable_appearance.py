"""Author Cable Easy's RSS-matching, visual-only room.

Run in AUTOLab's cable environment, with the pinned AUTOLab checkout as the
first argument and isaaclab_arena_vabar's cable_routing_assets as the second.
The source renderer authors the room geometry, procedural textures and lights.
No simulator or graph is changed and the resulting USD has no runtime source
dependency.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def build(source: Path, assets: Path) -> None:
    sys.path.insert(0, str(source))
    from run_cable import configure

    configure()
    import numpy as np
    from pxr import Sdf, Usd, UsdGeom
    from robosimstudio.core.render.finish import MATTE
    from robosimstudio.core.render.look import look_for
    from robosimstudio.core.render.palette import GROUND
    from robosimstudio.core.render.room import house_room_closed
    from robosimstudio.scene.builtin.benches import BENCH_075
    from robosimstudio.solvers.render.rtx import (
        _MaterialLibrary,
        _StageKit,
        _add_lab_lights,
    )

    appearance = assets / "native_appearance"
    appearance.mkdir(exist_ok=True)
    room_path = appearance / "room.usda"
    room = Usd.Stage.CreateNew(str(room_path))
    root = UsdGeom.Xform.Define(room, "/root")
    room.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(room, 1.0)
    UsdGeom.SetStageUpAxis(room, UsdGeom.Tokens.z)
    root.AddTranslateOp().Set((-0.5875, 0.0, 0.017))
    library = _MaterialLibrary(room)
    kit = _StageKit(room, library, cell=BENCH_075, surface_z=BENCH_075.surface_z)
    house_room_closed(kit)
    table = BENCH_075.statics[0]
    kit.box("table", table.xform[:3], table.half_extents, color=GROUND, finish=MATTE)
    look = look_for(None)
    _add_lab_lights(
        room,
        center=BENCH_075.center,
        surface_z=BENCH_075.surface_z,
        environment_map="",
        key=look.key_intensity,
        ambient=float(np.mean(look.ambient)),
    )
    # Localize RSS procedural texture files so the asset remains relocatable.
    for prim in room.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
                value = attr.Get()
                if value and value.path:
                    original = Path(value.path)
                    target = appearance / original.name
                    shutil.copyfile(original, target)
                    attr.Set(Sdf.AssetPath("./" + target.name))
    room.GetRootLayer().Save()

    print(f"Authored {room_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("assets", type=Path)
    args = parser.parse_args()
    build(args.source.resolve(), args.assets.resolve())
