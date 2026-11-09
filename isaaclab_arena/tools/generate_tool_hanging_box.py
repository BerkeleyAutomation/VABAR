"""Generate the AUTOLab Level-1 screwdriver cup as an open USDA compound.

Dimensions from AUTOLab 34c6f8d087c49beae924698de79edf4becbd5f72,
tasks/tool_hanging/assets/screwdriver_holders/generate_screwdriver_box.py.
Run with: uv run python tools/generate_tool_hanging_box.py
"""

from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

# Metres, in the RSS fixture frame; each convex part remains separate.
PARTS = [
    ((0.065, 0.110, 0.004), (0.0325, 0.0732, 0.002)),
    ((0.004, 0.110, 0.071), (0.002, 0.0732, 0.0395)),
    ((0.004, 0.110, 0.048), (0.063, 0.0732, 0.028)),
    ((0.065, 0.004, 0.071), (0.0325, 0.0202, 0.0395)),
    ((0.065, 0.004, 0.071), (0.0325, 0.1262, 0.0395)),
]
for y in (0.043, 0.1034):
    PARTS.extend(
        [
            ((0.008, 0.018, 0.040), (0.000, y, 0.090)),
            ((0.018, 0.018, 0.008), (-0.005, y, 0.107)),
            ((0.006, 0.018, 0.018), (-0.011, y, 0.101)),
        ]
    )


def generate(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(destination))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    material = UsdShade.Material.Define(stage, "/Asset/Material")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(0.5)
    physics.CreateDynamicFrictionAttr(0.5)
    physics.CreateRestitutionAttr(0)
    for index, (size, center) in enumerate(PARTS):
        cube = UsdGeom.Cube.Define(stage, f"/Asset/part_{index:02d}")
        cube.CreateSizeAttr(1)
        cube.AddTranslateOp().Set(Gf.Vec3d(*center))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(228 / 255, 214 / 255, 169 / 255)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
            material, materialPurpose="physics"
        )
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    generate(
        Path(__file__).resolve().parents[1]
        / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets/vabar__screwdriver_box/box.usda"
    )
