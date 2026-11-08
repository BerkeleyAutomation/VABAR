"""Rendering-only material contract from AUTOLab's cable task."""


def bind_cable_material(stage, prim, color) -> None:
    """Match EntityRod.apply_display / finish_for(..., 'deformables')."""
    from pxr import Gf, Sdf, UsdShade

    path = prim.GetPath().AppendChild("JacketMaterial")
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
