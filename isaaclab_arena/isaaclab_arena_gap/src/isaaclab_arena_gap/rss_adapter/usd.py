"""An Arena robot USD as one stage with a copy per arm, every label carrying the arm name.

RoboSimStudio's hand specs address their parts as ``{arm}_left_finger``,
``{arm}_link_6`` and so on, and RoboSimStudio finds joints and bodies by the
last label segment. Arena's YAM names them ``left_finger`` and ``link_6``, the
same in both copies. So each copy here is the flattened asset with every prim
renamed ``{arm}_<name>`` and its relationships retargeted, and RoboSimStudio's
registered hand applies as-is.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom


def _rename_subtree(layer: Sdf.Layer, root: Sdf.Path, prefix: str) -> dict[str, str]:
    """Rename every prim below ``root`` to ``prefix + name``; return old -> new paths."""
    renames: list[tuple[Sdf.Path, str]] = []

    def walk(spec: Sdf.PrimSpec) -> None:
        for child in spec.nameChildren:
            walk(child)
            renames.append((child.path, prefix + child.name))

    walk(layer.GetPrimAtPath(root))
    mapping: dict[str, str] = {}
    # Deepest first, so a parent's rename never invalidates a pending child path.
    for path, new_name in sorted(renames, key=lambda item: -item[0].pathElementCount):
        spec = layer.GetPrimAtPath(path)
        spec.name = new_name
        mapping[str(path)] = str(path.GetParentPath().AppendChild(new_name))
    return mapping


def _remap(path: Sdf.Path, renames: Sequence[tuple[str, str]]) -> Sdf.Path:
    """``path`` with each renamed ancestor replaced, deepest rename last applied."""
    text = str(path)
    for old, new in renames:
        if text == old or text.startswith(old + "/") or text.startswith(old + "."):
            text = new + text[len(old):]
    return Sdf.Path(text)


def _retarget(layer: Sdf.Layer, root: Sdf.Path, mapping: dict[str, str]) -> None:
    # Deepest renames first: each old path is spelt with its ancestors still unrenamed.
    renames = sorted(mapping.items(), key=lambda item: -item[0].count("/"))

    def fix(spec: Sdf.PrimSpec) -> None:
        for prop in spec.properties:
            if isinstance(prop, Sdf.RelationshipSpec):
                items = list(prop.targetPathList.GetAddedOrExplicitItems())
                if items:
                    prop.targetPathList.ClearEdits()
                    prop.targetPathList.explicitItems = [_remap(p, renames) for p in items]
            elif isinstance(prop, Sdf.AttributeSpec):
                items = list(prop.connectionPathList.GetAddedOrExplicitItems())
                if items:
                    prop.connectionPathList.ClearEdits()
                    prop.connectionPathList.explicitItems = [_remap(p, renames) for p in items]
        for child in spec.nameChildren:
            fix(child)

    fix(layer.GetPrimAtPath(root))


def build_bimanual(
    source: str | Path,
    out: str | Path,
    mounts: dict[str, Sequence[float]],
    gripper_variant: str | None = None,
) -> Path:
    """Write ``out`` with one renamed copy of ``source`` per arm, at ``mounts[arm]`` (x, y, z, qx, qy, qz, qw).

    ``gripper_variant`` selects the asset's ``gripper`` variant before copying.
    """
    stage = Usd.Stage.Open(str(source))
    if gripper_variant is not None:
        variants = stage.GetDefaultPrim().GetVariantSets().GetVariantSet("gripper")
        if gripper_variant not in variants.GetVariantNames():
            raise ValueError(f"{source} has no gripper variant {gripper_variant!r}; has {variants.GetVariantNames()}")
        variants.SetVariantSelection(gripper_variant)
    flat = stage.Flatten()
    src_root = Sdf.Path("/" + flat.defaultPrim)
    layer = Sdf.Layer.CreateNew(str(out))
    layer.pseudoRoot.SetInfo(UsdGeom.Tokens.upAxis, UsdGeom.Tokens.z)
    layer.pseudoRoot.SetInfo(UsdGeom.Tokens.metersPerUnit, 1.0)
    world = Sdf.CreatePrimInLayer(layer, "/World")
    world.specifier = Sdf.SpecifierDef
    world.typeName = "Xform"
    layer.defaultPrim = "World"
    # Flatten moves instanced geometry to root-level prototypes the copies reference.
    for spec in flat.rootPrims:
        if spec.path != src_root:
            Sdf.CopySpec(flat, spec.path, layer, spec.path)
    for arm, mount in mounts.items():
        dst = Sdf.Path(f"/World/{arm}_yam")
        if not Sdf.CopySpec(flat, src_root, layer, dst):
            raise RuntimeError(f"could not copy {src_root} for arm {arm!r}")
        _retarget(layer, dst, {str(src_root): str(dst)})
        mapping = _rename_subtree(layer, dst, f"{arm}_")
        _retarget(layer, dst, mapping)
        stage = Usd.Stage.Open(layer)
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(dst))
        xform.ClearXformOpOrder()
        x, y, z, qx, qy, qz, qw = (float(v) for v in mount)
        xform.AddTranslateOp().Set((x, y, z))
        xform.AddOrientOp().Set(Gf.Quatf(qw, qx, qy, qz))
    layer.Save()
    return Path(out)
