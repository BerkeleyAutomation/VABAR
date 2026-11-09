"""Discover and register benchmark-owned Arena assets."""

from __future__ import annotations

import pkgutil
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.utils.pose import Pose

from isaaclab_arena_vabar.registry import registered_component
from isaaclab_arena_vabar.numeric import is_finite_real


@dataclass(frozen=True)
class ConvertedAssetSpec:
    """Package-local asset facts emitted by a converter registration module."""

    registry_name: str
    role: Literal["object", "background"]
    relative_usd_path: str
    object_min_z: float | None


_registered = False


def register_assets() -> None:
    """Discover owner-local registration modules in deterministic order."""

    global _registered
    if _registered:
        return

    from . import registrations

    prefix = f"{registrations.__name__}."
    module_names = sorted(
        module.name for module in pkgutil.iter_modules(registrations.__path__, prefix)
    )
    for module_name in module_names:
        module = import_module(module_name)
        module.register_assets()

    _registered = True


def register_converted_assets(
    specs: Iterable[ConvertedAssetSpec], *, module_file: str
) -> None:
    """Register package-local converted assets with Arena."""

    asset_root = Path(module_file).resolve().parents[1]
    _register_converted_asset_specs(specs, asset_root=asset_root)


def _register_converted_asset_specs(
    specs: Iterable[ConvertedAssetSpec], *, asset_root: Path
) -> None:
    registry = AssetRegistry()
    root = asset_root.resolve()
    for spec in specs:
        path = (root / spec.relative_usd_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(
                "converted asset USD is missing or escapes assets package: "
                f"{spec.relative_usd_path}"
            )
        expected = _asset_class(
            spec.registry_name,
            spec.role,
            str(path),
            spec.object_min_z,
        )
        existing = registered_component(registry, spec.registry_name)
        if existing is not None:
            if _registered_asset_contract(existing) != _registered_asset_contract(
                expected
            ):
                raise RuntimeError(
                    f"conflicting converted asset registration for {spec.registry_name!r}"
                )
            continue
        registry.register(expected, spec.registry_name)


def _registered_asset_contract(asset: type) -> tuple[Any, ...]:
    return (
        getattr(asset, "asset_role", None),
        getattr(asset, "usd_path", None),
        getattr(asset, "object_min_z", None),
    )


def _validated_object_min_z(role: str, value: Any) -> float | None:
    if role == "background" and is_finite_real(value):
        return float(value)
    if role == "object" and value is None:
        return None
    raise ValueError(f"invalid object_min_z for converted {role} asset")


def _asset_class(name: str, role: str, usd_path: str, object_min_z: Any) -> type:
    if role not in {"background", "object"}:
        raise ValueError(f"invalid converted asset role: {role!r}")
    minimum = _validated_object_min_z(role, object_min_z)
    if role == "background":

        class ConvertedBackground(Background):
            tags = ["background"]

            def __init__(self, **kwargs: Any):
                initial_pose = kwargs.get("initial_pose")
                if isinstance(initial_pose, dict):
                    kwargs["initial_pose"] = _pose_from_mapping(initial_pose)
                super().__init__(
                    name=name, usd_path=usd_path, object_min_z=minimum, **kwargs
                )

        ConvertedBackground.name = name
        ConvertedBackground.usd_path = usd_path
        ConvertedBackground.asset_role = role
        ConvertedBackground.object_min_z = minimum
        return ConvertedBackground

    class ConvertedObject(Object):
        tags = ["object"]

        def __init__(self, instance_name: str | None = None, **kwargs: Any):
            initial_pose = kwargs.get("initial_pose")
            if isinstance(initial_pose, dict):
                kwargs["initial_pose"] = _pose_from_mapping(initial_pose)
            kwargs.setdefault("scale", root_scale(usd_path))
            super().__init__(
                name=instance_name or name,
                usd_path=usd_path,
                object_type=ObjectType.RIGID,
                **kwargs,
            )

        def get_world_bounding_box(self):
            """Return the world AABB enclosing an arbitrarily rotated fixed anchor."""
            initial_pose = self.get_initial_pose()
            if self.is_anchor and isinstance(initial_pose, Pose):
                return (
                    self.get_bounding_box()
                    .enclosing_after_rotation(initial_pose.rotation_xyzw)
                    .translated(initial_pose.position_xyz)
                )
            return super().get_world_bounding_box()

    ConvertedObject.name = name
    ConvertedObject.usd_path = usd_path
    ConvertedObject.asset_role = role
    ConvertedObject.object_min_z = minimum
    return ConvertedObject


@lru_cache(maxsize=None)
def root_scale(usd_path: str) -> tuple[float, float, float]:
    """The ``xformOp:scale`` authored on the USD's default prim, else unit scale.

    Isaac Lab's ``UsdFileCfg`` spawner writes ``Object.scale`` over the referenced
    default prim's own scale, so an asset sized on its root prim (the multimeter at
    0.83) would spawn at full size unless that scale is passed as ``Object.scale``.
    Arena's bounding-box helper unbakes the root scale and applies ``Object.scale``
    once, so the relation solver sees the same size.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim() if stage else None
    if not root:
        raise ValueError(f"converted asset USD has no default prim: {usd_path}")
    scales = [
        op
        for op in UsdGeom.Xformable(root).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeScale
    ]
    if not scales:
        return (1.0, 1.0, 1.0)
    if len(scales) > 1 or scales[0].GetOpName() != "xformOp:scale" or scales[0].IsInverseOp():
        raise ValueError(
            f"converted asset root carries a scale stack Object.scale cannot express: {usd_path}"
        )
    return tuple(float(v) for v in scales[0].Get())


def _pose_from_mapping(initial_pose: dict[str, Any]) -> Pose:
    try:
        position = tuple(initial_pose["position_xyz"])
        rotation = tuple(initial_pose["rotation_xyzw"])
    except (KeyError, TypeError) as error:
        raise ValueError("converted asset initial_pose is invalid") from error
    return Pose(position_xyz=position, rotation_xyzw=rotation)
