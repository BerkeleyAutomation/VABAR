"""Opt-in mesh instancing for benchmark embodiments and scene USD spawners."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("module", "embodiment", "usd_path", "robot_fields", "kwargs"),
    [
        (
            "fr3_robotiq_2f85",
            "IndustrialFr3Robotiq2f85Embodiment",
            "industrial_fr3_robotiq_2f85:ROBOT_ON_CART_INSTANCEABLE_USD_PATH",
            ("robot",),
            {"camera_rig": "gear_mesh"},
        ),
        (
            "bimanual_yam",
            "IndustrialBimanualYamEmbodiment",
            "industrial_i2rt_yam:ROBOT_INSTANCEABLE_USD_PATH",
            ("left_robot", "right_robot"),
            {},
        ),
    ],
    ids=("fr3", "yam"),
)
def test_embodiment_can_select_tiled_cameras_and_instanceable_meshes(
    module, embodiment, usd_path, robot_fields, kwargs
) -> None:
    import importlib

    embodiment_type = getattr(
        importlib.import_module(f"isaaclab_arena_vabar.embodiments.{module}"),
        embodiment,
    )
    registration, constant = usd_path.split(":")
    expected_usd = getattr(
        importlib.import_module(
            f"isaaclab_arena_vabar.assets.registrations.{registration}"
        ),
        constant,
    )

    instance = embodiment_type(
        enable_cameras=True,
        use_tiled_cameras=True,
        use_instanceable_meshes=True,
        **kwargs,
    )

    assert instance.camera_config.use_tiled_camera is True
    for field in robot_fields:
        assert getattr(instance.scene_config, field).spawn.usd_path == str(
            expected_usd
        )


def test_fr3_mesh_instancing_keeps_non_mesh_ancestors_writable() -> None:
    from isaaclab_arena_vabar.mesh_instancing import (
        make_referenced_mesh_leaves_instanceable,
    )
    from pxr import Gf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(stage, "/GeometryLibrary/mesh")
    robot = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    link = UsdGeom.Xform.Define(stage, "/Robot/link").GetPrim()
    referenced_mesh = UsdGeom.Mesh.Define(stage, "/Robot/link/visual").GetPrim()
    referenced_mesh.GetReferences().AddInternalReference("/GeometryLibrary/mesh")
    UsdGeom.Xformable(referenced_mesh).AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    material = UsdShade.Material.Define(stage, "/Robot/Materials/paint")
    UsdShade.MaterialBindingAPI.Apply(referenced_mesh).Bind(material)
    collision_mesh = UsdGeom.Mesh.Define(stage, "/Robot/link/collision").GetPrim()
    collision_mesh.GetReferences().AddInternalReference("/GeometryLibrary/mesh")
    UsdGeom.Imageable(collision_mesh).CreatePurposeAttr("guide")

    make_referenced_mesh_leaves_instanceable(robot)

    render_instance = stage.GetPrimAtPath("/Robot/link/visual__render_instance")
    assert not referenced_mesh.IsInstanceable()
    assert UsdGeom.Imageable(referenced_mesh).ComputeVisibility() == "invisible"
    assert render_instance.IsA(UsdGeom.Xform)
    assert render_instance.IsInstanceable()
    assert render_instance.IsInstance()
    assert (
        UsdGeom.Xformable(render_instance).GetLocalTransformation()
        == UsdGeom.Xformable(referenced_mesh).GetLocalTransformation()
    )
    bound_material, _ = UsdShade.MaterialBindingAPI(
        render_instance
    ).ComputeBoundMaterial()
    assert bound_material.GetPath() == material.GetPath()
    assert not collision_mesh.IsInstanceable()
    assert not stage.GetPrimAtPath("/Robot/link/collision__render_instance").IsValid()
    assert UsdGeom.Imageable(collision_mesh).ComputeVisibility() == "inherited"
    assert not link.IsInstanceable()
    assert not robot.IsInstanceable()


def test_scene_mesh_instancing_selects_only_compatible_usd_spawners() -> None:
    from isaaclab_arena_vabar.mesh_instancing import (
        configure_scene_mesh_instancing,
        spawn_multi_usd_as_instances,
        spawn_usd_as_instance,
    )
    from isaaclab.sim import MultiUsdFileCfg, UsdFileCfg
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.spawners.wrappers.wrappers import spawn_multi_usd_file

    usd = UsdFileCfg(usd_path="asset.usd")
    multi_usd = MultiUsdFileCfg(usd_path=["one.usd", "two.usd"])
    editable_usd = UsdFileCfg(usd_path="editable.usd", make_uninstanceable=True)

    def custom_spawner(*_args, **_kwargs):
        return None

    custom_usd = UsdFileCfg(usd_path="custom.usd", func=custom_spawner)
    assets = {
        name: SimpleNamespace(object_cfg=SimpleNamespace(spawn=spawn))
        for name, spawn in {
            "usd": usd,
            "multi_usd": multi_usd,
            "editable_usd": editable_usd,
            "custom_usd": custom_usd,
        }.items()
    }
    referenced_parent = SimpleNamespace(
        object_cfg=SimpleNamespace(spawn=UsdFileCfg(usd_path="parent.usd"))
    )
    assets["referenced_parent"] = referenced_parent
    assets["reference"] = SimpleNamespace(parent_asset=referenced_parent)
    scene = SimpleNamespace(assets=assets)

    configure_scene_mesh_instancing(scene, True)

    assert usd.func is spawn_usd_as_instance
    assert multi_usd.func is spawn_multi_usd_as_instances
    assert editable_usd.func != spawn_usd_as_instance
    assert custom_usd.func is custom_spawner
    assert referenced_parent.object_cfg.spawn.func != spawn_usd_as_instance

    configure_scene_mesh_instancing(scene, False)

    assert usd.func is spawn_from_usd
    assert multi_usd.func is spawn_multi_usd_file


def test_multi_usd_instancing_resolves_cloned_environment_paths(monkeypatch) -> None:
    from isaaclab.sim import MultiUsdFileCfg

    from isaaclab_arena_vabar import mesh_instancing

    prims = {
        f"/World/envs/env_{env_id}/Object_{object_id}": object()
        for env_id in range(2)
        for object_id in range(2)
    }

    class Stage:
        @staticmethod
        def GetPrimAtPath(path):
            return prims[str(path)]

    root_prim = SimpleNamespace(GetStage=lambda: Stage())
    matched_expressions = []
    instanced = []

    def find_matching(expression, *, stage):
        assert isinstance(stage, Stage)
        matched_expressions.append(expression)
        object_id = int(expression.rsplit("_", 1)[1])
        return [f"/World/envs/env_{env_id}/Object_{object_id}" for env_id in range(2)]

    monkeypatch.setattr(
        mesh_instancing, "spawn_multi_usd_file", lambda *_args, **_kwargs: root_prim
    )
    monkeypatch.setattr(
        mesh_instancing.sim_utils, "find_matching_prim_paths", find_matching
    )
    monkeypatch.setattr(
        mesh_instancing, "make_asset_root_instanceable", instanced.append
    )

    mesh_instancing.spawn_multi_usd_as_instances(
        "/World/envs/env_.*/Object_.*",
        MultiUsdFileCfg(usd_path=["one.usd", "two.usd"]),
    )

    assert matched_expressions == [
        "/World/envs/env_.*/Object_0",
        "/World/envs/env_.*/Object_1",
    ]
    assert instanced == [
        prims[f"/World/envs/env_{env_id}/Object_{object_id}"]
        for object_id in range(2)
        for env_id in range(2)
    ]


