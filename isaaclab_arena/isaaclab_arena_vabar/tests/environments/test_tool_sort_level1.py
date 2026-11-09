"""Tool-sorting scene and asset contracts."""

import numpy as np
import pytest

from ib_support import ASSETS_ROOT, AUTOLAB_ROOT, load_scene, scene_params


@pytest.mark.parametrize(
    "index,appearance,names",
    [
        # AUTOLab v1.0's level-1 suite (2F-85, this graph) never stages sort_electrical's
        # safety glasses or sort_bench's adjustable wrench; pliers it does stage stand in.
        (1, "electrical", ("battery", "multimeter", "flashlight", "cutting_pliers")),
        (
            2,
            "bench",
            (
                "combination_pliers",
                "slotted_screwdriver",
                "tape_measure",
                "insulating_tape",
            ),
        ),
        (
            3,
            "wiring",
            ("cutting_pliers", "breadboard", "wire_spool", "combination_pliers"),
        ),
    ],
)
def test_existing_assets_match_fixed_labels_and_randomized_placement(
    index, appearance, names
):
    from pxr import Usd

    scene = load_scene(f"tool_sorting/easy_{index}.yaml")
    objects = {o["id"]: o for o in scene["objects"]}
    assert scene["background"]["registry_name"] == "vabar__fr3_workcell_table"
    assert objects["bin2"]["params"]["appearance"] == appearance
    stage = Usd.Stage.Open(
        str(ASSETS_ROOT / f"vabar__tool_sort_bin/bin2_{appearance}.usda")
    )
    textures = [
        str(a.Get())
        for p in stage.Traverse()
        for a in p.GetAttributes()
        if a.GetName() in ("inputs:file", "inputs:diffuse_texture")
    ]
    assert scene_params(scene)["object_list"] == [
        n + "_0" for n in names
    ]
    for compartment, name in enumerate(names):
        obj = objects[name + "_0"]
        assert obj["registry_name"] == "vabar__" + name
        assert "initial_pose" not in obj["params"]
        assert any(f"bin2_{compartment}_{name}.png" in t for t in textures)
        # A 30 mm footprint margin keeps object centres clear of bin1's walls
        # so the gripper coupler fits; 10 mm let pliers settle against a wall.
        assert [r for r in scene["relations"] if r["subject"] == obj["id"]] == [
            {
                "kind": "on",
                "subject": obj["id"],
                "reference": "bin1",
                "params": {"clearance_m": 0.0, "edge_margin_m": 0.03},
            }
        ]
    # Every destination compartment within FR3 reach (AUTOLab's export puts
    # bin2 at x=0.1, where its far compartment centre is 0.86 m from the base).
    bin2 = next(r for r in scene["relations"] if r["subject"] == "bin2" and r["kind"] == "at_position")
    assert bin2["params"] == {"x": 0.0, "y": 0.28}
    assert objects["bin2"]["params"]["initial_pose"]["position_xyz"][:2] == [0.0, 0.28]


@pytest.mark.parametrize(
    "name,scale",
    [
        ("multimeter", (0.83,) * 3),
        ("measuring_tape", (0.84,) * 3),
        ("tape_measure", (0.81,) * 3),
        ("breadboard", (1.0, 1.0, 2.5)),
    ],
)
def test_existing_tool_scale(name, scale):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(
        str(ASSETS_ROOT / f"vabar__{name}/vabar__{name}.usda")
    )
    ops = UsdGeom.Xformable(stage.GetDefaultPrim()).GetOrderedXformOps()
    assert len(ops) == 1
    assert np.allclose(ops[0].Get(), scale)


@pytest.mark.parametrize(
    "name,scale,size_mm",
    [
        # AUTOLab v1.0 level-1 sizes, longest x middle x shortest.
        ("multimeter", (0.83,) * 3, (152, 73, 44)),
        ("tape_measure", (0.81,) * 3, None),
        ("measuring_tape", (0.84,) * 3, None),
        ("battery", (1.0,) * 3, None),
        ("breadboard", (1.0, 1.0, 2.5), (84, 55, 20)),
    ],
)
def test_registered_tool_spawns_at_its_root_scale(name, scale, size_mm):
    """Isaac Lab's spawner writes Object.scale over the USD root's own scale, so
    the registered object must carry it; otherwise the multimeter spawns 87 mm
    wide, past the 2F-85's 85 mm stroke."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena_vabar.registration import register_components

    register_components()
    obj = AssetRegistry().get_asset_by_name(f"vabar__{name}")(instance_name=f"{name}_0")
    assert np.allclose(obj.scale, scale)
    if size_mm is not None:
        box = obj.get_bounding_box()
        extent = sorted(
            (np.asarray(box.max_point) - np.asarray(box.min_point)).ravel() * 1000.0,
            reverse=True,
        )
        assert np.allclose(extent, size_mm, atol=1.5)


def test_explicit_object_scale_overrides_root_scale():
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena_vabar.registration import register_components

    register_components()
    obj = AssetRegistry().get_asset_by_name("vabar__multimeter")(scale=(1.0, 1.0, 1.0))
    assert tuple(obj.scale) == (1.0, 1.0, 1.0)


def test_root_scale_rejects_a_stack_object_scale_cannot_express(tmp_path):
    from isaaclab_arena_vabar.assets.registry import root_scale

    usd = tmp_path / "stacked.usda"
    usd.write_text(
        '#usda 1.0\n(\n    defaultPrim = "base"\n)\n\ndef Xform "base"\n{\n'
        "    double3 xformOp:scale = (0.5, 0.5, 0.5)\n"
        "    double3 xformOp:scale:unitsResolve = (100, 100, 100)\n"
        '    uniform token[] xformOpOrder = ["xformOp:scale", "xformOp:scale:unitsResolve"]\n}\n'
    )
    with pytest.raises(ValueError, match="scale stack"):
        root_scale(str(usd))


def test_every_registered_object_reads_its_root_scale():
    """Fails closed on any shipped asset whose root carries an unsupported scale stack."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena_vabar.assets.registry import root_scale
    from isaaclab_arena_vabar.registration import register_components

    register_components()
    registry = AssetRegistry()
    scaled = {}
    for key in registry.get_all_keys():
        asset = registry.get_asset_by_name(key)
        if getattr(asset, "asset_role", None) == "object" and key.startswith("vabar__"):
            scale = root_scale(asset.usd_path)
            if scale != (1.0, 1.0, 1.0):
                scaled[key] = scale
    assert set(scaled) == {
        "vabar__multimeter",
        "vabar__tape_measure",
        "vabar__measuring_tape",
        "vabar__breadboard",
    }


def _obj_vertices(path):
    return np.array(
        [[float(v) for v in line.split()[1:4]] for line in path.open() if line.startswith("v ")]
    )


@pytest.mark.skipif(not (AUTOLAB_ROOT / "tasks").is_dir(), reason="AUTOLab v1.0 sources are not materialized; run ./setup.sh")
def test_rss_level1_breadboard_is_the_sort_wiring_board_at_20_mm():
    """AUTOLab's 2F-85 level-1 suite stages breadboard_1.obj: sort_wiring's 8 mm board with its
    thickness stretched 2.5x, so the 2F-85's closing fingertips reach its sides at the graph's
    fixed TCP height (bin floor + 20 mm). vabar__breadboard's root scale reproduces it."""
    import xml.etree.ElementTree as ET

    assets = AUTOLAB_ROOT / "assets/vab_tool_sort/assets"
    board, level1 = _obj_vertices(assets / "breadboard.obj"), _obj_vertices(assets / "breadboard_1.obj")
    assert board.shape == level1.shape
    assert np.allclose(level1, board * [1.0, 1.0, 2.5], atol=1e-6)
    staged = {
        mesh.get("file").rsplit("/", 1)[-1]
        for scene in (AUTOLAB_ROOT / "assets/vab_tool_sort/level1").glob("level1_*/scene.xml")
        for mesh in ET.parse(scene).iter("mesh")
        if "breadboard" in mesh.get("name", "") and "esp32" not in mesh.get("name", "")
    }
    assert staged == {"breadboard_1.obj"}
    wiring = ET.parse(AUTOLAB_ROOT / "assets/vab_tool_sort/sort_wiring/scene.xml")
    assert {
        m.get("file").rsplit("/", 1)[-1]
        for m in wiring.iter("mesh")
        if m.get("name", "").startswith("industrial_breadboard")
    } == {"breadboard.obj"}


def test_breadboard_collides_as_the_level1_box():
    """One guide-purpose Cube at the visual mesh's bounds is the only enabled collider; with the
    root scale it is AUTOLab's level-1 box geom (20 x 55 x 84 mm)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(ASSETS_ROOT / "vabar__breadboard/vabar__breadboard.usda"))
    (collider,) = [
        p
        for p in stage.Traverse()
        if p.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False
    ]
    assert collider.IsA(UsdGeom.Cube)
    assert UsdGeom.Imageable(collider).GetPurposeAttr().Get() == UsdGeom.Tokens.guide
    mesh = stage.GetPrimAtPath(
        "/Asset/Geometry/industrial_breadboard_0_g0_visual_obj_00/industrial_breadboard_0_g0_visual_mesh_00"
    )
    points = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get())
    box = UsdGeom.Cube(collider).ComputeLocalBound(
        Usd.TimeCode.Default(), UsdGeom.Tokens.default_, UsdGeom.Tokens.guide
    ).ComputeAlignedRange()
    assert tuple(box.GetMin()) == pytest.approx(tuple(points.min(axis=0)), abs=2e-6)
    assert tuple(box.GetMax()) == pytest.approx(tuple(points.max(axis=0)), abs=2e-6)
    world = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.guide]).ComputeWorldBound(collider)
    size_mm = np.asarray(world.ComputeAlignedRange().GetSize()) * 1000.0
    assert np.allclose(sorted(size_mm), (20.0, 54.8, 84.1), atol=0.1)


@pytest.mark.skipif(not (AUTOLAB_ROOT / "tasks").is_dir(), reason="AUTOLab v1.0 sources are not materialized; run ./setup.sh")
def test_rss_level1_breadboard_box_matches_ours():
    import xml.etree.ElementTree as ET

    scene = AUTOLAB_ROOT / "assets/vab_tool_sort/level1/level1_11/scene.xml"
    body = ET.parse(scene).find('.//body[@name="industrial_breadboard_0_main"]')
    colliding = [
        g for g in body.findall("geom") if g.get("contype", "1") != "0" or g.get("conaffinity", "1") != "0"
    ]
    assert [g.get("type") for g in colliding] == ["box"]
    half_mm = sorted(float(v) * 1000.0 for v in colliding[0].get("size").split())
    assert np.allclose(half_mm, (10.0, 27.4, 42.05), atol=0.1)
