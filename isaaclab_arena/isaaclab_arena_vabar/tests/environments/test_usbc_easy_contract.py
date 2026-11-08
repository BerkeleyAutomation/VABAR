"""Source contracts for AUTOLab ``usbc_insert/insert_easy``."""

import pytest

from ib_support import ASSETS_ROOT, load_scene, relations_by, scene_params

ENVIRONMENT = "usbc_insertion/easy.yaml"
PORT = ASSETS_ROOT / "vabar__usbc_easy_port/vabar__usbc_easy_port.usda"
PLUG = ASSETS_ROOT / "vabar__usbc_easy_plug/vabar__usbc_easy_plug.usda"


def test_easy_scene_contains_only_the_fixed_port_task_closure() -> None:
    document = load_scene(ENVIRONMENT)
    objects = {entry["id"]: entry for entry in document["objects"]}

    assert document["embodiment"]["registry_name"] == "vabar__bimanual_yam"
    assert {"bench", "port", "plug"} <= objects.keys()
    assert not ({"bulkhead", "cradle_front", "cradle_rear", "detent"} & objects.keys())
    assert objects["port"]["registry_name"] == "vabar__usbc_easy_port"
    assert objects["plug"]["registry_name"] == "vabar__usbc_easy_plug"


def test_easy_randomizes_the_plug_and_the_bolted_port_through_relations() -> None:
    relations = relations_by(load_scene(ENVIRONMENT))
    plug_kinds = {kind for subject, kind in relations if subject == "plug"}

    assert plug_kinds == {
        "on",
        "at_position",
        "rotate_around_solution",
        "random_around_solution",
    }
    assert relations["plug", "on"]["reference"] == "bench"
    # Supports sit flush on their references, as in the source scene.
    assert relations["bench", "on"]["params"]["clearance_m"] == 0.0
    assert relations["plug", "on"]["params"]["clearance_m"] == 0.0
    # AUTOLab's bolted-port slop: PORT_JITTER_ACROSS, PORT_JITTER_ALONG, PORT_JITTER_YAW.
    assert relations["port", "random_around_solution"]["params"] == pytest.approx(
        dict(x_half_m=0.005, y_half_m=0.0015, yaw_half_rad=0.08726646259971647)
    )


def test_easy_success_requires_a_released_plug_without_a_tilt_gate() -> None:
    goal = scene_params(load_scene(ENVIRONMENT))["goal"]

    assert goal["require_released"] is True
    assert "tilt_max" not in goal
    assert "acknowledged_unscored" not in goal


def test_easy_port_bakes_the_source_rotation_and_bore_frame() -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(PORT))
    root = stage.GetDefaultPrim()
    mesh = UsdGeom.Mesh.Get(stage, "/Asset/Geometry")
    bounds = mesh.GetExtentAttr().Get()

    assert root.GetCustomDataByKey("autolab:bakedPose") == "PORT_QUAT"
    assert tuple(bounds[0]) == pytest.approx((-0.02, -0.013, -0.075))
    assert tuple(bounds[1]) == pytest.approx((0.02, 0.05, 0.075))
    assert UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get()
    material = UsdPhysics.MaterialAPI.Get(stage, "/Asset/PhysicsMaterial")
    assert material.GetStaticFrictionAttr().Get() == pytest.approx(0.35)
    assert material.GetDynamicFrictionAttr().Get() == pytest.approx(0.35)
    assert material.GetRestitutionAttr().Get() == pytest.approx(0.0)

    # The scored bore opens on the port mesh's -Y face and runs into the mesh.
    bore = scene_params(load_scene(ENVIRONMENT))["bore"]
    assert bore["axis"] == pytest.approx([0.0, 1.0, 0.0])
    assert bore["mouth"][1] == pytest.approx(bounds[0][1])
    assert bounds[0][0] < bore["mouth"][0] < bounds[1][0]
    assert bounds[0][2] < bore["mouth"][2] < bounds[1][2]
    assert bore["depth"] < bounds[1][1] - bounds[0][1]


def test_easy_plug_uses_the_source_mesh_appearance_and_physics() -> None:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(PLUG))
    root = stage.GetDefaultPrim()
    collision = UsdGeom.Mesh.Get(stage, "/Asset/ArtistFrame/SourceCollisionMesh")
    visual = UsdGeom.Mesh.Get(
        stage, "/Asset/ArtistFrame/Geometry_obj_00/Geometry_mesh_00"
    )

    points = collision.GetPointsAttr().Get()
    assert min(point[0] for point in points) == pytest.approx(-0.011)
    assert max(point[0] for point in points) == pytest.approx(0.011)
    assert UsdPhysics.CollisionAPI(collision.GetPrim()).GetCollisionEnabledAttr().Get()
    assert collision.ComputeVisibility() == UsdGeom.Tokens.invisible
    assert not UsdPhysics.CollisionAPI(visual.GetPrim()).GetCollisionEnabledAttr().Get()

    subsets = {
        subset.GetPrim().GetName(): UsdShade.MaterialBindingAPI(subset.GetPrim())
        .ComputeBoundMaterial()[0]
        .GetPath()
        for subset in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(visual))
    }
    assert set(subsets) == {
        "m_opaque__plastic__glossy_white_a01",
        "m_opaque__plastic__blue_a01",
        "m_opaque__metal__aluminiumBrushed_e01",
        "m_opaque__plastic__black_a01",
        "m_opaque__metal__bronzePolished_a01",
    }
    orange_materials = {
        "m_opaque__plastic__blue_a01": "t_plastic_orange_a01_tile_alb.png",
        "m_opaque__plastic__glossy_white_a01": (
            "t_plastic_glossy_orange_a01_tile_alb.png"
        ),
    }
    for material_name, texture_name in orange_materials.items():
        material_path = f"/Asset/ArtistFrame/Materials/{material_name}"
        albedo = UsdShade.Shader.Get(stage, f"{material_path}/Albedo")
        assert (
            albedo.GetInput("file")
            .Get()
            .resolvedPath.endswith(
                f"vabar__usbc_easy_plug/textures/{texture_name}"
            )
        )
        omni_pbr = UsdShade.Shader.Get(stage, f"{material_path}/OmniPBR_Shader")
        assert (
            omni_pbr.GetInput("diffuse_texture")
            .Get()
            .resolvedPath.endswith(
                f"vabar__usbc_easy_plug/textures/{texture_name}"
            )
        )
    nose_material = "/Asset/ArtistFrame/Materials/m_opaque__metal__aluminiumBrushed_e01"
    nose_albedo = UsdShade.Shader.Get(stage, f"{nose_material}/Albedo")
    assert (
        nose_albedo.GetInput("file")
        .Get()
        .resolvedPath.endswith(
            "vabar__plug/simready_usd/textures/"
            "t_metal_aluminiumBrushed_e01_tile_alb.png"
        )
    )
    nose_pbr = UsdShade.Shader.Get(stage, f"{nose_material}/OmniPBR_Shader")
    assert (
        nose_pbr.GetInput("diffuse_texture")
        .Get()
        .resolvedPath.endswith(
            "vabar__plug/simready_usd/textures/"
            "t_metal_aluminiumBrushed_e01_tile_alb.png"
        )
    )
    assert UsdPhysics.MassAPI(root).GetDensityAttr().Get() == pytest.approx(2000.0)
    material = UsdPhysics.MaterialAPI.Get(
        stage,
        "/Asset/ArtistFrame/PhysicsMaterials/"
        "physics_mat_m_opaque__plastic__glossy_white_a01",
    )
    assert material.GetStaticFrictionAttr().Get() == pytest.approx(0.35)
    assert material.GetDynamicFrictionAttr().Get() == pytest.approx(0.35)
    assert material.GetRestitutionAttr().Get() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "texture_name",
    [
        "t_plastic_orange_a01_tile_alb.png",
        "t_plastic_glossy_orange_a01_tile_alb.png",
    ],
)
def test_easy_plug_orange_is_saturated(texture_name: str) -> None:
    import numpy as np
    from PIL import Image

    texture = PLUG.parent / "textures" / texture_name
    rgb = np.asarray(Image.open(texture).convert("RGB"), dtype=np.float32)
    mean = rgb.mean(axis=(0, 1)) / 255.0

    assert mean[0] > 0.9
    assert mean[1] < 0.45
    assert mean[2] < 0.08
    assert rgb.std(axis=(0, 1)).max() == pytest.approx(0.0)


def test_easy_descriptor_has_one_source_cable() -> None:
    from isaaclab_arena_vabar.environments.usbc_insertion.registration import (
        ENVIRONMENTS,
    )

    descriptor = ENVIRONMENTS[0]
    cables = descriptor.cable_spec
    assert set(cables) == {"scale", "plug"}
    assert cables["plug"]["object"] == "plug"


def test_easy_cable_visual_uses_newton_render_sync(monkeypatch) -> None:
    import isaaclab.sim
    from isaaclab_arena_vabar.environments.usbc_insertion import (
        environment as module,
    )
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(isaaclab.sim, "get_current_stage", lambda: stage)

    body_paths, curve_path = module._author_cable_visual(
        "/World/envs/env_0/cable", links=16, radius=0.0032
    )

    curve = UsdGeom.BasisCurves.Get(stage, curve_path)
    assert curve
    assert body_paths == [
        f"/World/envs/env_0/cable/physics/link{index}" for index in range(16)
    ]
    assert curve.GetCurveVertexCountsAttr().Get() == [17]
    assert curve.GetWidthsAttr().Get() == pytest.approx([0.0064])
    assert curve.GetWidthsInterpolation() == UsdGeom.Tokens.constant
    assert curve.GetTypeAttr().Get() == UsdGeom.Tokens.cubic
    assert curve.GetBasisAttr().Get() == UsdGeom.Tokens.bspline
    assert curve.GetWrapAttr().Get() == UsdGeom.Tokens.pinned
    assert "PhysicsCurvesDeformableSimAPI" in (
        curve.GetPrim().GetPrimTypeInfo().GetAppliedAPISchemas()
    )


def test_easy_full_contact_rig_preserves_fingers_and_enables_housings(
    monkeypatch,
) -> None:
    import newton
    from isaaclab_arena_vabar.assets.registrations.industrial_i2rt_yam import (
        ROBOT_USD_PATH,
    )
    from isaaclab_arena_vabar.environments.usbc_insertion import (
        environment as module,
    )
    from isaaclab_newton.physics import NewtonManager
    from newton.solvers import SolverMuJoCo

    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    for side in ("LeftRobot", "RightRobot"):
        start = len(builder.body_label)
        builder.add_usd(str(ROBOT_USD_PATH), floating=False, verbose=False)
        builder.body_label[start:] = [
            f"/World/envs/env_0/{side}/{label.lstrip('/')}"
            for label in builder.body_label[start:]
        ]
    fingers = [
        i
        for i, b in enumerate(builder.shape_body)
        if int(b) >= 0 and "finger" in builder.body_label[int(b)]
    ]
    flags = [builder.shape_flags[i] for i in fingers]
    count = builder.shape_count
    monkeypatch.setattr(NewtonManager, "_builder", builder)
    from isaaclab_arena_vabar.embodiments.bimanual_yam.config import CRANK_4310

    module._apply_full_hand_contacts(hand=CRANK_4310)
    # Both imported fingers retain a passive equality, now stiff enough to hold.
    eq_solref = builder.custom_attributes["mujoco:eq_solref"].values
    assert len(eq_solref) == 2
    assert all(tuple(value) == pytest.approx((0.004, 1.0)) for value in eq_solref)
    assert builder.shape_count == count
    assert [builder.shape_flags[i] for i in fingers] == flags
    for i in fingers:
        assert builder.custom_attributes["mujoco:condim"].values[i] == (
            module.CONTACT_CONDIM
        )
        assert builder.shape_material_mu[i] == pytest.approx(module.FINGER_MU)
        assert builder.shape_material_mu_torsional[i] == pytest.approx(
            module.FINGER_MU_TORSIONAL
        )
        assert builder.shape_material_mu_rolling[i] == pytest.approx(
            module.FINGER_MU_ROLLING
        )
    housing = [
        i
        for i, b in enumerate(builder.shape_body)
        if int(b) >= 0
        and builder.body_label[int(b)].endswith("/link_6")
        and builder.shape_source[i] is not None
    ]
    assert housing
    assert all(builder.custom_attributes["mujoco:condim"].values[i] == 3 for i in housing)
    assert all(
        builder.shape_flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES) for i in housing
    )


def test_linear_hand_housing_stays_visual_only(monkeypatch) -> None:
    """The linear hand's housing collides spuriously in Newton 1.5; the rig leaves it off."""
    from dataclasses import replace

    import newton
    from isaaclab_arena_vabar.assets.registrations.industrial_i2rt_yam import (
        ROBOT_USD_PATH,
    )
    from isaaclab_arena_vabar.embodiments.bimanual_yam.config import (
        CRANK_4310,
        LINEAR_4310,
    )
    from isaaclab_arena_vabar.environments.usbc_insertion import (
        environment as module,
    )
    from isaaclab_newton.physics import NewtonManager
    from newton.solvers import SolverMuJoCo

    assert LINEAR_4310.housing_body is None
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    for side in ("LeftRobot", "RightRobot"):
        start = len(builder.body_label)
        builder.add_usd(str(ROBOT_USD_PATH), floating=False, verbose=False)
        builder.body_label[start:] = [
            f"/World/envs/env_0/{side}/{label.lstrip('/')}"
            for label in builder.body_label[start:]
        ]
    flags = list(builder.shape_flags)
    pairs = len(builder._shape_collision_filter_pairs)
    monkeypatch.setattr(NewtonManager, "_builder", builder)

    module._apply_full_hand_contacts(hand=replace(CRANK_4310, housing_body=None))

    housing = [
        i
        for i, b in enumerate(builder.shape_body)
        if int(b) >= 0
        and builder.body_label[int(b)].endswith("/link_6")
        and builder.shape_source[i] is not None
    ]
    assert housing
    assert [builder.shape_flags[i] for i in housing] == [flags[i] for i in housing]
    assert len(builder._shape_collision_filter_pairs) == pairs


def test_cable_retains_rss_sixteen_hinges(monkeypatch):
    import newton
    import warp as wp
    from isaaclab_arena_vabar.environments.usbc_insertion import (
        environment as module,
    )
    from newton.solvers import SolverMuJoCo

    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    parent = builder.add_link(label="/World/envs/env_0/plug")
    monkeypatch.setattr(
        module,
        "_author_cable_visual",
        lambda root, links, radius: (
            [f"{root}/link{i}" for i in range(links)],
            f"{root}/curve",
        ),
    )
    module._cable_chain(
        builder,
        parent=parent,
        root_xform=wp.transform_identity(),
        root_anchor=wp.transform_identity(),
        links=16,
        seg_half=0.0075,
        radius=0.0032,
        density=450.0,
        tag="cable",
    )
    assert builder.joint_type == [newton.JointType.REVOLUTE] * 16
    assert len(builder.shape_body) == 16
    assert builder.joint_limit_upper == pytest.approx([2.0] * 16)
    assert builder.joint_damping == pytest.approx([0.0] * 16)
    assert builder.joint_armature == pytest.approx([2e-5] * 16)
    assert builder.joint_friction == pytest.approx([0.002] * 16)


def test_linear_fingertip_pieces_collide_as_meshes(monkeypatch) -> None:
    """AUTOLab collides each CoACD fingertip piece as a triangle mesh, not its hull."""
    import newton
    import numpy as np
    from isaaclab_arena_vabar.embodiments.bimanual_yam.config import LINEAR_4310
    from isaaclab_arena_vabar.environments.usbc_insertion import (
        environment as module,
    )
    from isaaclab_newton.physics import NewtonManager
    from newton.solvers import SolverMuJoCo

    cube = newton.Mesh(
        np.array([[x, y, z] for x in (0, 0.01) for y in (0, 0.01) for z in (0, 0.01)], dtype=np.float32),
        np.array([0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1,
                  2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3], dtype=np.int32),
    )
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    pieces, drawn = [], []
    for side in ("LeftRobot", "RightRobot"):
        for tip in ("tip_left", "tip_right"):
            body = builder.add_body(label=f"/World/envs/env_0/{side}/Geometry/gripper/{tip}")
            pieces.append(builder.add_shape_convex_hull(body, mesh=cube, label=f"{tip}/CableConvex_00"))
            cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
            drawn.append(builder.add_shape_convex_hull(body, mesh=cube, cfg=cfg, label=f"{tip}/{tip}"))
    monkeypatch.setattr(NewtonManager, "_builder", builder)

    module._apply_full_hand_contacts(hand=LINEAR_4310)

    assert all(builder.shape_type[i] == newton.GeoType.MESH for i in pieces)
    assert all(builder.shape_type[i] == newton.GeoType.CONVEX_MESH for i in drawn)
