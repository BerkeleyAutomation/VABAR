"""Durable acceptance contract for every movable benchmark asset."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("isaaclab")

from asset_acceptance.asset_drop_cases import (
    ASSET_DROP_CASES,
    PRODUCTION_PHYSICS_YAML,
    case_for_asset,
)
from asset_acceptance.asset_drop_harness import (
    AssetDropObservation,
    evaluate_asset_drop,
    pose_deviation,
    settling_time,
)
from asset_acceptance.process_contracts import (
    is_accepted_isaac_teardown,
    run_fresh_process_with_retries,
)
from ib_support import TOOLS_ROOT, load_tool

LIVE = os.environ.get("ISAAC_CAP_RUN_LIVE_TESTS") == "1"
LIVE_CHECK = TOOLS_ROOT / "asset_acceptance/asset_drop_live_check.py"

# Pinned on purpose: the asset doc tools and quality report derive their
# expected set from ASSET_DROP_CASES, so only this list catches an asset that
# silently drops out of registration or loses its movable-object role.
EXPECTED_DROP_ASSETS = frozenset(
    {
        "vabar__adjustable_wrench",
        "vabar__battery",
        "vabar__breadboard",
        "vabar__clamp",
        "vabar__combination_pliers",
        "vabar__cutting_pliers",
        "vabar__drill",
        "vabar__flashlight",
        "vabar__flathead_screwdriver",
        "vabar__hammer",
        "vabar__insulating_tape",
        "vabar__instrument_tray",
        "vabar__measuring_tape",
        "vabar__multimeter",
        "vabar__phillips_screwdriver",
        "vabar__pliers",
        "vabar__round_nut",
        "vabar__safety_glasses",
        "vabar__screwdriver",
        "vabar__slotted_screwdriver",
        "vabar__syringe",
        "vabar__syringe_blank",
        "vabar__tape_measure",
        "vabar__utility_knife",
        "vabar__wire_spool",
        "vabar__hanging_hammer_1",
        "vabar__hanging_hammer_2",
        "vabar__plier_combination",
        "vabar__plier_cutter",
        "vabar__scissor_pair_1",
        "vabar__scissor_pair_2",
        "vabar__screwdriver_grip",
        "vabar__screwdriver_slim",
        "vabar__wrench_ring_17",
        "vabar__wrench_ring_19",
        "vabar__bulkhead",
        "vabar__plug",
    }
)


def _observation(**changes) -> AssetDropObservation:
    values = {
        "finite_state": True,
        "contact_count": 10,
        "final_lowest_point_z": 0.75,
        "tabletop_z": 0.75,
        "inside_tabletop": True,
        "tail_translations_m": (0.0, 0.0005),
        "tail_rotations_rad": (0.0, 0.01),
    }
    values.update(changes)
    return AssetDropObservation(**values)


def test_drop_verdict_accepts_a_settled_supported_asset() -> None:
    assert evaluate_asset_drop(_observation()).passed


def test_drop_verdict_fails_closed_on_physical_or_non_finite_state() -> None:
    for observation in (
        _observation(contact_count=0),
        _observation(final_lowest_point_z=0.747),
        _observation(tail_rotations_rad=(0.0, 0.02)),
        _observation(finite_state=False),
    ):
        assert not evaluate_asset_drop(observation).passed


# Poses are (x, y, z, qx, qy, qz, qw), as Isaac Lab's root_link_pose_w reports them.
REST = (0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 1.0)


def _shifted(dx: float) -> tuple[float, ...]:
    return (dx, *REST[1:])


def test_drop_verdict_ignores_chatter_in_place_but_fails_a_slow_slide() -> None:
    chatter = [_shifted(0.00004 * (-1) ** i) for i in range(50)]
    slide = [_shifted(0.0002 * i) for i in range(50)]  # 1 cm/s
    assert evaluate_asset_drop(
        _observation(**dict(zip(("tail_translations_m", "tail_rotations_rad"),
                                AssetDropObservation.tail_from_poses(chatter))))
    ).passed
    assert not evaluate_asset_drop(
        _observation(**dict(zip(("tail_translations_m", "tail_rotations_rad"),
                                AssetDropObservation.tail_from_poses(slide))))
    ).passed


def test_drop_settling_and_quaternion_conventions() -> None:
    assert settling_time((_shifted(0.01), _shifted(0.0), _shifted(0.0)), 0.1) == 0.1
    assert settling_time((REST, _shifted(math.nan)), 0.1) is None
    turned = (0.0, 0.0, 0.75, 0.0, 0.0, math.sin(0.5), math.cos(0.5))
    assert pose_deviation(REST, turned)[1] == pytest.approx(1.0)


def test_drop_spawns_the_orientation_its_height_and_verdict_assume() -> None:
    """The drop height and rest pose use rotation_xyzw, so the spawn must too.

    The pinned Isaac Lab reads InitialStateCfg.rot as (x, y, z, w); converting it to
    (w, x, y, z) spawned an identity case flipped 180 degrees about X.
    """
    from isaaclab.assets import RigidObjectCfg

    from asset_acceptance.asset_drop_live_check import _drop_initial_state

    identity = RigidObjectCfg.InitialStateCfg().rot
    assert identity == (0.0, 0.0, 0.0, 1.0)
    case = case_for_asset("vabar__hanging_hammer_2")
    assert case.rotation_xyzw == identity
    assert _drop_initial_state(case, (0.0, 0.0, 0.9)).rot == identity
    for case in ASSET_DROP_CASES:
        state = _drop_initial_state(case, (0.0, 0.0, 0.9))
        assert state.rot == case.rotation_xyzw, case.registry_name
        assert state.pos == (0.0, 0.0, 0.9)


def test_drop_inventory_is_unique_and_addressable() -> None:
    names = [case.registry_name for case in ASSET_DROP_CASES]
    assert len(names) == len(set(names))
    assert set(names) == EXPECTED_DROP_ASSETS, (
        "drop inventory changed: "
        f"missing={sorted(EXPECTED_DROP_ASSETS - set(names))} "
        f"extra={sorted(set(names) - EXPECTED_DROP_ASSETS)}; "
        "update EXPECTED_DROP_ASSETS only for an intended asset change"
    )
    assert all(case_for_asset(name).registry_name == name for name in names)


def test_asset_doc_tools_follow_the_drop_inventory(tmp_path: Path) -> None:
    names = tuple(case.registry_name for case in ASSET_DROP_CASES)
    assert load_tool("generate_asset_docs")._asset_names() == names

    report = load_tool("asset_quality_report")
    for name in names[1:]:
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"result": {"registry_name": name}})
        )
    with pytest.raises(ValueError, match="unique movable-asset results"):
        report._read_results(tmp_path, frozenset(names))
    (tmp_path / f"{names[0]}.json").write_text(
        json.dumps({"result": {"registry_name": names[0]}})
    )
    assets = report._read_results(tmp_path, frozenset(names))
    assert [asset["registry_name"] for asset in assets] == sorted(names)


def test_asset_doc_tools_launch_children_inside_the_venv(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_PYTHON", raising=False)
    python = load_tool("generate_asset_docs").ARENA_PYTHON
    assert python.parent.parent.name == ".venv", python


def test_drop_physics_profiles_replay_production_yaml(arena_manager) -> None:
    from asset_acceptance.asset_drop_cases import DROP_ONLY_SOLVER_OVERRIDES
    from asset_acceptance.asset_drop_live_check import (
        _ENVIRONMENT_ROOT,
        _drop_physics_cfg,
        _physics_cfg,
    )

    NewtonArenaMJWarpManager = arena_manager
    used_profiles = {case.physics_profile for case in ASSET_DROP_CASES}
    assert used_profiles <= set(PRODUCTION_PHYSICS_YAML)
    for profile, relative_yaml in PRODUCTION_PHYSICS_YAML.items():
        document = yaml.safe_load((_ENVIRONMENT_ROOT / relative_yaml).read_text())
        sim = document["env_cfg_override"]["sim"]
        dt, physics = _physics_cfg(profile)
        assert dt == sim["dt"]
        assert physics.num_substeps == sim["physics"]["num_substeps"]
        for key, value in sim["physics"]["solver_cfg"].items():
            if key != "_target_":
                assert getattr(physics.solver_cfg, key) == value, (profile, key)
        for key, value in document["benchmark_physics"]["solver_cfg"].items():
            assert getattr(physics.solver_cfg, key) == value, (profile, key)
        # Production resyncs the manager only when the YAML replaces solver_cfg.
        if "_target_" in sim["physics"]["solver_cfg"]:
            assert physics.class_type is not NewtonArenaMJWarpManager, profile
        else:
            assert physics.class_type is NewtonArenaMJWarpManager, profile
    assert DROP_ONLY_SOLVER_OVERRIDES == {
        "tool_sort_hammer_mjwarp": {"impratio": 100.0}
    }
    hammer_yaml_path = (
        _ENVIRONMENT_ROOT / PRODUCTION_PHYSICS_YAML["tool_sort_hammer_mjwarp"]
    )
    hammer_yaml = yaml.safe_load(hammer_yaml_path.read_text())
    production_impratio = hammer_yaml["env_cfg_override"]["sim"]["physics"][
        "solver_cfg"
    ]["impratio"]
    assert _physics_cfg("tool_sort_hammer_mjwarp")[1].solver_cfg.impratio == (
        production_impratio
    )
    assert _drop_physics_cfg("tool_sort_hammer_mjwarp")[1].solver_cfg.impratio == 100.0
    assert case_for_asset("vabar__syringe").physics_profile == "syringe_mjwarp"


def test_standalone_usbc_drop_stiffens_only_mating_part_contacts(monkeypatch):
    from isaaclab_arena_vabar.environments.usbc_insertion.environment import (
        CONTACT_KD,
        CONTACT_KE,
    )
    from isaaclab.physics import PhysicsEvent
    from isaaclab_newton.physics import NewtonManager

    from asset_acceptance.asset_drop_live_check import (
        _register_standalone_usbc_contact_stiffening,
    )

    builder = SimpleNamespace(
        shape_count=3,
        shape_label=["plug/collider", "bulkhead/collider", "table"],
        shape_material_ke=[0.0, 0.0, 0.0],
        shape_material_kd=[0.0, 0.0, 0.0],
    )
    callbacks = []
    monkeypatch.setattr(NewtonManager, "_builder", builder)
    monkeypatch.setattr(
        NewtonManager,
        "register_callback",
        lambda callback, event, **_: callbacks.append((callback, event)),
    )
    _register_standalone_usbc_contact_stiffening()
    assert len(callbacks) == 1
    callback, event = callbacks[0]
    assert event == PhysicsEvent.MODEL_INIT
    callback()
    assert builder.shape_material_ke == [CONTACT_KE, CONTACT_KE, 0.0]
    assert builder.shape_material_kd == [CONTACT_KD, CONTACT_KD, 0.0]


def _drop_command(case, tmp_path: Path, result_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(LIVE_CHECK),
        "--asset",
        case.registry_name,
        "--record-dir",
        str(tmp_path / "recordings"),
        "--result-dir",
        str(result_dir),
        "--no-record",
    ]


def test_asset_subprocess_teardown_codes_are_bounded() -> None:
    assert is_accepted_isaac_teardown(0)
    assert not is_accepted_isaac_teardown(1)


def test_fresh_process_retries_a_crash_without_evidence(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    marker = tmp_path / "crashed-once"
    # The first process crashes like a Kit SIGSEGV; the retry persists evidence.
    script = (
        "import pathlib, sys\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        "if not marker.exists():\n"
        "    marker.touch()\n"
        "    sys.exit(139)\n"
        f"pathlib.Path({str(result_path)!r}).write_text('{{}}')\n"
    )
    failures = []
    complete = run_fresh_process_with_retries(
        command=[sys.executable, "-c", script],
        cwd=tmp_path,
        environment=dict(os.environ),
        stale_outputs=(result_path,),
        result_path=result_path,
        validate=lambda: json.loads(result_path.read_text()),
        label="Asset test",
        on_failure=failures.append,
    )
    assert complete
    assert [(f.attempt, f.process_returncode, f.will_retry) for f in failures] == [
        (1, 139, True)
    ]
    assert failures[0].reason == "incomplete or invalid evidence"


@pytest.mark.live
@pytest.mark.skipif(
    not LIVE,
    reason="set ISAAC_CAP_RUN_LIVE_TESTS=1 to run fresh-process asset drops",
)
@pytest.mark.parametrize(
    "case",
    ASSET_DROP_CASES,
    ids=lambda case: case.registry_name,
)
def test_asset_passes_drop_acceptance(case, tmp_path: Path) -> None:
    """Run one asset in one process and make its physical verdict authoritative."""
    result_dir = tmp_path / "results"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(TOOLS_ROOT), environment.get("PYTHONPATH"))
        if value
    )
    completed = subprocess.run(
        _drop_command(case, tmp_path, result_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1200,
        check=False,
        env=environment,
    )
    assert is_accepted_isaac_teardown(completed.returncode), completed.stdout
    result_path = result_dir / f"{case.registry_name}.json"
    assert result_path.is_file(), completed.stdout
    payload = json.loads(result_path.read_text())
    assert payload["asset"] == case.registry_name
    assert payload["passed"], payload["result"]


def test_collision_points_keep_scale_authored_above_the_body(tmp_path: Path) -> None:
    """The simulated body pose carries no scale, so the points must keep it."""
    from pxr import Usd, UsdGeom, UsdPhysics

    from asset_acceptance.asset_drop_live_check import _composed_collision_points

    path = tmp_path / "scaled.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    root.AddScaleOp().Set((0.5, 0.5, 0.5))
    stage.SetDefaultPrim(root.GetPrim())
    body = UsdGeom.Xform.Define(stage, "/Asset/Body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/Asset/Body/Collider")
    cube.GetSizeAttr().Set(0.1)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.Save()

    zs = [point[2] for point in _composed_collision_points(path)]
    assert max(zs) - min(zs) == pytest.approx(0.05)
