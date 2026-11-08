"""Structural contracts of the FR3 gear-mesh and syringe-sorting scene YAMLs."""

from __future__ import annotations

import math

import pytest

from ib_support import load_scene, relations_by, scene_params


def test_gear_easy_is_a_single_20t_gear_placed_only_through_relations() -> None:
    scene = load_scene("gear_insertion/easy.yaml")
    objects = {item["id"]: item for item in scene["objects"]}
    assert objects["gear_a"]["registry_name"] == "vabar__gear_mesh_20t"
    assert objects["board"]["registry_name"] == "vabar__gear_mesh_board_20"
    assert not {"gear_small", "gear_medium", "gear_large", "gear_base"} & objects.keys()
    for name in ("gear_mat", "board", "gear_a"):
        assert objects[name]["params"] == {}, name
    assert scene["task"]["subtasks"][0]["kind"] == "GearMeshTask"
    params = scene_params(scene)
    assert (params["board"], params["gear"], params["gear_teeth"]) == (
        "board",
        "gear_a",
        20,
    )
    assert [ref["id"] for ref in scene["object_references"]] == ["table"]

    relations = relations_by(scene)
    kinds = {}
    for subject, kind in relations:
        kinds.setdefault(subject, set()).add(kind)
    assert kinds == {
        "table": {"is_anchor"},
        "gear_mat": {"on", "at_position", "rotate_around_solution"},
        "board": {
            "on",
            "position_limits_box",
            "rotate_around_solution",
            "random_around_solution",
        },
        "gear_a": {
            "on",
            "position_limits_box",
            "rotate_around_solution",
            "random_around_solution",
        },
    }
    for subject, support in (
        ("gear_mat", "table"),
        ("board", "table"),
        ("gear_a", "gear_mat"),
    ):
        assert relations[subject, "on"]["reference"] == support


@pytest.mark.parametrize(
    ("filename", "gear_names", "episode_length_s"),
    [
        ("easy_pair.yaml", ("gear_a", "gear_b"), 200.0),
        ("medium_train.yaml", ("gear_a", "gear_b", "gear_c"), 300.0),
    ],
)
def test_generated_gear_mesh_families_use_fixed_source_rows_per_rebuild(
    filename: str, gear_names: tuple[str, ...], episode_length_s: float
) -> None:
    scene = load_scene(f"gear_insertion/{filename}")
    objects = {item["id"] for item in scene["objects"]}
    assert {"gear_mat", "board", *gear_names} <= objects
    params = scene_params(scene)
    assert tuple(params["gears"]) == gear_names
    assert params["episode_length_s"] == episode_length_s
    assert len(params["target_offsets_xyz"]) == len(gear_names)
    relations = relations_by(scene)
    assert relations[("board", "on")]["reference"] == "table"
    for name in gear_names:
        assert relations[(name, "on")]["reference"] == "gear_mat"
        assert (
            relations[(name, "at_position")]["params"]["relation_loss_weight"] == 1000.0
        )
        assert "random_around_solution" not in {
            kind for subject, kind in relations if subject == name
        }



@pytest.mark.parametrize(
    "name,count,goals",
    [("single", 1, 1), ("both", 2, 2), ("designated", 2, 1), ("cluttered", 6, 6)],
)
def test_syringe_families_have_randomized_on_placements(name, count, goals):
    scene = load_scene(f"syringe_disposal/{name}.yaml")
    assert "routine" not in scene
    objects = {obj["id"]: obj for obj in scene["objects"]}
    assert (
        objects["sharps_container"]["registry_name"]
        == "vabar__syringe_sort_bin"
    )
    syringes = [f"syringe_{i}" for i in range(count)]
    assert {obj for obj in objects if obj.startswith("syringe_")} == set(syringes)
    relations = scene["relations"]
    kinds = {"on", "is_anchor", "position_limits_box"}
    if name in ("cluttered", "both"):
        # AUTOLab's clutter and tray_pair trials solve the tray at yaw 140.4 deg.
        kinds |= {"rotate_around_solution"}
        rotation = next(r for r in relations if r["kind"] == "rotate_around_solution")
        assert rotation["subject"] == "instrument_tray"
        assert rotation["params"]["yaw_rad"] == pytest.approx(math.radians(140.4), abs=1e-4)
    assert {r["kind"] for r in relations} == kinds
    for fixture in ("instrument_tray", "sharps_container"):
        limits = next(
            r["params"]
            for r in relations
            if r["subject"] == fixture and r["kind"] == "position_limits_box"
        )
        if fixture == "sharps_container":
            assert (limits["x_min"], limits["x_max"]) == pytest.approx((0.05, 0.12))
        else:
            assert limits["x_max"] - limits["x_min"] == pytest.approx(0.1)
        assert limits["y_max"] - limits["y_min"] == pytest.approx(0.1)
        assert (
            next(r for r in relations if r["subject"] == fixture and r["kind"] == "on")[
                "reference"
            ]
            == "table"
        )
    # Every syringe rests on the tray floor; a raised release lets the solver
    # stack syringes that then settle through each other.
    for syringe in syringes:
        assert "initial_pose" not in objects[syringe]["params"]
        on = next(r for r in relations if r["subject"] == syringe)
        assert on["reference"] == "instrument_tray"
        assert on["params"] == {"clearance_m": 0.0, "edge_margin_m": 0.01}
    params = scene_params(scene)
    assert params["object_list"] == syringes[:goals]
    assert params["region_list"] == ["sharps_container"] * goals
    assert len(params["bounds_xyzxyz"]) == goals
    assert objects["syringe_0"]["registry_name"] == "vabar__syringe"
    if name in ("designated", "cluttered"):
        # AUTOLab's spares are untextured: only syringe_0 wears the red cap that
        # perceive_target.py identifies its syringe by.
        for syringe in syringes[1:]:
            assert objects[syringe]["registry_name"] == "vabar__syringe_blank"
