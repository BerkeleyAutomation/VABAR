"""Medium scene and moving-receptacle contracts from AUTOLab bf985d0."""

from ib_support import load_scene, scene_params


def test_medium_uses_artist_assets_and_physical_support_relations():
    scene = load_scene("usbc_insertion/medium.yaml")
    objects = {obj["id"]: obj for obj in scene["objects"]}
    assert scene["embodiment"]["registry_name"] == "vabar__bimanual_yam"
    assert "port" not in objects
    assert objects["bulkhead"]["registry_name"] == "vabar__bulkhead"
    supports = {
        r["subject"]: r["reference"] for r in scene["relations"] if r["kind"] == "on"
    }
    assert supports == {
        "bench": "table",
        "cradle_front": "table",
        "cradle_rear": "table",
        "plug": "bench",
        "bulkhead": "cradle_front",
    }
    assert not any(
        r["kind"] == "is_anchor" and r["subject"] == "bulkhead"
        for r in scene["relations"]
    )
    task = scene_params(scene)
    assert task["bulkhead"] == "bulkhead"


def test_medium_cables_solver_and_passive_coupling_reuse_easy():
    from isaaclab_arena_vabar.environments.usbc_insertion.registration import (
        ENVIRONMENTS,
    )

    easy, medium = ENVIRONMENTS
    easy_scene = load_scene(easy.relative_graph_path)
    medium_scene = load_scene(medium.relative_graph_path)
    assert medium_scene["external_yaml"] == easy_scene["external_yaml"]
    assert medium.cable_spec["plug"] == easy.cable_spec["plug"]
    assert medium.cable_spec["bulkhead"]["links"] == 8
    assert medium.cable_spec["bulkhead"]["length"] * medium.cable_spec["scale"] == 0.1
