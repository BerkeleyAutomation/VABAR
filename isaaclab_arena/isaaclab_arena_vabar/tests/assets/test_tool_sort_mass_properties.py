"""Sorting asset masses follow the pinned RSS task bodies, not artist density."""

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from isaaclab_arena_gap.task_sources import autolab_root

Usd = pytest.importorskip("pxr.Usd")
UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets"
RSS_TOOL_SORT = autolab_root() / "assets/vab_tool_sort"

# Our asset -> (RSS scene, RSS body, our mass prim).
MASS_SOURCES = {
    "adjustable_wrench": ("sort_bench", "adjustable_wrench_0_main", "/Asset/Geometry/adjustable_wrench_obj_00"),
    "battery": ("sort_electrical", "industrial_d_battery_0_main", "/Asset/Geometry/industrial_d_battery_0_g0_visual_obj_00"),
    "breadboard": ("sort_wiring", "industrial_breadboard_0_main", "/Asset/Geometry/industrial_breadboard_0_g0_visual_obj_00"),
    "combination_pliers": (
        "sort_wiring",
        "industrial_combination_pliers_0_main",
        "/Asset/Geometry/industrial_combination_pliers_0_g0_visual_obj_00",
    ),
    "cutting_pliers": ("sort_electrical", "distractor_industrial_cutting_pliers_0_main", "/Asset/Geometry/base_01_obj_00"),
    "flashlight": (
        "sort_electrical",
        "industrial_tactical_flashlight_0_main",
        "/Asset/Geometry/industrial_tactical_flashlight_0_g0_visual_obj_00",
    ),
    "insulating_tape": ("sort_bench", "insulating_tape_0_main", "/Asset/Geometry/insulating_tape_0_g0_visual_obj_00"),
    "multimeter": (
        "sort_electrical",
        "industrial_multimeter_0_main",
        "/base/Geometry/vabar__multimeter/industrial_multimeter_0_g0",
    ),
    "slotted_screwdriver": ("sort_bench", "slotted_screwdriver_0_main", "/Asset/Geometry/slotted_screwdriver_01_obj_00"),
    "tape_measure": ("sort_bench", "tape_measure_0_main", "/Asset/Geometry/tape_measure_01_obj_00"),
    "safety_glasses": ("sort_electrical", "industrial_safety_glasses_0_main", "/Asset/Geometry/industrial_safety_glasses_01_obj_00"),
    "wire_spool": (
        "sort_wiring",
        "industrial_wire_spool_0_main",
        "/Asset/Geometry/industrial_wire_spool_0_g0_visual_obj_00/industrial_wire_spool_0_g0_visual_mesh_00",
    ),
}


def _rss_mass(scene: str, body: str) -> float:
    node = ET.parse(RSS_TOOL_SORT / scene / "scene.xml").find(f'.//body[@name="{body}"]')
    assert node is not None and node.find("inertial") is None
    geoms = node.findall(".//geom")
    assert geoms and all(g.get("mass") for g in geoms)
    return sum(float(g.get("mass")) for g in geoms)


@pytest.mark.parametrize("name", sorted(MASS_SOURCES))
def test_sorting_asset_mass_matches_rss(name):
    scene, body, prim = MASS_SOURCES[name]
    stage = Usd.Stage.Open(str(ASSETS / f"vabar__{name}" / f"vabar__{name}.usda"))
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(prim)).GetMassAttr().Get()
    assert mass == pytest.approx(_rss_mass(scene, body), rel=1e-6)
