"""Tests for coupled gear-family build-time variation."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

pytest.importorskip("isaaclab")

from ib_support import AUTOLAB_ROOT, PACKAGE_ROOT
from isaaclab_arena_vabar.assets.gear_insertion import GEAR_MESH_ASSET_PATHS
from isaaclab_arena_vabar.tasks.gear_insertion.variation import (
    GearFamilyVariation,
    GearFamilyVariationCfg,
    GearLayoutVariation,
    GearLayoutVariationCfg,
)

VARIANTS_PATH = PACKAGE_ROOT / "tasks/gear_insertion/variants.json"
RSS_VARIANTS_PATH = AUTOLAB_ROOT / "tasks/gear_mesh/assets/variants.json"


class _Asset:
    def __init__(self, relation=None):
        self.usd_path = "original.usda"
        self.spawn_cfg_addon = {}
        self.object_cfg = SimpleNamespace(
            spawn=SimpleNamespace(usd_path=self.usd_path, variants=None)
        )
        self.bounding_box = object()
        self._relations = [] if relation is None else [relation]

    def get_relations(self):
        return self._relations


def _placed_asset():
    return _Asset(
        SimpleNamespace(name="at_position", x=0.0, y=0.0, relation_loss_weight=1000.0)
    )


@pytest.mark.parametrize("teeth", [16, 20, 24])
def test_gear_family_variation_selects_matching_assets_and_bounds(teeth: int) -> None:
    limits = SimpleNamespace(
        name="position_limits_box", x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0
    )
    board = _Asset()
    gear = _Asset(limits)
    task = SimpleNamespace(
        set_gear_teeth=lambda value: setattr(task, "gear_teeth", value)
    )
    variation = GearFamilyVariation(
        board,
        gear,
        task,
        GearFamilyVariationCfg(tooth_counts=[teeth]),
    )

    variation.configure_at_build_time()

    assert gear.usd_path == str(GEAR_MESH_ASSET_PATHS[f"{teeth}t"])
    assert board.usd_path == str(GEAR_MESH_ASSET_PATHS["board"])
    assert board.spawn_cfg_addon["variants"] == {"layout": f"board_{teeth}"}
    assert board.object_cfg.spawn.variants == {"layout": f"board_{teeth}"}
    assert gear.object_cfg.spawn.usd_path == gear.usd_path
    assert board.object_cfg.spawn.usd_path == board.usd_path
    assert gear.bounding_box is None
    assert board.bounding_box is None
    assert task.gear_teeth == teeth

    radius = 0.0025 * (teeth + 2) / 2.0
    center_x, center_y = -0.0309982590563552, 0.0155650225211777
    assert limits.x_min == pytest.approx(center_x - (0.075 - radius - 0.008))
    assert limits.x_max == pytest.approx(center_x + (0.075 - radius - 0.008))
    assert limits.y_min == pytest.approx(center_y - (0.135 - radius - 0.008))
    assert limits.y_max == pytest.approx(center_y + (0.135 - radius - 0.008))


@pytest.mark.parametrize("tooth_counts", [[], [15], [16, 20, 25]])
def test_gear_family_variation_rejects_unsupported_choices(tooth_counts) -> None:
    variation = GearFamilyVariation(
        _Asset(),
        _Asset(SimpleNamespace(name="position_limits_box")),
        SimpleNamespace(),
        GearFamilyVariationCfg(tooth_counts=tooth_counts),
    )

    with pytest.raises(ValueError, match="16, 20, or 24"):
        variation.configure_at_build_time()


def test_generated_pair_and_train_manifest_matches_every_rss_row() -> None:
    generated = json.loads(VARIANTS_PATH.read_text())["variants"]
    rss_rows = [
        row
        for row in json.loads(RSS_VARIANTS_PATH.read_text())["variants"]
        if row["family"] in {"pair", "train"}
    ]

    assert len(generated) == len(rss_rows) == 160
    assert [row["name"] for row in generated] == [row["name"] for row in rss_rows]
    assert sum(row["family"] == "pair" for row in generated) == 60
    assert sum(row["family"] == "train" for row in generated) == 100
    for actual, expected in zip(generated, rss_rows, strict=True):
        assert actual["seed"] == expected["seed"]
        assert actual["stations"] == expected["stations"]
        assert actual["bends"] == expected["board"].get("bends", [])
        assert actual["board_xy"] == expected["board"]["xy"]
        assert actual["board_yaw"] == expected["board"]["yaw"]
        assert [gear["teeth"] for gear in actual["gears"]] == expected["stations"]
        assert [gear["xy"] for gear in actual["gears"]] == [
            gear["xy"] for gear in expected["gears"]
        ]
        assert [gear["yaw"] for gear in actual["gears"]] == [
            gear["yaw"] for gear in expected["gears"]
        ]
        assert len(actual["target_offsets_xyz"]) == len(expected["stations"])

        offsets = actual["target_offsets_xyz"]
        headings = []
        for index in range(1, len(offsets)):
            dx = offsets[index][0] - offsets[index - 1][0]
            dy = offsets[index][1] - offsets[index - 1][1]
            expected_distance = (
                0.0025
                * (actual["stations"][index - 1] + actual["stations"][index])
                / 2.0
            )
            assert math.hypot(dx, dy) == pytest.approx(expected_distance)
            headings.append(math.atan2(dy, dx))
        if actual["family"] == "pair":
            assert headings == pytest.approx([0.0])
        else:
            assert headings == pytest.approx([actual["bends"][0], sum(actual["bends"])])


def test_single_board_usd_contains_every_generated_layout() -> None:
    from pxr import Usd

    manifest = json.loads(VARIANTS_PATH.read_text())["variants"]
    expected = {row["board_variant"] for row in manifest}
    expected.update({"board_16", "board_20", "board_24"})
    assert len(expected) == 109
    path = GEAR_MESH_ASSET_PATHS["board"]
    stage = Usd.Stage.Open(str(path))
    assert stage and stage.GetDefaultPrim().IsValid(), path
    variants = stage.GetDefaultPrim().GetVariantSet("layout")
    assert set(variants.GetVariantNames()) == expected


def test_every_board_variant_composes_its_source_station_offsets() -> None:
    from pxr import Usd

    rows = json.loads(VARIANTS_PATH.read_text())["variants"]
    stage = Usd.Stage.Open(str(GEAR_MESH_ASSET_PATHS["board"]))
    root = stage.GetDefaultPrim()
    variants = root.GetVariantSet("layout")
    for row in rows:
        assert variants.SetVariantSelection(row["board_variant"])
        pegs = [
            root.GetChild("base").GetChild(f"station_{index}_peg") for index in range(3)
        ]
        active = [peg for peg in pegs if peg.IsActive()]
        assert len(active) == len(row["target_offsets_xyz"])
        for peg, offset in zip(active, row["target_offsets_xyz"], strict=True):
            assert tuple(peg.GetAttribute("xformOp:translate").Get()) == pytest.approx(
                offset[:2] + [0.0]
            )

        pinion_position = tuple(
            root.GetChild("pinion").GetAttribute("xformOp:translate").Get()
        )
        pinion_anchor = tuple(
            root.GetChild("pinion_joint").GetAttribute("physics:localPos0").Get()
        )
        assert pinion_position == pytest.approx(pinion_anchor)
        button_position = tuple(
            root.GetChild("button").GetAttribute("xformOp:translate").Get()
        )
        button_anchor = tuple(
            root.GetChild("button_joint").GetAttribute("physics:localPos0").Get()
        )
        assert button_position == pytest.approx(button_anchor)


def test_layout_variation_applies_one_complete_source_row(monkeypatch) -> None:
    board = _placed_asset()
    board._relations.append(SimpleNamespace(name="rotate_around_solution", yaw_rad=0.0))
    gears = [_placed_asset(), _placed_asset()]
    for gear in gears:
        gear._relations.append(
            SimpleNamespace(name="rotate_around_solution", yaw_rad=0.0)
        )
    task = SimpleNamespace(
        configure_layout=lambda teeth, offsets: setattr(
            task, "layout", (tuple(teeth), tuple(map(tuple, offsets)))
        )
    )
    variation = GearLayoutVariation(
        board, gears, task, GearLayoutVariationCfg(family="pair")
    )
    row = next(
        row
        for row in json.loads(VARIANTS_PATH.read_text())["variants"]
        if row["family"] == "pair"
    )
    monkeypatch.setattr(variation.sampler, "sample", lambda **_kwargs: [row["name"]])

    variation.configure_at_build_time()

    assert variation.selected_variant == row["name"]
    assert board.usd_path == str(GEAR_MESH_ASSET_PATHS["board"])
    assert board.spawn_cfg_addon["variants"] == {"layout": row["board_variant"]}
    assert task.layout == (
        tuple(row["stations"]),
        tuple(map(tuple, row["target_offsets_xyz"])),
    )
    for asset, source in zip(gears, row["gears"], strict=True):
        assert asset.usd_path == str(GEAR_MESH_ASSET_PATHS[f"{source['teeth']}t"])
        position, rotation = asset.get_relations()
        assert position.x == pytest.approx(0.0990017409436448 + source["xy"][0])
        assert position.y == pytest.approx(0.0155650225211777 + source["xy"][1])
        assert rotation.yaw_rad == pytest.approx(source["yaw"])


def _layout_variation(family: str, seed: int | None) -> GearLayoutVariation:
    gear_count = 2 if family == "pair" else 3
    board = _placed_asset()
    board._relations.append(SimpleNamespace(name="rotate_around_solution", yaw_rad=0.0))
    gears = [_placed_asset() for _ in range(gear_count)]
    for gear in gears:
        gear._relations.append(
            SimpleNamespace(name="rotate_around_solution", yaw_rad=0.0)
        )
    task = SimpleNamespace(configure_layout=lambda teeth, offsets: None)
    return GearLayoutVariation(
        board,
        gears,
        task,
        GearLayoutVariationCfg(family=family),
        seed_source=lambda: seed,
    )


def _drawn_layout(family: str, seed: int | None) -> str:
    variation = _layout_variation(family, seed)
    variation.configure_at_build_time()
    return variation.selected_variant


def _drawn_teeth(seed: int | None) -> int:
    task = SimpleNamespace(
        set_gear_teeth=lambda value: setattr(task, "gear_teeth", value)
    )
    variation = GearFamilyVariation(
        _Asset(),
        _Asset(SimpleNamespace(name="position_limits_box")),
        task,
        seed_source=lambda: seed,
    )
    variation.configure_at_build_time()
    return task.gear_teeth


@pytest.mark.parametrize("family", ["pair", "train"])
def test_layout_draw_is_reproducible_by_seed(family: str) -> None:
    """The same seed draws the same document in every build, whatever torch's RNG holds."""
    import torch

    seeds = range(42, 52)
    torch.manual_seed(0)
    first = [_drawn_layout(family, seed) for seed in seeds]
    torch.manual_seed(1234)
    torch.rand(97)
    second = [_drawn_layout(family, seed) for seed in reversed(seeds)][::-1]

    assert first == second
    assert len(set(first)) >= 8


def test_tooth_count_draw_is_reproducible_by_seed() -> None:
    import torch

    seeds = range(42, 52)
    torch.manual_seed(0)
    first = [_drawn_teeth(seed) for seed in seeds]
    torch.manual_seed(1234)
    second = [_drawn_teeth(seed) for seed in seeds]

    assert first == second
    assert set(first) == {16, 20, 24}


def test_seeded_draw_still_reaches_the_variation_recorder() -> None:
    variation = _layout_variation("pair", 42)
    seen = []
    variation.add_sample_listener(lambda sample, env_ids: seen.append(sample))

    variation.configure_at_build_time()

    assert seen == [[variation.selected_variant]]


def test_layout_draw_without_a_seed_uses_arena_sampling() -> None:
    import torch

    torch.manual_seed(7)
    first = _drawn_layout("train", None)
    torch.manual_seed(7)
    assert _drawn_layout("train", None) == first


@pytest.mark.parametrize(
    "names,match",
    [
        ([], "explicit"),
        (["missing"], "distinct"),
        (["missing", "missing"], "distinct"),
        (["missing", "other"], "family"),
    ],
)
def test_parallel_configuration_rejects_invalid_layout_names(names, match):
    from isaaclab_arena_vabar.tasks.gear_insertion.parallel_layouts import (
        configure_parallel_layouts,
    )

    cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=2))
    with pytest.raises(ValueError, match=match):
        configure_parallel_layouts(
            cfg, family="pair", gear_names=("gear_a", "gear_b"), layout_names=names
        )


@pytest.mark.parametrize("gear_count", [2, 3])
def test_clone_plan_keeps_every_board_and_gear_on_its_source_row(gear_count):
    import torch
    from isaaclab_arena_vabar.tasks.gear_insertion.parallel_layouts import (
        coupled_layout_clones,
    )
    from isaaclab.cloner.clone_plan import make_valid_clone_combinations

    combinations = make_valid_clone_combinations(
        ["robot", "board", *[f"gear_{i}" for i in range(gear_count)]],
        [1, *([20] * (gear_count + 1))],
    )
    result = coupled_layout_clones(combinations, 20, "cpu")
    assert result[:, 0].tolist() == [0] * 20
    assert torch.equal(
        result[:, 1:], torch.arange(20)[:, None].expand(-1, gear_count + 1)
    )


@pytest.mark.parametrize("family,gear_count", [("pair", 2), ("train", 3)])
def test_parallel_configuration_keeps_assets_poses_and_scoring_coupled(
    family, gear_count
):

    from isaaclab.managers import TerminationTermCfg
    from isaaclab.sim import UsdFileCfg
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.relations.placement_events import solve_and_place_objects
    from isaaclab_arena_vabar.fixed_layouts import ReplayFirstLayouts
    from isaaclab_arena_vabar.tasks.gear_insertion.parallel_layouts import (
        configure_parallel_layouts,
        gear_success_params,
    )
    from isaaclab_arena_vabar.tasks.gear_insertion.terminations import (
        gear_mesh_success,
    )

    rows = [
        row
        for row in json.loads(VARIANTS_PATH.read_text())["variants"]
        if row["family"] == family
    ][:40]
    gear_names = [f"gear_{letter}" for letter in "abc"[:gear_count]]
    assets = [_placed_asset() for _ in range(gear_count + 1)]
    for asset, name in zip(assets, ["board", *gear_names], strict=True):
        asset.name = name
    mat = _placed_asset()
    mat.name = "gear_mat"
    for batch in (rows[:20], rows[20:]):
        layouts = [
            [
                SimpleNamespace(
                    positions={
                        mat: (0.0, 0.0, 0.8),
                        **{
                            asset: (0.0, 0.0, 0.9 + i * 0.01)
                            for i, asset in enumerate(assets)
                        },
                    },
                    orientations={},
                )
            ]
            for _ in batch
        ]
        pool = SimpleNamespace(layouts_per_env=lambda layouts=layouts: layouts)
        scene = SimpleNamespace(num_envs=20, clone_cfg=SimpleNamespace())
        for name in ["board", *gear_names]:
            setattr(
                scene, name, SimpleNamespace(spawn=UsdFileCfg(usd_path="template.usda"))
            )
        cfg = SimpleNamespace(
            scene=scene,
            episode_recorders=SimpleNamespace(),
            events=SimpleNamespace(
                placement_reset=SimpleNamespace(
                    func=solve_and_place_objects,
                    params={"placement_pool": SimpleNamespace(pool=pool)},
                )
            ),
            task_progress=SimpleNamespace(
                success_criteria=[
                    CompletionCriteria(
                        name="gear_mesh",
                        predicate_sequence=[
                            TerminationTermCfg(func=gear_mesh_success, params={})
                        ],
                    )
                ],
                subtasks_are_sequential=False,
                desired_subtask_success_state=None,
            ),
            # Arena's success term reads task progress and takes no params.
            terminations=SimpleNamespace(
                success=TerminationTermCfg(func=None, params={})
            ),
        )
        configure_parallel_layouts(
            cfg,
            family=family,
            gear_names=gear_names,
            layout_names=[r["name"] for r in batch],
        )
        # Source rows are fixed starts: every reset replays the first layout.
        assert cfg.events.placement_reset.func is ReplayFirstLayouts
        for i, row in enumerate(batch):
            recorder = cfg.episode_recorders.gear_layout
            assert recorder.func(None, i, **recorder.params) == {
                "gear_layout": row["name"]
            }
            assert cfg.scene.board.spawn.assets_cfg[i].variants == {
                "layout": row["board_variant"]
            }
            for j, name in enumerate(gear_names):
                assert getattr(cfg.scene, name).spawn.assets_cfg[i].usd_path == str(
                    GEAR_MESH_ASSET_PATHS[f"{row['gears'][j]['teeth']}t"]
                )
            # The rows belong to the gear predicate nested inside the
            # task-progress criteria; Arena's success term stays param-free.
            assert cfg.terminations.success.params == {}
            params = gear_success_params(cfg)
            assert params["gear_teeth"][i] == row["stations"]
            assert (
                params["target_offsets_xyz"][i]
                == row["target_offsets_xyz"]
            )
            for j, asset in enumerate(assets):
                source = (
                    {"xy": row["board_xy"], "yaw": row["board_yaw"]}
                    if j == 0
                    else row["gears"][j - 1]
                )
                # The board keeps its solved height; gears rest on the mat's top face.
                assert layouts[i][0].positions[asset] == pytest.approx(
                    (
                        0.0990017409436448 + source["xy"][0],
                        0.0155650225211777 + source["xy"][1],
                        0.9 if j == 0 else 0.8 + 0.008 + 0.0005,
                    )
                )
                assert layouts[i][0].orientations[asset] == pytest.approx(source["yaw"])



def test_gear_rest_height_matches_the_mat_and_gear_colliders():
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    from isaaclab_arena_vabar.tasks.gear_insertion.parallel_layouts import (
        GEAR_HALF_THICKNESS_M,
        MAT_TOP_ABOVE_ORIGIN_M,
    )

    def collider_z(key):
        stage = Usd.Stage.Open(str(GEAR_MESH_ASSET_PATHS[key]))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
        ranges = [
            cache.ComputeWorldBound(prim).ComputeAlignedRange()
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        return min(r.GetMin()[2] for r in ranges), max(r.GetMax()[2] for r in ranges)

    assert collider_z("mat")[1] == pytest.approx(MAT_TOP_ABOVE_ORIGIN_M, abs=1e-4)
    for teeth in ("16t", "20t", "24t"):
        assert collider_z(teeth) == pytest.approx(
            (-GEAR_HALF_THICKNESS_M, GEAR_HALF_THICKNESS_M), abs=1e-4
        )

def test_scalar_layout_configuration_is_unchanged_without_explicit_names():
    from isaaclab_arena_vabar.tasks.gear_insertion.parallel_layouts import (
        configure_parallel_layouts,
    )

    cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=1))
    assert (
        configure_parallel_layouts(cfg, family="pair", gear_names=("gear_a", "gear_b"))
        is cfg
    )
    assert vars(cfg) == {"scene": cfg.scene}
