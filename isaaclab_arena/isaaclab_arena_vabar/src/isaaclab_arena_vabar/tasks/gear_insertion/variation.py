"""Coupled build-time asset variation for the easy gear-mesh family."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import field
import json
from pathlib import Path
import random

from isaaclab.utils.configclass import configclass
from isaaclab_arena.variations.choice_sampler import ChoiceSampler, ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import (
    BuildTimeVariationBase,
    VariationBaseCfg,
)

from isaaclab_arena_vabar.assets.gear_insertion import (
    GEAR_MESH_ASSET_PATHS,
)

_SURFACE_ORIGIN_ARENA = (0.0990017409436448, 0.0155650225211777)
_VARIANTS = json.loads(Path(__file__).with_name("variants.json").read_text())[
    "variants"
]


class SeededChoiceSampler(ChoiceSampler):
    """Arena's choice sampler, drawing from ``rng`` when one is set.

    Arena's ``ChoiceSampler`` draws from torch's global RNG, and build-time
    variations run before the builder applies the experiment seed, so the same
    seed drew a different gear layout in every process.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rng: random.Random | None = None

    def _sample(self, num_samples: int, choices: Sequence) -> list:
        if self.rng is None:
            return super()._sample(num_samples, choices)
        assert num_samples >= 0 and len(choices) >= 1
        return [choices[self.rng.randrange(len(choices))] for _ in range(num_samples)]


@configclass
class SeededChoiceSamplerCfg(ChoiceSamplerCfg):
    """Build a :class:`SeededChoiceSampler`."""

    def build(self) -> SeededChoiceSampler:
        return SeededChoiceSampler()


SeedSource = Callable[[], int | None]


class _SeededBuildTimeVariation(BuildTimeVariationBase):
    """Draw one build-time choice keyed on the run's seed.

    ``seed_source`` returns the seed when the draw happens; the gear environments
    read the placement seed, which the builder applies before build-time
    variations. Without a seed the draw falls back to Arena's global RNG.
    """

    def __init__(self, cfg, name: str, seed_source: SeedSource | None):
        super().__init__(cfg, name=name)
        self._seed_source = seed_source

    def _draw(self, choices: Sequence):
        seed = None if self._seed_source is None else self._seed_source()
        sampler = self.sampler
        assert sampler is not None
        if isinstance(sampler, SeededChoiceSampler):
            sampler.rng = None if seed is None else random.Random(f"{self.name}:{int(seed)}")
        return sampler.sample(num_samples=1, choices=choices)[0]


@configclass
class GearFamilyVariationCfg(VariationBaseCfg):
    """Configuration for paired gear and board tooth-count sampling."""

    enabled: bool = True
    tooth_counts: list[int] = field(default_factory=lambda: [16, 20, 24])
    sampler_cfg: ChoiceSamplerCfg = field(default_factory=SeededChoiceSamplerCfg)


class GearFamilyVariation(_SeededBuildTimeVariation):
    """Select one matching gear/board family before placement is solved."""

    cfg: GearFamilyVariationCfg

    def __init__(
        self,
        board,
        gear,
        task,
        cfg: GearFamilyVariationCfg | None = None,
        seed_source: SeedSource | None = None,
    ):
        super().__init__(cfg or GearFamilyVariationCfg(), "gear_family", seed_source)
        self._board = board
        self._gear = gear
        self._task = task

    @staticmethod
    def _set_usd(asset, path) -> None:
        asset.usd_path = str(path)
        asset.object_cfg.spawn.usd_path = str(path)
        asset.bounding_box = None

    @classmethod
    def _set_board_layout(cls, board, layout: str) -> None:
        cls._set_usd(board, GEAR_MESH_ASSET_PATHS["board"])
        variants = {"layout": layout}
        board.spawn_cfg_addon = {
            **getattr(board, "spawn_cfg_addon", {}),
            "variants": variants,
        }
        board.object_cfg.spawn.variants = variants

    def _realize_at_build_time(self) -> None:
        tooth_counts = [int(value) for value in self.cfg.tooth_counts]
        if not tooth_counts or any(value not in (16, 20, 24) for value in tooth_counts):
            raise ValueError("gear_family tooth_counts must contain only 16, 20, or 24")
        teeth = int(self._draw(tooth_counts))

        self._set_usd(self._gear, GEAR_MESH_ASSET_PATHS[f"{teeth}t"])
        self._set_board_layout(self._board, f"board_{teeth}")
        self._task.set_gear_teeth(teeth)

        radius = 0.0025 * (teeth + 2) / 2.0
        x_half = 0.075 - radius - 0.008
        y_half = 0.135 - radius - 0.008
        limits = next(
            relation
            for relation in self._gear.get_relations()
            if relation.name == "position_limits_box"
        )
        center_x, center_y = -0.0309982590563552, 0.0155650225211777
        limits.x_min, limits.x_max = center_x - x_half, center_x + x_half
        limits.y_min, limits.y_max = center_y - y_half, center_y + y_half


@configclass
class GearLayoutVariationCfg(VariationBaseCfg):
    """Configuration for selecting one exact generated RSS layout."""

    enabled: bool = True
    family: str = "pair"
    sampler_cfg: ChoiceSamplerCfg = field(default_factory=SeededChoiceSamplerCfg)


class GearLayoutVariation(_SeededBuildTimeVariation):
    """Select a coupled pair/train board, gear set, and source-authored poses."""

    cfg: GearLayoutVariationCfg

    def __init__(
        self,
        board,
        gears,
        task,
        cfg: GearLayoutVariationCfg,
        seed_source: SeedSource | None = None,
    ):
        super().__init__(cfg, "gear_layout", seed_source)
        self._board = board
        self._gears = tuple(gears)
        self._task = task

    @staticmethod
    def _relation(asset, name):
        return next(
            relation for relation in asset.get_relations() if relation.name == name
        )

    @classmethod
    def _set_pose(cls, asset, xy, yaw) -> None:
        position = cls._relation(asset, "at_position")
        position.x = _SURFACE_ORIGIN_ARENA[0] + float(xy[0])
        position.y = _SURFACE_ORIGIN_ARENA[1] + float(xy[1])
        cls._relation(asset, "rotate_around_solution").yaw_rad = float(yaw)

    def _realize_at_build_time(self) -> None:
        rows = {
            row["name"]: row for row in _VARIANTS if row["family"] == self.cfg.family
        }
        if self.cfg.family not in {"pair", "train"} or not rows:
            raise ValueError("gear_layout family must be 'pair' or 'train'")
        selected_variant = self._draw(tuple(rows))
        row = rows[selected_variant]
        if len(row["gears"]) != len(self._gears):
            raise ValueError("selected gear layout does not match the scene gear count")

        GearFamilyVariation._set_board_layout(self._board, row["board_variant"])
        self._set_pose(self._board, row["board_xy"], row["board_yaw"])
        for asset, gear in zip(self._gears, row["gears"], strict=True):
            GearFamilyVariation._set_usd(
                asset, GEAR_MESH_ASSET_PATHS[f"{int(gear['teeth'])}t"]
            )
            self._set_pose(asset, gear["xy"], gear["yaw"])
        self._task.configure_layout(row["stations"], row["target_offsets_xyz"])
        self.selected_variant = selected_variant
