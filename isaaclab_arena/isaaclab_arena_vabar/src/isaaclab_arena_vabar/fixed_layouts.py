# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay each environment's first placement layout on every later reset.

Arena 0.3.1's ``placement_reset`` consumes one pooled layout per reset and solves
fresh layouts once an env's queue runs dry. Scenes that rewrite the solved layouts
after the solve (gear source rows, AUTOLab tray trials) are fixed starts: a fresh
solve skips the rewrite, so it either fails outright or returns the template's
poses. :func:`replay_first_layouts` swaps the event's function for
:class:`ReplayFirstLayouts`, which draws each env's layout from the pool once, on
that env's first reset, and writes the same layout on every reset after it.

The event keeps its name and its ``placement_pool`` handle, so Arena's
``get_placement_pool`` and ``get_reset_placement_results`` work unchanged, and a
later reset term such as ``random_around_solution`` still perturbs around the
replayed layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg
    from isaaclab_arena.relations.placement_events import PlacementPoolHandle
    from isaaclab_arena.relations.placement_result import PlacementResult


class ReplayFirstLayouts(ManagerTermBase):
    """Pooled placement reset that keeps the layout each env drew first."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.first_layouts: dict[int, PlacementResult] = {}
        """Layout each absolute env id drew on its first reset."""

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        placement_pool: PlacementPoolHandle,
    ) -> None:
        from isaaclab_arena.relations.placement_events import (
            get_base_rotation_per_asset,
            write_layout_to_sim,
        )
        from isaaclab_arena.relations.relations import get_anchor_objects

        if env_ids is None or len(env_ids) == 0:
            return
        pool = placement_pool.pool
        assert pool.num_envs == env.scene.env_origins.shape[0], (
            f"Placement pool has {pool.num_envs} envs, but the scene has "
            f"{env.scene.env_origins.shape[0]} env origins."
        )
        reset_env_ids = env_ids.tolist()
        first_draws = [env_id for env_id in reset_env_ids if env_id not in self.first_layouts]
        if first_draws:
            self.first_layouts.update(pool.sample_for_envs(first_draws))
        assets = pool.objects
        anchor_assets = set(get_anchor_objects(assets))
        base_rotations = get_base_rotation_per_asset(assets)
        results = {env_id: self.first_layouts[env_id] for env_id in reset_env_ids}
        for env_id, layout in results.items():
            if env_id in first_draws and not layout.success:
                failed = layout.validation_results.get_failed_validation_check_names
                print(f"Warning: Writing best-loss fallback placement for env {env_id}; failed checks: {failed}.")
            write_layout_to_sim(env, env_id, layout, anchor_assets, base_rotations)
        placement_pool.last_results = results


def replay_first_layouts(env_cfg):
    """Make ``env_cfg``'s pooled placement reset replay each env's first layout."""
    from isaaclab_arena.relations.placement_events import (
        PLACEMENT_RESET_EVENT_NAME,
        solve_and_place_objects,
    )

    event = getattr(env_cfg.events, PLACEMENT_RESET_EVENT_NAME)
    if event.func is not ReplayFirstLayouts:
        # Fail loudly if Arena's placement reset changes shape under this override.
        assert event.func is solve_and_place_objects, (
            f"'{PLACEMENT_RESET_EVENT_NAME}' runs {event.func!r}, not Arena's pooled placement reset"
        )
        assert set(event.params) == {"placement_pool"}, (
            f"'{PLACEMENT_RESET_EVENT_NAME}' params changed: {sorted(event.params)}"
        )
        event.func = ReplayFirstLayouts
    return env_cfg
