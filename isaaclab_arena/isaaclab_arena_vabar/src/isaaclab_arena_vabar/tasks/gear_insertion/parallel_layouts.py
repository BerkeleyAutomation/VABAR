"""Assign complete RSS gear layouts to independent Arena slots."""

from __future__ import annotations

import copy

from .variation import _SURFACE_ORIGIN_ARENA, _VARIANTS

# Collider geometry of the gear-mesh assets: the kinematic mat's top face is at
# its origin, and every gear is 16 mm thick about its origin.
MAT_TOP_ABOVE_ORIGIN_M = 0.0
GEAR_HALF_THICKNESS_M = 0.008
# Gears start just above the mat and settle; the relation solver's support
# heights are only within its 5 mm "on" tolerance, and a gear starting 4 mm
# inside the mat is thrown out of the scene on the first physics step.
GEAR_SPAWN_CLEARANCE_M = 0.0005


def coupled_layout_clones(combinations, num_clones, device):
    """Keep board and gear variant indices coupled in the clone planner."""
    varying = combinations.amax(dim=0) > 0
    columns = combinations[:, varying]
    if not columns.numel():
        return combinations[:1].expand(num_clones, -1).to(device)
    rows = combinations[(columns == columns[:, :1]).all(dim=1)]
    if len(rows) != num_clones:
        raise ValueError("gear clone variants must match the environment count")
    return rows.to(device)


def apply_layout_poses(layout, row, gear_names):
    """Replace source XY/yaw; rest each gear on the mat's top face."""
    by_name = {asset.name: asset for asset in layout.positions}
    gear_z = (
        layout.positions[by_name["gear_mat"]][2]
        + MAT_TOP_ABOVE_ORIGIN_M
        + GEAR_HALF_THICKNESS_M
        + GEAR_SPAWN_CLEARANCE_M
    )
    for index, name in enumerate(("board", *gear_names)):
        asset = by_name[name]
        source = (
            {"xy": row["board_xy"], "yaw": row["board_yaw"]}
            if index == 0
            else row["gears"][index - 1]
        )
        layout.positions[asset] = (
            _SURFACE_ORIGIN_ARENA[0] + source["xy"][0],
            _SURFACE_ORIGIN_ARENA[1] + source["xy"][1],
            layout.positions[asset][2] if index == 0 else gear_z,
        )
        layout.orientations[asset] = source["yaw"]


def apply_selected_layout(env_cfg, *, layout_name, gear_names):
    """Write one variation-selected row's X/Y/yaw into the solved placement pool.

    Arena solves placement before build-time variations run, so the row the
    variation picked never reached the solver. The variation already swapped the
    gear and board assets and configured scoring; only the poses are missing.
    """
    from isaaclab_arena.relations.placement_events import PLACEMENT_RESET_EVENT_NAME

    from isaaclab_arena_vabar.fixed_layouts import replay_first_layouts

    row = next(row for row in _VARIANTS if row["name"] == layout_name)
    event = getattr(env_cfg.events, PLACEMENT_RESET_EVENT_NAME)
    pool = event.params["placement_pool"].pool
    for layouts in pool.layouts_per_env():
        for layout in layouts:
            apply_layout_poses(layout, row, gear_names)
    # The row is a fixed start: a refill would solve the template's poses.
    return replay_first_layouts(env_cfg)


def record_gear_layout(env, env_id, layout_names):
    """Stamp the RSS source row into Arena's authoritative episode record."""
    return {"gear_layout": layout_names[env_id]}


def gear_success_params(env_cfg):
    """Return the gear-mesh predicate's params inside Arena's task-progress criteria.

    Arena's success term (``task_success_from_progress``) takes no params; the
    criteria live on ``env_cfg.task_progress`` and the progress tracker builds its
    predicates from them when the environment starts.
    """
    from .terminations import gear_mesh_success

    progress = getattr(env_cfg, "task_progress", None)
    if progress is None:
        raise ValueError("gear mesh layouts need Arena task progress to carry the success criteria")
    for criteria in progress.success_criteria:
        for sequence in criteria.canonical_predicate_sequences.values():
            for predicate, _score in sequence:
                if getattr(predicate, "func", None) is gear_mesh_success:
                    return predicate.params
    raise ValueError("gear mesh success predicate is missing from the task-progress criteria")


def configure_parallel_layouts(env_cfg, *, family, gear_names, layout_names=()):
    """Bind geometry, placement and scoring to the same ordered source rows."""
    from isaaclab.sim import MultiAssetSpawnerCfg
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena.relations.placement_events import PLACEMENT_RESET_EVENT_NAME

    from isaaclab_arena_vabar.assets.gear_insertion import GEAR_MESH_ASSET_PATHS
    from isaaclab_arena_vabar.fixed_layouts import replay_first_layouts

    count = env_cfg.scene.num_envs
    if count == 1 and not layout_names:
        return env_cfg
    if not layout_names:
        raise ValueError(
            "parallel gear layouts require explicit layout_names, one per environment"
        )
    if len(layout_names) != count or len(set(layout_names)) != count:
        raise ValueError(
            "gear layout_names must contain one distinct name per environment"
        )
    available = {row["name"]: row for row in _VARIANTS if row["family"] == family}
    if any(name not in available for name in layout_names):
        raise ValueError(
            "gear layout_names must belong to the selected RSS family"
        )
    rows = [available[name] for name in layout_names]
    for index, name in enumerate(("board", *gear_names)):
        asset = getattr(env_cfg.scene, name)
        variants = []
        for row in rows:
            spawn = copy.deepcopy(asset.spawn)
            if index == 0:
                spawn.usd_path = str(GEAR_MESH_ASSET_PATHS["board"])
                spawn.variants = {"layout": row["board_variant"]}
            else:
                spawn.usd_path = str(
                    GEAR_MESH_ASSET_PATHS[f"{row['gears'][index - 1]['teeth']}t"]
                )
            variants.append(spawn)
        asset.spawn = MultiAssetSpawnerCfg(assets_cfg=variants, random_choice=False)
    env_cfg.scene.clone_cfg.clone_strategy = coupled_layout_clones
    event = getattr(env_cfg.events, PLACEMENT_RESET_EVENT_NAME)
    pool = event.params["placement_pool"].pool
    for layouts, row in zip(pool.layouts_per_env(), rows, strict=True):
        for layout in layouts:
            apply_layout_poses(layout, row, gear_names)
    # Source rows are fixed starts: a refill would solve the scalar template's poses.
    replay_first_layouts(env_cfg)
    env_cfg.episode_recorders.gear_layout = EpisodeRecorderTermCfg(
        func=record_gear_layout, params={"layout_names": list(layout_names)}
    )
    gear_success_params(env_cfg).update(
        target_offsets_xyz=[row["target_offsets_xyz"] for row in rows],
        gear_teeth=[row["stations"] for row in rows],
    )
    return env_cfg
