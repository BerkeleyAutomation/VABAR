"""Lay syringe trays out as AUTOLab's own trials do.

``clutter_layouts.json`` holds the six-syringe clutter trials, which keep the tray
at the YAML's heading. ``pair_layouts.json`` holds the two-syringe ``tray_pair``
trials: both barrels across the tray, 90 mm apart along it, with the whole tray
yawed per trial.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

TRAY_NAME = "instrument_tray"
LAYOUT_FILES = ("clutter_layouts.json", "pair_layouts.json")


@dataclass(frozen=True)
class TrayLayout:
    """One AUTOLab trial: tray-frame ``(x, y, yaw)`` per syringe, and the tray's world yaw if the trial sets it."""

    syringes: tuple[tuple[float, float, float], ...]
    tray_yaw: float | None = None


def load_tray_layouts(file_name: str) -> list[TrayLayout]:
    """Return every trial in ``file_name``, one of :data:`LAYOUT_FILES`."""
    if file_name not in LAYOUT_FILES:
        raise ValueError(f"unknown tray layout file {file_name!r}")
    doc = json.loads(Path(__file__).with_name(file_name).read_text())
    return [
        TrayLayout(
            syringes=tuple(tuple(map(float, s)) for s in layout["syringes"]),
            tray_yaw=None if layout.get("tray_yaw_rad") is None else float(layout["tray_yaw_rad"]),
        )
        for layout in doc["layouts"]
    ]


def trial_index(placement_seed: int | None, env_id: int, count: int) -> int:
    """Index of the trial env ``env_id`` starts from under ``placement_seed``.

    Seed ``s`` gives env ``e`` trial ``(s + e) mod count``: one seed always gives
    one trial, any ``count`` consecutive seeds give distinct trials, and env ``e``
    of a vector run starts from the trial a scalar run with seed ``s + e`` does.
    Without a placement seed the index counts from 0.
    """
    return ((placement_seed or 0) + env_id) % count


def apply_layout(layout, syringe_names: tuple[str, ...], chosen: TrayLayout) -> None:
    """Rewrite one solved layout's syringe X/Y/yaw from ``chosen`` and the tray pose; keep their Z.

    A trial that sets the tray's yaw turns the solved tray about its own centre first.
    """
    by_name = {asset.name: asset for asset in layout.positions}
    tray = by_name[TRAY_NAME]
    tx, ty, _ = layout.positions[tray]
    if len(chosen.syringes) != len(syringe_names):
        raise ValueError("tray layout does not match the scene's syringe count")
    if chosen.tray_yaw is not None:
        layout.orientations[tray] = chosen.tray_yaw
    tray_yaw = float(layout.orientations.get(tray, 0.0))
    cos, sin = math.cos(tray_yaw), math.sin(tray_yaw)
    for name, (x, y, yaw) in zip(syringe_names, chosen.syringes, strict=True):
        asset = by_name[name]
        z = layout.positions[asset][2]
        layout.positions[asset] = (tx + cos * x - sin * y, ty + sin * x + cos * y, z)
        layout.orientations[asset] = math.remainder(tray_yaw + yaw, math.tau)


def apply_tray_layouts(env_cfg, *, file_name: str, syringe_names: tuple[str, ...], placer_params):
    """Lay every pooled placement candidate out from the trial its env draws.

    The trial is keyed on the run's placement seed (see :func:`trial_index`), which
    the builder writes onto the environment's ``placer_params`` before it solves.
    """
    from isaaclab_arena.relations.placement_events import PLACEMENT_RESET_EVENT_NAME

    layouts = load_tray_layouts(file_name)
    pool = getattr(env_cfg.events, PLACEMENT_RESET_EVENT_NAME).params["placement_pool"].pool
    for env_id, per_env in enumerate(pool.layouts_per_env()):
        chosen = layouts[trial_index(placer_params.placement_seed, env_id, len(layouts))]
        for layout in per_env:
            apply_layout(layout, syringe_names, chosen)
    # These are fixed starts: a refill would solve layouts that skip the rewrite.
    from isaaclab_arena_vabar.fixed_layouts import replay_first_layouts

    return replay_first_layouts(env_cfg)
