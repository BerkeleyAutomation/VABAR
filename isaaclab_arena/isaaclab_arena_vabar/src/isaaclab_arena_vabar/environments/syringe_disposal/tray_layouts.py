"""Lay syringe trays out as AUTOLab's own trials do.

``clutter_layouts.json`` holds the clutter trials, which keep the tray at the YAML's
heading. A trial with fewer syringes than the scene parks the spares (:data:`PARKED_XY`). ``pair_layouts.json`` holds the two-syringe ``tray_pair``
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

PARKED_XY = (0.35, 0.88)
"""Environment-local X/Y where a trial's spare syringe lies, along the table's far corner.

On the work table, out of the overhead and agentview cameras' fields of view and about
1.3 m from the FR3's base, beyond its reach. Spares are laid along world x, side by side."""
PARKED_SPACING_Y = 0.03
"""Y step between parked spares [m]."""
PARKED_YAW = math.pi / 2
"""A syringe's long axis is its local +y; this lays it along world x."""
TRAY_FLOOR_ABOVE_TABLE_M = 0.003
"""The tray's cavity floor above the table [m] (12 mm tray, 9 mm rim-to-floor): a parked spare
lies this much lower than its solved place in the tray."""
PARKED_BOUNDS_XYZXYZ = (0.2, 0.8, 0.7, 0.5, 0.95, 0.9)
"""Environment-local box the scoring treats as "not in this trial" (``parked_bounds_xyzxyz``)."""


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

    A trial that sets the tray's yaw turns the solved tray about its own centre first. A trial
    with fewer syringes than ``syringe_names`` lays out the first ones and parks the rest.
    """
    by_name = {asset.name: asset for asset in layout.positions}
    tray = by_name[TRAY_NAME]
    tx, ty, _ = layout.positions[tray]
    if not 0 < len(chosen.syringes) <= len(syringe_names):
        raise ValueError("tray layout does not match the scene's syringe count")
    if chosen.tray_yaw is not None:
        layout.orientations[tray] = chosen.tray_yaw
    tray_yaw = float(layout.orientations.get(tray, 0.0))
    cos, sin = math.cos(tray_yaw), math.sin(tray_yaw)
    laid_out = syringe_names[: len(chosen.syringes)]
    for name, (x, y, yaw) in zip(laid_out, chosen.syringes, strict=True):
        asset = by_name[name]
        z = layout.positions[asset][2]
        layout.positions[asset] = (tx + cos * x - sin * y, ty + sin * x + cos * y, z)
        layout.orientations[asset] = math.remainder(tray_yaw + yaw, math.tau)
    for slot, name in enumerate(syringe_names[len(chosen.syringes):]):
        asset = by_name[name]
        z = layout.positions[asset][2] - TRAY_FLOOR_ABOVE_TABLE_M
        layout.positions[asset] = (PARKED_XY[0], PARKED_XY[1] - slot * PARKED_SPACING_Y, z)
        layout.orientations[asset] = PARKED_YAW


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
