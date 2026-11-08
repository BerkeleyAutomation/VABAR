"""Shared camera and recording contract for industrial benchmark tasks."""

from __future__ import annotations

from typing import Any

RESET_CAMERA_WARMUP_EVENT = "rtx_camera_warmup"
"""Name of the reset event that converges the RTX cameras on the reset scene."""

RESET_CAMERA_WARMUP_FRAMES = 60
"""RTX frames rendered after each reset, before the episode's first observation.

A frame rendered right after a reset is soft and loses thin detail; the
benchmark's RTX cameras converge after 20 to 40 frames, so 60 leaves margin.
Isaac Lab's ``num_rerenders_on_reset`` cannot do this headless:
``SimulationContext.render`` syncs poses and steps visualizers but never runs
Kit's update, and RTX renders a frame only when a camera reads its data, so
the configured rerenders collapse into one frame.
"""


def warm_up_rtx_cameras(env: Any, env_ids: Any, num_frames: int) -> None:
    """Render ``num_frames`` RTX frames of the reset scene without stepping physics.

    Runs as the last reset event, once per reset batch. Each iteration syncs
    the reset poses to the renderer and starts a new render generation
    (``sim.render``), then renders one frame. Simulation time and the episode's
    step count do not change. In a vector run, environments that are not
    resetting also receive these frames, at their unchanged poses.
    """
    del env_ids  # RTX renders every environment's view in the same frame.
    if num_frames <= 0 or not getattr(env, "has_rtx_sensors", False):
        return
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import (
        ensure_isaac_rtx_render_update,
    )

    for _ in range(num_frames):
        env.sim.render()
        ensure_isaac_rtx_render_update(force=True)


def install_reset_camera_warmup(
    env_cfg: Any, num_frames: int = RESET_CAMERA_WARMUP_FRAMES
) -> None:
    """Add the warm-up as the last reset event, after every pose-setting term."""
    from isaaclab.managers import EventTermCfg

    valid = isinstance(num_frames, int) and not isinstance(num_frames, bool)
    if not valid or num_frames < 0:
        raise ValueError("reset camera warm-up frames must be a nonnegative integer")
    events = env_cfg.events
    if hasattr(events, RESET_CAMERA_WARMUP_EVENT):
        # Re-adding moves the term behind any reset event added since.
        delattr(events, RESET_CAMERA_WARMUP_EVENT)
    setattr(
        events,
        RESET_CAMERA_WARMUP_EVENT,
        EventTermCfg(
            func=warm_up_rtx_cameras, mode="reset", params={"num_frames": num_frames}
        ),
    )


def configure_camera_contract(env_cfg: Any) -> int:
    """Apply the common Arena RTX backend without replacing camera calibration.

    Scenes with cameras also get the reset warm-up, so an episode's first
    observation shows the reset scene as converged as a later step does.
    """
    if not hasattr(env_cfg, "scene"):
        return 0

    from isaaclab.sensors import CameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    configured = 0
    for entity_cfg in vars(env_cfg.scene).values():
        if isinstance(entity_cfg, CameraCfg):
            entity_cfg.renderer_cfg = IsaacRtxRendererCfg()
            configured += 1
    if configured:
        install_reset_camera_warmup(env_cfg)
    return configured
