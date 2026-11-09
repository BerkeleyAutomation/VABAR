# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry point for the shared benchmark camera contract."""

from __future__ import annotations

from typing import Any


def configure_newton_camera_renderers(env_cfg: Any) -> int:
    """Apply the repository-wide Arena RTX camera contract."""
    # GaP-side tests and camera-free stand-ins do not install the benchmark
    # package. Preserve that lightweight import boundary until a real scene
    # needs camera finalization.
    if not hasattr(env_cfg, "scene"):
        return 0

    from isaaclab_arena_vabar.camera_contract import configure_camera_contract

    return configure_camera_contract(env_cfg)
