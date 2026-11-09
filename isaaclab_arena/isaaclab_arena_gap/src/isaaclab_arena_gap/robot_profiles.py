"""Supported ``robot_profile`` values.

Each is a robot the RSS adapter binds in
:data:`isaaclab_arena_gap.rss_adapter.runner.ROBOTS`.
"""

from __future__ import annotations

BIMANUAL_YAM_PROFILES = ("yam_bimanual", "yam_bimanual_wide", "yam_bimanual_linear")
"""Profiles served by ``cap_bimanual_yam_env`` (two arms, explicit base poses)."""

ROBOT_PROFILES = ("fr3", *BIMANUAL_YAM_PROFILES)
"""Every supported profile."""
