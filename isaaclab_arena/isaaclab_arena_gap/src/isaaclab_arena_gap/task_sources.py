"""Arena robot assets the RSS adapter builds its embodiments from, and AUTOLab's sources."""

from __future__ import annotations

import os
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_AUTOLAB_GITLINK = WORKSPACE_ROOT / "external/gap/autolab-vabar"


def autolab_root() -> Path:
    """AUTOLab v1.0's tasks, graphs and lock, resolved as ``scripts/autolab_layout.sh`` does.

    ``AUTOLAB_VABAR_ROOT`` wins; otherwise the ``external/gap/autolab-vabar`` gitlink, or, when
    this tree is a folder of the VABAR repository, the enclosing directory.
    """
    if explicit := os.environ.get("AUTOLAB_VABAR_ROOT"):
        return Path(explicit)
    return _AUTOLAB_GITLINK if _AUTOLAB_GITLINK.exists() else WORKSPACE_ROOT.parent


def autolab_python() -> Path:
    """The interpreter of the environment the RSS adapter runs in (``AUTOLAB_VABAR_VENV``)."""
    if explicit := os.environ.get("AUTOLAB_VABAR_VENV"):
        return Path(explicit) / "bin/python"
    root = autolab_root()
    if root.resolve() == WORKSPACE_ROOT.parent.resolve():
        return WORKSPACE_ROOT / ".venv-autolab/bin/python"
    return root / ".venv/bin/python"


def fr3_robotiq_usd_path() -> Path:
    return (
        WORKSPACE_ROOT
        / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets"
        / "vabar__fr3_robotiq_2f85/franka_fr3_robotiq_2f85.usda"
    )


def yam_usd_path() -> Path:
    """Arena's YAM USD; the RSS adapter selects each arm's ``gripper`` variant itself."""
    return (
        WORKSPACE_ROOT
        / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets"
        / "vabar__i2rt_yam" / "i2rt_yam_default.usda"
    )
