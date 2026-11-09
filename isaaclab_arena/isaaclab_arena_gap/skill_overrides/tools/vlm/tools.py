"""Load the pinned VLM tool bundle with this folder's qualified-model fix."""

from __future__ import annotations

from pathlib import Path

from isaaclab_arena_gap.vlm_compat import is_gpt5_model, patch_upstream_vlm_source


_CAP_ROOT = Path(__file__).resolve().parents[4]
_UPSTREAM_TOOLS = (
    _CAP_ROOT / "external/gap/open-robot-skills/tools/vlm/tools.py"
)
if not _UPSTREAM_TOOLS.is_file():
    raise RuntimeError(f"missing pinned VLM bundle: {_UPSTREAM_TOOLS}")

_source = patch_upstream_vlm_source(
    _UPSTREAM_TOOLS.read_text(encoding="utf-8")
)
exec(compile(_source, str(_UPSTREAM_TOOLS), "exec"), globals(), globals())
