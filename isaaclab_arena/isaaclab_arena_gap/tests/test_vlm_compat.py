from __future__ import annotations

from pathlib import Path

import pytest

from isaaclab_arena_gap.vlm_compat import is_gpt5_model, patch_upstream_vlm_source


ROOT = Path(__file__).resolve().parents[1]
CAP_ROOT = ROOT.parent


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.6-sol", True),
        ("openai/openai/gpt-5.6-sol", True),
        ("azure/openai/gpt-5.6-terra", True),
        ("google/gemini-3.1-flash-lite-preview", False),
    ],
)
def test_gpt5_detection_uses_final_qualified_model_component(
    model: str, expected: bool
) -> None:
    assert is_gpt5_model(model) is expected


def test_vlm_override_patches_the_exact_pinned_detection_seam() -> None:
    upstream_path = (
        CAP_ROOT
        / "external/gap/open-robot-skills/tools/vlm/tools.py"
    )
    if not upstream_path.is_file():
        pytest.skip("open-robot-skills gitlink is not materialized; run ./setup.sh")
    upstream = upstream_path.read_text(encoding="utf-8")

    patched = patch_upstream_vlm_source(upstream)

    assert 'if is_gpt5_model(model):' in patched
    assert 'if model.lower().startswith("gpt-5"):' not in patched
    # Outlasts a sweep's shared-key 429s instead of failing after ~3 s.
    assert "_MAX_RETRIES = 7\n" in patched
    assert "_MAX_RETRIES = 3\n" not in patched


def test_vlm_override_fails_closed_if_the_pin_changes() -> None:
    with pytest.raises(RuntimeError, match="expected exactly one GPT-5 detection seam"):
        patch_upstream_vlm_source("def unrelated():\n    pass\n")
    with pytest.raises(RuntimeError, match="expected exactly one retry count seam"):
        patch_upstream_vlm_source('if model.lower().startswith("gpt-5"):\n')
