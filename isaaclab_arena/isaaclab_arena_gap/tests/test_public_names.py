"""The published tree names only the VABAR repository it ships in.

This folder is published as ``isaaclab_arena/`` of the VABAR repository. Its tracked text, outside
the immutable ``external/`` pins, must not name the private development repository or its
organization, and calls the enclosing repository "the VABAR repository".
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Assembled from parts so this file does not match its own patterns.
_FORBIDDEN = (
    "vabar[-_ ]?" + "isaaclab[-_ ]?arena",
    "nvidia" + "-isaac",
    "autolab('s)?( vabar)? " + "repo",
)


@pytest.mark.parametrize("pattern", _FORBIDDEN)
def test_tracked_text_avoids_private_names(pattern: str) -> None:
    if shutil.which("git") is None or subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--git-dir"], capture_output=True
    ).returncode:
        pytest.skip("not a git checkout")
    result = subprocess.run(
        ["git", "-C", str(ROOT), "grep", "-n", "-I", "-i", "-E", pattern,
         "--", ".", ":(exclude)external"],
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1), result.stderr
    assert result.returncode == 1, f"tracked text matches {pattern!r}:\n{result.stdout}"
