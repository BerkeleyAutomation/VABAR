# SPDX-License-Identifier: Apache-2.0
"""On a GPU: a config's physics override reaches pool workers, which are fresh interpreters.

    VABAR_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=0 uv run pytest tests/test_gpu_config.py

Builds tool_hanging/wrenches_level1 in two pool workers and steps it five idle
steps, under ``paper`` (the variant's own tool_hanging_480hz@v1: 8 substeps) and
under a config that overrides the variant's physics to tool_hanging_960hz@v1
(16 substeps). Each worker imports vabar only through the suite's staging.py,
so the 16 in its ``run_meta.json`` build is the config arriving through
$VABAR_CONFIG and the registry hook. A few minutes on a warm cache.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(os.environ.get("VABAR_GPU_TESTS") != "1", reason="set VABAR_GPU_TESTS=1 on a GPU")


def pooled_substeps(tmp_path: Path, name: str, config: str) -> list[int]:
    cmd = [sys.executable, "-m", "vabar", "run", "--config", config, "--task", "tool_hanging/wrenches_level1"]
    cmd += ["--solve", "idle", "--workers", "2", "--episodes", "0:2", "--max-steps", "5"]
    cmd += ["--out", str(tmp_path), "--name", name]
    env = {k: v for k, v in os.environ.items() if k not in ("VABAR_CONFIG", "VABAR_SET")}
    subprocess.run(cmd, cwd=REPO, env=env, check=False, timeout=1800)
    meta = json.loads((tmp_path / name / "run_meta.json").read_text())
    assert meta["provenance"]["pooled"] and meta["provenance"]["workers"] == 2
    provenance = json.loads((tmp_path / name / "vabar_config.json").read_text())
    assert provenance["vabar_rev"] != "unknown"
    return [b["substeps"] for b in meta["provenance"]["builds"]]


def test_pool_workers_build_the_configs_physics(tmp_path):
    override = tmp_path / "th960.toml"
    override.write_text(
        'inherits = "default"\n[suites."tool_hanging/wrenches_level1"]\nphysics = "tool_hanging_960hz@v1"\n'
    )
    assert pooled_substeps(tmp_path, "paper", "paper") == [8, 8]
    assert pooled_substeps(tmp_path, "override", str(override)) == [16, 16]
