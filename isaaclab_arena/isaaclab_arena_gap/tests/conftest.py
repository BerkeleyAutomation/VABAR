"""Pytest defaults shared with the repository bootstrap."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# bootstrap.sh uses the same defaults for every Isaac Sim installation and run.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
# Newton's USD parser can abort while resolving heavily instanced assets in
# parallel. Keep all adapter USD-import tests on the established safe path.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "isaaclab_arena_gap/scripts/run_rss_graph.sh"


@pytest.fixture
def git_commit():
    """Return ``commit(repo, message)``: init if needed, commit everything, return HEAD."""

    def commit(repo: Path, message: str = "fixture") -> str:
        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        if not (repo / ".git").exists():
            git("init", "-q")
        git("add", "-A")
        git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-q",
            "-m",
            message,
        )
        return git("rev-parse", "HEAD")

    return commit


_FAKE_PYTHON = """#!{python}
import json, os, sys
print("FAKE_PYTHON=" + json.dumps({{
    "cwd": os.getcwd(),
    "argv": sys.argv[1:],
    "env": {{k: v for k, v in os.environ.items()
            if k in ("GAP_SKILLS_PATH", "GAP_TRACE_DIR", "PYTHONPATH") or k.startswith("GAP_VLM_")}},
}}))
"""


@pytest.fixture
def run_launcher(tmp_path):
    """Run ``run_rss_graph.sh`` against a fake AUTOLab Python that reports its invocation.

    Returns ``launch(*args, **env)`` -> ``(CompletedProcess, record)`` where
    ``record`` is the fake interpreter's ``{"cwd", "argv", "env"}`` or ``None``
    if it was never reached.
    """
    autolab = tmp_path / "fake-autolab"
    python = autolab / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(_FAKE_PYTHON.format(python=sys.executable))
    python.chmod(0o755)
    credentials = tmp_path / "empty-vlm.env"
    credentials.touch()
    base = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GAP_") and key not in {"CAP_BAKED_RUNTIME", "PYTHONPATH"}
    }
    base.update(
        AUTOLAB_VABAR_ROOT=str(autolab),
        SKILLS_ROOT=str(tmp_path / "skills"),
        GAP_GRAPH="tool_hanging/gap_perception",
        GAP_TRACE_DIR=str(tmp_path / "trace"),
        GAP_DRY_RUN="0",
        GAP_VLM_ENV=str(credentials),
    )

    def launch(*args: str, **env: str):
        result = subprocess.run(
            [str(LAUNCHER), *args],
            env={**base, **env},
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        record = next(
            (
                json.loads(line.removeprefix("FAKE_PYTHON="))
                for line in result.stdout.splitlines()
                if line.startswith("FAKE_PYTHON=")
            ),
            None,
        )
        return result, record

    return launch
