"""AUTOLab's sources resolve alike in shell and Python, as a gitlink or as the enclosing repository."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from isaaclab_arena_gap import task_sources

ROOT = Path(__file__).resolve().parents[2]
LAYOUT = ROOT / "isaaclab_arena_gap/scripts/autolab_layout.sh"


def _layout(tmp_path: Path, gitlink: bool) -> Path:
    autolab = tmp_path / "autolab"
    cap = autolab / "isaaclab_arena"
    (cap / "external/gap").mkdir(parents=True)
    if gitlink:
        (cap / "external/gap/autolab-vabar").mkdir()
    return cap


def _shell(cap: Path, **env: str) -> tuple[Path, Path]:
    base = {k: v for k, v in os.environ.items() if not k.startswith("AUTOLAB_VABAR_")}
    out = subprocess.run(
        ["bash", "-c", f'. "{LAYOUT}"; resolve_autolab_layout "$1"; '
         'printf "%s\\n%s\\n" "$AUTOLAB_VABAR_ROOT" "$AUTOLAB_VABAR_VENV"', "_", str(cap)],
        env={**base, **env}, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return Path(out[0]), Path(out[1])


def _python(cap: Path, monkeypatch: pytest.MonkeyPatch, **env: str) -> tuple[Path, Path]:
    monkeypatch.setattr(task_sources, "WORKSPACE_ROOT", cap)
    monkeypatch.setattr(task_sources, "_AUTOLAB_GITLINK", cap / "external/gap/autolab-vabar")
    for key in ("AUTOLAB_VABAR_ROOT", "AUTOLAB_VABAR_VENV"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return task_sources.autolab_root(), task_sources.autolab_python().parent.parent


@pytest.mark.parametrize("gitlink", [True, False], ids=["gitlink", "enclosing"])
def test_shell_and_python_resolve_the_same_layout(tmp_path, monkeypatch, gitlink) -> None:
    cap = _layout(tmp_path, gitlink)
    if gitlink:
        expected = (cap / "external/gap/autolab-vabar", cap / "external/gap/autolab-vabar/.venv")
    else:
        # The repository root's own .venv keeps its default groups; the adapter's lives beside it.
        expected = (cap.parent, cap / ".venv-autolab")
    shell_root, shell_venv = _shell(cap)
    assert (shell_root.resolve(), shell_venv.resolve()) == tuple(p.resolve() for p in expected)
    py_root, py_venv = _python(cap, monkeypatch)
    assert (py_root.resolve(), py_venv.resolve()) == tuple(p.resolve() for p in expected)


def test_an_explicit_root_keeps_its_own_environment(tmp_path, monkeypatch) -> None:
    cap = _layout(tmp_path, gitlink=False)
    elsewhere = tmp_path / "checkout"
    elsewhere.mkdir()
    assert _shell(cap, AUTOLAB_VABAR_ROOT=str(elsewhere)) == (elsewhere, elsewhere / ".venv")
    assert _python(cap, monkeypatch, AUTOLAB_VABAR_ROOT=str(elsewhere)) == (
        elsewhere, elsewhere / ".venv"
    )


def test_image_build_takes_autolab_from_a_named_context() -> None:
    build = (ROOT / "docker/build.sh").read_text(encoding="utf-8")
    dockerfile = (ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert '--build-context "autolab=$CAP_AUTOLAB_CONTEXT"' in build
    assert "COPY --from=autolab . ${AUTOLAB_VABAR_ROOT}/" in dockerfile
    # The in-image location stays where the gitlink layout always put it.
    assert "AUTOLAB_VABAR_ROOT=/workspace/isaaclab_arena/external/gap/autolab-vabar" in dockerfile
    # Only the named context carries AUTOLab, in either layout.
    assert "external/gap/autolab-vabar" in ignored
