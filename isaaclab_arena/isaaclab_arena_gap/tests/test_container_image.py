from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/IsaacLab-Arena"))

from isaaclab_arena_gap.container_cli import build_run_command, load_manifest


def _docker_module(name: str):
    """Load a Docker build helper by path; a top-level `docker` module would shadow the Docker SDK."""
    path = ROOT / f"docker/{name}.py"
    spec = importlib.util.spec_from_file_location(f"_docker_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_container_run_interface_hides_launcher_environment(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("runs: {}\n", encoding="utf-8")
    output = tmp_path / "outputs"
    command, environment = build_run_command(
        ROOT,
        experiment,
        output,
        record_camera_video=False,
        runner_args=("--headless",),
    )
    assert command == [
        str(ROOT / "isaaclab_arena_gap/scripts/run_arena_eval.sh"),
        "--headless",
    ]
    assert environment == {
        "ARENA_EXPERIMENT_CONFIG": str(experiment),
        "ARENA_OUTPUT_DIR": str(output),
        "ARENA_RECORD_CAMERA_VIDEO": "0",
    }


def test_image_manifest_schema_fails_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema_version": 2}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsupported image manifest schema"):
        load_manifest(manifest)


def test_dirty_source_manifest_records_and_verifies_runtime_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git_commit
) -> None:
    source_manifest = _docker_module("source_manifest")

    source = tmp_path / "source"
    source.mkdir()
    tracked = source / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    git_commit(source)
    tracked.write_text("dirty\n", encoding="utf-8")
    (source / "untracked.txt").write_text("included\n", encoding="utf-8")
    (source / "notes.txt").write_text("not copied into the image\n", encoding="utf-8")
    monkeypatch.setattr(source_manifest, "REQUIRED_GITLINKS", ())

    metadata = tmp_path / "metadata"
    manifest = source_manifest.create_manifest(
        source, metadata, runtime_paths=("tracked.txt", "untracked.txt")
    )
    assert manifest["dirty"] is True
    assert set(manifest["dirty_files"]) == {"tracked.txt", "untracked.txt"}
    assert manifest["runtime_paths"] == ["tracked.txt", "untracked.txt"]
    assert (metadata / "dirty.patch").read_bytes()
    (source / "notes.txt").write_text("still irrelevant\n", encoding="utf-8")
    source_manifest.verify_manifest(source, metadata / "source-manifest.json")

    tracked.write_text("different\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs at tracked.txt"):
        source_manifest.verify_manifest(source, metadata / "source-manifest.json")


def _build_runtime_paths() -> tuple[str, ...]:
    script = (ROOT / "docker/build.sh").read_text(encoding="utf-8")
    return tuple(re.findall(r"--runtime-path (\S+)", script))


def test_dirty_gate_covers_external_files_copied_into_the_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git_commit
) -> None:
    source_manifest = _docker_module("source_manifest")

    runtime_paths = _build_runtime_paths()
    # external/ holds only gitlinks today; a tracked file stands in for any added later.
    tracked_files = ["external/pinned.txt"]
    assert "external" in runtime_paths

    source = tmp_path / "source"
    (source / "external").mkdir(parents=True)
    for path in tracked_files:
        (source / path).write_text("pinned\n", encoding="utf-8")
    git_commit(source)
    monkeypatch.setattr(source_manifest, "REQUIRED_GITLINKS", ())
    untracked = ["external/scratch.txt", "external/gap/notes.md"]
    for path in [*tracked_files, *untracked]:
        subprocess.run(["git", "-C", str(source), "checkout", "-q", "."], check=True)
        subprocess.run(["git", "-C", str(source), "clean", "-qfd"], check=True)
        (source / path).parent.mkdir(parents=True, exist_ok=True)
        (source / path).write_text("edited\n", encoding="utf-8")
        manifest = source_manifest.create_manifest(
            source, tmp_path / "metadata", runtime_paths=runtime_paths
        )
        assert manifest["dirty"] is True, path
        assert set(manifest["dirty_files"]) == {path}


DOCKERFILE = ROOT / "docker/Dockerfile"


def test_overlay_dependency_extraction_excludes_local_distributions(
    tmp_path: Path,
) -> None:
    installer = _docker_module("install_overlay_dependencies")
    BUILD_REQUIREMENTS = installer.BUILD_REQUIREMENTS
    declared_dependencies = installer.declared_dependencies

    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        """
[project]
name = "example"
dependencies = [
  "isaaclab-arena-vabar",
  "msgpack>=1",
  "Some_Package[extra]~=2.0; python_version >= '3.11'",
]
""",
        encoding="utf-8",
    )
    assert declared_dependencies([manifest]) == [
        "msgpack>=1",
        "Some_Package[extra]~=2.0; python_version >= '3.11'",
    ]
    assert "editables" in BUILD_REQUIREMENTS


def test_overlay_install_keeps_declared_simulator_pin(tmp_path, monkeypatch):
    installer = _docker_module("install_overlay_dependencies")

    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        '[project]\nname="example"\ndependencies=["mujoco-warp~=3.11.0"]\n'
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["install", "--python", "runtime-python", str(manifest)],
    )
    calls = []
    monkeypatch.setattr(
        installer.subprocess, "run", lambda argv, **kwargs: calls.append(argv)
    )
    installer.main()
    assert calls[0][:5] == ["uv", "pip", "install", "--python", "runtime-python"]
    assert "mujoco-warp~=3.11.0" in calls[0]


def test_container_runner_publishes_image_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from isaaclab_arena_gap.container_runner import (
        output_directory_from_args,
        publish_image_provenance,
    )

    output = tmp_path / "output"
    output.mkdir()
    run_output = output / "hammer"
    run_output.mkdir()
    manifest = tmp_path / "image-manifest.json"
    manifest.write_text(
        json.dumps({"source": {"source_fingerprint": "abc"}}), encoding="utf-8"
    )
    monkeypatch.setenv("CAP_IMAGE_DIGEST", "registry/cap@sha256:" + "a" * 64)
    assert (
        output_directory_from_args(["--experiment_output_directory", str(output)])
        == output
    )
    publish_image_provenance(output, manifest)
    run_provenance = json.loads(
        (output / "run-provenance.json").read_text(encoding="utf-8")
    )
    assert run_provenance["source_fingerprint"] == "abc"
    assert run_provenance["image_digest"].endswith("a" * 64)
    assert (run_output / "image-manifest.json").is_file()
    assert (run_output / "run-provenance.json").is_file()
