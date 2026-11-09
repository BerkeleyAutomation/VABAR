from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from isaaclab_arena_gap import osmo_cache
from isaaclab_arena_gap.osmo_cache import CacheCompatibility


def _compatibility(**overrides: str) -> CacheCompatibility:
    values = {
        "recipe_epoch": "hold-v1",
        "kit_build": "f9bf0dda",
        "platform": "ovx-l40",
        "driver_branch": "580",
    }
    values.update(overrides)
    return CacheCompatibility(**values)


def test_cache_key_is_task_neutral_and_recipe_configurable() -> None:
    compatibility = _compatibility()
    assert compatibility.key == (
        "schema-1/recipe-hold-v1/kit-f9bf0dda/platform-ovx-l40/driver-580"
    )
    assert compatibility.artifact_url("swift://bucket/cap/cache/") == (
        "swift://bucket/cap/cache/" + compatibility.key
    )
    assert "gear" not in compatibility.key
    with pytest.raises(ValueError, match="driver_branch"):
        _compatibility(driver_branch="unknown")


class _HeadResponse:
    status = 200

    def __enter__(self) -> _HeadResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def test_persistent_artifact_heads_archive_and_manifest_over_https() -> None:
    urls: list[str] = []

    def open_url(request: urllib.request.Request, **_: object) -> _HeadResponse:
        urls.append(request.full_url)
        assert request.method == "HEAD"
        return _HeadResponse()

    assert osmo_cache.persistent_artifact_exists(
        "swift://swift.example.com/AUTH_example/cap/cache", open_url=open_url
    )
    assert urls == [
        "https://swift.example.com/v1/AUTH_example/cap/cache/cache-manifest.json",
        "https://swift.example.com/v1/AUTH_example/cap/cache/cache.tar.zst",
    ]


def test_persistent_artifact_404_is_miss_and_other_errors_surface() -> None:
    def missing(request: urllib.request.Request, **_: object) -> _HeadResponse:
        raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)

    assert not osmo_cache.persistent_artifact_exists(
        "https://swift.example.com/v1/AUTH_example/cache", open_url=missing
    )

    def forbidden(request: urllib.request.Request, **_: object) -> _HeadResponse:
        raise urllib.error.HTTPError(request.full_url, 403, "forbidden", {}, None)

    with pytest.raises(RuntimeError, match="HTTP 403"):
        osmo_cache.persistent_artifact_exists(
            "https://swift.example.com/v1/AUTH_example/cache", open_url=forbidden
        )


def test_publish_then_restore_verifies_manifest_and_extracts_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(osmo_cache, "_driver_branch", lambda: "580")
    source = tmp_path / "source-cache"
    (source / "kit").mkdir(parents=True)
    (source / "kit/compiled.bin").write_bytes(b"compiled shader")
    artifact = tmp_path / "downloaded-artifact"
    compatibility = _compatibility()

    osmo_cache.publish_cache_artifact(
        source,
        artifact,
        compatibility,
        "registry/evaluator@sha256:" + "a" * 64,
    )
    manifest = json.loads((artifact / osmo_cache.CACHE_MANIFEST_NAME).read_text())
    assert manifest["cache_key"] == compatibility.key
    assert manifest["driver_branch"] == "580"
    assert manifest["complete"] is True
    assert manifest["archive_size_bytes"] > 0
    assert len(manifest["archive_sha256"]) == 64

    destination = tmp_path / "local-cache"
    (destination / "stale").mkdir(parents=True)
    destination_inode = destination.stat().st_ino
    osmo_cache.restore_cache_artifact(artifact, destination, compatibility)
    assert destination.stat().st_ino == destination_inode
    assert (destination / "kit/compiled.bin").read_bytes() == b"compiled shader"
    assert not (destination / "stale").exists()
    assert (destination / "nvidia/ComputeCache").is_dir()
    output = capsys.readouterr().out
    assert "[cap-cache] published key=schema-1/" in output
    assert "[cap-cache] restored key=schema-1/" in output
    assert "archive_bytes=" in output
    assert "elapsed_seconds=" in output


def test_restore_rejects_corruption_and_driver_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(osmo_cache, "_driver_branch", lambda: "580")
    source = tmp_path / "source"
    source.mkdir()
    artifact = tmp_path / "artifact"
    compatibility = _compatibility()
    osmo_cache.publish_cache_artifact(source, artifact, compatibility, "image")
    archive = artifact / osmo_cache.CACHE_ARCHIVE_NAME
    archive.write_bytes(archive.read_bytes() + b"corrupt")

    with pytest.raises(RuntimeError, match="size mismatch"):
        osmo_cache.restore_cache_artifact(
            artifact, tmp_path / "destination", compatibility
        )


def test_restore_rejects_incomplete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(osmo_cache, "_driver_branch", lambda: "580")
    source = tmp_path / "source"
    source.mkdir()
    artifact = tmp_path / "artifact"
    compatibility = _compatibility()
    osmo_cache.publish_cache_artifact(source, artifact, compatibility, "image")
    manifest_path = artifact / osmo_cache.CACHE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="mismatch for complete"):
        osmo_cache.restore_cache_artifact(
            artifact, tmp_path / "destination", compatibility
        )

    monkeypatch.setattr(osmo_cache, "_driver_branch", lambda: "575")
    with pytest.raises(RuntimeError, match="runtime has 575"):
        osmo_cache.restore_cache_artifact(
            artifact, tmp_path / "destination", compatibility
        )


def test_driver_query_is_bounded(monkeypatch) -> None:
    class Process:
        returncode = 0

        def communicate(self, timeout=None):
            assert 0 < timeout < float("inf")
            return "580.65.06\n", ""

    monkeypatch.setattr(osmo_cache.subprocess, "Popen", lambda *_a, **_k: Process())
    assert osmo_cache._driver_branch() == "580"


def test_driver_query_never_waits_unbounded_on_an_unkillable_child(
    monkeypatch,
) -> None:
    import subprocess

    events = []

    class StuckProcess:
        returncode = None

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired("nvidia-smi", timeout)

        def kill(self):
            events.append("kill")

        def wait(self, timeout=None):
            assert timeout is not None, "unbounded wait on a D-state child"
            events.append("wait")
            raise subprocess.TimeoutExpired("nvidia-smi", timeout)

    monkeypatch.setattr(
        osmo_cache.subprocess, "Popen", lambda *_a, **_k: StuckProcess()
    )
    with pytest.raises(subprocess.TimeoutExpired):
        osmo_cache._driver_branch()
    assert events == ["kill", "wait"]
