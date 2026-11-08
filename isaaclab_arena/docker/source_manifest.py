#!/usr/bin/env python3
"""Create and verify provenance for an exact or deliberately dirty source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

REQUIRED_GITLINKS = (
    (".", "external/IsaacLab-Arena"),
    ("external/IsaacLab-Arena", "submodules/IsaacLab"),
    (".", "external/gap/graph-as-policy"),
    (".", "external/gap/open-robot-skills"),
    (
        "external/gap/graph-as-policy",
        "third_party/Variational-Automation-Benchmark",
    ),
    ("external/gap/graph-as-policy", "third_party/robosuite"),
)
# AUTOLab's sources are pinned as this gitlink, or, when this tree is a folder of AUTOLab's
# repository, are that repository's own revision (the manifest's `revision`).
AUTOLAB_GITLINK = (".", "external/gap/autolab-vabar")


def _git(root: Path, *args: str, text: bool = True) -> str | bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=text,
    ).stdout


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.exists() and not path.is_symlink():
        return {"kind": "missing"}
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        target = os.readlink(path).encode()
        return {
            "kind": "symlink",
            "target": os.fsdecode(target),
            "sha256": _sha256_bytes(target),
        }
    if not stat.S_ISREG(mode):
        raise ValueError(f"dirty source path is not a regular file or symlink: {path}")
    return {
        "kind": "file",
        "mode": stat.S_IMODE(mode),
        "size": path.stat().st_size,
        "sha256": _sha256_bytes(path.read_bytes()),
    }


def _nul_paths(data: bytes) -> list[str]:
    return [os.fsdecode(value) for value in data.split(b"\0") if value]


def _validated_runtime_paths(paths: object) -> tuple[str, ...]:
    if not isinstance(paths, (list, tuple)) or not all(
        isinstance(path, str) and path for path in paths
    ):
        raise ValueError("runtime source paths must be a list of paths")
    runtime_paths: list[str] = []
    for runtime_path in paths:
        path = Path(runtime_path)
        if path == Path(".") or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe runtime source path: {runtime_path!r}")
        runtime_paths.append(path.as_posix())
    if len(set(runtime_paths)) != len(runtime_paths):
        raise ValueError("runtime source paths must be unique")
    return tuple(runtime_paths)


def _gitlink_record(source_root: Path, parent: str, child: str) -> dict[str, str]:
    parent_root = (source_root / parent).resolve()
    child_root = parent_root / child
    stage = str(_git(parent_root, "ls-files", "--stage", "--", child)).strip()
    fields = stage.split()
    if len(fields) < 4 or fields[0] != "160000":
        raise RuntimeError(f"missing required gitlink {parent}/{child}")
    expected = fields[1]
    if not (child_root / ".git").exists():
        raise RuntimeError(f"required gitlink is not initialized: {child_root}")
    actual = str(_git(child_root, "rev-parse", "HEAD")).strip()
    if actual != expected:
        raise RuntimeError(f"gitlink {child_root} is at {actual}, expected {expected}")
    dirty = str(_git(child_root, "status", "--porcelain", "--untracked-files=all"))
    if dirty:
        raise RuntimeError(f"required immutable gitlink is dirty: {child_root}")
    relative = (Path(parent) / child).as_posix()
    return {"path": relative.removeprefix("./"), "revision": expected}


def _has_gitlink(source_root: Path, parent: str, child: str) -> bool:
    stage = str(_git((source_root / parent).resolve(), "ls-files", "--stage", "--", child))
    return stage.startswith("160000 ")


def create_manifest(
    source_root: Path,
    output_dir: Path,
    runtime_paths: tuple[str, ...] = (),
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_paths = _validated_runtime_paths(runtime_paths)
    pathspec = ("--", *runtime_paths) if runtime_paths else ()
    revision = str(_git(source_root, "rev-parse", "HEAD")).strip()
    try:
        branch = str(_git(source_root, "symbolic-ref", "--short", "-q", "HEAD")).strip()
    except subprocess.CalledProcessError:
        branch = ""
    status_text = str(
        _git(
            source_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            *pathspec,
        )
    )
    tracked_paths = _nul_paths(
        bytes(
            _git(
                source_root,
                "diff",
                "--name-only",
                "--relative",
                "-z",
                "HEAD",
                *pathspec,
                text=False,
            )
        )
    )
    untracked_paths = _nul_paths(
        bytes(
            _git(
                source_root,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                *pathspec,
                text=False,
            )
        )
    )
    dirty_paths = sorted(set(tracked_paths + untracked_paths))
    dirty_files = {
        relative: _file_record(source_root / relative) for relative in dirty_paths
    }
    patch = bytes(
        _git(
            source_root,
            "diff",
            "--binary",
            "--full-index",
            "--relative",
            "HEAD",
            *pathspec,
            text=False,
        )
    )
    required = list(REQUIRED_GITLINKS)
    if _has_gitlink(source_root, *AUTOLAB_GITLINK):
        required.insert(4, AUTOLAB_GITLINK)  # after open-robot-skills, in manifest order
    gitlinks = [_gitlink_record(source_root, parent, child) for parent, child in required]
    core: dict[str, Any] = {
        "schema_version": 1,
        "revision": revision,
        "branch": branch or None,
        "dirty": bool(status_text),
        "git_status": status_text.splitlines(),
        "tracked_patch_sha256": _sha256_bytes(patch),
        "dirty_files": dirty_files,
        "gitlinks": gitlinks,
        "runtime_paths": list(runtime_paths),
    }
    fingerprint_input = {
        "revision": revision,
        "tracked_patch_sha256": core["tracked_patch_sha256"],
        "dirty_files": dirty_files,
        "gitlinks": gitlinks,
        "runtime_paths": list(runtime_paths),
    }
    canonical = json.dumps(
        fingerprint_input, sort_keys=True, separators=(",", ":")
    ).encode()
    core["source_fingerprint"] = _sha256_bytes(canonical + b"\0" + patch)
    (output_dir / "source-manifest.json").write_text(
        json.dumps(core, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "dirty.patch").write_bytes(patch)
    return core


def verify_manifest(source_root: Path, manifest_path: Path) -> None:
    source_root = source_root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported source manifest schema")
    try:
        runtime_paths = _validated_runtime_paths(manifest.get("runtime_paths", []))
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    for relative, expected in manifest["dirty_files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError(f"unsafe source-manifest path: {relative}")
        if runtime_paths and not any(
            relative == runtime_path or relative.startswith(f"{runtime_path}/")
            for runtime_path in runtime_paths
        ):
            raise RuntimeError(
                f"source-manifest path is outside the runtime allowlist: {relative}"
            )
        actual = _file_record(source_root / relative)
        if actual != expected:
            raise RuntimeError(
                f"materialized dirty source differs at {relative}: "
                f"actual={actual}, expected={expected}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--source-root", type=Path, required=True)
    create.add_argument("--output-dir", type=Path, required=True)
    create.add_argument("--runtime-path", action="append", default=[])
    verify = subparsers.add_parser("verify")
    verify.add_argument("--source-root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        create_manifest(args.source_root, args.output_dir, tuple(args.runtime_path))
    else:
        verify_manifest(args.source_root, args.manifest)


if __name__ == "__main__":
    main()
