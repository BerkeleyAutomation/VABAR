#!/usr/bin/env python3
"""Bake and verify the exact GaP perception model snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--notice-dir", type=Path, required=True)
    parser.add_argument("--offline-verify", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    token = os.environ.get("HF_TOKEN")
    if not token and not args.offline_verify:
        raise RuntimeError("HF_TOKEN is required to bake gated SAM3 weights")

    provenance: dict[str, object] = {"schema_version": 1, "models": {}}
    for name, model in manifest["models"].items():
        revision = model["revision"]
        snapshot = Path(
            snapshot_download(
                repo_id=model["repo_id"],
                revision=revision,
                allow_patterns=sorted(model["files"]),
                cache_dir=args.cache_dir,
                token=token,
                local_files_only=args.offline_verify,
            )
        )
        actual_files = {
            path.relative_to(snapshot).as_posix()
            for path in snapshot.rglob("*")
            if path.is_file()
        }
        expected_files = set(model["files"])
        if actual_files != expected_files:
            raise RuntimeError(
                f"{name} file mismatch: missing={sorted(expected_files - actual_files)}, "
                f"unexpected={sorted(actual_files - expected_files)}"
            )
        for relative, expected_hash in model["files"].items():
            actual_hash = sha256(snapshot / relative)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"{name}/{relative} sha256={actual_hash}, expected {expected_hash}"
                )

        if not args.offline_verify:
            main_ref = snapshot.parents[1] / "refs" / "main"
            main_ref.parent.mkdir(parents=True, exist_ok=True)
            main_ref.write_text(revision, encoding="utf-8")

        notice_dir = args.notice_dir / name
        notice_dir.mkdir(parents=True, exist_ok=True)
        for relative in model.get("notice_files", []):
            shutil.copy2(snapshot / relative, notice_dir / Path(relative).name)

        provenance["models"][name] = {
            "repo_id": model["repo_id"],
            "revision": revision,
            "files": model["files"],
        }

    provenance_path = args.cache_dir / "CAP_MODEL_PROVENANCE.json"
    expected_provenance = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    if args.offline_verify:
        actual_provenance = provenance_path.read_text(encoding="utf-8")
        if actual_provenance != expected_provenance:
            raise RuntimeError("baked model provenance does not match the manifest")
    else:
        provenance_path.write_text(expected_provenance, encoding="utf-8")


if __name__ == "__main__":
    main()
