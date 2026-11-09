#!/usr/bin/env python3
"""Compose the source, dependency, model, and base-image provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LOCKS = (
    "external/IsaacLab-Arena/submodules/IsaacLab/uv.lock",
    "external/gap/graph-as-policy/uv.lock",
    "external/gap/open-robot-skills/tools/curobo/uv.lock",
    "external/gap/open-robot-skills/tools/geometry/uv.lock",
    "external/gap/open-robot-skills/tools/grounding-dino/uv.lock",
    "external/gap/open-robot-skills/tools/sam3/uv.lock",
    "external/gap/open-robot-skills/tools/vlm/uv.lock",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compose(
    source_root: Path,
    source_manifest: Path,
    model_manifest: Path,
    base_image: str,
) -> dict:
    if "@sha256:" not in base_image:
        # Mutable bases are intentionally allowed for local development, but
        # the manifest must make that reduced provenance explicit.
        base_immutable = False
    else:
        base_immutable = True
    return {
        "schema_version": 1,
        "base_image": base_image,
        "base_image_immutable": base_immutable,
        "source": json.loads(source_manifest.read_text(encoding="utf-8")),
        "locks": {relative: sha256(source_root / relative) for relative in LOCKS},
        "models": json.loads(model_manifest.read_text(encoding="utf-8"))["models"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = compose(
        args.source_root,
        args.source_manifest,
        args.model_manifest,
        args.base_image,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
