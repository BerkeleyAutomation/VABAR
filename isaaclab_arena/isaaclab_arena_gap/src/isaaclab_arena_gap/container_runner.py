"""OSMO Arena runner that publishes evaluator-image provenance with results."""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path


def output_directory_from_args(args: Sequence[str]) -> Path | None:
    try:
        index = args.index("--experiment_output_directory")
    except ValueError:
        return None
    if index + 1 >= len(args):
        return None
    return Path(args[index + 1])


def write_run_provenance(
    destination: Path, manifest_path: Path, manifest: dict
) -> None:
    """Copy the image manifest into ``destination`` and record run provenance."""
    shutil.copy2(manifest_path, destination / "image-manifest.json")
    provenance = {
        "schema_version": 1,
        "image_digest": os.environ.get("CAP_IMAGE_DIGEST"),
        "source_fingerprint": manifest["source"]["source_fingerprint"],
    }
    (destination / "run-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def publish_image_provenance(output_dir: Path, manifest_path: Path) -> None:
    if not output_dir.is_dir() or not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Arena's OSMO collector copies each completed Run directory rather than
    # arbitrary files at the Experiment root. Keep a root copy for direct task
    # inspection and a copy in every Run so aggregate publication retains it.
    destinations = [output_dir]
    destinations.extend(path for path in output_dir.iterdir() if path.is_dir())
    for destination in destinations:
        write_run_provenance(destination, manifest_path, manifest)


def main() -> None:
    output_dir = output_directory_from_args(sys.argv[1:])
    manifest_path = Path(
        os.environ.get("CAP_IMAGE_MANIFEST", "/opt/cap/image-manifest.json")
    )
    try:
        from isaaclab_arena_gap.isolated_runs import plan_isolation, run_isolated

        plan = plan_isolation(sys.argv[1:])
        if plan is not None:
            exit_code = run_isolated(plan)
            if exit_code:
                raise SystemExit(exit_code)
            return
        from isaaclab_arena_gap.arena_runner import main as arena_main

        arena_main()
    finally:
        if output_dir is not None:
            publish_image_provenance(output_dir, manifest_path)


if __name__ == "__main__":
    main()
