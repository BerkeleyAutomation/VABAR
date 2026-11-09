"""Narrow runtime interface for the exact-pin Arena/GaP evaluator image."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from isaaclab_arena_gap.container_runner import write_run_provenance

DEFAULT_ROOT = Path(os.environ.get("CAP_ROOT", "/workspace/isaaclab_arena"))
DEFAULT_MANIFEST = Path(
    os.environ.get("CAP_IMAGE_MANIFEST", "/opt/cap/image-manifest.json")
)


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError(f"unsupported image manifest schema in {path}")
    return manifest


def build_run_command(
    root: Path,
    experiment: Path,
    output_dir: Path,
    *,
    record_camera_video: bool,
    runner_args: Sequence[str] = (),
) -> tuple[list[str], dict[str, str]]:
    command = [str(root / "isaaclab_arena_gap/scripts/run_arena_eval.sh"), *runner_args]
    environment = {
        "ARENA_EXPERIMENT_CONFIG": str(experiment),
        "ARENA_OUTPUT_DIR": str(output_dir),
        "ARENA_RECORD_CAMERA_VIDEO": "1" if record_camera_video else "0",
    }
    return command, environment


def verify_image(
    root: Path,
    manifest_path: Path,
    *,
    quick: bool,
    require_immutable_base: bool,
) -> dict:
    manifest = load_manifest(manifest_path)
    if require_immutable_base and not manifest["base_image_immutable"]:
        raise RuntimeError("image was built from a mutable base reference")
    expected_fingerprint = os.environ.get("CAP_IMAGE_SOURCE_FINGERPRINT")
    actual_fingerprint = manifest["source"]["source_fingerprint"]
    if expected_fingerprint and expected_fingerprint != actual_fingerprint:
        raise RuntimeError("image label and embedded source fingerprint differ")

    required = {
        "Arena Python": root / ".venv/bin/python",
        "GaP Python": root / "external/gap/graph-as-policy/.venv/bin/python",
        "source manifest": Path("/opt/cap/provenance/source-manifest.json"),
        "model manifest": root / "docker/runtime_dependencies.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"image runtime is incomplete: {', '.join(missing)}")
    forbidden_tokens = (
        Path(os.environ.get("HF_HOME", "/opt/cap/models/huggingface")) / "token",
        Path(os.environ.get("HF_HOME", "/opt/cap/models/huggingface"))
        / "stored_tokens",
    )
    leaked = [str(path) for path in forbidden_tokens if path.exists()]
    if leaked:
        raise RuntimeError(f"credential material exists in the image: {leaked}")

    subprocess.run(
        [
            "python3",
            str(root / "docker/source_manifest.py"),
            "verify",
            "--source-root",
            str(root),
            "--manifest",
            str(required["source manifest"]),
        ],
        check=True,
    )
    if not quick:
        subprocess.run(
            [
                str(required["Arena Python"]),
                "-c",
                "import isaaclab_arena_gap.gap_remote_policy, isaaclab, isaaclab_arena, msgpack_numpy",
            ],
            check=True,
        )
        subprocess.run(
            [
                str(required["GaP Python"]),
                "-c",
                "import gap, isaaclab_arena_gap",
            ],
            check=True,
            env={**os.environ, "PYTHONPATH": str(root / "isaaclab_arena_gap/src")},
        )
        dino_python = (
            root
            / "external/gap/open-robot-skills/tools/grounding-dino/.venv/bin/python"
        )
        with tempfile.TemporaryDirectory(prefix="cap-model-notices-") as notice_dir:
            subprocess.run(
                [
                    str(dino_python),
                    str(root / "docker/bake_models.py"),
                    "--manifest",
                    str(required["model manifest"]),
                    "--cache-dir",
                    os.environ.get(
                        "HF_HUB_CACHE", "/opt/cap/models/huggingface/hub"
                    ),
                    "--notice-dir",
                    notice_dir,
                    "--offline-verify",
                ],
                check=True,
            )
    return {
        "ok": True,
        "revision": manifest["source"]["revision"],
        "dirty": manifest["source"]["dirty"],
        "source_fingerprint": actual_fingerprint,
        "base_image_immutable": manifest["base_image_immutable"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("manifest", help="Print embedded build provenance.")
    verify = subparsers.add_parser("verify", help="Verify the baked runtime offline.")
    verify.add_argument(
        "--quick", action="store_true", help="Skip imports and model hashes."
    )
    verify.add_argument(
        "--require-immutable-base",
        action="store_true",
        help="Reject local-development images built from a mutable base tag.",
    )
    run = subparsers.add_parser("run", help="Run one Arena job with managed GaP.")
    run.add_argument("--experiment", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, default=Path("/outputs"))
    run.add_argument("--no-record-camera-video", action="store_true")
    run.add_argument("runner_args", nargs=argparse.REMAINDER)
    return parser


def main(cli_args: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(cli_args)
    if args.command == "manifest":
        print(json.dumps(load_manifest(args.manifest_path), indent=2, sort_keys=True))
        return 0
    if args.command == "verify":
        result = verify_image(
            args.root,
            args.manifest_path,
            quick=args.quick,
            require_immutable_base=args.require_immutable_base,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    experiment = args.experiment.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_provenance(
        output_dir, args.manifest_path, load_manifest(args.manifest_path)
    )
    runner_args = args.runner_args
    if runner_args[:1] == ["--"]:
        runner_args = runner_args[1:]
    command, updates = build_run_command(
        args.root,
        experiment,
        output_dir,
        record_camera_video=not args.no_record_camera_video,
        runner_args=runner_args,
    )
    os.execvpe(command[0], command, {**os.environ, **updates})
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
