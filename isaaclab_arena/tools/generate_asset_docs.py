"""Run every asset drop test and generate the local README/media tree."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from asset_acceptance.process_contracts import (
    FailedAttempt,
    run_fresh_process_with_retries,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ARENA_PYTHON = Path(
    os.environ.get("ARENA_PYTHON", WORKSPACE_ROOT / ".venv/bin/python")
).expanduser().absolute()  # resolve() would follow the venv symlink out of the venv
DROP_CHECK = WORKSPACE_ROOT / "tools/asset_acceptance/asset_drop_live_check.py"
REPORT_GENERATOR = WORKSPACE_ROOT / "tools/asset_quality_report.py"
COLLISION_RENDERER = WORKSPACE_ROOT / "tools/render_asset_collisions.py"


def _asset_names() -> tuple[str, ...]:
    from asset_acceptance.asset_drop_cases import ASSET_DROP_CASES

    names = tuple(case.registry_name for case in ASSET_DROP_CASES)
    if not names or len(set(names)) != len(names):
        raise ValueError(f"expected unique movable assets, got {names}")
    return names


def _workspace_path(path: Path) -> Path:
    path = path.expanduser()
    return (
        (WORKSPACE_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    )


def _command_text(command: list[str]) -> str:
    return "$ " + " ".join(shlex.quote(part) for part in command)


def _preserve_drop_evidence(
    *,
    asset: str,
    attempt: int,
    result_path: Path,
    media_dir: Path,
    run_dir: Path,
) -> dict[str, str]:
    preserved: dict[str, str] = {}
    blocked_dir = run_dir / "blocked" / asset / f"attempt-{attempt}"
    sources = {
        "result": result_path,
        "video": media_dir / f"{asset}.mp4",
        "thumbnail": media_dir / f"{asset}.thumbnail.png",
    }
    for label, source in sources.items():
        if not source.is_file():
            continue
        blocked_dir.mkdir(parents=True, exist_ok=True)
        destination = blocked_dir / source.name
        shutil.move(source, destination)
        preserved[f"preserved_{label}"] = str(destination.resolve())
    return preserved


def _run_drop_asset(
    *,
    asset: str,
    command: list[str],
    result_path: Path,
    media_dir: Path,
    run_dir: Path,
    environment: dict[str, str],
) -> tuple[bool, list[dict]]:
    from asset_acceptance.asset_drop_harness import validate_drop_evidence

    blockers: list[dict] = []

    def record_blocker(failure: FailedAttempt) -> None:
        blocker = {
            "asset": asset,
            "attempt": failure.attempt,
            "process_returncode": failure.process_returncode,
            "validator_returncode": failure.validator_returncode,
            "reason": failure.reason,
        }
        blocker.update(
            _preserve_drop_evidence(
                asset=asset,
                attempt=failure.attempt,
                result_path=result_path,
                media_dir=media_dir,
                run_dir=run_dir,
            )
        )
        blockers.append(blocker)

    complete = run_fresh_process_with_retries(
        command=command,
        cwd=WORKSPACE_ROOT,
        environment=environment,
        stale_outputs=(
            result_path,
            media_dir / f"{asset}.mp4",
            media_dir / f"{asset}.thumbnail.png",
        ),
        result_path=result_path,
        validate=lambda: validate_drop_evidence(result_path, media_dir, asset),
        label=f"Asset {asset}",
        on_failure=record_blocker,
    )
    return complete, blockers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=WORKSPACE_ROOT / "outputs/asset-quality",
        help="Intermediate JSON and MP4 directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets",
        help="Asset catalog whose registry directories receive README.md and media/.",
    )
    parser.add_argument(
        "--fail-on-drop-failure",
        action="store_true",
        help="Generate all documentation, then fail if any drop verdict failed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the child commands without running Isaac Sim.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    work_root = _workspace_path(args.work_dir)
    output_dir = _workspace_path(args.output)
    if args.dry_run:
        run_dir = work_root / "fresh-run"
    else:
        if not ARENA_PYTHON.is_file():
            raise SystemExit(
                f"Pinned Arena Python does not exist: {ARENA_PYTHON}\n"
                "Run `uv run vabar-arena setup` first."
            )
        work_root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=work_root))
    media_dir = run_dir / "media"
    result_dir = run_dir / "results"
    environment = os.environ.copy()
    environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    environment.setdefault("ACCEPT_EULA", "Y")
    environment["ASSET_SOURCE_ROOT"] = str(WORKSPACE_ROOT)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(WORKSPACE_ROOT / "isaaclab_arena_vabar/src"),
            str(WORKSPACE_ROOT / "tools"),
        )
    )

    asset_names = _asset_names()
    asset_commands = [
        [
            str(ARENA_PYTHON),
            str(DROP_CHECK),
            "--asset",
            asset,
            "--record-dir",
            str(media_dir),
            "--result-dir",
            str(result_dir),
        ]
        for asset in asset_names
    ]
    report_command = [
        str(ARENA_PYTHON),
        str(REPORT_GENERATOR),
        "--results",
        str(result_dir),
        "--recordings",
        str(media_dir),
        "--output",
        str(output_dir),
    ]
    collision_command = [
        str(ARENA_PYTHON),
        str(COLLISION_RENDERER),
        "--output",
        str(media_dir),
    ]
    if args.dry_run:
        for command in (*asset_commands, collision_command, report_command):
            print(_command_text(command))
        return 0

    media_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    blockers: list[dict] = []
    incomplete_assets: list[str] = []
    for asset, command in zip(asset_names, asset_commands, strict=True):
        result_path = result_dir / f"{asset}.json"
        complete, asset_blockers = _run_drop_asset(
            asset=asset,
            command=command,
            result_path=result_path,
            media_dir=media_dir,
            run_dir=run_dir,
            environment=environment,
        )
        blockers.extend(asset_blockers)
        if not complete:
            incomplete_assets.append(asset)
            print(
                f"Asset {asset} has no complete drop evidence; continuing.",
                flush=True,
            )
    blockers_path = run_dir / "drop-blockers.json"
    blockers_path.write_text(json.dumps(blockers, indent=2, sort_keys=True) + "\n")
    campaign_path = run_dir / "drop-campaign.json"
    campaign_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "blockers": str(blockers_path.resolve()),
                "asset_count": len(asset_names),
                "completed_asset_count": len(asset_names) - len(incomplete_assets),
                "incomplete_assets": incomplete_assets,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if incomplete_assets:
        raise RuntimeError(
            "drop campaign incomplete after attempting every asset; "
            f"incomplete_assets={incomplete_assets}; summary={campaign_path}"
        )
    subprocess.run(collision_command, cwd=WORKSPACE_ROOT, env=environment, check=True)
    subprocess.run(report_command, cwd=WORKSPACE_ROOT, check=True)
    print(f"Generated per-asset documentation: {output_dir}")
    print(f"Intermediate evidence: {run_dir}")
    drop_failed = any(
        json.loads(path.read_text())["result"]["passed"] is not True
        for path in result_dir.glob("*.json")
    )
    return int(args.fail_on_drop_failure and drop_failed)


if __name__ == "__main__":
    raise SystemExit(main())
