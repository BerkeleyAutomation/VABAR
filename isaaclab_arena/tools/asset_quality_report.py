"""Build one Markdown drop-test report beside every movable asset."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _gif(source: Path, destination: Path) -> None:
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required; run this with the pinned Arena Python or install ffmpeg"
        )
    subprocess.run(
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop",
            "0",
            str(destination),
        ],
        check=True,
    )


def _read_results(results_root: Path, expected_assets: frozenset[str]) -> list[dict]:
    assets: list[dict] = []
    for path in sorted(results_root.glob("*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
            raise TypeError(f"invalid asset result: {path}")
        assets.append(payload["result"])
    names = [asset.get("registry_name") for asset in assets]
    if len(set(names)) != len(names) or set(names) != expected_assets:
        raise ValueError(
            f"expected {len(expected_assets)} unique movable-asset results, got {len(names)}"
        )
    return sorted(assets, key=lambda asset: asset["registry_name"])


def _measurement(value, *, format_spec: str, unit: str) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "Not finite"
    return f"{value:{format_spec}} {unit}"



def _readme(asset: dict) -> str:
    status = "PASS" if asset["passed"] else "FAIL"
    failures = "; ".join(asset["failures"]) if asset["failures"] else "None"
    settling = asset["settling_time_s"]
    settling_text = (
        "Not settled"
        if settling is None
        else _measurement(settling, format_spec=".3f", unit="s")
    )
    final_lowest = asset["final_lowest_point_z"]
    tabletop = asset["tabletop_z"]
    penetration = (
        tabletop - final_lowest
        if isinstance(final_lowest, (int, float)) and isinstance(tabletop, (int, float))
        else None
    )
    return f"""# {asset["registry_name"]}

Generated with a fresh-process 10 cm drop test.

![Asset thumbnail with authored pivot and local XYZ axes](media/thumbnail.png)

![Enabled collision body](media/collision.png)

![Drop test](media/drop.gif)

| Property | Value |
|---|---|
| Mass | {asset["mass_kg"]:.6g} kg |
| Friction | {asset["friction"]} |
| Restitution | {asset["restitution"]} |
| Collider | {asset["collider"]} |
| Physics profile | `{asset["physics_profile"]}` |
| Drop-test status | **{status}** |
| Settling time | {settling_text} |
| Final penetration | {_measurement(None if penetration is None else penetration * 1000.0, format_spec=".3f", unit="mm")} |
| Final-second movement | {_measurement(asset["max_tail_translation_m"], format_spec=".6g", unit="m")} |
| Final-second rotation | {_measurement(asset["max_tail_rotation_rad"], format_spec=".6g", unit="rad")} |
| Failure reasons | {failures} |
"""


def _overview_readme(assets: list[dict]) -> str:
    rows = "\n".join(
        f"| [{asset['registry_name']}]({asset['registry_name']}/README.md) "
        f'| <a href="{asset["registry_name"]}/README.md"><img '
        f'src="{asset["registry_name"]}/media/thumbnail.png" '
        f'alt="{asset["registry_name"]} thumbnail" width="240" height="135"></a> '
        f'| <img src="{asset["registry_name"]}/media/drop.gif" '
        f'alt="{asset["registry_name"]} drop test" width="240" height="135"> '
        f'| <img src="{asset["registry_name"]}/media/collision.png" '
        f'alt="{asset["registry_name"]} collision body" width="240" height="135"> |'
        for asset in assets
    )
    return f"""# Movable asset catalog

Each row links to the asset's generated physical-property report.

| Asset | Thumbnail | Drop test | Collision body |
|---|---|---|---|
{rows}
"""


def build_report(
    results_root: Path,
    recordings_root: Path,
    asset_root: Path,
    *,
    expected_assets: frozenset[str],
    gif_converter: Callable[[Path, Path], None] = _gif,
) -> list[dict]:
    assets = _read_results(results_root, expected_assets)
    asset_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".asset-docs-", dir=asset_root.parent))
    try:
        staged_assets = staging / "assets"
        for asset in assets:
            name = asset["registry_name"]
            asset_dir = asset_root / name
            if not (asset_dir / f"{name}.usda").is_file():
                raise FileNotFoundError(
                    f"missing canonical asset entry point for {name}"
                )
            source_thumbnail = recordings_root / f"{name}.thumbnail.png"
            source_collision = recordings_root / f"{name}.collision.png"
            source_video = recordings_root / f"{name}.mp4"
            if (
                not source_thumbnail.is_file()
                or not source_collision.is_file()
                or not source_video.is_file()
            ):
                raise FileNotFoundError(f"missing media for {name}")
            destination = staged_assets / name
            media = destination / "media"
            media.mkdir(parents=True)
            shutil.copy2(source_thumbnail, media / "thumbnail.png")
            shutil.copy2(source_collision, media / "collision.png")
            gif_converter(source_video, media / "drop.gif")
            (destination / "README.md").write_text(_readme(asset))
        staged_overview = staging / "README.md"
        staged_overview.write_text(_overview_readme(assets))
        _publish_reports(staged_assets, staged_overview, asset_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return assets


def _publish_reports(
    staged_assets: Path,
    staged_overview: Path,
    asset_root: Path,
) -> None:
    """Replace only generated files, restoring every target on failure."""

    backup_root = staged_assets.parent / "previous"
    published: list[tuple[Path, Path, bool]] = []
    try:
        for staged_asset in sorted(staged_assets.iterdir()):
            target_asset = asset_root / staged_asset.name
            for relative in (Path("README.md"), Path("media")):
                source = staged_asset / relative
                target = target_asset / relative
                backup = backup_root / staged_asset.name / relative
                had_previous = target.exists()
                if had_previous:
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    target.rename(backup)
                published.append((target, backup, had_previous))
                os.replace(source, target)
        overview = asset_root / "README.md"
        overview_backup = backup_root / "catalog-README.md"
        had_previous = overview.exists()
        if had_previous:
            overview.rename(overview_backup)
        published.append((overview, overview_backup, had_previous))
        os.replace(staged_overview, overview)
    except BaseException:
        for target, backup, had_previous in reversed(published):
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            if had_previous and backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.rename(target)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-drop-failure", action="store_true")
    args = parser.parse_args()
    from asset_acceptance.asset_drop_cases import ASSET_DROP_CASES

    assets = build_report(
        args.results,
        args.recordings,
        args.output,
        expected_assets=frozenset(case.registry_name for case in ASSET_DROP_CASES),
    )
    return int(
        args.fail_on_drop_failure and any(not asset["passed"] for asset in assets)
    )


if __name__ == "__main__":
    raise SystemExit(main())
