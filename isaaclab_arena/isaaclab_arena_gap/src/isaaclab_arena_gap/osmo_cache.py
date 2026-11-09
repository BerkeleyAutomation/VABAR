"""Persistent, task-neutral Isaac runtime cache artifacts for OSMO."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

CACHE_ARCHIVE_NAME = "cache.tar.zst"
CACHE_MANIFEST_NAME = "cache-manifest.json"
CACHE_MANIFEST_SCHEMA = 1
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DRIVER_BRANCH = re.compile(r"^[0-9]+$")
_NVIDIA_SMI_KILL_WAIT_S = 5.0
_NVIDIA_SMI_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class CacheCompatibility:
    """Inputs that determine whether a warmed Isaac cache can be reused."""

    recipe_epoch: str
    kit_build: str
    platform: str
    driver_branch: str

    def __post_init__(self) -> None:
        for name in ("recipe_epoch", "kit_build", "platform"):
            value = getattr(self, name)
            if not _SAFE_COMPONENT.fullmatch(value):
                raise ValueError(f"cache {name} must be a non-empty safe path component: {value!r}")
        if not _DRIVER_BRANCH.fullmatch(self.driver_branch):
            raise ValueError(
                "cache driver_branch must be the explicitly configured numeric major branch; "
                f"got {self.driver_branch!r}"
            )

    @property
    def key(self) -> str:
        return "/".join((
            f"schema-{CACHE_MANIFEST_SCHEMA}",
            f"recipe-{self.recipe_epoch}",
            f"kit-{self.kit_build}",
            f"platform-{self.platform}",
            f"driver-{self.driver_branch}",
        ))

    def artifact_url(self, root_url: str) -> str:
        if not root_url.startswith(("swift://", "https://")):
            raise ValueError(f"cache artifact root must be an object-store URL: {root_url!r}")
        return f"{root_url.rstrip('/')}/{self.key}"


def persistent_artifact_exists(
    artifact_url: str,
    *,
    open_url: Callable[..., Any] = urllib.request.urlopen,
) -> bool:
    """Check both immutable objects through the anonymously readable PDX endpoint."""
    base_url = _artifact_https_url(artifact_url)
    for filename in (CACHE_MANIFEST_NAME, CACHE_ARCHIVE_NAME):
        url = f"{base_url.rstrip('/')}/{filename}"
        request = urllib.request.Request(url, method="HEAD")
        try:
            with open_url(request, timeout=20) as response:
                status = getattr(response, "status", 200)
                if not 200 <= status < 300:
                    raise RuntimeError(f"cache artifact HEAD returned HTTP {status}: {url}")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return False
            raise RuntimeError(
                f"cache artifact HEAD returned HTTP {error.code}: {url}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"cache artifact lookup failed: {url}: {error.reason}") from error
    return True


def _artifact_https_url(artifact_url: str) -> str:
    if artifact_url.startswith("https://"):
        return artifact_url
    if artifact_url.startswith("swift://"):
        authority_and_path = artifact_url.removeprefix("swift://")
        host, separator, path = authority_and_path.partition("/")
        if not separator or not host or not path:
            raise ValueError(f"invalid Swift cache artifact URL: {artifact_url!r}")
        return f"https://{host}/v1/{path}"
    raise ValueError(
        "cache artifact URL must use swift:// for OSMO transport or https:// "
        f"for direct transport: {artifact_url!r}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _driver_branch() -> str:
    command = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        stdout, stderr = process.communicate(timeout=_NVIDIA_SMI_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        # A child stuck in the driver may not die on SIGKILL; never wait on it
        # unbounded the way subprocess.run does after its timeout.
        process.kill()
        try:
            process.wait(timeout=_NVIDIA_SMI_KILL_WAIT_S)
        except subprocess.TimeoutExpired:
            pass
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode, command, stdout, stderr
        )
    version = stdout.splitlines()[0].strip()
    branch = version.partition(".")[0]
    if not _DRIVER_BRANCH.fullmatch(branch):
        raise RuntimeError(f"nvidia-smi returned an invalid driver version: {version!r}")
    return branch


def _assert_runtime_compatible(expected: CacheCompatibility) -> None:
    actual_driver = _driver_branch()
    if actual_driver != expected.driver_branch:
        raise RuntimeError(
            "Isaac cache driver mismatch: "
            f"artifact expects branch {expected.driver_branch}, runtime has {actual_driver}"
        )


def _find_artifact_files(artifact_directory: Path) -> tuple[Path, Path]:
    manifests = list(artifact_directory.rglob(CACHE_MANIFEST_NAME))
    if len(manifests) != 1:
        raise RuntimeError(
            f"cache artifact must contain exactly one {CACHE_MANIFEST_NAME}; found {len(manifests)}"
        )
    manifest = manifests[0]
    archive = manifest.with_name(CACHE_ARCHIVE_NAME)
    if not archive.is_file():
        raise RuntimeError(f"cache artifact is missing {CACHE_ARCHIVE_NAME} beside its manifest")
    return manifest, archive


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cache manifest is unreadable: {path}") from error
    if not isinstance(data, dict):
        raise RuntimeError("cache manifest must be a JSON object")
    return data


def _expected_manifest_fields(expected: CacheCompatibility) -> dict[str, Any]:
    return {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "complete": True,
        **asdict(expected),
    }


def _verify_manifest(
    manifest: dict[str, Any], archive: Path, expected: CacheCompatibility
) -> None:
    for name, value in _expected_manifest_fields(expected).items():
        if manifest.get(name) != value:
            raise RuntimeError(
                f"cache manifest mismatch for {name}: expected {value!r}, got {manifest.get(name)!r}"
            )
    archive_size = archive.stat().st_size
    if manifest.get("archive_size_bytes") != archive_size:
        raise RuntimeError(
            "cache archive size mismatch: "
            f"manifest has {manifest.get('archive_size_bytes')!r}, file has {archive_size}"
        )
    archive_sha256 = _sha256(archive)
    if manifest.get("archive_sha256") != archive_sha256:
        raise RuntimeError("cache archive checksum mismatch")


def _validate_tar_members(archive: Path) -> None:
    result = subprocess.run(
        ["tar", "--zstd", "-tf", str(archive)],
        check=True,
        capture_output=True,
        text=True,
    )
    for member in result.stdout.splitlines():
        member_path = Path(member)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise RuntimeError(f"cache archive contains an unsafe path: {member!r}")


def _initialize_local_cache(local_cache: Path) -> None:
    for relative in (
        "kit",
        "data",
        "logs",
        "documents",
        "xdg",
        "ov",
        "nvidia/ComputeCache",
        "nvidia/GLCache",
    ):
        (local_cache / relative).mkdir(parents=True, exist_ok=True)


def restore_cache_artifact(
    artifact_directory: Path,
    local_cache: Path,
    expected: CacheCompatibility,
) -> None:
    """Verify a downloaded artifact and atomically promote its local extraction."""
    started = time.monotonic()
    _assert_runtime_compatible(expected)
    manifest_path, archive = _find_artifact_files(artifact_directory)
    _verify_manifest(_read_manifest(manifest_path), archive, expected)
    _validate_tar_members(archive)
    local_cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cap-cache-restore.") as temporary:
        staging = Path(temporary)
        subprocess.run(
            ["tar", "--zstd", "-xf", str(archive), "-C", str(staging)],
            check=True,
        )
        for existing in local_cache.iterdir():
            if existing.is_dir() and not existing.is_symlink():
                shutil.rmtree(existing)
            else:
                existing.unlink()
        for extracted in staging.iterdir():
            shutil.move(str(extracted), local_cache / extracted.name)
    _initialize_local_cache(local_cache)
    print(
        f"[cap-cache] restored key={expected.key} "
        f"archive_bytes={archive.stat().st_size} "
        f"elapsed_seconds={time.monotonic() - started:.3f}",
        flush=True,
    )


def publish_cache_artifact(
    local_cache: Path,
    artifact_directory: Path,
    expected: CacheCompatibility,
    image_digest: str,
) -> None:
    """Create an archive and publish its manifest last into the OSMO output."""
    started = time.monotonic()
    _assert_runtime_compatible(expected)
    _initialize_local_cache(local_cache)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    archive = artifact_directory / CACHE_ARCHIVE_NAME
    manifest = artifact_directory / CACHE_MANIFEST_NAME
    temporary_archive = archive.with_suffix(archive.suffix + ".tmp")
    temporary_manifest = manifest.with_suffix(manifest.suffix + ".tmp")
    subprocess.run(
        ["tar", "--zstd", "-cf", str(temporary_archive), "-C", str(local_cache), "."],
        check=True,
    )
    temporary_archive.replace(archive)
    data = {
        **_expected_manifest_fields(expected),
        "cache_key": expected.key,
        "image_digest": image_digest,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
    }
    temporary_manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest)
    print(
        f"[cap-cache] published key={expected.key} "
        f"archive_bytes={data['archive_size_bytes']} "
        f"elapsed_seconds={time.monotonic() - started:.3f}",
        flush=True,
    )


def _compatibility_from_args(args: argparse.Namespace) -> CacheCompatibility:
    return CacheCompatibility(
        recipe_epoch=args.recipe_epoch,
        kit_build=args.kit_build,
        platform=args.platform,
        driver_branch=args.driver_branch,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("restore", "publish"))
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--local-cache", type=Path, required=True)
    parser.add_argument("--recipe-epoch", required=True)
    parser.add_argument("--kit-build", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--driver-branch", required=True)
    parser.add_argument("--image-digest", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    compatibility = _compatibility_from_args(args)
    if args.command == "restore":
        restore_cache_artifact(args.artifact_directory, args.local_cache, compatibility)
    else:
        if not args.image_digest:
            raise ValueError("publish requires --image-digest provenance")
        publish_cache_artifact(
            args.local_cache,
            args.artifact_directory,
            compatibility,
            args.image_digest,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
