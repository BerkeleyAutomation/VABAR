"""Install container dependencies declared by repository-owned overlays."""

from __future__ import annotations

import argparse
import subprocess
import tomllib
from pathlib import Path

LOCAL_DISTRIBUTIONS = {"isaaclab-arena-vabar", "vabar"}
BUILD_REQUIREMENTS = (
    "editables",
    "hatchling",
    "idna==3.10",
    "Pillow>=10",
    "pytest>=8",
)


def normalized_name(requirement: str) -> str:
    """Return enough of a normalized PEP 508 name to identify local packages."""
    name = requirement.split(";", 1)[0].split("[", 1)[0]
    for separator in (" ", "<", ">", "=", "!", "~", "@"):
        name = name.split(separator, 1)[0]
    return name.strip().lower().replace("_", "-")


def declared_dependencies(manifests: list[Path]) -> list[str]:
    requirements: list[str] = []
    for manifest in manifests:
        project = tomllib.loads(manifest.read_text(encoding="utf-8"))["project"]
        for requirement in project.get("dependencies", []):
            if normalized_name(requirement) not in LOCAL_DISTRIBUTIONS:
                requirements.append(requirement)
    return requirements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()
    requirements = [*BUILD_REQUIREMENTS, *declared_dependencies(args.manifests)]
    subprocess.run(
        ["uv", "pip", "install", "--python", args.python, *requirements],
        check=True,
    )


if __name__ == "__main__":
    main()
