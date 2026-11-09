#!/usr/bin/env python3
"""Hardlink only byte-identical files that repeat across immutable roots."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
from collections import defaultdict
from pathlib import Path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deduplicate(roots: list[Path]) -> tuple[int, int]:
    candidates: dict[tuple[str, int, int], list[tuple[int, Path]]] = defaultdict(list)
    for root_index, root in enumerate(roots):
        for directory, _, filenames in os.walk(root):
            for filename in filenames:
                path = Path(directory, filename)
                if path.is_symlink() or not path.is_file():
                    continue
                metadata = path.stat()
                candidates[(filename, metadata.st_size, stat.S_IMODE(metadata.st_mode))].append(
                    (root_index, path)
                )

    groups: dict[tuple[int, int, str], list[tuple[int, Path]]] = defaultdict(list)
    for (_, size, mode), paths in candidates.items():
        if len({root_index for root_index, _ in paths}) < 2:
            continue
        for root_index, path in paths:
            groups[(size, mode, _digest(path))].append((root_index, path))

    linked_files = 0
    saved_bytes = 0
    for (size, _, _), paths in groups.items():
        if len({root_index for root_index, _ in paths}) < 2:
            continue
        canonical_root, canonical = paths[0]
        for root_index, duplicate in paths[1:]:
            if root_index == canonical_root or os.path.samefile(canonical, duplicate):
                continue
            temporary = duplicate.with_name(f".{duplicate.name}.cap-dedup-{os.getpid()}")
            os.link(canonical, temporary)
            os.replace(temporary, duplicate)
            linked_files += 1
            saved_bytes += size
    return linked_files, saved_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    linked_files, saved_bytes = deduplicate(args.roots)
    print(f"Hardlinked {linked_files} byte-identical files ({saved_bytes} bytes)")


if __name__ == "__main__":
    main()
