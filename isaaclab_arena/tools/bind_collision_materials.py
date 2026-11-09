"""Bind each stored collision hull directly to its parent's physics material.

The Newton USD importer reads a collider's physics material only from a binding on
the collider prim itself. A binding on an ancestor (here the ``StoredConvexCollision``
Xform that references the hull payload) is ignored, and the hull falls back to the
default friction. This adds an ``over`` with the parent's
``material:binding:physics`` for every collider the Xform references. Text is inserted
in place so the hand-authored wrapper is otherwise unchanged. Idempotent.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ASSETS_ROOT = Path(__file__).resolve().parents[1] / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets"
_COLLISION = re.compile(r'^(?P<indent>\s*)def Xform "StoredConvexCollision" \($')
_REFERENCE = re.compile(r"prepend references = @(?P<asset>[^@]+)@<(?P<prim>[^>]*)>")
_BINDING = re.compile(r"^(?P<indent>\s*)rel material:binding:physics = <(?P<target>[^>]+)>(?P<meta> \()?$")


def _hull_names(wrapper: Path, asset: str, prim: str) -> list[str]:
    from pxr import Sdf, Usd  # noqa: F401  (importing Usd registers the file-format plugins)

    payload = Sdf.Layer.FindOrOpen(str((wrapper.parent / asset).resolve()))
    if payload is None:
        raise ValueError(f"{wrapper}: cannot open {asset}")
    root = payload.GetPrimAtPath(prim or f"/{payload.defaultPrim}")
    return [
        child.name
        for child in root.nameChildren
        if child.HasInfo("apiSchemas")
        and "PhysicsCollisionAPI" in child.GetInfo("apiSchemas").GetAddedOrExplicitItems()
    ]


def bind_file(path: Path) -> int:
    """Return how many hull bindings were added to one wrapper USD."""
    lines = path.read_text().splitlines(keepends=True)
    out: list[str] = []
    added = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        out.append(line)
        index += 1
        if not _COLLISION.match(line.rstrip("\n")):
            continue
        reference = binding = None
        while index < len(lines):
            line = lines[index]
            out.append(line)
            index += 1
            reference = reference or _REFERENCE.search(line)
            binding = _BINDING.match(line.rstrip("\n"))
            if binding or line.strip() == "}":  # bound, or an Xform without a binding
                break
        if not binding or reference is None:
            continue
        if binding["meta"]:  # copy the binding's metadata block through to its ")"
            while index < len(lines):
                out.append(lines[index])
                index += 1
                if lines[index - 1].strip() == ")":
                    break
        rest = "".join(lines[index:])
        indent, inner = binding["indent"], binding["indent"] + "    "
        for name in _hull_names(path, reference["asset"], reference["prim"]):
            if re.search(rf'^\s*over "{name}" \(', rest, re.MULTILINE):
                continue
            out.append(
                f'\n{indent}over "{name}" (\n'
                f'{inner}prepend apiSchemas = ["MaterialBindingAPI"]\n'
                f"{indent}) {{\n"
                f"{inner}rel material:binding:physics = <{binding['target']}>\n"
                f"{indent}}}\n"
            )
            added += 1
    if added:
        path.write_text("".join(out))
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    for path in args.paths or sorted(ASSETS_ROOT.glob("*/*.usda")):
        added = bind_file(path.resolve())
        if added:
            print(f"{path.resolve().relative_to(ASSETS_ROOT)}: bound {added} hulls")


if __name__ == "__main__":
    main()
