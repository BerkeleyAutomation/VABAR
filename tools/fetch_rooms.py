# SPDX-License-Identifier: Apache-2.0
"""Download the Isaac Sim rooms ``vabar/rooms.py`` draws around each suite's bench.

    uv run python tools/fetch_rooms.py [--out ~/.cache/vabar/isaac_rooms]

Fetches four USD stages from NVIDIA's public Omniverse content bucket -- the full
warehouse, the simple warehouse, the hospital and one data-hall pod -- and,
recursively, every sublayer, reference, payload, texture and MDL material they
name, mirroring the bucket's layout under ``--out`` (about 2 GB). The rooms'
Unreal-exported materials also import ``OmniUe4Function``, ``OmniUe4Base`` and
``OmniUe4Translucent``, which no stage names; they are fetched beside every MDL
file that needs them. Without them those materials compile to nothing and the
room renders flat red.

The assets are NVIDIA's, under the Isaac Sim asset terms; they are downloaded
to this machine and never added to the repository. Re-running skips files
already present.
"""

from __future__ import annotations

import argparse
import posixpath
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
STAGES = [  # (bucket prefix, path under it, local subdirectory) -- the layout vabar/rooms.py reads
    ("Assets/Isaac/5.0/Isaac", "Environments/Simple_Warehouse/full_warehouse.usd", "assets"),
    ("Assets/Isaac/5.0/Isaac", "Environments/Simple_Warehouse/warehouse.usd", "assets"),
    ("Assets/Isaac/5.0/Isaac", "Environments/Hospital/hospital.usd", "assets"),
    ("", "Assets/DigitalTwin/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd", "dc"),
]
UE4_HELPERS = ("OmniUe4Function.mdl", "OmniUe4Base.mdl", "OmniUe4Translucent.mdl")


def _get(url: str, dst: Path) -> bool:
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(dst)
    return True


def fetch_stage(prefix: str, rel: str, out: Path) -> tuple[int, list[str]]:
    """``rel`` and everything it names, recursively. Returns (files present, paths the bucket does not have)."""
    from pxr import UsdUtils

    root = f"{BUCKET}/{prefix}".rstrip("/")
    seen: set[str] = set()
    todo, missing = [rel], []
    while todo:
        r = posixpath.normpath(todo.pop())
        if r in seen or r.startswith(".."):
            continue
        seen.add(r)
        dst = out / r
        if not _get(f"{root}/{urllib.parse.quote(r)}", dst):
            missing.append(r)
            continue
        if not r.lower().endswith((".usd", ".usda", ".usdc")):
            continue
        try:
            subs, refs, pays = UsdUtils.ExtractExternalReferences(str(dst))
        except Exception as exc:  # a layer that will not parse still leaves the rest usable
            print(f"  cannot parse {r}: {exc}")
            continue
        for a in (*subs, *refs, *pays):
            if not a or a.startswith(("omniverse:", "http")) or "<UDIM>" in a:
                continue
            if a.endswith(".mdl") and "/" not in a:  # core MDL (OmniPBR.mdl, ...) ships with the renderer
                continue
            todo.append(posixpath.join(posixpath.dirname(r), a))
    for d in sorted({posixpath.dirname(r) for r in seen if r.endswith(".mdl") and r not in missing}):
        for helper in UE4_HELPERS:
            _get(f"{root}/{urllib.parse.quote(posixpath.join(d, helper))}", out / d / helper)
    return len(seen) - len(missing), missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path.home() / ".cache" / "vabar" / "isaac_rooms"),
                    help="where the rooms go; vabar/rooms.py reads $VABAR_ROOMS_DIR, which defaults to this")
    out = Path(ap.parse_args().out).expanduser()
    for prefix, rel, sub in STAGES:
        n, missing = fetch_stage(prefix, rel, out / sub)
        print(f"{rel}: {n} files" + (f", {len(missing)} not on the bucket (the stage renders without them)" if missing else ""))
    print(f"rooms in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
