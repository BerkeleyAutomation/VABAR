# SPDX-License-Identifier: Apache-2.0
"""Split the robosuite tool-sort export into a robot and a room.

The export is one MJCF holding an arena, two bins, four tools and a Franka on a
pedestal. That is fine for the suite it came with and useless for anything else:
the Panda cannot stand anywhere but this arena, and no other robot can work this
bench.

So it is cut in two, **without touching what the native pairing builds**:

* ``panda_standalone.xml`` -- the ``robot0_base`` subtree at its own origin, with
  the ``mount0_*`` pedestal chain dropped, plus the nine actuators and two
  sensors that drive it. The pedestal goes because it is furniture: it exists to
  hold the arm at 0.912 m over *this* bench, and a robot asset that carries its
  own plinth cannot be mounted anywhere else. That height becomes
  ``EmbodimentSpec.base_offset``.
* ``scene_norobot.xml`` -- everything else, so another embodiment can be stood in
  front of the same bins.

Both are written **beside the export**, because ``compiler`` sets
``meshdir="."`` and every mesh and texture is named relative to that directory.
Writing them elsewhere would mean copying 66 meshes and 17 textures to say the
same thing.

The native ``tool_sort`` pairing still imports ``scene.xml`` whole and is
unaffected by any of this -- its binding says ``scene_resident=True``, so the
compiler indexes the Panda already in the arena instead of importing the
standalone one beside it. The extraction exists for every *other* pairing.

    uv run python tasks/tool_sort/rigs/split_tool_sort.py
    uv run python tasks/tool_sort/rigs/split_tool_sort.py --check
"""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
"""The checkout. The export is the shared vendored one under ``assets/``, three
directories up from the suite's ``rigs/``."""
EXPORT = ROOT / "assets" / "vab_tool_sort"


def exports() -> list[Path]:
    """Every tool-sort export directory, the shared one first.

    Five, not one: ``sort_all`` and ``hammer`` share the top-level export and
    the other four variants carry their own object sets, so a stripped arena
    has to exist beside each of them or those variants cannot host a foreign
    robot at all. Discovered rather than listed, so a sixth export needs no
    edit here.
    """
    return [EXPORT, *sorted(p.parent for p in EXPORT.glob("*/scene.xml"))]


ROBOT_ROOT = "robot0_base"
"""Worldbody child that is the arm."""
PEDESTAL = "mount0_base"
"""Its child that is the plinth, and therefore the arena's rather than the robot's."""
ROBOT_PREFIXES = ("robot0_", "gripper0_")
"""What the actuators and sensors that drive the arm are named."""


def _worldbody(root: ET.Element) -> ET.Element:
    body = root.find("worldbody")
    if body is None:
        raise ValueError("no <worldbody>")
    return body


def _named_child(parent: ET.Element, tag: str, name: str) -> ET.Element | None:
    return next((c for c in parent if c.tag == tag and c.get("name") == name), None)


def _referenced(element: ET.Element) -> tuple[set[str], set[str]]:
    """Every mesh and material name used anywhere under ``element``."""
    meshes, materials = set(), set()
    for node in element.iter():
        if node.get("mesh"):
            meshes.add(node.get("mesh"))
        if node.get("material"):
            materials.add(node.get("material"))
    return meshes, materials


def _prune_assets(root: ET.Element, keep_meshes: set[str], keep_materials: set[str]) -> int:
    """Drop asset entries nothing left in the file refers to.

    Materials are followed to their textures, so a kept material never loses its
    image -- which is the one way this can go wrong quietly, since MuJoCo
    tolerates an unresolved material name by drawing untextured.
    """
    asset = root.find("asset")
    if asset is None:
        return 0
    keep_textures = {
        m.get("texture") for m in asset if m.tag == "material" and m.get("name") in keep_materials and m.get("texture")
    }
    dropped = 0
    for entry in list(asset):
        name = entry.get("name")
        if entry.tag == "mesh" and name not in keep_meshes:
            asset.remove(entry)
        elif entry.tag == "material" and name not in keep_materials:
            asset.remove(entry)
        elif entry.tag == "texture" and name is not None and name not in keep_textures:
            asset.remove(entry)
        else:
            continue
        dropped += 1
    return dropped


def build_panda(source: ET.ElementTree) -> tuple[ET.ElementTree, dict[str, int]]:
    """The arm alone, at its own base origin."""
    tree = copy.deepcopy(source)
    root = tree.getroot()
    world = _worldbody(root)

    robot = _named_child(world, "body", ROBOT_ROOT)
    if robot is None:
        raise ValueError(f"no <body name={ROBOT_ROOT!r}> to extract")

    pedestal = _named_child(robot, "body", PEDESTAL)
    if pedestal is not None:
        robot.remove(pedestal)

    was = robot.get("pos", "0 0 0")
    robot.set("pos", "0 0 0")

    for child in list(world):
        if child is not robot:
            world.remove(child)

    # Actuators and sensors that name a joint or site the arm no longer has
    # would fail the compile, so keep exactly the arm's own.
    kept = {"actuator": 0, "sensor": 0}
    for tag in ("actuator", "sensor"):
        section = root.find(tag)
        if section is None:
            continue
        for entry in list(section):
            target = entry.get("joint") or entry.get("site") or entry.get("name") or ""
            if not target.startswith(ROBOT_PREFIXES):
                section.remove(entry)
            else:
                kept[tag] += 1

    meshes, materials = _referenced(root)
    dropped = _prune_assets(root, meshes, materials)
    return tree, {"pedestal": int(pedestal is not None), "assets_dropped": dropped, "was_at": was, **kept}


def build_norobot(source: ET.ElementTree) -> tuple[ET.ElementTree, dict[str, int]]:
    """The arena, the bins and the tools -- with nothing to drive them."""
    tree = copy.deepcopy(source)
    root = tree.getroot()
    world = _worldbody(root)

    robot = _named_child(world, "body", ROBOT_ROOT)
    if robot is None:
        raise ValueError(f"no <body name={ROBOT_ROOT!r}> to remove")
    world.remove(robot)

    removed = {"actuator": 0, "sensor": 0}
    for tag in ("actuator", "sensor"):
        section = root.find(tag)
        if section is None:
            continue
        for entry in list(section):
            section.remove(entry)
            removed[tag] += 1
        if not list(section):
            root.remove(section)

    meshes, materials = _referenced(root)
    dropped = _prune_assets(root, meshes, materials)
    return tree, {"assets_dropped": dropped, **removed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report what would be written, write nothing")
    args = parser.parse_args()

    for directory in exports():
        source_path = directory / "scene.xml"
        if not source_path.is_file():
            raise SystemExit(f"missing {source_path}")
        source = ET.parse(source_path)
        rel = directory.relative_to(ROOT / "assets")

        # The arm is the same in every export, so it is extracted once from the
        # shared one; every export gets its own stripped arena, because that is
        # what differs between them.
        builds = [("scene_norobot.xml", build_norobot)]
        if directory == EXPORT:
            builds.insert(0, ("panda_standalone.xml", build_panda))

        for name, build in builds:
            tree, report = build(source)
            target = directory / name
            world = _worldbody(tree.getroot())
            print(f"[{rel}/{name}] worldbody now {[c.get('name') or f'<{c.tag}>' for c in world]}")
            print(f"[{rel}/{name}] {report}")
            if not args.check:
                tree.write(target, encoding="unicode")
                print(f"[{rel}/{name}] wrote {target.relative_to(ROOT)}")
    print("[split_tool_sort] done" + (" (check only)" if args.check else ""))


if __name__ == "__main__":
    main()
