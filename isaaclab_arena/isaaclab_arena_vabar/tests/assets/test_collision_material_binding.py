"""Stored collision hulls carry their asset's physics material directly.

The Newton USD importer ignores a physics-material binding that lives only on an
ancestor of the collider, so each hull must be bound itself
(``tools/bind_collision_materials.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

Usd = pytest.importorskip("pxr.Usd")
from pxr import UsdPhysics, UsdShade  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "src/isaaclab_arena_vabar/assets"
WRAPPERS = sorted(
    path for path in ROOT.glob("*/*.usda") if 'def Xform "StoredConvexCollision"' in path.read_text()
)


def _physics_binding(prim):
    return UsdShade.MaterialBindingAPI(prim).GetDirectBinding("physics").GetMaterialPath()


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=lambda path: path.parent.name)
def test_every_stored_hull_is_bound_to_its_parent_material(wrapper: Path) -> None:
    stage = Usd.Stage.Open(str(wrapper))
    bound = checked = 0
    for collision in stage.Traverse():
        if collision.GetName() != "StoredConvexCollision" or not collision.IsActive():
            continue
        material = _physics_binding(collision)
        if material.isEmpty:
            continue
        bound += 1
        for hull in Usd.PrimRange(collision):
            if hull != collision and hull.HasAPI(UsdPhysics.CollisionAPI):
                assert _physics_binding(hull) == material, hull.GetPath()
                checked += 1
    assert checked or not bound, f"{wrapper.name}: bound collision Xform without hulls"
