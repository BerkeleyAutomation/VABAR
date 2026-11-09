# Wrench hooks

The visible hook meshes are preserved from `payloads/meshes.usdc`.

`wrench_hook_2.usda` uses `hook_2_collision.usdc`, a 20-part convex decomposition
of its visual geometry. MuJoCo contact generation convexifies each mesh collider;
a single concave J-hook mesh therefore fills the cavity needed for hanging.
The replacement disables collision on the visual mesh and references the hulls
under the same fixture transform. It changes contact geometry only.

Generated with the repository utility:

```bash
uv run python tools/build_stored_convex_collision.py \
  isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets/vabar__wrench_hooks/wrench_hook_2.usda \
  /Asset/fixture/Geometry /tmp/hook_2_collision.usdc --threshold 0.05 --merge
```

The generator preserves mesh-local coordinates. Referencing its payload under
`fixture` applies the original 90-degree rotation exactly once. The cavity regression is in
`isaaclab_arena_vabar/tests/assets/test_wrench_hook_collision.py`. Approximation
expands the original bounds by about 0.33 mm. Collider validation alone is not task success.
