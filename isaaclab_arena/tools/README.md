# Tools

This directory contains independently movable benchmark-authoring utilities. They are deliberately
outside the root UV workspace: each tool owns the dependencies needed for authoring without adding
MuJoCo, RoboSuite, or Isaac Sim to the generic Arena/GaP runtime environment.

| Tool | Purpose |
|---|---|
| [`bind_collision_materials.py`](bind_collision_materials.py) | Bind each stored collision hull directly to its asset's physics material; the Newton importer ignores ancestor-only bindings. Idempotent. |
| [`build_screwdriver_slim_appearance.py`](build_screwdriver_slim_appearance.py) | Author `vabar__screwdriver_slim`'s visual-only AUTOLab v1.0 look (MTL colours and red tip marker) from the pinned AUTOLab checkout. Run in AUTOLab's environment; idempotent. |
| [`generate_environment_docs.py`](generate_environment_docs.py) | Render a fresh five-second embodiment-specific GIF per industrial environment and regenerate the environment README. |
| [`generate_asset_docs.py`](generate_asset_docs.py) | Run every movable-asset drop test in a fresh Isaac Sim process and regenerate the per-asset README/media tree and the catalog ([ASSET_DOCS.md](../docs/ASSET_DOCS.md)). |
| [`asset_quality_report.py`](asset_quality_report.py) | Build one Markdown drop-test report beside every movable asset. |
| [`render_asset_collisions.py`](render_asset_collisions.py) | Render deterministic thumbnails of each asset's enabled collider geometry. |
| [`asset_acceptance/`](asset_acceptance/) | Fresh-process drop acceptance inventory and harness used by the asset tools and tests. |
| [`build_stored_convex_collision.py`](build_stored_convex_collision.py) | Build a deterministic stored CoACD collision payload from one USD mesh. |
| [`collide_with_visual_mesh.py`](collide_with_visual_mesh.py) | Collide the tool-sort objects with their visible shape instead of the RSS collision box. |
| [`build_cable_appearance.py`](build_cable_appearance.py) | Author Cable Easy's visual-only room from the pinned AUTOLab checkout. Run in AUTOLab's cable environment. |
| [`generate_gear_mesh_assets.py`](generate_gear_mesh_assets.py) | Generate the package-owned USDA closure for the AUTOLab gear-mesh families from the pinned AUTOLab source. |
| [`generate_tool_hanging_box.py`](generate_tool_hanging_box.py) | Generate the AUTOLab Level-1 screwdriver cup as an open USDA compound. |
