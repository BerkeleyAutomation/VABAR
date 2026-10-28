# Scissor 1 automatic CoACD collision proxy

This folder contains the automatic CoACD collision proxy currently selected by
the scissors task; `../collision_compound/` remains the handcrafted alternative.

`tools/build_coacd_collisions.py` sends the original fragmented OBJ to
CoACD with manifold preprocessing forced on, a fixed seed, and the parameters
recorded in `manifest.json`. Hull simplification is enabled with at most 64
vertices per hull. The convex hulls are saved separately so the rigid importer
attaches them all to one scissor body.
