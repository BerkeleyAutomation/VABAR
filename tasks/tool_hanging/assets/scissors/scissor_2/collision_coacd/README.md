# Scissor 2 automatic CoACD collision proxy

`tools/build_coacd_collisions.py` generates these convex hulls from
the original `scissor_2.obj`. CoACD manifold preprocessing is forced because
the supplied source is open and fragmented. Hull simplification is enabled
with at most 64 vertices per hull.

Exact parameters, source counts, bounds, and generated hulls are recorded in
`manifest.json`.
