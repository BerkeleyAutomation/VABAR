# ManipulationNet cable-management board

`board.usdc` is the ManipulationNet Cable Management Benchmark board: the four task-board segment meshes as render geometry and one `0.30 x 0.40 x 0.00635 m` active-workspace box collider.

The triangle meshes do not collide; only the box does, which keeps the asset cheap for the cable solver while preserving the printed appearance. It is a self-contained, metre-scale, Z-up, kinematic rigid component. The lower panels keep their `30 mm` front branding strips, so the render envelope extends to `X=-0.18 m`, outside the collision box. The board's alignment clips are not included; they are cosmetic.

License: Apache-2.0, see [`LICENSE`](LICENSE).
