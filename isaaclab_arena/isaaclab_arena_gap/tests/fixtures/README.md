`artist_wrench_cloud.npz` is the overhead RGB-D point cloud from the seed-42
wrench-easy evaluation before grasping. It contains
1,804 observed/support-completed XYZ points in meters, without RGB or simulator
object state. The measured closed-ring center was approximately (0.7665, 0.001)
in world XY. The unadapted adjustable-wrench profile selected the opposite end
near (0.568, 0.002). The regression replays this observation through the RSS
planar and held CAD registration using the artist asset's actual mesh.

`artist_hook_cloud.npz` contains the 172 hook points from the same camera rig in
the artist-CAD evaluation. The RSS oblique-depth heuristic placed the shaft
center at Z=1.161609 m; the authored/measured centerline was Z=1.154716 m. The
hook-CAD regression fits the observed points and requires centerline accuracy
within 1.5 mm, without passing the measured fixture pose to the fit.

`usbc_shell_gray_tip.npz` contains camera-derived orange-shell points, nearby
unmasked depth points, and measured top-plane axis/up hints from one USB-C
evaluation. Segmentation uses `orange plug` plus the existing HSV shell gate.
No simulated object poses are included. `expected_point` is a regression snapshot
of the shell-edge estimate, not independent ground truth. Tests also remove tip
depth and truncate the shell to ensure unsupported shoulders are rejected.
