# YAM kinematics layers

Two layers copied unchanged from the generated robot USD of the YAM. Their prim paths
match this asset's arm, so they apply as-is.

| Layer | What it does |
| --- | --- |
| `base_frame.usda` | Puts the base frame on the underside of the base, as I2RT does from v1.2.4 on; MuJoCo Menagerie's frame is 10 mm higher. |
| `autolab_v1.usda` | Reshapes the chain into AUTOLab's `yam_i2rt`: shoulder axis 112.5 mm, joint4→joint5 offset 70.385 mm, wrist offset 3.589 mm laterally. |

`../i2rt_yam_default.usda` has a `kinematics` variant set with two choices.
Each hand selects the one AUTOLab plans that hand's cells on.

| Choice | Selected by | What the arm is | AUTOLab embodiment |
| --- | --- | --- | --- |
| `menagerie` | `crank_4310`, `crank_4310_wide` | MuJoCo Menagerie's YAM, unchanged; base frame 10 mm above the underside | `yam_tool_hanging` and its cuRobo URDF (tool hanging) |
| `autolab_v1` | `linear_4310` | `base_frame.usda` + `autolab_v1.usda`, referenced; base frame on the underside | `yam_i2rt` (cable routing, USB-C) |

The `autolab_v1` chain matches the generated robot USD of the whole YAM
(`gripper=linear_4310`, `kinematics=autolab_v1`) joint for joint.

Why: AUTOLab GaP v1.0's PyRoKi IK checks `yam_i2rt_arm.urdf` against the simulated
arm. It refuses a tool-link-to-TCP drift over 1.0 mm and falls back to Newton IK.
On the linear hand the drift is 2.71 mm with MuJoCo Menagerie's chain and 0.001 mm with
`autolab_v1`. Tool hanging's cuRobo plans on a MuJoCo Menagerie-frame URDF, so the crank
hands keep that frame. Moving it costs 2.1 mm of drift there.

Mounting: with `autolab_v1` the base frame is the underside, so cable routing and
USB-C, which mount at the table top, stand the base on the table. Tool hanging keeps
MuJoCo Menagerie's frame and its 0.81 mount.

The variant set restates the arm body's centre of mass for each frame: z = 0.026
for `menagerie`, 0.036 for `autolab_v1`.
