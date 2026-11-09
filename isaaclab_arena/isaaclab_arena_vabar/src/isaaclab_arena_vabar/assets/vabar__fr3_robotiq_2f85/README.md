# FR3 with Robotiq 2F-85

A generated robot USD of three MuJoCo Menagerie models
(<https://github.com/google-deepmind/mujoco_menagerie>). Each part keeps its upstream licence,
in a `LICENSE` file beside it, copied unchanged from Menagerie at `71f066ad`:

| Files | Menagerie model | Licence |
| --- | --- | --- |
| `franka_fr3_v1.usda`, `franka_fr3_v1/`, `franka_fr3_robotiq_2f85*.usda` | `franka_fr3` (Franka Research 3, Franka Robotics) | Apache-2.0, [`franka_fr3_v1/LICENSE`](franka_fr3_v1/LICENSE) |
| `robotiq_2f85/` | `robotiq_2f85_v4` (Robotiq 2F-85) | BSD-2-Clause, Copyright (c) 2013, ROS-Industrial, [`robotiq_2f85/LICENSE`](robotiq_2f85/LICENSE) |
| `panda_hand/` | `franka_emika_panda/hand.xml` (Franka hand) | Apache-2.0, [`panda_hand/LICENSE`](panda_hand/LICENSE) |

`franka_fr3_v1.usda` has a `gripper` variant set: `franka_hand` (the base layer's default,
`panda_hand/`), `robotiq_2f85` and `none`. `franka_fr3_robotiq_2f85.usda` selects
`robotiq_2f85`; the benchmark uses that layer.
