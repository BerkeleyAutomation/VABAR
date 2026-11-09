"""VABAR on Isaac Lab Arena — glue for running the GaP agent against Isaac Lab Arena.

See README.md and docs/DESIGN.md.
"""

__version__ = "0.0.1"

# Installed at first import of the package, not by individual policies, so no
# spawn path can miss it and run the caging linkage geometry (see
# gripper_collision_fix). Inert outside Isaac Lab: registers an import hook.
from isaaclab_arena_gap import gripper_collision_fix as _gripper_collision_fix

_gripper_collision_fix.install_gripper_fix()
