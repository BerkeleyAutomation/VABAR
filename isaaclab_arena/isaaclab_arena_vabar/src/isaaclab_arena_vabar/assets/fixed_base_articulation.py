"""Fixed-base articulation whose root-pose writes reach the MuJoCo solver."""

from __future__ import annotations

from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import NewtonManager
from newton import ModelFlags


class PlacedFixedBaseArticulation(Articulation):
    """Re-sync MuJoCo's welded root after every root-pose write.

    A fixed-base articulation's root pose is the parent transform of its world
    joint (``Model.joint_X_p``). Writing it updates Newton's model and bodies, so
    Newton's collision pipeline sees the new pose, but SolverMuJoCo copies that
    transform into its welded ``body_pos`` only when it is told that joint
    properties changed. Arena's reset placement writes the solved pose this way.
    Without the notification, MuJoCo keeps the root, and every joint anchored to
    it, at the construction pose while the colliders sit at the placed pose
    (the gear board's pinion anchor ended up 20-40 mm away from its teeth).
    """

    def _sync_fixed_root(self) -> None:
        if self.is_fixed_base:
            NewtonManager.add_model_change(ModelFlags.JOINT_PROPERTIES)

    def write_root_link_pose_to_sim_index(self, **kwargs) -> None:
        super().write_root_link_pose_to_sim_index(**kwargs)
        self._sync_fixed_root()

    def write_root_link_pose_to_sim_mask(self, **kwargs) -> None:
        super().write_root_link_pose_to_sim_mask(**kwargs)
        self._sync_fixed_root()

    def write_root_com_pose_to_sim_index(self, **kwargs) -> None:
        super().write_root_com_pose_to_sim_index(**kwargs)
        self._sync_fixed_root()

    def write_root_com_pose_to_sim_mask(self, **kwargs) -> None:
        super().write_root_com_pose_to_sim_mask(**kwargs)
        self._sync_fixed_root()
