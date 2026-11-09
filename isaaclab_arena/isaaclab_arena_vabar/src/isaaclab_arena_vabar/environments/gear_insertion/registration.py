"""Gear insertion environments."""

from isaaclab_arena_vabar.environments.gear_insertion.environment import (
    GearInsertionEasyNewtonEnvironment,
    GearInsertionEasyNewtonEnvironmentCfg,
    GearMeshPairNewtonEnvironment,
    GearMeshPairNewtonEnvironmentCfg,
    GearMeshTrainNewtonEnvironment,
    GearMeshTrainNewtonEnvironmentCfg,
)

ENVIRONMENTS = (
    (GearInsertionEasyNewtonEnvironment, GearInsertionEasyNewtonEnvironmentCfg),
    (GearMeshPairNewtonEnvironment, GearMeshPairNewtonEnvironmentCfg),
    (GearMeshTrainNewtonEnvironment, GearMeshTrainNewtonEnvironmentCfg),
)
