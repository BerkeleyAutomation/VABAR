"""Cable routing environments."""

from isaaclab_arena_vabar.environments.cable_routing.environment import (
    CableRoutingEasyEnvironment,
    CableRoutingEasyEnvironmentCfg,
    CableRoutingMediumEnvironment,
    CableRoutingMediumEnvironmentCfg,
)

ENVIRONMENTS = (
    (CableRoutingEasyEnvironment, CableRoutingEasyEnvironmentCfg),
    (CableRoutingMediumEnvironment, CableRoutingMediumEnvironmentCfg),
)
