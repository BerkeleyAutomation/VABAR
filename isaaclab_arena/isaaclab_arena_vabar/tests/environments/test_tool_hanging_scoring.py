"""Every tool-hanging scene scores AUTOLab v1.0's goal on the final state."""

import pytest

from ib_support import load_scene, scene_params

_TOOL_HANGING = [
    f"tool_hanging/{tool}_{level}.yaml"
    for tool in ("wrench", "plier", "scissors", "screwdriver")
    for level in ("easy", "medium")
]


@pytest.mark.parametrize("scene", _TOOL_HANGING)
def test_every_goal_carries_autolabs_level_clauses(scene):
    # tasks/tool_hanging/*_level{1,2}.task: contact-force >= 0.2 and speed <= 0.15, no hold-for
    # and no release check.
    params = scene_params(load_scene(scene))
    assert set(params) == {"episode_length_s", "tools", "fixtures", "goals"}
    for goal in params["goals"]:
        assert goal["contact_force_n"] == 0.2
        assert goal["speed_m_s"] == 0.15
