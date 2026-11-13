"""Every tool-hanging scene scores the RSS version's goal: hung, at rest and released, held for 1 s."""

import pytest

from ib_support import load_scene, scene_params

_TOOL_HANGING = [
    f"tool_hanging/{tool}_{level}.yaml"
    for tool in ("wrench", "plier", "scissors", "screwdriver")
    for level in ("easy", "medium")
]

# The RSS version's (> (hands-clear TOOL) R), tasks/tool_hanging/*_level{1,2}.task, by its rule
# (radius off the collision hulls plus 15 mm, rounded up to the centimetre) on Arena's own tools,
# measured from each tool's physical centre. The RSS documents state 0.11, 0.10, 0.20 and 0.21
# about their own tools' origins.
_HANDS_CLEAR_M = {"plier": 0.10, "scissors": 0.10, "screwdriver": 0.11, "wrench": 0.14}


@pytest.mark.parametrize("scene", _TOOL_HANGING)
def test_every_goal_carries_the_rss_level_clauses(scene):
    # contact-force >= 0.2, speed <= 0.15 and the release clause; the 1 s hold is the task's.
    params = scene_params(load_scene(scene))
    assert set(params) == {"episode_length_s", "tools", "fixtures", "goals"}
    tool = scene.split("/")[1].split("_")[0]
    for goal in params["goals"]:
        assert goal["contact_force_n"] == 0.2
        assert goal["speed_m_s"] == 0.15
        assert goal["hands_clear_m"] == _HANDS_CLEAR_M[tool]
        assert len(goal["release_point_xyz"]) == 3
