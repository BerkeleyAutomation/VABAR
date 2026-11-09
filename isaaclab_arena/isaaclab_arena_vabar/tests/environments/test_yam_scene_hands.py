"""Every YAM scene fits the hand AUTOLab authored its task on, and scores with it."""

import pytest

from ib_support import load_scene, scene_params

from isaaclab_arena_vabar.embodiments.bimanual_yam.config import (
    CRANK_4310_WIDE,
    LINEAR_4310,
)

_TOOL_HANGING = [
    f"tool_hanging/{tool}_{level}.yaml"
    for tool in ("wrench", "plier", "scissors", "screwdriver")
    for level in ("easy", "medium")
]
_USBC = ["usbc_insertion/easy.yaml", "usbc_insertion/medium.yaml"]


@pytest.mark.parametrize("scene", _TOOL_HANGING)
def test_tool_hanging_fits_autolabs_wide_crank_hand(scene):
    document = load_scene(scene)
    assert document["embodiment"]["params"]["gripper"] == CRANK_4310_WIDE.variant


@pytest.mark.parametrize("scene", _USBC)
def test_usbc_fits_and_scores_autolabs_linear_hand(scene):
    document = load_scene(scene)
    assert document["embodiment"]["params"]["gripper"] == LINEAR_4310.variant
    hand = scene_params(document)["work_hand"]
    assert hand["gripper_joint_name"] == LINEAR_4310.driven_joints[0]
    assert hand["open_joint_m"] == pytest.approx(LINEAR_4310.open_position)
    assert hand["span_m"] == pytest.approx(2.0 * LINEAR_4310.open_position)
