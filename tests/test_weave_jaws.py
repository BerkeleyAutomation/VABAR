# SPDX-License-Identifier: Apache-2.0
"""The weave graph's crossing take keeps its open fingertips off the pegs.

cable_routing/L2's pegs stand 72-80 mm apart and each crossing takes the cable
50-140 mm past its station, so a fully open jaw (47.5 mm a finger) put a fingertip
on the next spool on most station-1 takes. See docs/validation/cable_routing_l2.md.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "tasks" / "cable_route" / "graphs" / "weave" / "scripts"


@pytest.fixture(scope="module")
def plan_crossing():
    pytest.importorskip("gap")
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("weave_plan_crossing", SCRIPTS / "plan_crossing.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(SCRIPTS))


def _rod_along_y(x: float, y0: float = -0.40, y1: float = 0.40, n: int = 42) -> list[list[float]]:
    return [[x, float(y), 0.7535] for y in np.linspace(y0, y1, n)]


def test_full_opening_where_nothing_is_near(plan_crossing):
    # port_easy's spacing: the pegs are far from any take.
    posts = [[0.53, -0.20], [0.56, 0.04]]
    plan = {"grasp_m": [0.44, -0.10, 0.76], "grips": {"open": {"command_m": 0.0475}}}
    opening, note = plan_crossing.take_opening(_rod_along_y(0.44), plan, posts)
    assert opening == pytest.approx(0.0475)
    assert note["tip_to_post_m"] >= plan_crossing.PEG_KEEPOUT


def test_narrows_until_the_tips_clear_the_next_spool(plan_crossing):
    # A station-1 take just short of peg2: the +x fingertip would land 15 mm off its axis.
    posts = [[0.53, -0.28], [0.555, -0.206], [0.515, -0.130], [0.565, -0.054]]
    grasp = [0.4675, -0.145]
    plan = {"grasp_m": [*grasp, 0.76], "grips": {"open": {"command_m": 0.0475}}}
    rod = _rod_along_y(0.4675)
    t = plan_crossing.rod_tangent(rod, grasp)
    assert plan_crossing.nearest_tip_to_post(grasp, t, posts, 0.0475) < 0.02
    opening, note = plan_crossing.take_opening(rod, plan, posts)
    assert plan_crossing.JAW_MIN_REACH <= opening < 0.0475
    assert plan_crossing.jaws_clear_posts(grasp, t, posts, opening)
    assert note["tip_to_post_m"] >= plan_crossing.PEG_KEEPOUT - 1e-9


def test_no_opening_clears_a_post_beside_the_cable(plan_crossing):
    # The post sits inside the closing stroke, 20 mm off the rod: the tip clears it
    # when open, but the finger would close onto it, so no opening is accepted.
    posts = [[0.46, -0.10]]
    grasp = [0.44, -0.10]
    rod = _rod_along_y(0.44)
    t = plan_crossing.rod_tangent(rod, grasp)
    assert plan_crossing.jaw_reach_clearing_posts(grasp, t, posts, 0.0475) is None
