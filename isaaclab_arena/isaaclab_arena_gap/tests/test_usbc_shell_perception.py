"""Recorded RGB-D shell fits preserve a measured shoulder with a gray tip."""

from pathlib import Path

import numpy as np
import pytest

from isaaclab_arena_gap.task_sources import autolab_root
from isaaclab_arena_gap.usbc_shell_perception import fit_plug_shell

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def shape():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("rtree")
    mesh = trimesh.load(autolab_root() / "tasks/usbc_insert/assets/USB_C_Plug.stl")
    return {
        "surface_vertices": np.asarray(mesh.vertices) * 2,
        "surface_triangles": mesh.faces,
    }


def test_recorded_shell_and_gray_tip_fit(shape):
    with np.load(FIXTURES / "usbc_shell_gray_tip.npz") as sample:
        fit = fit_plug_shell(
            sample["points"],
            shape,
            sample["axis"],
            sample["up"],
            sample["scene_points"],
        )
        assert fit is not None
        assert fit["residual_m"] < 0.0001
        assert fit["inlier_count"] > 800
        assert fit["point"] == pytest.approx(sample["expected_point"], abs=0.0003)
        assert fit["tip_support_count"] >= 40


def test_missing_tip_depth_rejects_inferred_shoulder(shape):
    with np.load(FIXTURES / "usbc_shell_gray_tip.npz") as sample:
        assert (
            fit_plug_shell(
                sample["points"], shape, sample["axis"], sample["up"], np.empty((0, 3))
            )
            is None
        )


def test_truncated_shell_edge_does_not_become_a_shoulder(shape):
    with np.load(FIXTURES / "usbc_shell_gray_tip.npz") as sample:
        points = sample["points"]
        along = points @ sample["axis"]
        cropped = points[along < along.max() - 0.006]
        assert (
            fit_plug_shell(
                cropped, shape, sample["axis"], sample["up"], sample["scene_points"]
            )
            is None
        )
