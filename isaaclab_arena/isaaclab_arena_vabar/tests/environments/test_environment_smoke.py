"""Fresh-process construction/reset/step smoke for every environment YAML.

The live smoke is opt-in; the launcher helpers it relies on are unit-tested here.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch

from environments.environment_smoke_check import (
    _assert_finite,
    _factory_name_for,
    _with_safe_openusd_thread_limit,
)
from ib_support import ENVIRONMENT_YAMLS

LIVE = os.environ.get("ISAAC_CAP_RUN_LIVE_TESTS") == "1"
LIVE_CHECK = Path(__file__).with_name("environment_smoke_check.py")


@pytest.mark.parametrize("with_cameras", [False, True])
def test_smoke_launcher_enables_camera_support_when_requested(with_cameras):
    """Camera scene creation and AppLauncher support must be enabled together."""
    args = SimpleNamespace(with_cameras=with_cameras)

    configured = _with_safe_openusd_thread_limit(args)

    assert configured is args
    assert args.enable_cameras is with_cameras
    assert args.limit_cpu_threads == 1


def test_every_environment_requires_a_registered_factory() -> None:
    registry = SimpleNamespace(get_all_keys=lambda: ())
    with pytest.raises(
        RuntimeError, match="factory-backed environment is not registered"
    ):
        _factory_name_for("tool_sorting/missing.yaml", registry)


@pytest.mark.parametrize(
    ("value", "path", "accepted"),
    [
        pytest.param(
            torch.tensor([1.0, float("inf")]),
            "observation.camera.rgb",
            False,
            id="rgb-inf",
        ),
        pytest.param(
            torch.tensor([1.0, float("inf")]),
            "observation.camera.distance_to_image_plane",
            True,
            id="depth-positive-inf",
        ),
        pytest.param(
            torch.tensor([1.0, float("-inf")]),
            "observation.camera.distance_to_image_plane",
            False,
            id="depth-negative-inf",
        ),
        pytest.param(
            torch.tensor([1.0, float("nan")]),
            "observation.camera.distance_to_image_plane",
            False,
            id="depth-nan",
        ),
    ],
)
def test_smoke_finite_check_accepts_only_unbounded_depth(value, path, accepted):
    """Unbounded camera depth may use positive infinity, but no other tensor may."""
    if accepted:
        _assert_finite(value, path)
    else:
        with pytest.raises(AssertionError, match=path):
            _assert_finite(value, path)


@pytest.mark.live
@pytest.mark.skipif(
    not LIVE,
    reason="set ISAAC_CAP_RUN_LIVE_TESTS=1 to run fresh-process environment smokes",
)
@pytest.mark.parametrize("relative_yaml", ENVIRONMENT_YAMLS)
def test_environment_builds_resets_and_steps(relative_yaml: str) -> None:
    subprocess_utils = pytest.importorskip("isaaclab_arena.tests.utils.subprocess")
    result = subprocess_utils.run_subprocess(
        [os.sys.executable, str(LIVE_CHECK), relative_yaml],
        timeout_sec=1800,
        capture_output=True,
    )
    assert result is not None
    assert "INDUSTRIAL_ENVIRONMENT_SMOKE" in result.stdout, result.stderr
