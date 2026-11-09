"""Regression coverage for generated environment camera artifacts."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import re
import sys

from PIL import Image
import yaml

from ib_support import TOOLS_ROOT


SPEC = importlib.util.spec_from_file_location(
    "generate_environment_docs", TOOLS_ROOT / "generate_environment_docs.py"
)
assert SPEC is not None and SPEC.loader is not None
DOCS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DOCS
SPEC.loader.exec_module(DOCS)


def test_discovery_excludes_shared_physics_yaml() -> None:
    entries = DOCS._discover_environments()
    assert all(not entry.yaml_path.name.endswith("_env_config.yaml") for entry in entries)


def test_committed_readme_text_matches_generator() -> None:
    DOCS.check_environment_docs(include_media=False)


def test_fr3_environments_document_their_rig_context_view() -> None:
    cameras = {
        (entry.family, entry.name): entry.camera_name
        for entry in DOCS._discover_environments()
    }
    assert cameras["gear_insertion", "easy"] == "iso_camera"
    assert cameras["syringe_disposal", "single"] == "oblique_camera"
    assert cameras["tool_sorting", "easy_1"] == "oblique_camera"
    assert cameras["usbc_insertion", "medium"] == "top_camera"


def test_render_children_run_inside_the_venv(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_PYTHON", raising=False)
    monkeypatch.delenv("ARENA_ENV_ROOT", raising=False)
    python = DOCS._arena_roots()[2]
    assert python.parent.parent.name == ".venv", python


def _cm(value_m: float) -> str:
    return f"{round(value_m * 100, 2):g}"


def _yaw_phrase(yaw_half_rad: float) -> str:
    if math.isclose(yaw_half_rad, math.pi):
        return "full yaw"
    return f"+/-{round(math.degrees(yaw_half_rad), 1):g} deg yaw"


_PROSE_CLAIM = re.compile(
    r"\d+(?:\.\d+)? x \d+(?:\.\d+)? cm"
    r"|\+/-\d+(?:\.\d+)? cm"
    r"|\+/-\d+(?:\.\d+)? deg yaw"
    r"|full yaw"
    r"|fixed heading"
)


def _relation_phrases(relation: dict) -> tuple[str, ...]:
    """Return the prose phrases one placement relation supports."""
    params = relation.get("params") or {}
    if relation["kind"] in ("position_limits", "position_limits_box"):
        return (
            f"{_cm(params['x_max'] - params['x_min'])} x "
            f"{_cm(params['y_max'] - params['y_min'])} cm",
        )
    if relation["kind"] != "random_around_solution":
        return ()
    phrases = [
        f"+/-{_cm(params[axis])} cm"
        for axis in ("x_half_m", "y_half_m")
        if axis in params
    ]
    if "yaw_half_rad" in params:
        phrases.append(_yaw_phrase(params["yaw_half_rad"]))
    return tuple(phrases)


def _factory_yaw_claims() -> dict[str, str]:
    """Return the full-yaw resets the tool-sort factories add outside the YAML."""
    from isaaclab_arena_vabar.environments.tool_sorting.registration import (
        ENVIRONMENTS,
    )

    return {
        descriptor.relative_graph_path: "full yaw"
        for descriptor in ENVIRONMENTS
        if descriptor.random_yaw_init
    }


def test_reset_randomization_prose_matches_yaml_ranges() -> None:
    # The gear-family variation resizes gear_a's box per sampled tooth count.
    variation_owned = {
        ("gear_insertion/easy.yaml", "gear_a", "position_limits_box")
    }
    factory_claims = _factory_yaw_claims()
    for entry in DOCS._discover_environments():
        relative_yaml = entry.relative_yaml.as_posix()
        prose = entry.reset_randomization
        document = yaml.safe_load(entry.yaml_path.read_text())
        supported = set()
        randomized_yaw = relative_yaml in factory_claims
        if randomized_yaw:
            supported.add(factory_claims[relative_yaml])
        for relation in document.get("relations") or ():
            subject = relation.get("subject", "")
            kind = relation["kind"]
            if subject.startswith("distractor_"):
                continue
            phrases = _relation_phrases(relation)
            supported.update(phrases)
            randomized_yaw |= "yaw_half_rad" in (relation.get("params") or {})
            if (relative_yaml, subject, kind) in variation_owned:
                continue
            # Every YAML range must be described ...
            for phrase in phrases:
                assert phrase in prose, (relative_yaml, subject, phrase)
        # ... and every quantitative prose claim must come from the YAML.
        for claim in _PROSE_CLAIM.findall(prose):
            if claim == "fixed heading":
                assert not randomized_yaw, (relative_yaml, claim)
            else:
                assert claim in supported, (relative_yaml, claim)


def test_render_artifact_accepts_authored_capture_resolution(tmp_path: Path) -> None:
    entry = DOCS.EnvironmentEntry(
        family="tool_hanging",
        name="wrench_easy",
        yaml_path=tmp_path / "wrench_easy.yaml",
        relative_yaml=Path("tool_hanging/wrench_easy.yaml"),
    )
    animation_path = tmp_path / "wrench_easy.gif"
    frames = [
        Image.new("RGB", DOCS.GIF_SIZE, (index, 0, 0))
        for index in range(DOCS.GIF_FRAME_COUNT)
    ]
    frames[0].save(
        animation_path,
        save_all=True,
        append_images=frames[1:],
        duration=DOCS.GIF_FRAME_DURATION_MS,
        loop=0,
    )
    report = {
        "camera": entry.camera_name,
        "environment": entry.name,
        "relative_yaml": entry.relative_yaml.as_posix(),
        "render_contract": DOCS.RENDER_CONTRACT,
        "capture_shape": [480, 640, 3],
        "encoded_size": list(DOCS.GIF_SIZE),
        "duration_s": DOCS.GIF_DURATION_SECONDS,
        "simulated_duration_s": DOCS.GIF_DURATION_SECONDS,
        "step_dt_s": 1.0 / 60.0,
        "fps": DOCS.GIF_FPS,
        "frame_count": DOCS.GIF_FRAME_COUNT,
        "min": 0,
        "max": 255,
        "mean": 127.5,
        "std": 2.0,
    }
    report_path = tmp_path / "wrench_easy.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    assert DOCS._validate_render_artifact(entry, animation_path, report_path) == report
