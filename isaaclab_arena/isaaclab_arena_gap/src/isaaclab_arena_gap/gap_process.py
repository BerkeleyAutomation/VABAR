"""GaP child-process plumbing: ports, launcher spawn, and child environment.

One GaP process runs per Arena episode and environment slot. The launcher
(``scripts/run_rss_graph.sh``, AUTOLab's graph through the RSS adapter) is
started in its own session so
``isaaclab_arena_gap.process_group.terminate_process_group`` can reap the whole group.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena_gap.gap_remote_policy import GapRemotePolicyArgs

DEFAULT_LAUNCHER = "run_rss_graph.sh"
"""``scripts/`` launcher a config without ``gap_launcher`` runs: AUTOLab's graph through the RSS adapter."""


def session_artifact_path(raw_path: str, config: GapRemotePolicyArgs) -> str:
    """Namespace a fleet artifact by immutable slot and episode identity."""
    if not raw_path or not config.gap_fleet_session:
        return raw_path
    path = Path(raw_path)
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    name = (
        f"{stem}.env-{config.gap_env_id:03d}."
        f"episode-{config.gap_episode_id:04d}{suffix}"
    )
    return str(path.with_name(name))


def free_port() -> int:
    """Allocate an ephemeral free TCP port (unique session/port per episode)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def default_spawn_gap(config: GapRemotePolicyArgs, port: int):
    """Spawn ONE GaP process via the launcher on a unique GAP_PORT.

    The bridge (``cap_franka_real_env.make_env`` / ``cap_bimanual_yam_env.make_env``)
    reads ``GAP_PORT``, so the spawned process binds the unique port the policy
    connects to. The launcher handles the VLM environment and AUTOLab's runtime.
    """
    arena_gap_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    launcher = config.gap_launcher or os.path.join(arena_gap_root, "scripts", DEFAULT_LAUNCHER)
    if not os.path.isabs(launcher):
        # Experiment configs name launchers relative to the repository root.
        launcher = os.path.join(os.path.dirname(arena_gap_root), launcher)
    env = dict(os.environ)
    env["GAP_PORT"] = str(port)
    env["GAP_GRAPH"] = config.gap_graph
    if config.graph_result_path:
        env["CAP_GRAPH_RESULT_PATH"] = session_artifact_path(
            config.graph_result_path, config
        )
    else:
        env.pop("CAP_GRAPH_RESULT_PATH", None)
    apply_adapter_tool_env(config, env)
    if config.gap_fleet_session:
        env["GAP_ENV_ID"] = str(config.gap_env_id)
        env["GAP_SEED"] = str(config.gap_session_seed)
        env["PYTHONHASHSEED"] = str(config.gap_session_seed)
        trace_root = config.gap_trace_dir or env.get("GAP_TRACE_DIR", "")
        if trace_root:
            env["GAP_TRACE_DIR"] = os.path.join(
                trace_root,
                f"env-{config.gap_env_id:03d}",
                f"episode-{config.gap_episode_id:04d}",
            )
    # start_new_session=True => the launcher (uv -> python -> ...) gets its own
    # process group/session, so terminate_process_group can SIGTERM/SIGKILL the WHOLE
    # group and nothing orphans.
    command = [launcher]
    if config.gap_inputs:
        command.extend(
            [
                "--inputs",
                *(
                    f"{key}={json.dumps(value)}"
                    for key, value in config.gap_inputs.items()
                ),
            ]
        )
    return subprocess.Popen(command, env=env, start_new_session=True)


def apply_adapter_tool_env(config: GapRemotePolicyArgs, env: dict[str, str]) -> None:
    """Stamp child-process settings for the selected pair and embodiment."""
    if config.gap_workspace:
        env["CAP_GAP_WORKSPACE"] = json.dumps(config.gap_workspace)
    else:
        env.pop("CAP_GAP_WORKSPACE", None)
    env["CAP_GAP_ARM_BASE_POSES"] = json.dumps(config.gap_arm_base_poses)
    statics = getattr(config, "gap_scene_static_poses", None)
    if statics:
        env["CAP_GAP_SCENE_STATICS"] = json.dumps(statics)
    else:
        env.pop("CAP_GAP_SCENE_STATICS", None)
    env["CAP_GAP_CONTROL_FREQUENCY_HZ"] = str(config.gap_control_frequency_hz)
    if config.gap_work_arm:
        env["CAP_GAP_WORK_ARM"] = config.gap_work_arm
    else:
        env.pop("CAP_GAP_WORK_ARM", None)
    base_position = config.gap_arm_base_position
    if base_position is None and config.gap_arm_base_positions:
        base_position = config.gap_arm_base_positions[0]
    if base_position is None:
        env.pop("CAP_GAP_ARM_BASE_POSITION", None)
    else:
        env["CAP_GAP_ARM_BASE_POSITION"] = ",".join(
            str(float(value)) for value in base_position
        )
    if config.gap_adaptation:
        env["GAP_ADAPTATION"] = config.gap_adaptation
    else:
        env.pop("GAP_ADAPTATION", None)

    camera_names = []
    if config.gap_camera:
        camera_names.append(config.gap_bridge_camera or "robot0_robotview")
    if config.gap_wrist_camera:
        camera_names.append(config.gap_wrist_bridge_camera or "eye_in_hand")
    if config.gap_aux_camera:
        camera_names.append(config.gap_aux_bridge_camera or config.gap_aux_camera)
    camera_names.extend(config.gap_extra_cameras.values())
    if camera_names:
        env["CAP_GAP_CAMERA_NAME"] = ",".join(camera_names)
    else:
        env.pop("CAP_GAP_CAMERA_NAME", None)
