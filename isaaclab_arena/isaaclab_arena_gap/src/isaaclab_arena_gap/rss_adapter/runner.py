"""GaP process entry: an AUTOLab graph through RoboSimStudio's policy agent, on Arena.

``GapRemotePolicy`` starts this through ``scripts/run_rss_graph.sh`` with the
environment it gives every GaP process (GAP_PORT, GAP_GRAPH,
CAP_GAP_ARM_BASE_POSITION, CAP_GAP_CAMERA_NAME, CAP_GRAPH_RESULT_PATH). The loop
is RoboSimStudio's rollout -- ``get_action``, ``env.step``, ``on_step`` -- with
each ``env.step`` one Arena control period.
"""

from __future__ import annotations

import os
import signal
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GraphBinding:
    """How one AUTOLab graph runs on its Arena task."""

    task: str
    """AUTOLab task document the graph belongs to, unless the experiment names
    another (``gap_inputs.autolab_task``). Identity and instruction only:
    AUTOLab's world frame is not Arena's, so none of its geometry is used. The
    document also resolves the graph name: an L2 document's ``gap_perception``
    is ``graphs/gap_perception_level2``, a cable document's ``gt`` is
    ``graphs/weave``."""
    robot_profile: str
    flags: dict
    """``SolutionGraph`` options, as AUTOLab's benchmark cell runs the graph."""
    calibrations: tuple[str, ...] = ()
    """Task entities that calibrate the hand, built on AUTOLab's own robot
    (see :func:`~isaaclab_arena_gap.rss_adapter.embodiments.hand_calibrations`)."""
    environment: tuple[tuple[str, str], ...] = ()
    """Process environment the benchmark cell sets (its ``env``), where Arena keeps it."""
    task_base_midpoint: tuple[float, float, float] | None = None
    """Where AUTOLab's cell stands the midpoint of its arm bases, in AUTOLab's world frame.

    Set for graphs whose task code plans against world constants of AUTOLab's own frame.
    The shadow then runs in that frame: Arena's mounts, cameras, fixtures and surface are
    translated by the offset that puts Arena's base midpoint here (:func:`frame_offset`).
    Unset, the shadow runs in Arena's world."""


BINDINGS = {
    # benchmark/vabar_v1.json gear_meshing: --motion curobo --no-privileged --perception vision --bus-timeout 180
    "gear_mesh/gap_perception": GraphBinding(
        task="gear_mesh/gearmesh-easy-single-01",
        robot_profile="fr3",
        flags={"privileged": False, "perception": "vision", "motion": "curobo", "bus_timeout": 180},
    ),
    # benchmark/vabar_v1.json syringe_packing: --motion curobo --no-privileged --perception vision --bus-timeout 180.
    # The graph reads no instruction, so one document serves every Arena syringe variant.
    "syringe_packing/gap_perception": GraphBinding(
        task="syringe_packing/tray_pair_both_t00",
        robot_profile="fr3",
        flags={"privileged": False, "perception": "vision", "motion": "curobo", "bus_timeout": 180},
    ),
    # benchmark/vabar_v1.json tool_sorting: --no-privileged --perception vision --bus-timeout 180, default motion.
    # Its instruction is a literal in the workflow, so the document is identity only.
    "tool_sort/gap_perception_loop": GraphBinding(
        task="tool_sort/level1_1",
        robot_profile="fr3",
        flags={"privileged": False, "perception": "vision", "bus_timeout": 180},
    ),
    # benchmark/vabar_v1.json tool_hanging: --motion curobo --no-privileged --perception vision --bus-timeout 180.
    "tool_hanging/gap_perception": GraphBinding(
        task="tool_hanging/wrenches_level1",
        robot_profile="yam_bimanual_wide",
        flags={"privileged": False, "perception": "vision", "motion": "curobo", "bus_timeout": 180},
    ),
    # benchmark/vabar_v1.json cable_insertion: --no-privileged --perception vision --renderer rtx
    # --bus-timeout 600 --settle-steps 600. Arena renders, so --renderer has no counterpart. The cell's
    # ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette is not carried: that segmenter thresholds hue for
    # AUTOLab's synthetic scenes and, on Arena's concrete floor, ranks the floor as the "orange plug";
    # the graph's own sam3.segment_text (the SAM 3 bundle) answers instead. Medium experiments name
    # usbc_insert/insert_medium.
    "usbc_insert/mate": GraphBinding(
        task="usbc_insert/insert_easy",
        robot_profile="yam_bimanual_linear",
        flags={"privileged": False, "perception": "vision", "bus_timeout": 600, "settle_steps": 600},
        calibrations=("rig",),
    ),
    # benchmark/vabar_v1.json cable_routing L1/L2: --no-privileged --perception vision --renderer rtx
    # --bus-timeout 600 --settle-steps 60, ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette. Arena renders, so
    # --renderer has no counterpart. The palette is kept: Arena's cable scene carries AUTOLab's cable
    # appearance (orange fixtures, white jacket, its room), the colours its hue thresholds were set on.
    # `gt` resolves to graphs/weave through the document; medium experiments name cable_route/port_medium.
    # Its tools read the pegs off the compiled scene when they register, so experiments pass the
    # fixtures' poses (gap_scene_statics).
    "cable_route/gt": GraphBinding(
        task="cable_route/port_easy",
        robot_profile="yam_bimanual_linear",
        flags={"privileged": False, "perception": "vision", "bus_timeout": 600, "settle_steps": 60},
        environment=(("ROBOSIMSTUDIO_SEGMENTATION_BACKEND", "palette"),),
        # The weave's final retreat parks the work hand at x = layout.WEAVE_CABLE_X - 0.12 = 0.30,
        # an AUTOLab world x 47.5 mm in front of the bases: behind the rod, out of the cable
        # camera's view. Arena stands the bases at x -0.335, so read in Arena's world that x is
        # 635 mm in front of them, under the cable camera, and the parked arm hid the port from
        # the plug's look. AUTOLab's yam_i2rt stands its bases at x 0.2525, y +-0.31, level with
        # bench_075's 0.75 m surface; Arena's cable cell stands the same rig 0.3475 m behind its
        # table centre and maps the fixtures off that centre, so one translation registers both.
        task_base_midpoint=(0.2525, 0.0, 0.75),
    ),
}


def frame_offset(binding: GraphBinding | None, mounts) -> tuple[float, float, float]:
    """Translation from Arena's world to the frame the binding's shadow runs in.

    ``mounts`` are Arena's arm base poses (x, y, z, qx, qy, qz, qw). Only a translation
    is applied, so the bases must stand unrotated, as both cells stand them.
    """
    if binding is None or binding.task_base_midpoint is None:
        return (0.0, 0.0, 0.0)
    poses = [tuple(float(v) for v in pose) for pose in mounts]
    if not poses:
        raise ValueError("a task frame needs Arena's arm base poses")
    for pose in poses:
        qx, qy, qz, qw = pose[3:7]
        if max(abs(qx), abs(qy), abs(qz)) > 1.0e-6 or abs(abs(qw) - 1.0) > 1.0e-6:
            raise ValueError(f"Arena base pose {pose} is rotated; the task frame is a translation only")
    midpoint = [sum(pose[k] for pose in poses) / len(poses) for k in range(3)]
    return tuple(float(t - a) for t, a in zip(binding.task_base_midpoint, midpoint, strict=True))


FR3_ARM = tuple(f"left/fr3_joint{i}" for i in range(1, 8))
FR3_GRIPPER = "left/gripper_open_fraction"
FR3_PASSIVE = ("driver_joint", "spring_link_joint", "follower")


def _surface_z() -> float | None:
    """The work surface the Arena experiment declares (``gap_workspace.surface_z``), if any.

    In Arena's world frame, the one the shadow, its cameras and every perceived point share.
    """
    import json

    value = json.loads(os.environ.get("CAP_GAP_WORKSPACE", "{}") or "{}").get("surface_z")
    return None if value is None else float(value)


def _scene_statics() -> dict:
    """The task's static fixtures at Arena's poses (``gap_scene_statics``), by fixture name."""
    import json

    return json.loads(os.environ.get("CAP_GAP_SCENE_STATICS", "{}") or "{}")


def _fr3(task, calibrations=(), binding=None):
    """Arena's FR3: the Franka bridge, one wire arm, the hand's linkage reported as passive joints."""
    from isaaclab_arena_gap.bridge_robot_io import ArenaBridgeRobotIO
    from isaaclab_arena_gap.cap_franka_real_env import _arm_base_position, make_env
    from isaaclab_arena_gap.rss_adapter.arena_env import ArenaEnv, translated
    from isaaclab_arena_gap.rss_adapter.embodiments import (
        FR3_KINEMATICS,
        fr3_robotiq,
        fr3_robotiq_home_like,
        hand_calibrations,
    )

    bridge, config = make_env("franka_real", 0)
    io = ArenaBridgeRobotIO(bridge, control_freq=config.control_freq, camera_names=tuple(bridge.camera_names),
                            passive_joints=True)
    x, y, z = _arm_base_position()
    offset = frame_offset(binding, [(x, y, z, 0.0, 0.0, 0.0, 1.0)])
    base = translated((x, y, z, 0.0, 0.0, 0.0, 1.0), offset)
    # Arena always fits the 2F-85; where AUTOLab authored the task on another hand, rest it as that hand rests.
    home = None if task.embodiment == FR3_KINEMATICS else fr3_robotiq_home_like(task.embodiment, base)
    # Arena reports the hand's mimic and linkage joints by their own names, which the shadow shares.
    passive = {name: name for name in io.joint_names if "/" not in name and any(p in name for p in FR3_PASSIVE)}
    return ArenaEnv(
        io, fr3_robotiq(base, home), task,
        arm_joints={"right": list(FR3_ARM)}, gripper_joint={"right": FR3_GRIPPER},
        passive_joints=passive, cameras=tuple(bridge.camera_names), surface_z=_surface_z(),
        calibrations=hand_calibrations(task, calibrations), statics=_scene_statics(), frame_offset=offset,
    )


def _yam_bimanual(task, gripper_variant=None, hand=None, kinematics=None, calibrations=(), binding=None):
    """Arena's two YAMs: the bimanual bridge, work arm first on the wire, mounts from the session."""
    import json

    from isaaclab_arena_gap.bridge_robot_io import ArenaBridgeRobotIO
    from isaaclab_arena_gap.cap_bimanual_yam_env import make_env
    from isaaclab_arena_gap.rss_adapter.arena_env import ArenaEnv, translated
    from isaaclab_arena_gap.rss_adapter.embodiments import YAM_HAND, YAM_KINEMATICS, hand_calibrations, yam_bimanual

    bridge, config = make_env("franka_real", 0)
    io = ArenaBridgeRobotIO(bridge, control_freq=config.control_freq, camera_names=tuple(bridge.camera_names),
                            arm_dof=6, wire_arms=bridge.arm_names, joint_prefix="yam")
    poses = dict(zip(("left", "right"), json.loads(os.environ["CAP_GAP_ARM_BASE_POSES"]), strict=True))
    mounts = {arm: (*poses[arm][0], *poses[arm][1]) for arm in ("left", "right")}
    offset = frame_offset(binding, mounts.values())
    mounts = {arm: translated(pose, offset) for arm, pose in mounts.items()}
    work, support = bridge.arm_names
    return ArenaEnv(
        io, yam_bimanual(mounts, gripper_variant, hand or YAM_HAND, kinematics or YAM_KINEMATICS), task,
        arm_joints={arm: [f"{arm}/yam_joint{i}" for i in range(1, 7)] for arm in mounts},
        gripper_joint={arm: f"{arm}/gripper_open_fraction" for arm in mounts},
        cameras=tuple(bridge.camera_names), surface_z=_surface_z(), work_arm=work, support_arm=support,
        calibrations=hand_calibrations(task, calibrations), statics=_scene_statics(), frame_offset=offset,
    )


ROBOTS = {
    "fr3": _fr3,
    "yam_bimanual": _yam_bimanual,
    # The 47.5 mm crank hand Arena's tool-hanging scenes select; RoboSimStudio's yam_linear_arena spec is that hand.
    "yam_bimanual_wide": lambda task, calibrations=(), binding=None: _yam_bimanual(
        task, "crank_4310_wide", calibrations=calibrations, binding=binding),
    # Arena's linear_4310 variant is RoboSimStudio's linear_4310 hand, on the I2RT YAM AUTOLab's USB-C cell uses.
    "yam_bimanual_linear": lambda task, calibrations=(), binding=None: _yam_bimanual(
        task, "linear_4310", "linear_4310", "yam_i2rt", calibrations, binding),
}
"""Each Arena robot profile's bridge client and shadow."""


def experiment_inputs(argv: list[str]) -> dict:
    """The experiment's ``gap_inputs``, as GapRemotePolicy passes them: ``--inputs key=<json> ...``."""
    import json

    if "--inputs" not in argv:
        return {}
    inputs = {}
    for item in argv[argv.index("--inputs") + 1:]:
        if item.startswith("--"):
            break
        key, _, value = item.partition("=")
        inputs[key] = json.loads(value)
    return inputs


GRAPH_ADAPTATIONS = Path(__file__).resolve().parents[3] / "graph_adaptations"


def adapted_graph(source: str | Path, adaptation: str) -> str:
    """A scratch copy of AUTOLab's graph at ``source`` with ``graph_adaptations/<adaptation>.patch`` applied.

    For graphs whose perception names how AUTOLab's scene looks where Arena's scene
    looks different on purpose (USB-C's plug and receptacle). The pinned checkout is
    never touched, and a patch that no longer applies stops the run.
    """
    import shutil
    import subprocess
    import tempfile

    patch = GRAPH_ADAPTATIONS / f"{adaptation}.patch"
    if not patch.is_file():
        raise SystemExit(f"no graph adaptation {adaptation!r} in {GRAPH_ADAPTATIONS}")
    source = Path(source)
    # A graph that reuses a sibling graph's scripts (tool hanging's level-2 documents run
    # ``../gap_perception/scripts``) is patched together with that sibling: such a patch names
    # paths relative to the graphs directory and must patch this graph's own workflow, so the
    # whole directory is copied and patched there.
    touched = {line[6:].strip() for line in patch.read_text().splitlines() if line.startswith("+++ b/")}
    graphs_root = f"{source.name}/workflow.json" in touched
    scratch = Path(tempfile.mkdtemp(prefix=f"rss_{adaptation}_"))
    if graphs_root:
        shutil.copytree(source.parent, scratch / source.parent.name)
        root, copy = scratch / source.parent.name, scratch / source.parent.name / source.name
    else:
        copy = root = scratch / source.name
        shutil.copytree(source, copy)
    applied = subprocess.run(["git", "apply", "-p1", str(patch)], cwd=root, capture_output=True, text=True)
    if applied.returncode:
        raise SystemExit(f"graph adaptation {adaptation} does not apply to {source}: {applied.stderr.strip()}")
    return str(copy)


def document_for(graph: str, inputs: dict) -> str:
    """The AUTOLab document a run uses: the experiment's ``autolab_task``, else the binding's.

    One binding serves every Arena variant of a graph. Where AUTOLab splits a family across
    documents the graph reads differently (USB-C easy and medium, tool hanging's levels), the
    experiment names its own.
    """
    name = str(inputs.get("autolab_task") or BINDINGS[graph].task)
    if name.split("/", 1)[0] != graph.split("/", 1)[0]:
        raise SystemExit(f"autolab_task {name!r} is not in {graph!r}'s suite")
    return name


def main() -> int:
    import vabar

    vabar.install()
    from robosimstudio_bench.taskbuild.registry import get_task
    from robosimstudio_policy_agent.graph_solution import SolutionGraph

    from isaaclab_arena_gap.bridge_contract import atomic_write_json, final_graph_result, started_graph_result

    graph = os.environ["GAP_GRAPH"]
    binding = BINDINGS.get(graph)
    if binding is None:
        raise SystemExit(f"no Arena binding for AUTOLab graph {graph!r}; known: {sorted(BINDINGS)}")
    result_path = Path(os.environ["CAP_GRAPH_RESULT_PATH"]) if os.environ.get("CAP_GRAPH_RESULT_PATH") else None
    if result_path:
        atomic_write_json(result_path, started_graph_result())

    os.environ.update(dict(binding.environment))
    inputs = experiment_inputs(sys.argv[1:])
    task = get_task(document_for(graph, inputs))
    env = ROBOTS[binding.robot_profile](task, calibrations=binding.calibrations, binding=binding)
    # Some graphs perceive what the instruction names (tool hanging's target tool), so the
    # experiment's own instruction wins over the document's.
    instruction = inputs.get("instruction")
    if instruction:
        env.episode.instruction = str(instruction)
    graph_name = graph.split("/", 1)[1]
    if os.environ.get("GAP_ADAPTATION"):
        graph_name = adapted_graph(SolutionGraph._graph_source(env, graph_name), os.environ["GAP_ADAPTATION"])
    solution = SolutionGraph(env, graph=graph_name, trace_dir=os.environ.get("GAP_TRACE_DIR", ""),
                             **binding.flags)
    print(f"[rss-adapter] {graph} on {task.full_name} ({solution.source_graph}): arms {list(env.handles.arms)}, "
          f"cameras {list(env._camera_names)}, statics {sorted(env.handles.statics)}, "
          f"frame offset {[round(float(v), 4) for v in env.frame_offset]}", flush=True)

    def finish(*_args) -> None:
        if result_path:
            atomic_write_json(result_path, final_graph_result(solution._result))
        solution.write_trace()

    # Arena stops the session at episode end; record what the graph got to first.
    signal.signal(signal.SIGTERM, lambda *a: (finish(), os._exit(0)))
    obs, info = env.reset()
    solution.reset(obs, info)
    try:
        while True:
            action = solution.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action, mode=solution.control_mode)
            solution.on_step(obs, reward, info)
            if terminated or truncated:
                break
    except Exception:
        traceback.print_exc()
    finally:
        print(f"[rss-adapter] graph {solution.trace.get('graph_result')} error={solution.trace.get('graph_error')}",
              flush=True)
        finish()
        solution.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
