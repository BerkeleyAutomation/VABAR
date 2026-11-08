"""Task setup and terminal recording through the current Arena interfaces."""

from types import SimpleNamespace

import pytest

from ib_support import load_scene, scene_params

pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")

from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg


@pytest.mark.parametrize("variant", ["easy", "medium"])
def test_usbc_termination_uses_arena_criteria_and_preserves_goal(variant):
    from isaaclab_arena_vabar.tasks.usbc_insertion.task import UsbcInsertionTask
    from isaaclab_arena_vabar.tasks.usbc_insertion.terminations import (
        usbc_insertion_seated,
    )

    params = scene_params(load_scene(f"usbc_insertion/{variant}.yaml"))
    params["plug"] = SimpleNamespace(name=params["plug"])
    params["bulkhead"] = SimpleNamespace(name=params["bulkhead"])
    task = UsbcInsertionTask(**params)
    cfg = task.get_termination_cfg()
    assert isinstance(cfg, TaskTerminationCfg)
    assert cfg.timeout_s == params["episode_length_s"]
    assert len(cfg.success) == 1
    predicate = cfg.success[0].predicate_sequence[0]
    assert predicate.func is usbc_insertion_seated
    assert predicate.params == {
        "goal": task.goal_spec,
        "work_hand": task.work_hand_spec,
    }


def test_hanging_metrics_record_first_completed_episode_and_partial_ids(monkeypatch):
    from isaaclab_arena_vabar.tasks.tool_hanging import metrics
    from isaaclab_arena_vabar.tasks.tool_hanging.task import ToolHangingTask

    task = ToolHangingTask.__new__(ToolHangingTask)
    task.goal_specs = ()
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        termination_manager=SimpleNamespace(
            active_terms=["success"],
            get_term=lambda _: torch.tensor([False, True]),
        ),
    )
    monkeypatch.setattr(
        metrics,
        "tool_hanging_goal_completion",
        lambda *a, **k: torch.tensor([[False, True], [True, True]]),
    )
    recorders = []
    for metric in task.get_metrics():
        cfg = metric.get_recorder_term_cfg()
        recorders.append(cfg.class_type(cfg, env))
    for ids, expected in [([1], [1.0, 1.0]), ([0], [0.0, 0.5])]:
        for recorder, value in zip(recorders, expected, strict=True):
            name, values = recorder.record_pre_reset(ids)
            assert name is not None
            assert values.tolist() == [value]


def test_legacy_sorting_task_remains_registered_with_root_position_scoring():
    from isaaclab_arena_vabar.tasks.registry import register_tasks
    from isaaclab_arena.assets.registries import TaskRegistry

    register_tasks()
    task_type = TaskRegistry().get_component_by_name("ObjectsInRegionsTask")
    task = task_type(
        object_list=[SimpleNamespace(name="tool")],
        region_list=[SimpleNamespace(name="bin")],
        bounds_xyzxyz=[(-1, -1, -1, 1, 1, 1)],
    )
    cfg = task.get_termination_cfg()
    assert cfg.timeout_s == 400
    predicate = cfg.success[0].predicate_sequence[0]
    def data(**kw):
        return SimpleNamespace(data=SimpleNamespace(**{
            k: SimpleNamespace(torch=torch.tensor(v, dtype=torch.float32))
            for k, v in kw.items()
        }))
    env = SimpleNamespace(scene={
        "tool": data(root_pos_w=[[0, 0, 0], [2, 0, 0]]),
        "bin": data(root_pos_w=[[0, 0, 0], [0, 0, 0]],
                    root_quat_w=[[1, 0, 0, 0], [1, 0, 0, 0]]),
    })
    assert predicate.func(env, **predicate.params).tolist() == [True, False]
