"""USB-C construction preserves Newton world ordering and partial reset isolation."""

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("count", [0, 1, 5])
@pytest.mark.parametrize("medium", [False, True])
def test_cables_are_constructed_inside_each_cloned_world(monkeypatch, count, medium):
    import newton
    import torch
    import warp as wp
    from isaaclab_arena_vabar.environments.usbc_insertion import (
        environment as module,
    )
    from isaaclab_newton.cloner.newton_clone_utils import (
        rename_builder_labels,
        replicate_builder_mapping,
    )

    visuals = []

    def author(root, links, radius):
        visuals.append(root)
        return [
            f"{root}/physics/link{i}" for i in range(links)
        ], f"{root}/geometry/mesh"

    monkeypatch.setattr(module, "_author_cable_visual", author)
    spec = {
        "scale": 1.0,
        "plug": {
            "object": "plug",
            "links": 8,
            "length": 0.1,
            "rear_z": -0.01,
            "radius": 0.002,
            "density": 1000,
        },
    }
    if medium:
        spec["bulkhead"] = {
            "object": "bulkhead",
            "links": 8,
            "length": 0.1,
            "radius": 0.002,
            "density": 1000,
        }
    builder, prototype = newton.ModelBuilder(), newton.ModelBuilder()
    for name, pos in [
        ("plug", (0.0, 0.0, 1.0)),
        ("bulkhead/Geometry/bulkhead_01", (0.5, 0.1, 1.0)),
    ]:
        body = prototype.add_body(
            xform=wp.transform(pos, wp.quat_identity()),
            label=f"/World/envs/env_0/{name}",
        )
        prototype.add_shape_sphere(body, radius=0.01)
    if count == 0:
        module._build_cables(spec, prototype, 0, [0, 0, 0], [0, 0, 0, 1])
        prototype.finalize(device="cpu")
        assert set(prototype.body_world) == {-1}
        assert len(visuals) == (2 if medium else 1)
        return
    mapping = torch.ones((1, count), dtype=torch.bool)
    replicate_builder_mapping(
        builder,
        ("/World/envs/env_0",),
        mapping,
        torch.tensor([[i * 30.0, 0.0, 0.0] for i in range(count)]),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]] * count),
        {"/World/envs/env_0": prototype},
        per_world_builder_hooks=[partial(module._build_cables, spec)],
    )
    rename_builder_labels(
        builder,
        ("/World/envs/env_0",),
        ("/World/envs/env_{}",),
        torch.arange(count),
        mapping,
    )
    builder.finalize(device="cpu")  # Newton rejects non-contiguous world bodies.
    assert len(set(visuals)) == count * (2 if medium else 1)
    for label, world in zip(builder.body_label, builder.body_world):
        assert label.startswith(f"/World/envs/env_{world}/")
    for i in range(count):
        root = f"/World/envs/env_{i}/vabar_usbc_insert_"
        link = builder.body_label.index(root + "cable/physics/link0")
        np.testing.assert_allclose(
            builder.body_q[link][:3], [30 * i, 0.0, 0.98375], atol=1e-5
        )
        if medium:
            link = builder.body_label.index(root + "pcable/physics/link0")
            np.testing.assert_allclose(
                builder.body_q[link][:3], [30 * i + 0.44, 0.08255, 0.8234], atol=1e-5
            )


@pytest.mark.parametrize("count, selected", [(1, [0]), (5, [1, 3]), (5, [])])
def test_reset_changes_only_selected_worlds(monkeypatch, count, selected):
    import warp as wp
    from isaaclab_arena_vabar.environments.usbc_insertion import (
        environment as module,
    )
    from isaaclab_newton.physics import NewtonManager

    n = 4 * count
    array = lambda v, dtype: wp.array(v, dtype=dtype, device="cpu")
    model = SimpleNamespace(
        joint_label=[f"cable{i}_bend" for i in range(n)],
        joint_child=array(np.arange(n), wp.int32),
        body_label=[
            f"/World/envs/env_{i}/cable/link{j}" for i in range(count) for j in range(4)
        ],
        joint_q_start=array(np.arange(n + 1), wp.int32),
        joint_qd_start=array(np.arange(n + 1), wp.int32),
    )
    states = [
        SimpleNamespace(
            joint_q=array(np.full(n, 9.0), wp.float32),
            joint_qd=array(np.full(n, 7.0), wp.float32),
        )
        for _ in range(2)
    ]
    monkeypatch.setattr(NewtonManager, "_model", model)
    monkeypatch.setattr(NewtonManager, "_state_0", states[0])
    monkeypatch.setattr(NewtonManager, "_state_1", states[1])
    monkeypatch.setattr(NewtonManager, "invalidate_fk", lambda: None)
    module._reset_cables(SimpleNamespace(num_envs=count), selected, expected_links=4)
    for state in states:
        for i in range(count):
            np.testing.assert_array_equal(
                state.joint_q.numpy()[4 * i : 4 * i + 4], 0.0 if i in selected else 9.0
            )
            np.testing.assert_array_equal(
                state.joint_qd.numpy()[4 * i : 4 * i + 4], 0.0 if i in selected else 7.0
            )
