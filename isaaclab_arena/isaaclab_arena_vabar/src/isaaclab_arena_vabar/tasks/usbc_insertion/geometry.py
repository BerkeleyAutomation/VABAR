# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched insertion geometry for the USB-C task.

Pure: it takes poses and returns numbers, so the arithmetic can be tested without
a simulator. Everything is measured in the RECEPTACLE's frame, carried to world
each step, because the bulkhead is a free body here -- the other arm holds it --
and a bore stated in world coordinates stops describing the hole the moment it
moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Bore:
    """The receptacle's slot, in the receptacle's own frame.

    NOT frozen, deliberately. This travels inside a ``TerminationTermCfg``'s
    params, and Isaac Lab's configclass machinery assigns to the fields of what
    it finds there -- a frozen dataclass raises ``FrozenInstanceError`` during
    environment setup. The tool-hanging specs are plain dataclasses for the same
    reason.
    """

    mouth: tuple[float, float, float]
    axis: tuple[float, float, float]
    depth: float
    half_extents: tuple[float, float]
    tip_offset: float

    def __post_init__(self) -> None:
        if self.depth <= 0.0:
            raise ValueError("bore depth must be positive")
        if any(value <= 0.0 for value in self.half_extents):
            raise ValueError("bore half extents must be positive")
        if sum(component * component for component in self.axis) <= 0.0:
            raise ValueError("bore axis must be non-zero")


@dataclass
class InsertionMeasurement:
    """One reading per environment, all in metres or radians."""

    depth: torch.Tensor
    lateral: torch.Tensor
    tilt: torch.Tensor
    speed: torch.Tensor


def measure_insertion(
    plug_position: torch.Tensor,
    plug_orientation: torch.Tensor,
    plug_velocity: torch.Tensor,
    port_position: torch.Tensor,
    port_orientation: torch.Tensor,
    bore: Bore,
) -> InsertionMeasurement:
    """Depth, lateral offset, tilt and speed.

    Quaternions are WXYZ, which is what Isaac Lab's ``math_utils`` and the
    ``root_link_quat_w`` buffers use. Passing XYZW here silently rotates every
    reading.
    """
    from isaaclab.utils import math as math_utils

    def local(vector: tuple[float, float, float]) -> torch.Tensor:
        return torch.tensor(vector, dtype=plug_position.dtype, device=plug_position.device).expand(
            plug_position.shape[0], 3
        )

    mouth_w, _ = math_utils.combine_frame_transforms(
        port_position, port_orientation, local(bore.mouth)
    )
    axis_w = math_utils.quat_apply(port_orientation, local(bore.axis))
    axis_w = axis_w / torch.linalg.vector_norm(axis_w, dim=-1, keepdim=True)

    tip_w, _ = math_utils.combine_frame_transforms(
        plug_position, plug_orientation, local((0.0, 0.0, bore.tip_offset))
    )
    offset = tip_w - mouth_w
    depth = (offset * axis_w).sum(dim=-1)
    lateral = torch.linalg.vector_norm(offset - depth.unsqueeze(-1) * axis_w, dim=-1)

    # The plug's own approach axis against the bore's. Absolute, because a plug
    # inserted "backwards" along the same line is still aligned with it.
    plug_axis_w = math_utils.quat_apply(plug_orientation, local((0.0, 0.0, 1.0)))
    plug_axis_w = plug_axis_w / torch.linalg.vector_norm(plug_axis_w, dim=-1, keepdim=True)
    cosine = (plug_axis_w * axis_w).sum(dim=-1).abs().clamp(max=1.0)
    tilt = torch.arccos(cosine)

    return InsertionMeasurement(
        depth=depth,
        lateral=lateral,
        tilt=tilt,
        # Linear only. The source's clause is `(speed plug)`, and folding angular
        # velocity in would fail a seated plug that is merely spinning about its
        # own axis.
        speed=torch.linalg.vector_norm(plug_velocity, dim=-1),
    )
