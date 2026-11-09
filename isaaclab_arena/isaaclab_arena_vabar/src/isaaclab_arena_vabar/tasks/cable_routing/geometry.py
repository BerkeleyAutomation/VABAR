"""Cable centerline geometry for cable-routing scoring."""

from __future__ import annotations

import torch


def capsule_centerline(
    body_poses_xyzw: torch.Tensor, half_lengths: torch.Tensor
) -> torch.Tensor:
    """Return each capsule's start followed by the final end, as AUTOLab's RSS version scores it."""
    centers = body_poses_xyzw[..., :3]
    x, y, z, w = body_poses_xyzw[..., 3:7].unbind(dim=-1)
    axes = torch.stack(
        (2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)),
        dim=-1,
    )
    offsets = axes * half_lengths[..., None]
    return torch.cat(
        (centers - offsets, centers[..., -1:, :] + offsets[..., -1:, :]), dim=-2
    )
