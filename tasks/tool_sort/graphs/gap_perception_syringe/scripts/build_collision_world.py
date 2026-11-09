"""Reconstruct a coarse camera-only planner world and preserve the aperture."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    world_config: dict[str, Any]


def _mesh_resolution() -> float:
    """Return the collision-mesh sampling resolution.

    A 16 mm default deliberately represents the scene more coarsely than the
    old 8 mm mesh.  This removes small RGB-D surface ripples and sliver
    triangles that motion planners can interpret as narrow collisions.  The
    Alpha-shape topology and clustering are intentionally kept independent
    from this sampling resolution.
    """
    return 0.016


def _robot_mask(ctx: NodeContext, camera: dict[str, Any]) -> np.ndarray | None:
    merged = None
    for query in ("robot arm", "robot gripper"):
        result = ctx.tool("sam3.segment_text", image=camera["rgb"], query=query, max_results=2)
        for mask, score in zip(result.get("masks") or [], result.get("scores") or []):
            if float(score) < 0.10:
                continue
            candidate = np.asarray(mask) > 0
            merged = candidate if merged is None else merged | candidate
    return None if merged is None else merged.astype(np.uint8) * 255


def _distance_to_segment_xy(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    edge = end - start
    t = float(np.clip((point - start) @ edge / max(float(edge @ edge), 1.0e-12), 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * edge)))


def _point_in_triangle_xy(point: np.ndarray, triangle: np.ndarray) -> bool:
    """Whether point lies in a projected triangle, including its boundary."""
    a, b, c = triangle[:, :2]
    v0, v1, v2 = c - a, b - a, point - a
    dot00, dot01, dot02 = v0 @ v0, v0 @ v1, v0 @ v2
    dot11, dot12 = v1 @ v1, v1 @ v2
    denominator = float(dot00 * dot11 - dot01 * dot01)
    if abs(denominator) < 1.0e-12:
        return False
    u = float((dot11 * dot02 - dot01 * dot12) / denominator)
    v = float((dot00 * dot12 - dot01 * dot02) / denominator)
    return u >= -1.0e-8 and v >= -1.0e-8 and u + v <= 1.0 + 1.0e-8


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    target_mask: Any,
    aperture: dict[str, float],
    rim_z: float,
    aperture_radius: float,
) -> Output:
    overhead = next(
        (camera for camera in observation.get("cameras", []) if camera.get("name") == "overhead"),
        None,
    )
    if overhead is None:
        raise ValueError("collision reconstruction requires the overhead RGB-D camera")
    # The overhead view is the sole scene-perception source.  In particular,
    # agentview is a recording/debug view and must not affect the graph.
    cameras = [overhead]
    object_masks = [{"name": "grasp_target", "mask": target_mask, "camera_index": 0}]
    mask = _robot_mask(ctx, overhead)
    if mask is not None:
        object_masks.append({"name": "robot_view_overhead", "mask": mask, "camera_index": 0})
    try:
        robot_spheres = list(ctx.tool("motion.get_robot_collision_spheres", arm_id=-1).get("spheres") or [])
    except Exception:
        robot_spheres = []
    voxel_size = _mesh_resolution()
    response = ctx.tool(
        "geometry.build_world_config", cameras=cameras, object_masks=object_masks,
        voxel_size=voxel_size, noise_eps=0.025, noise_min_samples=4,
        mesh_alpha=0.04,
        robot_spheres=robot_spheres, robot_sphere_margin=0.015,
    )
    config = response.get("config") or {"meshes": []}

    # Alpha-shape reconstruction may bridge the real lid opening. Preserve a
    # corridor measured from the aperture mask rather than using an object- or
    # fixture-specific diameter.
    center = np.array([aperture["x"], aperture["y"]], dtype=np.float64)
    corridor_radius = max(0.005, 0.85 * float(aperture_radius))
    carved = []
    for mesh in config.get("meshes") or []:
        raw_vertices, raw_faces = mesh.get("vertices"), mesh.get("faces")
        vertices = np.asarray([] if raw_vertices is None else raw_vertices, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray([] if raw_faces is None else raw_faces, dtype=np.int64).reshape(-1, 3)
        if not len(vertices) or not len(faces):
            carved.append(mesh); continue
        # Alpha-shape reconstruction can produce detached, nearly planar
        # depth-edge fragments. CuRobo treats each fragment as exact occupied
        # geometry, so even a tiny false sheet can block the swept volume of
        # the long syringe during reorientation. Preserve small real geometry
        # unless the component is simultaneously sparse, very thin and local.
        extent = np.ptp(vertices, axis=0)
        depth_edge_sliver = bool(
            len(vertices) < 32
            and float(extent.min()) < 0.004
            and float(extent.max()) < 0.15
        )
        if depth_edge_sliver:
            continue
        keep = []
        for face in faces:
            triangle = vertices[face]
            z_overlap = float(triangle[:, 2].max()) >= rim_z - 0.035 and float(triangle[:, 2].min()) <= rim_z + 0.18
            distances = [float(np.linalg.norm(vertex[:2] - center)) for vertex in triangle]
            for start, end in ((triangle[0, :2], triangle[1, :2]), (triangle[1, :2], triangle[2, :2]), (triangle[2, :2], triangle[0, :2])):
                distances.append(_distance_to_segment_xy(center, start, end))
            crosses_corridor = min(distances) <= corridor_radius or _point_in_triangle_xy(center, triangle)
            keep.append(not (z_overlap and crosses_corridor))
        kept = faces[np.asarray(keep, dtype=bool)]
        if not len(kept):
            continue
        used = np.unique(kept)
        remap = np.full(len(vertices), -1, dtype=np.int64); remap[used] = np.arange(len(used))
        updated = dict(mesh)
        updated["vertices"] = vertices[used].astype(np.float32)
        updated["faces"] = remap[kept].astype(np.int32)
        carved.append(updated)
    # CuRobo's configured scene cache admits 32 meshes.  RGB-D clustering can
    # legitimately return more disconnected components; collision checking
    # does not require each component to be a separate named obstacle, so pack
    # them into one indexed mesh without adding any bridging triangles.
    if carved:
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        for mesh in carved:
            vertices = np.asarray(mesh["vertices"], dtype=np.float32).reshape(-1, 3)
            faces = np.asarray(mesh["faces"], dtype=np.int32).reshape(-1, 3)
            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            vertex_offset += len(vertices)
        config["meshes"] = [{
            "name": "perceived_scene",
            "vertices": np.concatenate(all_vertices, axis=0),
            "faces": np.concatenate(all_faces, axis=0),
            "pose": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            },
        }]
    else:
        config["meshes"] = []
    if not carved:
        raise ValueError("RGB-D collision reconstruction produced no scene mesh")
    return {"world_config": config}
