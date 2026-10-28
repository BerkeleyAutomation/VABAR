"""Canonical typed feature mating from computing-feature-mating-poses."""
from typing import Any, TypedDict
from gap import NodeContext

class Output(TypedDict):
    mate_pose: dict[str, Any]
    approach_pose: dict[str, Any]
    engaged_pose: dict[str, Any]
    approach_axis: dict[str, float]
    seating_distance: float
    minimum_clearance: float

def run(ctx: NodeContext, held_feature_in_tcp: dict[str, Any],
        fixture_feature: dict[str, Any], relation: str,
        attached_object: dict[str, Any]) -> Output:
    axis = fixture_feature.get("axis")
    if not axis:
        raise ValueError("fixture feature has no directed axis")
    reference_pose = ctx.tool("robot.get_ee_pose")["pose"]
    return ctx.tool(
        "geometry.compute_feature_mate",
        held_feature_in_tcp=held_feature_in_tcp,
        fixture_pose=fixture_feature["pose"], fixture_axis=axis,
        relation=relation,
        held_radius=float(held_feature_in_tcp.get("radius_outer", 0.0)),
        fixture_radius=float(fixture_feature.get("radius_inner", 0.0)),
        reference_tcp_pose=reference_pose, attached_object=attached_object,
    )
