"""Fail-closed inventory for reusable industrial-asset drop acceptance."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Literal

import yaml

PhysicsProfile = Literal[
    "tool_sort_mjwarp",
    "syringe_mjwarp",
    "tool_sort_hammer_mjwarp",
    "tool_hanging_mjwarp",
    "usbc_native",
]

DROP_HEIGHT_M = 0.10


@dataclass(frozen=True)
class AssetDropCase:
    registry_name: str
    relative_usd_path: str
    physics_profile: PhysicsProfile
    support_asset: str
    rotation_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if (
            not self.registry_name
            or not self.relative_usd_path
            or len(self.rotation_xyzw) != 4
            or not all(math.isfinite(value) for value in self.rotation_xyzw)
            or math.isclose(sum(value * value for value in self.rotation_xyzw), 0.0)
        ):
            raise ValueError("asset drop case is invalid")


@dataclass(frozen=True)
class _AssetSpec:
    registry_name: str
    role: str
    relative_usd_path: str


_SOURCE_ROOT = Path(
    os.environ.get("ASSET_SOURCE_ROOT", Path(__file__).resolve().parents[2])
)
_BENCHMARK_ROOT = _SOURCE_ROOT / "isaaclab_arena_vabar"


def _registered_specs(module_name: str) -> tuple[_AssetSpec, ...]:
    """Read the same declarative records consumed by runtime registration."""

    module = import_module(
        f"isaaclab_arena_vabar.assets.registrations.{module_name}"
    )
    return tuple(
        _AssetSpec(
            registry_name=spec.registry_name,
            role=spec.role,
            relative_usd_path=spec.relative_usd_path,
        )
        for spec in module.ASSETS
    )


_CANONICAL_ROTATIONS = {
    # Medium scissors pitches this shared hook; asset-drop tests retain its standard mount.
    "vabar__wrench_hook_2": (0.0, 0.0, 1.0, 0.0),
    # Randomized pile scenes intentionally have no authored object pose. Keep
    # their asset-level drop-test orientations independent of task placement.
    "vabar__adjustable_wrench": (
        -0.0030774192418903112,
        0.0026711411774158478,
        0.5894669890403748,
        -0.8077821731567383,
    ),
    "vabar__battery": (
        -0.00015629001427441835,
        -0.0006150670233182609,
        0.046391911804676056,
        -0.9989231824874878,
    ),
    "vabar__breadboard": (
        0.0002929068577941507,
        -0.00032268904033116996,
        0.6946227550506592,
        0.7193741202354431,
    ),
    "vabar__clamp": (
        -0.014119562692940235,
        -0.08092429488897324,
        0.5865128636360168,
        0.805763304233551,
    ),
    "vabar__combination_pliers": (
        -0.05029385909438133,
        0.002936960430815816,
        0.99422287940979,
        0.09477785229682922,
    ),
    "vabar__cutting_pliers": (
        0.042890168726444244,
        -0.09747745841741562,
        0.5096211433410645,
        -0.8537826538085938,
    ),
    "vabar__drill": (
        -0.1799403727054596,
        -0.17290718853473663,
        0.7333788275718689,
        0.6323607563972473,
    ),
    "vabar__flashlight": (
        -0.0006317686638794839,
        -0.0002455524809192866,
        0.6756804585456848,
        -0.7371944189071655,
    ),
    "vabar__flathead_screwdriver": (
        0.7105623483657837,
        -0.6296243667602539,
        -0.17638231813907623,
        0.25992992520332336,
    ),
    "vabar__hammer": (
        -0.4180590510368347,
        -0.4858432114124298,
        -0.44771748781204224,
        -0.6234837770462036,
    ),
    "vabar__insulating_tape": (
        0.0005987313925288618,
        -0.00040096629527397454,
        0.6710759997367859,
        0.7413882613182068,
    ),
    "vabar__measuring_tape": (
        -0.1009659692645073,
        0.03144553676247597,
        0.9554615616798401,
        0.2755182087421417,
    ),
    "vabar__multimeter": (
        -0.0021523048635572195,
        -0.010737620294094086,
        0.9664853811264038,
        -0.25648823380470276,
    ),
    "vabar__phillips_screwdriver": (
        -0.42946746945381165,
        0.22663837671279907,
        0.8233973979949951,
        0.2936146855354309,
    ),
    "vabar__pliers": (
        -0.007719389628618956,
        0.02536432258784771,
        0.8861774802207947,
        0.462586909532547,
    ),
    "vabar__round_nut": (
        0.006598883308470249,
        0.009126810356974602,
        0.9815715551376343,
        0.1907627433538437,
    ),
    "vabar__safety_glasses": (
        0.6098843812942505,
        0.5996133089065552,
        0.3687264323234558,
        0.36406832933425903,
    ),
    "vabar__screwdriver": (
        0.01724683679640293,
        -0.005092320963740349,
        0.011954821646213531,
        0.9997668862342834,
    ),
    "vabar__slotted_screwdriver": (
        0.09379097074270248,
        0.06392800807952881,
        -0.4213390052318573,
        0.8997721076011658,
    ),
    "vabar__syringe": (
        0.05101299285888672,
        0.04162578284740448,
        0.7660786509513855,
        0.6393656730651855,
    ),
    "vabar__tape_measure": (
        -0.14913351833820343,
        0.664914608001709,
        0.6548227071762085,
        -0.32688701152801514,
    ),
    "vabar__utility_knife": (
        -0.00041280261939391494,
        0.0006503418553620577,
        0.7543618083000183,
        0.6564585566520691,
    ),
    "vabar__wire_spool": (
        -0.00032227064366452396,
        -0.00030606469954364,
        0.9269546866416931,
        0.3751731216907501,
    ),
}


def _authored_rotations() -> dict[str, tuple[float, float, float, float]]:
    root = _BENCHMARK_ROOT / "src/isaaclab_arena_vabar/environments"
    candidates: dict[str, list[tuple[str, str, tuple[float, float, float, float]]]] = {}
    for path in sorted(root.rglob("*.yaml")):
        document = yaml.safe_load(path.read_text())
        if not isinstance(document, dict):
            continue
        for entry in document.get("objects", ()):
            if not isinstance(entry, dict):
                continue
            name, params = entry.get("registry_name"), entry.get("params")
            if not isinstance(name, str) or not isinstance(params, dict):
                continue
            rotation = params.get("initial_pose", {}).get("rotation_xyzw")
            if isinstance(rotation, list) and len(rotation) == 4:
                candidates.setdefault(name, []).append(
                    (
                        str(path.relative_to(root)),
                        str(entry.get("id")),
                        tuple(float(value) for value in rotation),
                    )
                )
    rotations = dict(_CANONICAL_ROTATIONS)
    for name, values in candidates.items():
        if name in rotations:
            continue
        if len(values) > 1:
            if len({value[2] for value in values}) > 1:
                raise ValueError(f"{name} needs an explicit canonical rotation")
            values = [values[0]]
        rotations[name] = values[0][2]
    return rotations


_ROTATIONS = _authored_rotations()
# The blank syringe references the same geometry as the red-cap syringe.
_ROTATIONS["vabar__syringe_blank"] = _ROTATIONS["vabar__syringe"]
_ROTATIONS["vabar__instrument_tray"] = (0.0, 0.0, 0.0, 1.0)
# Tool-hanging medium scenes express movable placement through on + position_limits
# relations and carry no initial_pose. Preserve each asset's formerly authored
# orientation as the canonical drop-test orientation.
_ROTATIONS["vabar__hanging_hammer_1"] = (
    -0.7071067811865476,
    0.0,
    0.0,
    0.7071067811865476,
)
_ROTATIONS["vabar__hanging_hammer_2"] = (0.0, 0.0, 0.0, 1.0)
_ROTATIONS["vabar__plier_combination"] = (0.0, 0.0, 0.0, 1.0)
_ROTATIONS["vabar__plier_cutter"] = (
    0.5,
    -0.5,
    -0.5,
    0.5,
)
_ROTATIONS["vabar__scissor_pair_1"] = (
    0.5,
    0.5,
    0.5,
    0.5,
)
_ROTATIONS["vabar__scissor_pair_2"] = (
    0.0,
    0.0,
    -0.7071067811865476,
    0.7071067811865476,
)
_ROTATIONS["vabar__screwdriver_slim"] = (0.0, 0.0, 1.0, 0.0)
_ROTATIONS["vabar__screwdriver_grip"] = (0.0, 0.0, 0.0, 1.0)
_ROTATIONS["vabar__wrench_ring_19"] = (0.0, 0.0, 1.0, 0.0)
_ROTATIONS["vabar__wrench_ring_17"] = (0.0, 0.0, 1.0, 0.0)
# The fixed-port Easy scene retired the free bulkhead object, so its reusable
# asset-level acceptance pose no longer comes from an environment YAML.
_ROTATIONS["vabar__bulkhead"] = (0.5, 0.5, 0.5, 0.5)
_ROTATIONS["vabar__plug"] = (0.5, 0.5, 0.5, -0.5)


# Each profile replays the Newton settings of the production family that uses
# the asset.
PRODUCTION_PHYSICS_YAML: dict[str, str] = {
    "tool_sort_mjwarp": "tool_sorting/tool_sorting_env_config.yaml",
    "tool_sort_hammer_mjwarp": "tool_sorting/tool_sorting_env_config.yaml",
    "syringe_mjwarp": "syringe_disposal/syringe_env_config.yaml",
    "tool_hanging_mjwarp": "tool_hanging/tool_hanging_env_config.yaml",
    "usbc_native": "usbc_insertion/usbc_env_config.yaml",
}
# Drop acceptance only; production keeps the family YAML values.
DROP_ONLY_SOLVER_OVERRIDES: dict[str, dict[str, float]] = {
    "tool_sort_hammer_mjwarp": {"impratio": 100.0},
}
_SYRINGE_TASK_OBJECTS = frozenset(
    {
        "vabar__instrument_tray",
        "vabar__syringe",
        "vabar__syringe_blank",
    }
)


def _tool_sort_profile(registry_name: str) -> PhysicsProfile:
    if registry_name == "vabar__hammer":
        return "tool_sort_hammer_mjwarp"
    if registry_name in _SYRINGE_TASK_OBJECTS:
        return "syringe_mjwarp"
    return "tool_sort_mjwarp"


def _case(spec, profile: PhysicsProfile, support: str) -> AssetDropCase:
    return AssetDropCase(
        registry_name=spec.registry_name,
        relative_usd_path=_RUNTIME_USD_OVERRIDES.get(
            spec.registry_name, spec.relative_usd_path
        ),
        physics_profile=profile,
        support_asset=support,
        rotation_xyzw=_ROTATIONS[spec.registry_name],
    )


_SORT_OBJECTS = tuple(
    spec for spec in _registered_specs("vabar_tool_sort_v1") if spec.role == "object"
)
_HANGING_OBJECTS = tuple(
    spec for spec in _registered_specs("vabar_tool_hanging_v1") if spec.role == "object"
)
_USBC_OBJECTS = tuple(
    spec for spec in _registered_specs("vabar_usbc_insert_v1") if spec.role == "object"
)

_RUNTIME_USD_OVERRIDES = {
    # Both production USB-C environments use the artist closure in the
    # source-task frame. Acceptance must exercise that composed runtime asset,
    # not the raw delivery wrapper named by the generated import record.
    "vabar__plug": "vabar__plug/task_frame_plug.usda",
}

ASSET_DROP_CASES = (
    *(
        _case(
            spec,
            _tool_sort_profile(spec.registry_name),
            "vabar__fr3_workcell_table",
        )
        for spec in _SORT_OBJECTS
    ),
    *(
        _case(spec, "tool_hanging_mjwarp", "vabar__yam_workcell_table")
        for spec in _HANGING_OBJECTS
    ),
    *(
        _case(spec, "usbc_native", "vabar__yam_workcell_table")
        for spec in _USBC_OBJECTS
    ),
)


def case_for_asset(registry_name: str) -> AssetDropCase:
    matches = tuple(
        case for case in ASSET_DROP_CASES if case.registry_name == registry_name
    )
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate movable asset: {registry_name}")
    return matches[0]
