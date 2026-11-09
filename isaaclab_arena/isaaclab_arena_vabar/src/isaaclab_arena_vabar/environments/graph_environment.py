"""Shared build steps for benchmark environments defined by an Arena graph YAML."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from isaaclab_arena.environments.arena_environment_factory import (
    ArenaEnvironmentCfg,
    ArenaEnvironmentCfgT,
    ArenaEnvironmentFactory,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import (
        IsaacLabArenaEnvironment,
    )

ENVIRONMENTS_ROOT = Path(__file__).resolve().parent


def build_graph_environment(
    graph_path: Path,
    cfg: Any,
) -> IsaacLabArenaEnvironment:
    """Build one graph YAML with the runner's camera and mesh-instancing options.

    ``cfg`` supplies ``enable_cameras``, ``use_tiled_cameras`` and
    ``use_instanceable_meshes``. A graph that declares ``random_around_solution``
    gets the reset term that applies it, ahead of any callback its family adds.
    """
    from isaaclab_arena.environment_spec.arena_env_graph_spec import (
        ArenaEnvGraphSpec,
    )

    from isaaclab_arena_vabar.environments import random_around_solution
    from isaaclab_arena_vabar.mesh_instancing import configure_scene_mesh_instancing

    spec = ArenaEnvGraphSpec.from_yaml(str(graph_path))
    arena_env = spec.to_arena_env(enable_cameras=cfg.enable_cameras)
    arena_env.embodiment.set_use_tiled_cameras(cfg.use_tiled_cameras)
    set_robot_instancing = getattr(
        arena_env.embodiment, "set_use_instanceable_meshes", None
    )
    if set_robot_instancing is not None:
        set_robot_instancing(cfg.use_instanceable_meshes)
    elif cfg.use_instanceable_meshes:
        raise ValueError(
            "the registered Arena embodiment does not expose an "
            "instanceable robot asset"
        )
    configure_scene_mesh_instancing(arena_env.scene, cfg.use_instanceable_meshes)
    random_around_solution.install(arena_env)
    return arena_env


class EnvironmentDescriptor(Protocol):
    """Immutable binding from a registry name to a package-relative graph YAML."""

    @property
    def environment_name(self) -> str: ...

    @property
    def relative_graph_path(self) -> str: ...


_GENERATED_FACTORIES: dict[tuple[type, str], type] = {}


class DescriptorEnvironmentFactory(ArenaEnvironmentFactory[ArenaEnvironmentCfgT]):
    """Factory family whose generated subclasses each bind one descriptor."""

    descriptor_type: ClassVar[type]
    cfg_type: ClassVar[type[ArenaEnvironmentCfg]]
    family: ClassVar[str]
    """Human-readable family name used in errors."""
    generated_class_prefix: ClassVar[str]

    name: str | None = None
    descriptor: Any = None

    @classmethod
    def for_descriptor(cls, descriptor: EnvironmentDescriptor) -> type:
        """Return one stable concrete factory type per descriptor name."""
        # Keyed by name, not by descriptor: some descriptors carry mappings and
        # are therefore unhashable. Generators reject duplicate names.
        key = (cls, descriptor.environment_name)
        existing = _GENERATED_FACTORIES.get(key)
        if existing is not None:
            if existing.descriptor != descriptor:
                raise RuntimeError(
                    f"conflicting descriptors for {descriptor.environment_name!r}"
                )
            return existing
        factory = type(
            cls.generated_class_prefix + descriptor.environment_name,
            (cls,),
            {
                "__module__": cls.__module__,
                "name": descriptor.environment_name,
                "descriptor": descriptor,
            },
        )
        _GENERATED_FACTORIES[key] = factory
        return factory

    def graph_path(self) -> Path:
        if self.descriptor is None:
            raise RuntimeError(f"{self.family} environment factory has no descriptor")
        return ENVIRONMENTS_ROOT / self.descriptor.relative_graph_path
