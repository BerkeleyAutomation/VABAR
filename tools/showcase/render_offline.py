# SPDX-License-Identifier: Apache-2.0
"""Re-render a recorded run with a better renderer.

Ported from RoboSimStudio-dev ``tools/render_offline.py`` (e94e85f2). Here it
also imports :mod:`hero_run`, so ``--cameras`` may name a ``HERO_CAMS``
camera and a re-render is framed exactly as a live run.

Collect cheap, re-render expensive. An episode recorded through the tiled ray
tracer costs a fraction of one recorded through a path tracer, and during
collection those pixels bought nothing -- the policy was not looking at them and
the evaluator was not scoring them. This rebuilds the deterministic scene and
*replays* the recorded body and particle state into it, so nothing is
re-simulated and the re-render is the same trajectory, photographed differently.

    python tools/render_offline.py \\
        --run outputs/peel_handle_scripted --renderer rtx --fps 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hero_run

import vabar

vabar.install()  # VABAR's task root, before the registry first loads
from robosimstudio.core.render.offline import OfflineRenderer, resolve_live_index
from robosimstudio.core.render.video import VideoSet
from robosimstudio.record import EpisodeStore
from robosimstudio.solvers.render import supported_channels
from robosimstudio_bench.envs.env import Env


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="a run directory written by tools/bench_run.py")
    parser.add_argument("--episode", default="", help="one episode id; empty re-renders all of them")
    parser.add_argument("--renderer", default="rtx", help="tiled | gl | rtx | pyrender | pybullet")
    parser.add_argument("--out", default="", help="where the videos go; defaults to <run>/rerender")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--every", type=int, default=1, help="re-render one recorded step in this many")
    parser.add_argument(
        "--cameras",
        default="",
        help=(
            "comma-separated camera names; empty re-renders every camera the task declares. "
            "A path tracer charges per camera per frame, so a re-render wanted for one view "
            "should say so rather than pay for three."
        ),
    )
    parser.add_argument("--width", type=int, default=0, help="override the camera width")
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument(
        "--environment",
        default="",
        help=(
            "RTX lighting preset -- 'default', 'studio', 'none' or 'lab'. --renderer rtx only: "
            "'default' is a dark key with a visible glare, 'studio' an even fill several stops "
            "brighter with a sphere light in shot, 'lab' a soft panel over a neutral dome. Empty "
            "keeps what the task's own look states."
        ),
    )
    parser.add_argument(
        "--quality",
        default="",
        help="RTX 'preview' or 'final' -- passes per frame; empty keeps the task's own look",
    )
    parser.add_argument(
        "--scenery",
        default="",
        help="RTX scenery override by registered name; 'none' strips the task's own backdrop",
    )
    parser.add_argument(
        "--environment-map",
        default="",
        dest="environment_map",
        help=(
            "RTX dome HDR override -- a path, or a name relative to the asset root. --renderer rtx "
            "only, and only under the 'lab' preset, which is the one that builds a dome. 'none' "
            "strips the task's own; empty keeps it."
        ),
    )
    parser.add_argument(
        "--channels",
        default="",
        help=(
            "comma-separated channels to re-render, e.g. 'rgb,depth'. Empty narrows each "
            "camera's own channels to what the backend can produce, which is what lets a run "
            "collected with segmentation be re-rendered by a path tracer that has none."
        ),
    )
    return parser


class Example:
    """Replays a run's recorded state into another backend."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # Before the scene build, which is the expensive half of this
        # constructor: a flag combination that cannot be honoured should not
        # cost a staged workcell to find out.
        for flag in ("environment", "quality", "scenery", "environment_map"):
            if getattr(args, flag) and args.renderer != "rtx":
                raise ValueError(
                    f"--{flag.replace('_', '-')} is an RTX setting; the {args.renderer!r} backend has none"
                )
        self.store = EpisodeStore(args.run)
        meta = self.store.run_meta
        task = meta.get("task")
        if not task:
            raise ValueError(f"{args.run}/run_meta.json does not say which task was run")
        # Deterministic rebuild: the same task, scene, embodiment and gripper
        # produce the same model, which is what makes replaying recorded
        # arrays into it sound. A run recorded on a non-default binding (a
        # foreign gripper, say) rebuilt on the task's own default instead
        # would compile a model with a different body count -- StateReplay
        # then has no recorded body_q to match and every frame renders the
        # rebuild's own reset pose instead of the run's trajectory.
        self.env = Env(
            task,
            scene=str(meta.get("scene") or ""),
            embodiment=str(meta.get("embodiment") or ""),
            gripper=str(meta.get("gripper") or ""),
            verbose=False,
        )
        cameras = self._cameras()
        if args.width and args.height:
            cameras = [c.scaled(args.width, args.height) for c in cameras]
        # Through Env.make_renderer rather than create_renderer directly: the
        # scene-derived arguments a backend needs (the tiled segmentation table,
        # the visual backends' part list) are stated once there, and a second
        # copy here is a second thing to forget when a backend gains a third.
        # Overrides ride on the task's own look rather than replacing it, so a
        # re-render that only asks for 'final' keeps the scenery the task named.
        render = self.env.look
        overrides = {}
        if args.environment:
            overrides["environment"] = args.environment
        if args.quality:
            overrides["quality"] = args.quality
        if args.scenery:
            overrides["scenery"] = None if args.scenery == "none" else args.scenery
        if args.environment_map:
            overrides["environment_map"] = "" if args.environment_map == "none" else args.environment_map
        if overrides:
            from dataclasses import replace

            render = replace(render, **overrides)
        self.renderer = OfflineRenderer(
            self.env.model,
            self.env.make_renderer(
                args.renderer,
                cameras=cameras,
                channels=self._channels(cameras),
                render=render,
            ),
            strict=False,
        )
        self.out = Path(args.out or Path(args.run) / "rerender")

    def _cameras(self) -> list:
        """The task's cameras, or the subset ``--cameras`` named, in that order."""
        cameras = [*self.env.cameras, *hero_run.HERO]
        if not self.args.cameras:
            return cameras
        wanted = [name.strip() for name in self.args.cameras.split(",") if name.strip()]
        have = {c.name: c for c in cameras}
        missing = [name for name in wanted if name not in have]
        if missing:
            raise ValueError(f"--cameras names {missing}, and this task declares {sorted(have)}")
        return [have[name] for name in wanted]

    def _channels(self, cameras) -> list[str]:
        """Which channels to ask the backend for.

        A re-render is a deliberate change of backend, so the channels a run was
        *collected* with are a starting point rather than a requirement: a run
        recorded through the tiled ray tracer carries segmentation, and a path
        tracer has none. Narrowing here rather than letting the constructor
        raise is what makes ``--renderer rtx`` work on any run at all -- but it
        says what it dropped, because a silently missing channel in a re-render
        is indistinguishable from a backend that renders it badly.
        """
        supported = set(supported_channels(self.args.renderer))
        if self.args.channels:
            wanted = [c.strip() for c in self.args.channels.split(",") if c.strip()]
            missing = sorted(set(wanted) - supported)
            if missing:
                raise ValueError(
                    f"--channels asks for {missing} but the {self.args.renderer!r} backend produces {sorted(supported)}"
                )
            return wanted
        asked = {c for cam in cameras for c in cam.channels}
        kept = [c for c in ("rgb", "depth", "seg") if c in asked and c in supported]
        dropped = sorted(asked - supported)
        if dropped:
            print(f"[rerender] the {self.args.renderer!r} backend cannot produce {dropped}; re-rendering {kept}")
        if not kept:
            raise ValueError(
                f"the {self.args.renderer!r} backend produces none of the recorded channels {sorted(asked)}"
            )
        return kept

    def run(self) -> int:
        ids = [self.args.episode] if self.args.episode else self.store.episode_ids()
        if not ids:
            print(f"[rerender] no episodes in {self.args.run}")
            return 1
        total = 0
        for episode_id in ids:
            episode = self.store.episode(episode_id)
            # Which shapes the recorded live colours address is an anchor of
            # this episode, not of the run: set it before the first step lands.
            self.renderer.replay_state.live_index = resolve_live_index(
                self.env.model, episode.privileged, fallback=self.env.scene.handles.live_shapes
            )
            steps = list(episode.replay_steps())[:: max(1, self.args.every)]
            if not steps:
                print(f"[rerender] episode {episode_id} recorded no privileged state; skipping")
                continue
            target = self.out / str(episode_id)
            with VideoSet(target, fps=self.args.fps) as videos:
                for frames in self.renderer.replay(steps):
                    videos.append_frames(frames)
            print(f"[rerender] episode {episode_id}: {len(steps)} frames -> {target}")
            total += len(steps)
        if self.renderer.replay_state.skipped:
            print(
                f"[rerender] skipped channels that did not match the model: "
                f"{sorted(self.renderer.replay_state.skipped)}"
            )
        print(f"[rerender] {total} frames through the {self.args.renderer} backend")
        return 0

    def close(self) -> None:
        self.renderer.close()
        self.env.close()


def main() -> int:
    args = create_parser().parse_args()
    example = Example(args)
    try:
        return example.run()
    finally:
        example.close()


if __name__ == "__main__":
    raise SystemExit(main())
