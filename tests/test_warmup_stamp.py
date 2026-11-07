# SPDX-License-Identifier: Apache-2.0
"""A cell's first graph run on an install compiles its scenes, once. No GPU: the warmup itself is a stand-in."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vabar import benchmark, warmup
from vabar import config as cfg


@pytest.fixture
def stamped(monkeypatch, tmp_path):
    """The stamp in a temporary directory, and a list of the warmups asked for."""
    calls: list[list[str]] = []
    monkeypatch.setattr(warmup, "stamp_path", lambda cell, config: tmp_path / "vabar_warm" / "stamp")
    monkeypatch.setattr(warmup, "main", lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(warmup, "pipelines_compiled", lambda: bool(calls))
    return calls, tmp_path / "vabar_warm" / "stamp"


def test_a_cell_is_warmed_once(stamped) -> None:
    calls, stamp = stamped
    config = cfg.resolve("fast", [])
    warmup.ensure_warm("tool_hanging/L1", config)
    warmup.ensure_warm("tool_hanging/L1", config)
    assert len(calls) == 1
    assert calls[0][:4] == ["--cell", "tool_hanging/L1", "--config", config.ref]
    assert stamp.exists()


def test_a_failed_warmup_is_tried_again(stamped, monkeypatch) -> None:
    _, stamp = stamped
    monkeypatch.setattr(warmup, "main", lambda argv: 1)
    warmup.ensure_warm("tool_hanging/L1", cfg.resolve("fast", []))
    assert not stamp.exists()


def test_a_changed_scene_is_warmed_again(stamped, monkeypatch) -> None:
    calls, _ = stamped
    config = cfg.resolve("fast", [])
    monkeypatch.setattr(warmup, "scene_digest", lambda cell: "before")
    warmup.ensure_warm("tool_hanging/L1", config)
    warmup.ensure_warm("tool_hanging/L1", config)
    monkeypatch.setattr(warmup, "scene_digest", lambda cell: "after a git pull")
    warmup.ensure_warm("tool_hanging/L1", config)
    assert len(calls) == 2


def test_a_stamp_from_before_digests_warms_once_more(stamped) -> None:
    calls, stamp = stamped
    config = cfg.resolve("fast", [])
    stamp.parent.mkdir(parents=True)
    stamp.touch()
    warmup.ensure_warm("tool_hanging/L1", config)
    warmup.ensure_warm("tool_hanging/L1", config)
    assert len(calls) == 1


def test_the_digest_reads_the_suites_own_files() -> None:
    assert warmup.scene_digest("tool_hanging/L1") == warmup.scene_digest("tool_hanging/L2")
    assert warmup.scene_digest("tool_hanging/L1") != warmup.scene_digest("syringe_packing/L1")


@pytest.mark.parametrize("config", ["default", "fast", "paper"])
def test_a_suite_that_renders_tiled_is_neither_warmed_nor_stamped(stamped, config) -> None:
    calls, stamp = stamped
    warmup.ensure_warm("syringe_packing/L1", cfg.resolve(config, []))
    assert calls == [] and not stamp.exists()


def test_a_tiled_suite_asked_for_rtx_warms_rtx(stamped) -> None:
    calls, stamp = stamped
    warmup.ensure_warm("syringe_packing/L1", cfg.resolve("fast", []), ["--renderer", "rtx"])
    assert calls[0][-2:] == ["--renderer", "rtx"] and stamp.exists()


def test_warmup_compiles_each_suite_on_its_own_renderer(capsys) -> None:
    assert warmup.main(["--cell", "syringe_packing/L1", "--config", "fast", "--print"]) == 0
    assert "--renderer tiled" in capsys.readouterr().out.splitlines()[0]
    assert warmup.main(["--cell", "tool_hanging/L1", "--config", "fast", "--print"]) == 0
    assert "--renderer rtx" in capsys.readouterr().out.splitlines()[0]


@pytest.mark.parametrize("extra", [["--renderer", "tiled"], ["--renderer=gl"]])
def test_another_renderer_is_not_warmed(stamped, extra) -> None:
    calls, _ = stamped
    warmup.ensure_warm("tool_hanging/L1", cfg.resolve("fast", []), extra)
    assert calls == []


def test_an_install_without_rtx_is_not_warmed(stamped, monkeypatch) -> None:
    calls, _ = stamped
    monkeypatch.setattr(warmup, "stamp_path", lambda cell, config: None)
    warmup.ensure_warm("tool_hanging/L1", cfg.resolve("fast", []))
    assert calls == []


@pytest.mark.parametrize(
    ("argv", "warmed"),
    [
        ([], True),
        (["--no-warmup"], False),
        (["--solve", "idle"], False),
    ],
)
def test_bench_warms_only_a_graph_run(monkeypatch, argv, warmed) -> None:
    asked: list[str] = []
    monkeypatch.setattr(warmup, "ensure_warm", lambda cell, config, extra=(): asked.append(cell))
    monkeypatch.setattr(benchmark, "plan_cell", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(benchmark, "run_plan", lambda plan, flags: 0)
    assert benchmark.main(["--cell", "tool_hanging/L1", "--config", "fast", *argv]) == 0
    assert asked == (["tool_hanging/L1"] if warmed else [])


def test_a_scene_that_finishes_is_not_killed(tmp_path: Path) -> None:
    cmd = [sys.executable, "-c", "print('compiled')"]
    assert warmup.run_scene(cmd, tmp_path / "scene.log", dict(os.environ), 30.0)
    assert "compiled" in (tmp_path / "scene.log").read_text()


def test_a_hung_scene_is_killed_with_its_children(tmp_path: Path) -> None:
    child_pid = tmp_path / "child.pid"
    # A process that starts a child of its own and then hangs, as a renderer's helpers would.
    script = (
        "import subprocess, sys, time;"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']);"
        f"open({str(child_pid)!r}, 'w').write(str(c.pid));"
        "time.sleep(120)"
    )
    started = time.perf_counter()
    assert not warmup.run_scene([sys.executable, "-c", script], tmp_path / "scene.log", dict(os.environ), 2.0)
    assert time.perf_counter() - started < 30.0
    pid = int(child_pid.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        # A killed child stays a zombie of pid 1 for a moment; a zombie is no longer running.
        try:
            if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
                return
        except OSError:
            return
        time.sleep(0.1)
    raise AssertionError("the hung scene's child outlived it")


@pytest.fixture
def driver_cache(monkeypatch, tmp_path):
    """A render cache in a temporary directory, on a GPU that says who it is, with nothing compiled yet."""
    monkeypatch.setattr(warmup, "_render_cache", lambda: tmp_path)
    monkeypatch.setattr(warmup, "_nvidia_smi", lambda: ["RTX, 580.126.09"])
    return tmp_path / "nv_shadercache" / "vabar_compiled"


def _driver_file(cache: Path, name: str = "9cd601fd3f14d49d", body: bytes = b"pipelines") -> Path:
    """One of the driver's files, as a compile leaves it."""
    directory = cache / "nv_shadercache" / "vulkan" / "GLCache" / "driver" / "app"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.bin").write_bytes(body)
    (directory / f"{name}.toc").write_bytes(b"table " + body)
    return directory


def test_the_compile_happens_once(driver_cache) -> None:
    calls: list[int] = []
    assert warmup.compile_pipelines(lambda: calls.append(1) or True) is True
    assert warmup.compile_pipelines(lambda: calls.append(1) or True) is True
    assert calls == [1]


def test_a_failed_compile_is_tried_again(driver_cache) -> None:
    assert warmup.compile_pipelines(lambda: False) is False
    assert warmup.pipelines_compiled() is False
    assert warmup.compile_pipelines(lambda: True) is True


def test_a_new_driver_compiles_again(driver_cache, monkeypatch) -> None:
    warmup.compile_pipelines(lambda: True)
    monkeypatch.setattr(warmup, "_nvidia_smi", lambda: ["RTX, 590.1"])
    assert warmup.pipelines_compiled() is False


def test_deleting_the_driver_files_forgets_the_compile(driver_cache) -> None:
    warmup.compile_pipelines(lambda: True)
    driver_cache.unlink()
    assert warmup.pipelines_compiled() is False


def test_no_gpu_tool_means_no_compile_on_record(driver_cache, monkeypatch) -> None:
    warmup.compile_pipelines(lambda: True)
    monkeypatch.setattr(warmup, "_nvidia_smi", list)
    assert warmup.pipelines_compiled() is False


def test_the_compile_is_copied_under_the_names_the_driver_tries_next(driver_cache, tmp_path) -> None:
    directory = _driver_file(tmp_path)
    assert warmup.spread_pipelines(4) == 3
    names = sorted(p.stem for p in directory.glob("*.bin"))
    assert names == ["9cd601fd3f14d49d", "9cd601fd3f14d49e", "9cd601fd3f14d49f", "9cd601fd3f14d4a0"]
    assert {(directory / f"{n}.bin").read_bytes() for n in names} == {b"pipelines"}
    assert {(directory / f"{n}.toc").read_bytes() for n in names} == {b"table pipelines"}
    assert not list(directory.glob("*.partial"))


def test_a_file_already_there_is_left_alone(driver_cache, tmp_path) -> None:
    directory = _driver_file(tmp_path)
    _driver_file(tmp_path, "9cd601fd3f14d49e", b"written by a compile")
    assert warmup.spread_pipelines(3) == 1
    assert (directory / "9cd601fd3f14d49e.bin").read_bytes() == b"written by a compile"


def test_copies_are_made_each_gpu_up_to_a_limit(driver_cache, tmp_path, monkeypatch) -> None:
    directory = _driver_file(tmp_path)
    warmup.compile_pipelines(lambda: True)
    assert len(list(directory.glob("*.bin"))) == warmup.FILES_PER_GPU
    driver_cache.unlink()
    monkeypatch.setattr(warmup, "_nvidia_smi", lambda: ["RTX, 580.126.09"] * 8)
    warmup.compile_pipelines(lambda: True)
    assert len(list(directory.glob("*.bin"))) == warmup.MOST_FILES


def test_no_driver_files_means_nothing_to_copy(driver_cache) -> None:
    assert warmup.spread_pipelines(4) == 0


def test_a_warmed_cell_from_before_the_one_compile_warms_once_more(stamped) -> None:
    calls, stamp = stamped
    config = cfg.resolve("fast", [])
    stamp.parent.mkdir(parents=True)
    stamp.write_text(warmup.scene_digest("tool_hanging/L1"))
    warmup.ensure_warm("tool_hanging/L1", config)
    assert len(calls) == 1


def test_a_cells_first_run_warms_one_scene(stamped) -> None:
    calls, _ = stamped
    warmup.ensure_warm("tool_hanging/L1", cfg.resolve("fast", []))
    assert "--first-scene" in calls[0]


def _fake_run(started: list[tuple[str, bool]]):
    def run_scene(cmd, log, env, timeout_s) -> bool:
        started.append((cmd[cmd.index("--task") + 1], warmup.pipelines_compiled()))
        meta = Path(cmd[cmd.index("--out") + 1]) / cmd[cmd.index("--name") + 1] / "run_meta.json"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps({"provenance": {"builds": [{"build_s": 1.0}]}}))
        return True

    return run_scene


def test_the_first_scene_compiles_alone_and_is_not_warmed_twice(driver_cache, monkeypatch, tmp_path) -> None:
    started: list[tuple[str, bool]] = []
    monkeypatch.setattr(warmup, "run_scene", _fake_run(started))
    out = str(tmp_path / "out")
    assert warmup.main(["--cell", "tool_hanging/L1", "--config", "fast", "--jobs", "4", "--out", out]) == 0
    assert started[0][1] is False and len(started) > 2 and all(done for _, done in started[1:])
    assert len({task for task, _ in started}) == len(started)


def test_first_scene_warms_only_the_first_instance(driver_cache, monkeypatch, tmp_path) -> None:
    started: list[tuple[str, bool]] = []
    monkeypatch.setattr(warmup, "run_scene", _fake_run(started))
    out = str(tmp_path / "out")
    assert warmup.main(["--cell", "tool_hanging/L1", "--first-scene", "--config", "fast", "--out", out]) == 0
    assert len(started) == 1


def test_a_suite_that_renders_tiled_compiles_nothing(driver_cache, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(warmup, "run_scene", lambda *args: False)
    monkeypatch.setattr(warmup, "compile_pipelines", lambda compile_one: pytest.fail("tiled needs no driver pipelines"))
    warmup.main(["--cell", "syringe_packing/L1", "--config", "fast", "--out", str(tmp_path / "out")])
