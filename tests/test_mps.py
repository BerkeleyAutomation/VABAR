"""``runtime.mps``: the private MPS daemon a multi-worker bench run shares its GPU through."""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import time

import pytest

from vabar import config as cfg
from vabar import mps


def _fake_control(tmp_path, monkeypatch, *, fail_start: bool = False):
    """Put a fake ``nvidia-cuda-mps-control`` first on PATH; it logs argv, stdin and the env it saw."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    script = bin_dir / mps.CONTROL
    script.write_text(
        "#!/bin/sh\n"
        f'echo "argv=$* pipe=$CUDA_MPS_PIPE_DIRECTORY gpu=$CUDA_VISIBLE_DEVICES stdin=$(cat -)" >> {log}\n'
        + ("exit 1\n" if fail_start else "exit 0\n")
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def test_off_and_inherited_leave_the_environment_alone(monkeypatch):
    monkeypatch.delenv(mps.PIPE_ENV, raising=False)
    with mps.daemon(False) as record:
        assert record == {"mps": "off"} and mps.PIPE_ENV not in os.environ
    monkeypatch.setenv(mps.PIPE_ENV, "/somewhere")
    with mps.daemon(True) as record:
        assert record["mps"].startswith("inherited") and os.environ[mps.PIPE_ENV] == "/somewhere"


def test_daemon_starts_exports_and_stops(tmp_path, monkeypatch):
    log = _fake_control(tmp_path, monkeypatch)
    monkeypatch.delenv(mps.PIPE_ENV, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1234")
    with mps.daemon(True) as record:
        assert record["mps"] == "on" and record["gpu"] == "GPU-1234"
        pipe = os.environ[mps.PIPE_ENV]
        assert pipe == record["pipe"] and os.path.isdir(pipe) and len(pipe) < 60  # unix socket path limit
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-1234"  # by UUID: the daemon renumbers its one GPU 0
    assert mps.PIPE_ENV not in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-1234"
    assert not os.path.exists(pipe)  # the private directory is cleaned up after quit
    start, stop = log.read_text().splitlines()
    assert start.startswith(f"argv=-d pipe={pipe} gpu=GPU-1234")
    assert stop.startswith(f"argv= pipe={pipe}") and stop.endswith(f"stdin=quit -t {mps.QUIT_TIMEOUT_S}")


def test_a_daemon_quit_cannot_stop_is_terminated_by_its_pid(tmp_path, monkeypatch):
    # ``quit`` fails (a client still attached); the pid file names a live
    # ``nvidia-cuda-mps-control`` -- here a sleeping stand-in of that name.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / mps.CONTROL
    script.write_text(
        '#!/bin/sh\nif [ "$1" = "-d" ]; then exit 0; fi\nif [ "$1" = "sleep" ]; then while :; do sleep 0.1; done; fi\nexit 1\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv(mps.PIPE_ENV, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1234")
    with mps.daemon(True) as record:
        stand_in = subprocess.Popen([str(script), "sleep"])
        time.sleep(0.2)
        with open(os.path.join(record["pipe"], "nvidia-cuda-mps-control.pid"), "w") as f:
            f.write(str(stand_in.pid))
    assert stand_in.wait(timeout=10) == -signal.SIGTERM
    assert not os.path.exists(record["pipe"])
    # A pid that is no longer the daemon is never signalled.
    other = subprocess.Popen(["sleep", "60"])
    try:
        pipe = tmp_path / "pipe"
        pipe.mkdir()
        (pipe / "nvidia-cuda-mps-control.pid").write_text(str(other.pid))
        assert mps._terminate(str(pipe)) is False and other.poll() is None
    finally:
        other.kill()


def test_index_is_translated_to_uuid(tmp_path, monkeypatch):
    _fake_control(tmp_path, monkeypatch)
    monkeypatch.delenv(mps.PIPE_ENV, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    monkeypatch.setattr(mps, "gpu_uuid", lambda visible: f"GPU-of-{visible}")
    with mps.daemon(True) as record:
        assert record["gpu"] == "GPU-of-3" and os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-of-3"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,5"


def test_missing_binary_or_failed_start_runs_without(tmp_path, monkeypatch):
    monkeypatch.delenv(mps.PIPE_ENV, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))  # no control binary anywhere
    with mps.daemon(True) as record:
        assert record["mps"].startswith("unavailable") and mps.PIPE_ENV not in os.environ
    _fake_control(tmp_path, monkeypatch, fail_start=True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1234")
    with mps.daemon(True) as record:
        assert record["mps"].startswith("unavailable") and mps.PIPE_ENV not in os.environ


def test_config_keys():
    assert cfg.load("default", ["mps=true"]).runtime["mps"] is True
    with pytest.raises(cfg.ConfigError, match="mps"):
        cfg.load("default", ["mps=3"])


def test_suite_workers_win_for_their_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "free_gpu_gb", lambda: (93.0, "GPU 0"))
    path = tmp_path / "w.toml"
    path.write_text('inherits = "default"\n[runtime]\nworkers = 2\n[suites.cable_route]\nworkers = 6\nworker_gb = 5\n')
    config = cfg.load(str(path))
    assert cfg.resolve_workers(config, 30, suite="cable_route") == (6, f"config {config.name} [suites.cable_route]")
    assert cfg.resolve_workers(config, 30)[0] == 2  # no suite: the runtime count
    assert cfg.resolve_workers(cfg.load(str(path), ["workers=3"]), 30, suite="cable_route")[0] == 3
    both = cfg.load(str(path), ["workers=3", "suites.cable_route.workers=5"])
    assert cfg.resolve_workers(both, 30, suite="cable_route")[0] == 5
    assert cfg.resolve_workers(config, 4, suite="cable_route")[0] == 4  # never more than the units of work
    bad = tmp_path / "bad.toml"
    bad.write_text('inherits = "default"\n[suites."cable_route/port_easy"]\nworkers = 6\n')
    with pytest.raises(cfg.ConfigError, match="a cell is one suite"):
        cfg.load(str(bad))


def test_suite_workers_are_held_to_free_memory(tmp_path, monkeypatch):
    path = tmp_path / "w.toml"
    path.write_text('inherits = "default"\n[suites.cable_route]\nworkers = 6\nworker_gb = 5\nshared_gb = 0\n')
    config = cfg.load(str(path))
    monkeypatch.setattr(cfg, "free_gpu_gb", lambda: (93.0, "GPU 0"))
    assert cfg.resolve_workers(config, 30, suite="cable_route")[0] == 6
    monkeypatch.setattr(cfg, "free_gpu_gb", lambda: (22.0, "GPU 0"))  # a 24 GB card
    n, why = cfg.resolve_workers(config, 30, suite="cable_route")
    assert n == 4 and "held to 4 by 22 GB free" in why
    monkeypatch.setattr(cfg, "free_gpu_gb", lambda: None)
    assert cfg.resolve_workers(config, 30, suite="cable_route")[0] == 6  # no nvidia-smi: the config's count
