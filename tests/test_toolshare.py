"""``runtime.share_tool_servers``: a private share directory for the run's brokers, cleaned up after."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from vabar import config as cfg
from vabar import toolshare


def test_off_and_inherited_leave_the_environment_alone(monkeypatch):
    monkeypatch.delenv(toolshare.DIR_ENV, raising=False)
    with toolshare.shared_tool_servers(False) as record:
        assert record == {"tool_servers": "per worker"} and toolshare.DIR_ENV not in os.environ
    monkeypatch.setenv(toolshare.DIR_ENV, "/somewhere")
    with toolshare.shared_tool_servers(True) as record:
        assert record["tool_servers"].startswith("shared, inherited")
        assert os.environ[toolshare.DIR_ENV] == "/somewhere"


def test_the_run_owns_the_brokers_and_stops_them(monkeypatch):
    monkeypatch.delenv(toolshare.DIR_ENV, raising=False)
    monkeypatch.delenv(toolshare.OWNER_ENV, raising=False)
    with toolshare.shared_tool_servers(True) as record:
        root = os.environ[toolshare.DIR_ENV]
        assert record == {"tool_servers": "shared", "dir": root} and len(root) < 40
        assert os.environ[toolshare.OWNER_ENV] == str(os.getpid())
        broker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        (Path(root) / "sam3-0123456789.sock.pid").write_text(str(broker.pid))
    assert toolshare.DIR_ENV not in os.environ and toolshare.OWNER_ENV not in os.environ
    assert not Path(root).exists()
    deadline = time.monotonic() + 10
    while broker.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    assert broker.poll() is not None, "a broker outlived the run"


def test_config_key():
    assert cfg.load("default", ["share_tool_servers=true"]).runtime["share_tool_servers"] is True
    with pytest.raises(cfg.ConfigError, match="share_tool_servers"):
        cfg.load("default", ["share_tool_servers=2"])
