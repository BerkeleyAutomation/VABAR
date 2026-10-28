# SPDX-License-Identifier: Apache-2.0
"""``python -m vabar <command> ...``

list     every registered suite/variant
run      run a solution over a task's episodes (RoboSimStudio's bench_run)
studio   open a task in the RoboSimStudio Studio (browser UI)
bench    run a solution over a frozen benchmark cell (task class x level)
probe    check that every suite loads on the installed RoboSimStudio
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "list": ("vabar.run", "list_main"),
    "run": ("vabar.run", "main"),
    "studio": ("vabar.studio", "main"),
    "bench": ("vabar.benchmark", "main"),
    "probe": ("vabar.probe", "main"),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__.strip())
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    module, func = COMMANDS[argv[0]]
    return getattr(importlib.import_module(module), func)(argv[1:])
