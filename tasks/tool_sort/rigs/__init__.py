# SPDX-License-Identifier: Apache-2.0
"""The measurement rigs behind ``tool_sort``'s constants.

Run by path from the repository root -- ``uv run python tasks/tool_sort/rigs/<rig>.py --help``
-- and never executed by the task loader: a rig measures the suite and is not
part of it. Each reaches the suite's own modules through
:func:`robosimstudio_bench.taskbuild.discovery.task_module`, because a task
directory has no importable name.
"""
