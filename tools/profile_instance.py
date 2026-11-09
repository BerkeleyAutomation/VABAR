# SPDX-License-Identifier: Apache-2.0
"""Where a run's wall time went: build, stepping, and the graph's nodes.

    python tools/profile_instance.py outputs/default/tool_hanging/L1/tool_hanging_wrenches_level1_graph_gap_perception
    python tools/profile_instance.py outputs/default/tool_hanging/L1          # every run directory under it
    python tools/profile_instance.py --exec -- --task tool_hanging/wrenches_level1 --episodes 1 \\
        --solve graph --graph gap_perception --motion curobo --no-privileged --perception vision --max-steps 8500

For each run directory (one holding ``run_meta.json``) it prints:

* from ``run_meta.json``: build_s, control steps, steps/s and wall_s (and, for a
  fleet, the waves and how long the worlds were held for their graphs);
* from ``episode_results.jsonl``: each episode's success, length and close kind;
* from ``traces/*/dag_trace.json``: the graph's wall time split by node --
  perception scripts (``perceive*``, ``register*``, ``build_collision*``,
  ``sight*``, ``ground*``, ``relook*``), the other (motion) scripts, robot tools,
  and planning (nodes named ``*.plan``) -- with the slowest nodes and the tools
  the scripts called.

``--exec`` runs ``python -m vabar run <args>`` first (from this checkout, into
``--out``, default a fresh directory under the system temp dir) and then profiles
what it wrote. Reads run directories only; never writes into them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PERCEPTION = ("perceive", "register", "build_collision", "sight", "ground", "relook")
GROUPS = ("perception scripts", "motion scripts", "robot tools", "planning")


# ----------------------------------------------------------------------------- reading
def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _rows(run: Path) -> list[dict]:
    path = run / "episode_results.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _tool_names(workflow: dict) -> dict[str, str]:
    """``node name`` (``sg.node`` inside a subgraph) -> the tool it calls, from a trace's ``workflow.json``."""
    out = {}
    for name, node in (workflow.get("nodes") or {}).items():
        if isinstance(node, dict) and node.get("tool"):
            out[name] = node["tool"]
    for sg_name, sg in (workflow.get("subgraphs") or {}).items():
        for name, node in (sg.get("nodes") or {}).items() if isinstance(sg, dict) else ():
            if isinstance(node, dict) and node.get("tool"):
                out[f"{sg_name}.{name}"] = node["tool"]
    return out


def group_of(node: dict) -> str | None:
    """Which wall-time group a trace node belongs to; ``None`` for one that runs nothing (subgraph, noop, end)."""
    name = str(node.get("name") or "")
    kind = node.get("node_type")
    if kind not in ("script", "tool"):
        return None
    if name.rsplit(".", 1)[-1] == "plan":
        return "planning"
    if kind == "tool":
        return "robot tools"
    stem = Path(str(node.get("script") or "")).stem
    leaf = name.rsplit(".", 1)[-1]
    if stem.startswith(PERCEPTION) or leaf.startswith(PERCEPTION):
        return "perception scripts"
    return "motion scripts"


def trace_profile(trace_dir: Path) -> dict:
    """One episode's graph: wall time, per-group seconds, per-node rows, the tools scripts called."""
    trace = _json(trace_dir / "dag_trace.json")
    tools = _tool_names(_json(trace_dir / "workflow.json"))
    groups: dict[str, float] = dict.fromkeys(GROUPS, 0.0)
    nodes = []
    starts, ends = [], []
    for node in trace.get("nodes", []):
        group = group_of(node)
        started, finished = float(node.get("started_at") or 0.0), float(node.get("finished_at") or 0.0)
        if group is None or started <= 0.0:
            continue
        seconds = float(node.get("duration_ms") or 0.0) / 1000.0
        groups[group] += seconds
        starts.append(started)
        ends.append(max(finished, started))
        what = node.get("script") or tools.get(node["name"]) or ""
        nodes.append(
            {
                "name": node["name"],
                "type": node.get("node_type"),
                "what": what,
                "group": group,
                "s": seconds,
                "visits": int(node.get("visits") or 0),
            }
        )
    subcalls = Counter(
        (e.get("detail") or {}).get("tool", "?")
        for e in trace.get("events", [])
        if e.get("event_type") == "subcall_recorded"
    )
    wall = (max(ends) - min(starts)) if starts else 0.0
    return {"wall_s": wall, "groups": groups, "nodes": nodes, "subcalls": subcalls}


# ----------------------------------------------------------------------------- printing
def _pct(part: float, whole: float) -> str:
    return f"{part / whole * 100:5.1f}%" if whole > 0 else "    -"


def _print_groups(groups: dict[str, float], wall: float, indent: str = "  ") -> None:
    accounted = sum(groups.values())
    for name in GROUPS:
        print(f"{indent}{name:20s} {groups[name]:9.1f} s {_pct(groups[name], wall)}")
    print(f"{indent}{'between nodes':20s} {wall - accounted:9.1f} s {_pct(wall - accounted, wall)}")
    print(f"{indent}{'graph wall':20s} {wall:9.1f} s")


def profile(run: Path, *, top: int = 8) -> None:
    """Print one run directory's profile."""
    meta = _json(run / "run_meta.json")
    prov = meta.get("provenance", {})
    builds = prov.get("builds", [])
    config = _json(run / "vabar_config.json")
    print(f"run {run}")
    print(
        f"  task {meta.get('task', '?')}   solution {meta.get('solution', '?')}   renderer {meta.get('renderer', '') or '-'}"
        f"   device {meta.get('device', '?')}"
    )
    workers = prov.get("workers", 1)
    worlds = prov.get("num_worlds", 1)
    line = f"  workers {workers}   num_worlds {worlds}   episodes {meta.get('episodes', '?')}"
    line += f"   successes {meta.get('successes', '?')}"
    if config:
        line += f"   config {config.get('config')} ({config.get('sha256', '')[:12]})"
    print(line)
    for b in builds:
        extra = f", {b['num_worlds']} worlds" if b.get("num_worlds") else ""
        print(
            f"  build_s {b.get('build_s', 0):7.1f}   ({b.get('task')}, {b.get('control_hz', 0):g} Hz x "
            f"{b.get('substeps', '?')} substeps{extra}, cuda graph {'yes' if b.get('captured') else 'no'})"
        )
    print(
        f"  steps {prov.get('control_steps', 0)}   steps/s {prov.get('steps_per_s', 0):.1f}"
        f" (stepping {prov.get('steps_per_s_stepping', 0):.1f})   wall_s {prov.get('wall_s', 0):.1f}"
        f" (run {prov.get('run_wall_s', prov.get('pool_wall_s', 0)):.1f})"
    )
    if prov.get("waves"):
        held = (prov.get("policy_provenance") or {}).get("held_s")
        waves = ", ".join(f"{w.get('episodes')} eps {w.get('wall_s', 0):.0f} s" for w in prov["waves"])
        print(
            f"  waves: {waves}; utilization {prov.get('utilization', 0):.2f}"
            + (f"; worlds held for their graphs {held:.0f} s" if held is not None else "")
        )

    rows = sorted(_rows(run), key=lambda r: int(r.get("episode_id", 0)))
    if rows:
        print("  episode  success  reward  length  close_kind        wall_s")
        for r in rows:
            print(
                f"  {r.get('episode_id', '?'):>7}  {'yes' if r.get('success') else 'no':7s}  {float(r.get('reward', 0)):6.2f}"
                f"  {int(r.get('episode_length', 0)):6d}  {r.get('close_kind', '')!s:16s}  {float(r.get('wall_s', 0)):6.1f}"
                + (f"  {r['error'][:80]}" if r.get("error") else "")
            )
    wall_by_episode = {int(r.get("episode_id", -1)): float(r.get("wall_s", 0)) for r in rows}

    traces = (
        sorted(p for p in (run / "traces").glob("*") if (p / "dag_trace.json").is_file())
        if (run / "traces").is_dir()
        else []
    )
    if not traces:
        print("  (no graph traces)")
        return
    total: dict[str, float] = dict.fromkeys(GROUPS, 0.0)
    total_wall = 0.0
    by_node: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    subcalls: Counter = Counter()
    for trace_dir in traces:
        p = trace_profile(trace_dir)
        episode = int(trace_dir.name) if trace_dir.name.isdigit() else None
        ep_wall = wall_by_episode.get(episode, 0.0) if episode is not None else 0.0
        cover = f", {p['wall_s'] / ep_wall * 100:.0f}% of the episode's {ep_wall:.0f} s" if ep_wall else ""
        print(f"  dag {trace_dir.relative_to(run)}: graph wall {p['wall_s']:.1f} s{cover}, {len(p['nodes'])} nodes run")
        _print_groups(p["groups"], p["wall_s"], indent="    ")
        for g in GROUPS:
            total[g] += p["groups"][g]
        total_wall += p["wall_s"]
        for n in p["nodes"]:
            by_node[(n["name"], n["group"], n["what"])].append(n["s"])
        subcalls.update(p["subcalls"])
    if len(traces) > 1:
        print(f"  all {len(traces)} traces:")
        _print_groups(total, total_wall, indent="    ")
    print(f"  slowest nodes (summed over {len(traces)} trace{'s' if len(traces) != 1 else ''}):")
    for (name, group, what), secs in sorted(by_node.items(), key=lambda kv: -sum(kv[1]))[:top]:
        print(f"    {sum(secs):8.1f} s  x{len(secs):<3d} {name:32s} {group:18s} {what}")
    by_tool: dict[str, float] = defaultdict(float)
    for (_, group, what), secs in by_node.items():
        if group == "robot tools":
            by_tool[what or "?"] += sum(secs)
    if by_tool:
        print(
            "  robot tools by tool: "
            + ", ".join(f"{t} {s:.1f} s" for t, s in sorted(by_tool.items(), key=lambda kv: -kv[1]))
        )
    if subcalls:
        print("  tools called from scripts: " + ", ".join(f"{t} x{c}" for t, c in subcalls.most_common(12)))


def run_dirs(path: Path) -> list[Path]:
    """``path`` itself if it is a run directory, else every run directory below it."""
    if (path / "run_meta.json").is_file():
        return [path]
    return sorted(p.parent for p in path.rglob("run_meta.json") if "shards" not in p.parts)


def execute(argv: list[str], out: Path) -> Path:
    """Run ``python -m vabar run <argv>`` from this checkout; the run directory it wrote."""
    if argv[:3] == ["python", "-m", "vabar"]:
        argv = argv[3:]
    if argv[:1] == ["run"]:
        argv = argv[1:]
    if "--out" not in argv:
        argv += ["--out", str(out)]
    else:
        out = Path(argv[argv.index("--out") + 1]).resolve()
    if "--name" not in argv:
        argv += ["--name", f"profile_{time.strftime('%Y%m%d_%H%M%S')}"]
    name = argv[argv.index("--name") + 1]
    cmd = [sys.executable, "-m", "vabar", "run", *argv]
    print("$ python -m vabar run " + " ".join(argv), flush=True)
    subprocess.run(cmd, cwd=REPO, check=False)
    return out / name


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, command = argv[:cut], argv[cut + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*", type=Path, help="run directories, or directories holding them")
    parser.add_argument("--exec", action="store_true", help="run `vabar run <args after -->` first, then profile it")
    parser.add_argument("--out", type=Path, default=None, help="with --exec, where the run goes")
    parser.add_argument("--top", type=int, default=8, help="how many of the slowest nodes to list")
    args = parser.parse_args(argv)
    targets: list[Path] = []
    if args.exec:
        if not command:
            parser.error("--exec needs the vabar run arguments after --")
        out = (args.out or Path(tempfile.mkdtemp(prefix="vabar_profile_"))).resolve()
        targets.append(execute(command, out))
    for path in args.runs:
        found = run_dirs(path)
        if not found:
            print(f"{path}: no run_meta.json here or below", file=sys.stderr)
        targets += found
    if not targets:
        parser.error("name a run directory, or --exec -- <vabar run arguments>")
    for i, run in enumerate(targets):
        if i:
            print()
        profile(run, top=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
