# Tool sorting: which Gemini model for the VLM

Measured 2026-11-05 on GPUs 4 and 5 (RTX PRO 6000 Blackwell), with Vertex AI project `autolab-res`, region `global`,
and `google-genai` 1.59.0 (the vlm bundle's own venv). The graph is the `tool_sorting` cell's `gap_perception_loop`.
It ran under `--config fast` with the flags `vabar bench --print` gives: lockstep, on-demand RTX, a 14k budget, and 5
workers under MPS for a cell. Raw output, scripts and patches are under `/home/shadeform/speed_ws/agent_scratch/vlm/`.

## Answer

**Pin `gemini-3.1-flash-lite` in `fast`.** It needs no client change: it does not think, and `GAP_VLM_MODEL` is
enough. Against today's `gemini-2.5-flash`:

| | `gemini-2.5-flash` | `gemini-3.1-flash-lite` |
| --- | --- | --- |
| Per call, replayed queries (median / p90) | 2.98 / 6.16 s; A/B sheets 4.20 s | **0.88 / 1.19 s**; A/B sheets 0.87 s |
| Per call, inside the graph (median) | 4.9 s | 1.95 s; 1.36 s with the client patch |
| A/B sheets whose answer changes on repeat (5×) | 3 of 36, and 4 of 15 on a second set | **0 of 36, 0 of 15** |
| Same object picked when A and B swap | 64% (it answers "A" 67% of the time) | **86%** (answers "A" 48%) |
| Prose replies to "reply A or B" (read as "B") | 34 of 2,383 in its cells | **0 of 2,944** |
| `level1_9`, end to end | 6/6 today (9/14 earlier) | **5/5**, flashlight first every time |
| Full L1 cell, two runs | 2/30 and 8/30; 50.5 and 46.2 min | **5/30 and 9/30; 37.7 min (28.2 with the client patch)** |
| Mean reward over the 60 episodes | 0.63 | 0.58 |

**Caveats.**
- **The success gain is within noise.** 14/60 against 10/60 gives p = 0.49 (Fisher). The same model on the same cell
  gave 2/30 and 8/30 today, and 1/19 to 9/30 in earlier runs. The firm gains are speed, the same answer on repeat,
  and replies the graph can parse.
- **Lower partial credit.** Being deterministic, 3.1 Flash-Lite keeps choosing an object that the grasp cannot finish,
  an AA battery or a breadboard. It assigned the same label 20 or more times in 16 of 60 episodes, against 7 for 2.5
  Flash (p = 0.06), and those episodes end near reward 0.2-0.4. 2.5 Flash's randomness lets it wander off such an
  object. A label-level exclusion in `select_pair` (stop offering a label after K failed attempts) would fix this.
  Today's exclusion is by pixel position, and a bumped object moves.

**Runner-up: `gemini-3.6-flash` with thinking at `MINIMAL`.** Its two cells reached 5/30 and 7/30 with the best
partial credit (0.68), only 4 stuck episodes, and 31.0 / 30.3 min. But it needs the client patch below (the shipped
client cannot turn its thinking down), it still sends 1% prose replies, and it went 2/5 on `level1_9`. Prefer it if
partial credit matters more than determinism, once `vlm_client.patch` is upstream.

**Do not use** these:
- 3.5 / 3.6 / 3.7 / 3.8 Flash at their default thinking: 2.6-3.4 s a call, and up to a third of A/B sheets flip.
  3.5 Flash answers 14% of A/B sheets in prose, and went 1/3 on `level1_9`.
- 3.5 Flash-Lite: it says "A" on 79% of sheets and went 0/3 on `level1_9`.
- 3.1 Pro: 10 s a call, and 14 of 36 sheets flip.

**How to pin.** Put it in the config's `[suites.tool_sort] env`. That only works once `apply_env` lets a suite's pin
beat the shell, because this shell exports `GAP_VLM_MODEL=gemini-2.5-flash`. See "Pinning" below; the patch is
`vabar_pin.patch`.

## Models this project can call

Listed with `client.models.list(config={"query_base": True})`, then each probed with a one-line `generate_content`
call (`scripts/probe_models.py`).

| Model id | Answers | Thinks by default | Turning thinking down |
| --- | --- | --- | --- |
| `gemini-2.5-flash` (today) | yes | yes (dynamic) | `thinking_budget=0` turns it off |
| `gemini-2.5-flash-lite` | yes | no | — |
| `gemini-2.5-pro` | yes | yes | — |
| `gemini-3-flash-preview` | yes | yes | `thinking_level=MINIMAL` (0 thought tokens) |
| `gemini-3.1-flash-lite` | yes | no | — |
| `gemini-3.1-pro-preview` (this shell's `GAP_LLM_MODEL`) | yes | yes | `LOW` at the least |
| `gemini-3.5-flash` | yes | yes | `MINIMAL` |
| `gemini-3.5-flash-lite` | yes | no | — |
| `gemini-3.6-flash` | yes | yes | `MINIMAL` |
| `gemini-3.7-flash` | yes | yes | `LOW` at the least: `MINIMAL` returns 400 and `thinking_budget=0` is ignored |
| `gemini-3.8-flash` | yes | yes | as 3.7 |
| `gemini-4-argon` | yes, 5-27 s for "OK" | barely (about 20 thought tokens) | not a candidate (unannounced name, 4.9 s a call) |

These return 404 for this project:
- `gemini-3.1-flash-lite-preview`;
- `gemini-3-pro-preview`, `gemini-3.5-pro`, `gemini-3.8-pro`;
- `gemini-3.1-flash`, and the Flash-Lite ids after 3.5;
- the `*-latest` aliases.

The first of these is the vlm bundle's code default (`DEFAULT_MODEL` in `tools/vlm/tools.py`). A shell that sets
neither `GAP_VLM_MODEL` nor `GAP_LLM_MODEL` therefore fails every call.

`gemini-omni-1.1-flash-preview` is listed but serves only the Interactions API. The TTS, image, live, transcribe and
embedding models in the list do not fit this job.

## Why 2.5 Flash flips at temperature 0

`_query_vertex` sets `temperature=0` and `max_output_tokens=1024`, and no thinking config. Without one, 2.5 Flash
and every 3.x Flash think dynamically, and the amount of thinking varies from call to call. On one A/B sheet, two
identical calls used 436 and 683 thought tokens and answered B and A.

Thought tokens also count against `max_output_tokens`. When a call thinks for about 980 tokens, its reply is cut off
mid-sentence (`finish_reason=MAX_TOKENS`), and `select_pair._tournament` reads any reply that does not start with "A"
as "B". In `level1_9` run c1 here, the flashlight-versus-tape sheet got:

> Object A is a tactical flashlight. Object B is painters tape. Both objects are clearly visible and appear
> graspable by a parallel-jaw gripper. Neither object is "topmost"

That counted as "B": the tape knocked the flashlight out, and the pliers were picked first. This is the failure path
`workers_mps.md` describes. That run still succeeded.

Inside the graph, these shares of A/B replies were prose:

| Model | Prose A/B replies | Where |
| --- | --- | --- |
| 2.5 Flash | 1.4% (34 of 2,383) | its two cells |
| 3.6 Flash at `MINIMAL` | 1.0% (21 of 2,073) | its two cells |
| 3.5 Flash at its default | 19% (10 of 53) | `level1_9` |
| 3.1 Flash-Lite | 0 of 2,944 | its two cells |

A typical one reads "Based on the accessibility ... the object in **A** is ...", which also counts as "B".

There is a second source of variation outside the VLM. The overhead render is not bit-identical between runs, so
DINO's candidates and the crops change: 4 candidates in one `level1_9` run, 5 in another. This is why a deterministic
model still does not give identical cells: across two runs, 26 of 30 instances had the same outcome, against 24 for
2.5 Flash. The replay below holds the images fixed, so it measures the model alone.

## Replay: fixed real queries, repeated

**Set** (`replay/final.json`, 61 queries). Each is a `vlm.query` captured from a graph run, prompt plus image:
- 24 from `level1_9`, including every query of the first pick in three runs;
- 20 from `level1_6`, from earlier traced runs;
- 17 from `level1_1`-`level1_5`.

That makes 36 A/B sheets, 20 label questions and 5 layout reads.

Traces alone were not enough to capture them. The trace writer keeps only a node's last iteration under
`calls/NNN_*`, so most of a select node's queries are lost even with `GAP_TRACE_ASSETS=1`. The capture used a
private copy of the vlm bundle that saves every request (see "Client patch").

**Method** (`scripts/replay.py`). The request is built as `_query_vertex` builds it, plus the variant's thinking
setting. Each variant sent each query 5 times, with calls shuffled across variants and 12 in flight. The table uses
three measures:
- **Flip:** the 5 answers to a query, parsed as the graph parses them, are not all the same. An A/B answer is read
  by whether it starts with "A"; a label answer by the `LABEL:` value matched to the list.
- **Agreement:** the share of calls that match 2.5 Flash's majority answer.
- **Order consistency** (`replay/final_swap.json`): each A/B sheet again with its two tiles swapped, 3 calls per
  variant. A sheet is consistent when the majority now names the other letter, that is, the same object. This needs
  no ground truth: a model that always says "A" scores 0%.

Every variant answered all 20 label questions and all 5 layout reads correctly, on every call. The labels were
checked by eye on `replay/label_grid.png`. So the model only matters on the A/B sheets.

| Variant | Median, all / A/B (s) | p90 (s) | A/B sheets that flip (of 36) | Same object when swapped | P("A") | Agrees with 2.5 Flash | Prose A/B replies | `level1_9` first pick (from replay) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gemini-2.5-flash` (today) | 2.98 / 4.20 | 6.16 | 3 (4 of 15 on the screen) | 64% | 0.67 | 99% (itself) | 5 of 180 | flashlight 67%, pliers 22% |
| `gemini-2.5-flash`, thinking off | 1.14 / 1.16 | 1.53 | 0 | 64% | 0.72 | 87% | 0 | flashlight 100% |
| **`gemini-3.1-flash-lite`** | **0.88 / 0.87** | **1.19** | **0 (0 of 15)** | **86%** | **0.48** | 85% | 0 | **flashlight 100%** |
| `gemini-3.5-flash-lite` | 0.81 / 0.80 | 1.09 | 4 | 47% | 0.79 | 84% | 0 | pliers 100% |
| `gemini-3-flash-preview`, MINIMAL | 1.20 / 1.19 | 1.61 | 0 | 64% | 0.67 | 84% | 0 | pliers 100% |
| `gemini-3.5-flash` (default) | 2.58 / 4.19 | 6.82 | 3 | 75% | 0.51 | 83% | 26 of 180 | tape or glue |
| `gemini-3.5-flash`, LOW | 1.85 / 2.42 | 3.56 | 1 | 78% | 0.58 | 85% | 0 | tape or glue |
| `gemini-3.5-flash`, MINIMAL | 1.11 / 1.18 | 1.49 | 1 | 69% | 0.56 | 85% | 0 | tape 100% |
| `gemini-3.6-flash`, MINIMAL | 1.07 / 1.05 | 2.15 | 1 | 83% | 0.61 | 81% | 0 | tape 100% |
| `gemini-3.7-flash`, LOW | 2.58 / 3.24 | 6.27 | 5 | 78% | 0.52 | 86% | 0 | pliers 100% |
| `gemini-3.8-flash`, LOW | 3.18 / 5.76 | 10.91 | 5 | 94% | 0.56 | 84% | 0 | pliers 100% |
| `gemini-3.1-pro-preview` (default) | 9.77 / 12.48 | 15.59 | 14 | 67% | 0.58 | 80% | 2 of 180 | flashlight 80% |

Reading the columns:
- "Same object when swapped" and P("A") pool the original and the swapped calls. On that balanced set, 0.5 means no
  position bias.
- The last column comes from `scripts/l19_first_pick.py`. It works out the first pick's bracket (pliers vs glue,
  flashlight vs tape, then the final) from each variant's answers on the `level1_9` sheets. The earlier MPS study
  found that the flashlight first means success, and the pliers first usually means failure.

A first screen covered 23 variants on 32 `level1_6` queries, 3 calls each (`replay/screen_l16.jsonl`). Beyond the
table it showed:
- At their defaults, 3.6 Flash, 3.8 Flash and 3.1 Pro each flipped on 5 of 15 A/B sheets.
- A fixed `seed` did not stabilise 2.5 Flash: 4 of 15, as without it.
- 2.5 Flash-Lite and `gemini-4-argon` never flipped. They took 1.0 s and 4.9 s a call.

What the replay shows:
- **Thinking drives both the latency and the flips.** The variants that do not think are the Lite models, 2.5 Flash
  at budget 0 and 3.x Flash at `MINIMAL`. They answer in 0.8-1.2 s and flip on 0-1 of 36 sheets. The variants that
  think take 2-12 s and flip on 3-14.
- **Determinism is not quality.**
  - 2.5 Flash with its thinking off never flips, but it still picks by position: 64% order consistency, as with
    thinking on.
  - 3.5 Flash-Lite is the fastest and the most bound to position (47%).
  - 3.1 Flash-Lite is the only fast variant with no position bias and high order consistency.
- **Agreement with 2.5 Flash is 81-87% for every other variant,** 3.1 Pro included. The sheets they disagree on are
  close calls between two isolated, graspable objects. Agreement with 2.5 Flash does not measure who is right.

## `level1_9`, end to end

`scripts/run_l19.sh` runs one instance with the brief's flags under `--config fast`, three copies at once on one
GPU. "Wall" is the episode's own `wall_s`. "VLM" is the time in `vlm.*` subcalls, taken from `dag_trace.json`.

| Model | Success | Steps | Wall (s) | VLM calls / time per episode | First pick |
| --- | --- | --- | --- | --- | --- |
| `gemini-2.5-flash` | 4/4, plus 2/2 in its cells | 4,919-4,953 | 158-190 | 13-15 / 65-85 s | flashlight 3, pliers 1 (pliers run still succeeded) |
| `gemini-3.1-flash-lite` | **3/3, plus 2/2 in its cells** | 4,935-5,335 | **142-167** | 13-14 / **28-33 s** | flashlight 3/3 |
| `gemini-3.6-flash`, MINIMAL | 1/3, plus 1/2 in its cells | 3,124; 14,001; 14,001 | 124-344 | 13-52 / 34-136 s | tape, pliers, tape |
| `gemini-3.5-flash`, MINIMAL | 3/3 | 4,938-12,565 | 158-262 | 15-48 / 31-106 s | tape, tape, pliers |
| `gemini-3.5-flash` (default) | 1/3 | 5,486; 6,494; 14,001 | 210-486 | 19-49 / 84-236 s | pliers, tape, tape |
| `gemini-3.5-flash-lite` | 0/3 (all at the budget, reward 0.75) | 14,000 | 251-279 | 38-40 / 79-82 s | pliers 3/3 |

The 6,494-step run of 3.5 Flash finished early with the pliers outside their bin.

The replay predicted both Lite models' first pick exactly. 3.5 and 3.6 Flash at `MINIMAL` sometimes picked the pliers
where the replay said tape every time, because their candidates differed in those runs (the render variation above).

## Full cell, `tool_sorting/L1`

`scripts/run_cell.sh GPU MODEL TAG [THINKING]` runs `vabar bench --cell tool_sorting/L1 --solve graph --config
fast`, with nothing else changed. Every cell ran through a copy of the client that dumps each request (`ors_overlay/`).
With `GAP_VLM_THINKING_*` unset, that copy behaves as the shipped client. The 3.6 Flash cells set `minimal` through
the patch's thinking knob. `cell_g31fl_v2` used `ors_overlay_v2/`, which adds the reused connection.

Wall times:
- "Bench wall" is the runner's "all in". The script's total, start-up included, is 0.3-0.4 min more.
- "VLM share" is time in `vlm.*` subcalls over episode time.

GPUs: the 2.5 Flash cells ran on GPU 5, the 3.1 Flash-Lite cells on GPU 4 (which also carried another tenant's 3.6
GB process), and the 3.6 Flash cells on one of each. The two GPUs gave 3.6 Flash 31.0 and 30.3 min.

| Run | Model / client | Success | Mean reward | Bench wall (min) | VLM calls | Per call, median / p90 (s) | VLM share | Stuck episodes (one label 20+ times) | Successes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cell_g25f_a` | 2.5 Flash, shipped | 2/30 | 0.64 | 50.5 | 1,733 | 4.96 / 7.91 | 67% | 4 | 6, 9 |
| `cell_g25f_b` | 2.5 Flash, shipped | 8/30 | 0.62 | 46.2 | 1,649 | 4.93 / 8.15 | 70% | 3 | 5, 6, 9, 14, 15, 20, 24, 30 |
| `cell_g31fl_a` | 3.1 Flash-Lite, shipped | 5/30 | 0.53 | 37.7 | 2,085 | 1.95 / 2.21 | 46% | 8 | 4, 6, 9, 14, 15 |
| `cell_g31fl_v2` | 3.1 Flash-Lite, + reused client | 9/30 | 0.64 | **28.2** | 1,839 | 1.36 / 1.62 | 35% | 8 | 4, 5, 6, 9, 14, 15, 18, 20, 30 |
| `cell_g36fmin_a` | 3.6 Flash `MINIMAL` | 5/30 | 0.67 | 31.0 | 1,460 | 2.08 / 3.03 | 43% | 3 | 14, 20, 22, 24, 30 |
| `cell_g36fmin_b` | 3.6 Flash `MINIMAL` | 7/30 | 0.69 | 30.3 | 1,531 | 2.08 / 2.95 | 43% | 1 | 4, 5, 9, 14, 22, 24, 30 |

Pooled over two runs each:

| Model | Success | Mean reward (± s.e.) | Stuck episodes | Instances that succeeded in both runs |
| --- | --- | --- | --- | --- |
| 2.5 Flash | 10/60 | 0.63 ± 0.03 | 7/60 | 2 |
| 3.1 Flash-Lite | 14/60 | 0.58 ± 0.04 | 16/60 | 5 |
| 3.6 Flash `MINIMAL` | 12/60 | 0.68 ± 0.03 | 4/60 | 4 |

Earlier 2.5 Flash cells scored 6/20, 9/30, 1/19 and 7/30 (45.5 min).

The stuck episodes are the graph looping select, grasp, recover on one object (`scripts/stuck.py`). The object is the
AA battery in `level1_8/13/16/20/22`, and a breadboard in `level1_11/12/25/28`. 2.5 Flash gets stuck on the same
objects, less often. The faster models also cut the cell's wall time through fewer VLM-heavy loops: 3.6 Flash made
1,500 calls a cell against 3.1 Flash-Lite's 1,840-2,085.

## Pinning

Today no config names a `GAP_VLM_*` variable. So the model a run used depends on the shell, and `vabar_config.json`
does not record it.

**`setdefault` matters here.** `apply_env` uses `os.environ.setdefault`, so a variable already in the shell wins.
This machine's shell exports `GAP_VLM_MODEL=gemini-2.5-flash`. So `[suites.tool_sort] env = { GAP_VLM_MODEL =
"gemini-3.1-flash-lite" }` changes nothing here, and neither does `--set suites.tool_sort.env.GAP_VLM_MODEL=...` on
the command line. Checked: `Config.env` asks for 3.1 Flash-Lite, but `apply_env` returns 2.5 Flash and the run uses
it.

Unsetting the variable is no fix: the client then falls back to `GAP_LLM_MODEL`, which is `gemini-3.1-pro-preview`
here (10 s a call, 14 of 36 sheets flip).

**`vabar_pin.patch`** is in `/home/shadeform/speed_ws/agent_scratch/vlm/`, against the current `speed/default-fast`
working tree; apply it with `patch -p1` from the repo root (a dry run applies cleanly). It changes three things:
- **`apply_env`** assigns the variables that a suite table names, or that a `--set` of one names, instead of using
  `setdefault`, and prints a line when this replaces a shell value. `runtime.env` and a cell's own env keep "the shell
  wins". `bench` passes its jobs' tasks, so the pin holds there too.
- **`default.toml`**: `[suites.tool_sort] env = { GAP_VLM_MODEL = "gemini-2.5-flash" }`. That is the model VABAR's
  tool-sort numbers so far were measured with, so `default` stops depending on the shell without changing what it
  measures.
- **`fast.toml`**: `[suites.tool_sort] env = { GAP_VLM_MODEL = "gemini-3.1-flash-lite" }`, with the evidence in a
  comment. Changing the VLM is a measured-behaviour tradeoff (partial credit above), so it goes in `fast`.

Checks:
- With the shell's 2.5 Flash export in place, `default` resolves to 2.5 Flash and `fast` to 3.1 Flash-Lite.
- Another suite keeps the shell's value.
- `tests/test_config.py` and `tests/test_bench_config.py` pass, except `test_provenance`, which needs a git checkout
  and fails only in the copy they ran in.
- To try another model, use `--set suites.tool_sort.env.GAP_VLM_MODEL=<id>`. The run's provenance records it.

**Why the config, not the graph.** Passing `model=` in `select_pair.py` and `discover_regions.py` would also beat
every variable. But it would tie the graph to Vertex's id (OpenRouter's slug is `google/gemini-3.1-flash-lite`), and
a graph cannot set thinking. The provider and credentials (`GAP_VLM_PROVIDER`, `GAP_VLM_PROJECT_ID`) stay the user's.

**Caveats:**
- The pin assumes Vertex. An OpenRouter user has to `--set` the slug.
- `paper.toml` does not inherit `default` and was left alone; the paper's VLM is not recorded here.
- If 3.6 Flash is pinned instead, pin `GAP_VLM_THINKING_LEVEL = "minimal"` in the same table, and only after the
  client patch is upstream. An older client ignores the variable, and 3.6 Flash then thinks: 2.9 s a call, and 5 of
  15 sheets flip.

## Client patch

`/home/shadeform/speed_ws/agent_scratch/vlm/vlm_client.patch` is against open-robot-skills `tools/vlm/tools.py` at
`752dfbd`. It applies cleanly in a dry run and was not applied. It makes two changes:

1. **Reuse one `genai.Client`** per (project, region) for the server's life. Today `_query_vertex` builds a client for
   every call, and the first request on a new client opens a TLS connection and fetches an access token.
   - Direct calls (`scripts/client_overhead.py`, one 3.1 Flash-Lite A/B sheet, 10 calls each way, alternating): 1.52 s
     per call with a fresh client, 0.91 s reused.
   - In the graph: 3.1 Flash-Lite's median call fell from 1.95 s to 1.36 s, and the cell from 37.7 to 28.2 min.
   - It applies to any model. Only 3.1 Flash-Lite was measured; for others the saving should be similar per call,
     but it is a smaller share of a slower call.
2. **`GAP_VLM_THINKING_LEVEL` or `GAP_VLM_THINKING_BUDGET` sets a `ThinkingConfig`.** The level is `minimal`, `low`,
   `medium` or `high` (Gemini 3.x); the budget is an int, and `0` turns off 2.5 Flash's thinking. Unset, nothing
   changes. 3.1 Flash-Lite does not need it. It is what made 3.x Flash at `MINIMAL` and 2.5 Flash at budget 0
   possible above: 2.5 Flash, for example, went from 2.98 s with 3 flips to 1.14 s with none.

Two more changes are suggested but not in the patch:
- `DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"` should become `gemini-3.1-flash-lite`, because the preview id is
  gone from Vertex.
- For a thinking model, `max_output_tokens=1024` is too small, because thought tokens count against it. It would need
  a thinking cap or a larger limit.

**Running against a patched client without touching the live checkout:** put a registry that holds only `tools/vlm`
ahead of open-robot-skills:

```bash
GAP_SKILLS_PATH=$VABAR/robot_skills:<copy>/open-robot-skills:/home/shadeform/speed_ws/open-robot-skills
```

The first registry wins, and gap logs the bundle it shadows. The copy needs a `graph-as-policy` symlink beside it,
because its `pyproject.toml` points at `../../../graph-as-policy/gap-core`. `uv` builds the copy's own `.venv` on
first use. This study's two copies:
- `ors_overlay/`: the thinking knob plus a request dump (`vlm_capture_measurement_only.patch`). Every graph run used
  it, except `cell_g31fl_v2` and the first two trace-asset captures (`runs/l19_g25f_assets_1`,
  `runs/cap_g25f_l1x5`), which used the live client.
- `ors_overlay_v2/`: the same plus the reused client. `cell_g31fl_v2` used it.

## Files

All under `/home/shadeform/speed_ws/agent_scratch/vlm/` (2.5 GB):
- `scripts/`:
  - `probe_models.py`;
  - `replay.py` (variants), `build_set.py` and `build_from_capture.py` (replay sets);
  - `analyze.py`, `swap_consistency.py`, `l19_first_pick.py`, `client_overhead.py`;
  - `tally.py` and `cell_table.py` (cells), `capstats.py` (in-graph calls), `stuck.py`, `episode_vlm.py`
    (one episode's VLM sequence);
  - `run_l19.sh`, `run_cell.sh`, `run_cell_v.sh` (`OVERLAY=`);
  - `tools_patched.py`, the client with the patch applied.
- `replay/`: `final.json` and `final.jsonl` (61 queries, 3,660 calls), `final_swap.*` (1,296 calls),
  `screen_l16.*` (2,208 calls), `img/`, `label_grid.png`, `l19_ab_grid.png`.
- `capture/<run>/`: every in-graph request, with prompt, PNG, reply and latency.
- `runs/<run>/`: run directories and logs. `runs/l19_summary.txt` and `runs/cell_summary.txt` hold the wall clocks.
- `vlm_client.patch`, `vabar_pin.patch`, `vlm_capture_measurement_only.patch`.
