# dreamkit

Plugins around the Dream-1 control loop. Everyone who drives a robot with Dream-1 writes the same code around the
model: camera checks, motion limits, retries, logs, "did it actually work?". dreamkit makes each of those a plugin you
switch on with one line, and adds the one nobody has: **failure-monitor**, which tells you *why* an attempt failed and
*when* ("dropped apple at 9.3 s"), with the video already at that second and a short recap clip of what happened.

```
$ dreamkit run --mock
  ep1   5.25s · failure-monitor  left gripper closing on apple (0.3 cm away)
  ep1   6.65s · failure-monitor  lifted apple 3 cm
  ep1   9.35s ⚠ failure-monitor  apple released at 12 cm
  ep1  25.00s ✗ failure-monitor  dropped apple at 9.3 s
  ep1  25.00s · retry            retrying scene 1 after 'dropped' (retry 1/2)
  ep2  13.15s ✓ success-check    apple is in the bowl
  ep2  13.15s ✓ recap            1 of 2 attempts worked; 1 of 1 scenes solved. Retries saved 1 scene.
```

## Install

You need **Python 3.11–3.13**, **git**, **[uv](https://docs.astral.sh/uv/)** (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
and, only to rebuild the web panel, Node 20+. macOS or Linux.

dreamkit drives Dream Labs' MuJoCo simulator unchanged, so clone both repositories **side by side**:

```bash
mkdir dream && cd dream
git clone https://github.com/dreamlabs-co/dreamlabs.git
git clone https://github.com/chlorei/hackathon26.git dreamkit

cd dreamkit
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install ../dreamlabs/dream_library/dream_client -e ".[dev]"
cp dreamkit.example.yaml dreamkit.yaml      # sim_path already points at ../dreamlabs/...
```

The folder layout this expects:

```
dream/
├── dreamlabs/   Dream Labs' repository (the simulator lives in dream_library/examples/yam_mujoco)
└── dreamkit/    this repository
```

If your `dreamlabs` checkout is somewhere else, set `sim_path` in `dreamkit.yaml`.

**Check it works** (no API key, no credits):

```bash
dreamkit run --mock --episodes 1     # a scripted arm drops the apple, retry solves it: ✗ then ✓
pytest -q                            # 15 passed
```

**Dream-1 itself** needs an API key from [platform.dreamlabs.ai](https://platform.dreamlabs.ai/api-keys), either in
`~/.dreamlabs/credentials` (the file the platform's Setup page gives you) or as `export DREAM_API_KEY=dl_live_...`.
Every model call uses credits.

## Use

```bash
dreamkit ui                         # the panel: http://127.0.0.1:8800
dreamkit run --mock                 # in the terminal, scripted policy, free
dreamkit run --episodes 10          # Dream-1 (uses credits)
dreamkit list                       # installed plugins
dreamkit install <what>             # add a plugin (see below)
dreamkit new my-check               # scaffold a plugin package
dreamkit review --limit 20          # failed attempts -> labels.csv, each with its video at the failure second
dreamkit accuracy labels.csv        # after you label them: "failure-monitor: 17/20 reasons correct"
```

**Where things are saved.** Every run gets a folder, `runs/<id>/` next to `dreamkit.yaml` (the panel shows the full
path and an *Open folder* button):

| file | what |
|---|---|
| `ep<N>/top.mp4`, `left.mp4`, `right.mp4` | each camera, per attempt; video time = simulated time |
| `ep<N>/recap.mp4` | the key moments of the attempt, captioned |
| `ep<N>/trace.json` | what failure-monitor measured, step by step |
| `run.json`, `events.jsonl` | results and the full event feed |
| `report.html`, `recap.json` | the run summary |

**Less CPU.** Switch on *Light mode* in the panel (or `light: true` in `dreamkit.yaml`): cameras are rendered at
5 fps instead of 10. Recording fewer cameras helps most: `- recorder: {cameras: [top]}`. The panel itself redraws
cameras and graphs outside React and pauses when its tab is hidden.

## Add plugins from GitHub

Any of these work, in the terminal or on the panel's Plugins page:

```bash
dreamkit install gripper-check                                   # a name from registry.yaml
dreamkit install you/dreamkit-my-check                           # owner/repo
dreamkit install https://github.com/you/repo/tree/main/plugins/x # a folder inside a repo
dreamkit install https://github.com/you/repo/blob/main/my.py     # a single file -> ~/.dreamkit/plugins/
dreamkit install ./my-local-plugin                               # a folder on disk
```

Then switch it on in the panel or add it under `plugins:` in `dreamkit.yaml`. A plugin is code that runs on your
machine: install what you trust.

To publish yours: `dreamkit new my-check`, edit `my_check.py`, push it to GitHub, and add an entry to `registry.yaml`
here with a pull request. `plugins-external/gripper-check/` is a complete example.

## The whole config

```yaml
task: put apple into the bowl
episodes: 5
plugins:
  - camera-check
  - latency-monitor: {warn_ms: 700}
  - success-check
  - failure-monitor
  - retry: {attempts: 2, on: [not_reached, grasped_air]}
  - recorder: {fps: 10}
  - recap
  - report
```

## Built-in plugins

| plugin | what it does |
|---|---|
| `success-check` | object in the bowl / on the plate with the gripper released → success; ends the attempt early |
| `failure-monitor` | first cause in time: `fell_off_table`, `dropped`, `missed_target`, `grasped_air`, `collision`, `stalled`, `not_reached`, `slow_model`, `timeout`; live milestones and graphs |
| `recap` | per attempt a captioned highlights clip and one-sentence story; per run a plain-language summary |
| `camera-check` | stops before the next model call when a camera is black / uniform / frozen (`simulate_failure: top` for a demo) |
| `retry` | same scene again for chosen failure reasons |
| `latency-monitor` | warns above `warn_ms`; mean and p95 per attempt and per run |
| `recorder` | MP4 per camera per attempt; `top.mp4#t=9.3` is the moment |
| `safety` | joint speed cap and a workspace box for the tool points, only ever *stricter* than the simulator |
| `report` | self-contained `report.html` |

Every threshold is a plugin setting: on the Plugins page, or `- failure-monitor: {reach_m: 0.05}`.

## Write a plugin

```python
from dreamkit import Plugin

class GripperCheck(Plugin):
    name = "gripper-check"
    defaults = {"closed_below": 0.2}

    def on_step(self, ctx):                       # 20 Hz, after every env.step
        if ctx.state["right"][6] < self.params["closed_below"]:
            ctx.emit("warn", "right gripper closed")
```

Hooks: `on_run_start`, `on_episode_start`, `before_model`, `after_model` (edit `ctx.actions`), `on_step`,
`on_episode_end` (write `ctx.result`), `on_run_end`. `ctx.stop(reason)` ends the attempt, `ctx.request_retry()` asks
for another, `ctx.telemetry(name=value)` draws a live graph. A plugin that raises never stops the run: the error shows
up in the feed under its name.

## Rules we keep

The robot is Dream Labs' as shipped: dreamkit never edits the simulator's files, the arm model or the cameras, and
every chunk goes through `dream_control.validate_chunk`, twice when a plugin edited it. Plugins may only move objects
(scene randomisation) and tighten actions.

## How it works

```
dreamkit ui → FastAPI (:8800) → Runner subprocess (python -m dreamkit.runner --json-events)
                                   YamEnv (headless) ⇄ loop ⇄ dream.DreamClient ⇄ DreamAPI
                                   PluginManager: hooks → plugins
```

The runner is `dream_control.main`'s loop (same frame windows, `capture`, `feed`, `validate_chunk`, physics paused
while the model thinks) with hooks around each stage. It prints one JSON line per event; the server relays them to
the panel over a WebSocket. MuJoCo never runs in the web server's process.

## Develop the panel

```bash
cd ui && npm install
npm run dev        # http://localhost:5173, proxies the API to `dreamkit ui` on :8800
npm run build      # writes dreamkit/ui_dist/, which `dreamkit ui` serves
```
