# dreamkit

Plugins around the Dream-1 control loop. Everyone who drives a robot with Dream-1 writes the same code around the
model: camera checks, motion limits, retries, logs, "did it actually work?". dreamkit makes each of those a plugin you
switch on with one line, and adds the one nobody has: **failure-monitor**, which tells you *why* an attempt failed and
*when* ("dropped apple at 9.3 s"), with the video already at that second.

```
dreamkit run --mock
  ep1   5.25s · failure-monitor  left gripper closing on apple (0.3 cm away)
  ep1   6.65s · failure-monitor  lifted apple 3 cm
  ep1   9.35s ⚠ failure-monitor  apple released at 12 cm
  ep1  25.00s ✗ failure-monitor  dropped apple at 9.3 s
  ep1  25.00s · retry            retrying scene 1 after 'dropped' (retry 1/2)
  ep2  13.15s ✓ success-check    apple is in the bowl
```

## Install

dreamkit drives Dream Labs' MuJoCo simulator, unchanged, from a checkout next to this folder:

```
some-folder/
├── dreamlabs/   git clone https://github.com/dreamlabs-co/dreamlabs
└── dreamkit/    this repository
```

```bash
cd dreamkit
uv venv --python 3.12 .venv
uv pip install --python .venv ../dreamlabs/dream_library/dream_client -e ".[dev]"
source .venv/bin/activate
cp dreamkit.example.yaml dreamkit.yaml        # sim_path points at ../dreamlabs/... already
```

Dream-1 needs a key in `~/.dreamlabs/credentials` or `DREAM_API_KEY`. `--mock` needs nothing.

## Use

```bash
dreamkit run --mock                 # scripted policy: free, for building and testing plugins
dreamkit run --episodes 10          # Dream-1 (uses credits)
dreamkit ui                         # panel at http://127.0.0.1:8800
dreamkit list                       # installed plugins
dreamkit search gripper             # the catalogue (registry.yaml)
dreamkit install gripper-check      # from GitHub, by catalogue name, URL or local folder
dreamkit new my-check               # scaffold a plugin package
dreamkit review --limit 20          # failed attempts -> labels.csv (video links at the failure second)
dreamkit accuracy labels.csv        # after labelling: "failure-monitor: 17/20 reasons correct"
```

Every run is saved in `runs/<id>/`: `run.json` (results), `events.jsonl` (the feed), `ep<N>/<cam>.mp4` (video, in
simulated seconds), `ep<N>/trace.json` (what failure-monitor measured) and `report.html`.

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
  - report
```

## Built-in plugins

| plugin | what it does |
|---|---|
| `success-check` | object in the bowl / on the plate with the gripper released → success; ends the attempt early |
| `failure-monitor` | first cause in time: `fell_off_table`, `dropped`, `missed_target`, `grasped_air`, `collision`, `stalled`, `not_reached`, `slow_model`, `timeout`; live milestones in the feed |
| `camera-check` | stops before the next model call when a camera is black / uniform / frozen (`simulate_failure: top` for the demo) |
| `retry` | same scene again for chosen failure reasons |
| `latency-monitor` | warns above `warn_ms`, mean and p95 per attempt and run |
| `recorder` | MP4 per camera per attempt; video time = sim time, so `top.mp4#t=9.3` is the moment |
| `safety` | joint speed cap and a workspace box for the tool points — only ever *stricter* than the simulator |
| `report` | self-contained `report.html` |

Every threshold is a plugin parameter (`dreamkit list`, the Plugins page, or `- failure-monitor: {reach_m: 0.05}`).

## Write a plugin

```python
from dreamkit import Plugin

class GripperCheck(Plugin):
    name = "gripper-check"
    defaults = {"closed_below": 0.2}

    def on_step(self, ctx):                       # 20 Hz, after every env.step
        opening = ctx.state["right"][6]
        if opening < self.params["closed_below"]:
            ctx.emit("warn", "right gripper closed")
```

Hooks: `on_run_start`, `on_episode_start`, `before_model`, `after_model` (edit `ctx.actions`), `on_step`,
`on_episode_end` (write `ctx.result`), `on_run_end`. `ctx.stop(reason)` ends the attempt, `ctx.request_retry()` asks
for another. A plugin that raises never stops the run: the error shows up in the feed under its name.

Drop the file in `./plugins/`, or publish it: `dreamkit new gripper-check` makes a package that registers itself under
the `dreamkit.plugins` entry point; push it to GitHub and add it to `registry.yaml`.

## Rules we keep

The robot is Dream Labs' as shipped: dreamkit never edits the simulator's files, the arm model or the cameras, and every
chunk goes through `dream_control.validate_chunk` — twice when a plugin edited it. Plugins may only move objects
(scene randomisation) and tighten actions.

## How it works

```
dreamkit ui → FastAPI (:8800) → Runner subprocess (python -m dreamkit.runner --json-events)
                                   YamEnv (headless) ⇄ loop ⇄ dream.DreamClient ⇄ DreamAPI
                                   PluginManager: hooks → plugins
```

The runner is `dream_control.main`'s loop (same window filling, `capture`, `feed`, `validate_chunk`, physics paused
while the model thinks) with hooks around each stage. It prints one JSON event per line; the server relays them to the
panel over a WebSocket.

## Demo recipes

```bash
# retry: a drop, then the same scene again, solved
dreamkit run --mock                                    # mock.behaviors: [dropped, success] in dreamkit.yaml
# camera fault: in the panel, Plugins → camera-check → simulate_failure: "top" → Start → the robot stops
# plugin from GitHub
dreamkit install gripper-check && dreamkit list
```

## Tests

```bash
pytest -q          # mock runs, no credits
```
