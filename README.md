# dreamkit

A plugin library and monitor for Dream-1, **inside your own control loop**. Everyone who drives a robot with Dream-1
writes the same code around the model: camera checks, motion limits, retries, logs, "did it actually work?". dreamkit
makes each of those a plugin you switch on with one line, and adds the story nobody has: **Detect → Recover → Learn**.

- **failure-monitor** says *why* an attempt failed and *when* ("dropped apple at 8.8 s"), with the video at that second;
- **recovery** fixes it from where the robot is — arms home and ask again, or back off and continue — no scene reset;
- **harvester** saves every attempt as a LeRobot v2.1 dataset; the successful ones pass Dream Labs' own
  `validate_dataset` and are ready for fine-tuning.

**On top of what you already run.** Install the base arm from Dream Labs, put dreamkit over it, change nothing:

```bash
pip install dreamlabs            # the base: Dream Labs' client, station and YAM simulator
pip install -e path/to/dreamkit  # the overlay

dreamkit sim                     # Dream Labs' simulator window + REPL, unchanged; type a task, dreamkit watches
dreamkit sim --mock dropped,grasped_air,success   # same, no model calls: scripted attempts (free)
dreamkit run -- python my_loop.py                 # any program on dream.DreamClient: your loop, a station script
```

`dreamkit run` starts the program as is; when it imports `dream`, its `DreamClient` gets the plugins, and when it
imports Dream Labs' `yam_env`, every `YamEnv.step()` becomes a dreamkit step with the simulator's world. No world
(a real robot): the steps come from the observations the program sends. One `policy()` … `close()` is one attempt; when
a plugin ends it (success, camera failure), the program's next `get_action()` stops that task the way any error does.
The monitor opens at http://127.0.0.1:8800. Writing your own loop? The same in one line:

```python
import dream, dreamkit

client = dreamkit.wrap(dream.DreamClient(), config="dreamkit.yaml")   # the one line
client.policy({"checkpoint": "dreamlabs/dream-1-beta"})

with client.episode("put apple into the bowl", env=env):     # one attempt
    while client.t < 60:
        client.set_observation(frames, observation)             # as before
        reply = client.get_action()                             # plugins run around the model call
        for left, right in rows(reply):
            env.step(left, right)
            client.step(env, images=env.images(), command=(left, right))   # plugins that read the robot
client.close()
```

The panel (`dreamkit ui`) never starts anything: it watches whatever your loop is doing (Live), manages plugins, and
keeps the history with videos, recaps and harvested datasets.

> **About `--mock`.** The mock policy is a scripted arm that **fails on purpose** — it drops the apple, grasps air,
> lets go off target, stalls, hits the table — so every plugin can be built and tested without the model or credits.
> Its numbers (failure rates, "recovery 20% → 100%") say that the *mechanism* works, nothing about Dream-1. Mock runs
> carry a **mock: scripted failures** badge in the panel and the report. Real numbers come from runs without `--mock`.

## Quickstart

See it first: `./demo.sh` opens the panel; on **Live**, **Try demo** runs a free simulated arm that drops the apple
on purpose, so you watch the plugins notice it and recovery step in. Then, for your own robot, the same three steps
as on that page:

1. **Install dreamkit and make a config**
   ```bash
   pip install dreamkit && dreamkit init      # dreamkit.yaml: success-check, failure-monitor, camera-check,
                                              # recovery, reliability, monitor-stats (an existing one is kept)
   ```
2. **Wrap your Dream-1 client: one line**
   ```python
   import dream, dreamkit
   client = dreamkit.wrap(dream.DreamClient())          # the one line
   with client.episode("put apple into the bowl", env=env):
       reply = client.get_action()                       # plugins run around the model
       env.step(left, right)                             # each row of the reply
       client.step(env, command=(left, right))           # robot: state=... instead of env
   ```
3. **Run your loop**: `python my_robot.py`, and `dreamkit ui` (Live comes alive with it).

Keys go in `.env` next to `dreamkit.yaml`: `DREAM_API_KEY` for Dream-1, `GEMINI_API_KEY` for monitor-stats and the
Create tab.

## On a real robot

dreamkit does not need the simulator. Give `step()` what your robot reports instead of an env:

```python
with client.episode("put apple into the bowl", state=read_state()):
    while client.t < 60:
        client.set_observation(frames, observation)
        reply = client.get_action()
        for left, right in rows(reply):
            robot.command(left, right)
            client.step(state=read_state(),          # {"left": 6 joints + gripper, "right": …}
                        images=read_cameras(),       # optional, for camera-check / recorder / harvester
                        command=(left, right))
```

Plugins then see `ctx.robot` (joints, gripper, command vs measured, tracking error, cameras, model latency) but no
`ctx.world` (positions, contacts). What changes:

| | simulator (`env=`) | real robot (`state=`) |
|---|---|---|
| failure-monitor | all reasons, precise times | `grasped_air` (the fingers met), `dropped` (holding, then the fingers closed in without being told to open), `collision` (the arm can't follow a steady command), `stalled`, `camera_failure`; `slow_model` as a warning |
| success-check | measured | unknown: ✓ / ✗ in History, or `client.label(True)` from your loop |
| recovery | arms home via the simulator's own path; lift 5 cm on grasped air | eases to `home_joints_rad`; opens the gripper on grasped air |
| harvester | success/ and failed/ | unlabeled/ until labelled, then moved |
| plugins with `needs: [robot, world]` | on | off, marked **sim only** |

Try it without a robot: `python examples/sim_loop.py --mock --no-world` gives dreamkit only state, cameras and
commands from the simulator, exactly what a robot would. `python examples/world_vs_robot.py --mock` judges the same
attempts both ways:

```
attempt                with world                   robot only (state, gripper, commands)
success                success                      unknown (to label)
dropped                dropped @9.7s                dropped @9.6s
grasped_air            grasped_air @5.3s            grasped_air @5.3s
released_off_target    released_off_target @9.7s    unknown (to label)
not_reached            not_reached @4.6s            grasped_air @5.2s
stalled                stalled @2.2s                stalled @2.2s
collision              collision @4.0s              collision @4.0s
fell_off_table         fell_off_table @8.8s         dropped @8.1s
```

(Mock: the failures are scripted on purpose. Where the robot cannot know — where the fruit landed, whether it is in
the bowl — it says unknown instead of guessing.)

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
python examples/sim_loop.py --mock --episodes 2   # a scripted arm drops the apple; recovery fixes it
dreamkit ui                                       # in another terminal: http://127.0.0.1:8800, Live page
pytest -q                                         # every test, no credits
dreamkit check --all                              # every plugin against the standard
```

**Dream-1 itself** needs an API key from [platform.dreamlabs.ai](https://platform.dreamlabs.ai/api-keys), either in
`~/.dreamlabs/credentials` (the file the platform's Setup page gives you) or as `export DREAM_API_KEY=dl_live_...`.
Every model call uses credits.

## Use

```bash
python examples/sim_loop.py --mock          # Dream Labs' simulator loop with dreamkit.wrap, scripted failures (free)
python examples/sim_loop.py --episodes 10   # the same with Dream-1 (uses credits)
python examples/sim_loop.py --mock --behaviors dropped,grasped_air,released_off_target,success \
       --fault-camera top --fault-episode 4 # the demo: every plugin gets something to do
python examples/recovery_ab.py --episodes 10  # Dream-1, same seeds with and without recovery (uses credits)
python examples/ab_test.py --tune --episodes 10   # prompt tuner: paraphrases of the task, same seeds, ranked with proof
python examples/ab_test.py --models dreamlabs/dream-1-beta,<fine-tune id> --episodes 20   # base vs fine-tune
dreamkit ui                                 # the panel: Live / Plugins / History
dreamkit where                              # where runs and datasets are, and how big
dreamkit datasets [check|train|status <name>]   # every dataset; check: validate_dataset (local); train: upload +
                                            # fine-tune (fine-tuning credits); status: the job, the model id when done
dreamkit migrate [--dry-run]                # older runs/, gym/, harvest/ into the data folder (moves, never deletes)
dreamkit list | search | install | new      # plugins
dreamkit review --limit 20                  # failed attempts -> labels.csv, each with its video at the failure second
dreamkit accuracy labels.csv                # after you label them: "failure-monitor: 17/20 reasons correct"
```

`examples/sim_loop.py` is the reference integration: Dream Labs' own `dream_control` functions (policy_config, feed,
capture, validate_chunk) drive the simulator unchanged; dreamkit adds `wrap`, `episode` and `step`.

## Where things are saved

One folder, `data_dir` in `dreamkit.yaml` (default `./dreamkit-data`, beside the config). `dreamkit where` prints it
with sizes; `dreamkit migrate` moves older layouts into it.

| path | what |
|---|---|
| `runs/<run-id>/run.json`, `events.jsonl` | one run (your loop, `dreamkit sim`/`run`, the Gym tab): results and the full event feed |
| `runs/<run-id>/ep003/` | one attempt: `top.mp4` … (each camera; video time = simulated time), `recap.mp4` (the key moments, captioned), `trace.json` (what failure-monitor measured), `result.json` |
| `runs/<run-id>/history/` | `summary.json` (the run in plain words), `report.html` (self-contained) |
| `runs/<run-id>/<plugin>/` | a plugin's own data, the files its page reads |
| `runs/gym-<id>/` | a Gym evaluation (base vs fine-tune on the same seeds) |
| `datasets/<name>/` | every dataset, LeRobot v2.1 (`meta/`, `data/`, `videos/`), plus `dataset.json`: source (harvester, Gym, Teleop, export), each episode's run and attempt, the `validate_dataset` result, the upload and fine-tune job, the model id |
| `datasets/harvest-<task>/` | the harvester's: each successful attempt of the task, appended as it ends |

History shows an attempt's files (*Show in Finder*) and the dataset it joined; the Data tab lists every dataset with
its source, size and status (collected → checked → uploaded → training → model id), and the storage it takes.

**Less CPU.** Switch on *Light mode* in the panel (or `light: true` in `dreamkit.yaml`): cameras are rendered at
5 fps instead of 10. Recording fewer cameras helps most: `- recorder: {cameras: [top]}`. The panel itself redraws
cameras and graphs outside React and pauses when its tab is hidden.

## Gym (an app plugin)

DreamGym is a dreamkit **app plugin** in [plugins-external/dreamgym](plugins-external/dreamgym/README.md): a tab in
the panel with **Arena** (success rate with a 95% interval over seeded, randomised scenes, failure GIFs), **Compare**
and **Prompt Tuner** (same seeds, McNemar's test), **Data & fine-tune** and **Teleop**. Every episode runs through
`dreamkit.wrap`, so the plugins of dreamkit.yaml work in it, and every run shows in History with a **Gym** badge.
Tick **Recovery on vs off** to measure recovery itself: the same seeds without and with it, McNemar on the pairs.

```bash
dreamkit install ./plugins-external/dreamgym     # check + tests, then pip; restart dreamkit ui -> tab "Data"
dreamkit dreamgym arena --task banana_plate --policy scripted --n 10 --recovery-ab
```

App plugins (a page, an API and commands of their own) are in [PLUGIN_STANDARD.md §13](docs/PLUGIN_STANDARD.md).

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

## Create a plugin by describing it

The panel's **Create** tab: say what to watch ("warn when a hand moves faster than 1.5 rad/s near the bowl"), a model
writes the plugin, and dreamkit runs it through the same filter as `dreamkit install` (the standard, its replay
tests, the 1 ms budget) before you see it; refused packages go back to the model with the problems, three tries at
most. Keep chatting to change it ("make the threshold 0.5"). **Add to catalog** puts it in `plugins-generated/<name>/`
next to `dreamkit.yaml` (listed as *generated* on the Plugins page, where Delete removes it); **Download .zip** gives
you the package. Keys come from `.env`: `GEMINI_API_KEY` (and `ANTHROPIC_API_KEY` for Claude when Gemini is out of
quota).

Generated plugins only watch: observe capability, diagnosis or safety group, no network, files or subprocesses. The
code is scanned before it runs and checked in a separate process without your API keys; it is still code a model
wrote, so read it (the Code panel) before you keep it. The checks prove it follows the standard, not that its idea
is right: a plugin can pass and still measure the wrong thing.

## The whole config

`dreamkit.yaml` says which plugins run and how — nothing else. What to run (task, attempts, model, mock, seed) is
your loop's business: `python examples/sim_loop.py --help`.

```yaml
sim_path: ../dreamlabs/dream_library/examples/yam_mujoco
data_dir: dreamkit-data      # runs/ and datasets/ (see Where things are saved)
plugins:
  - camera-check
  - success-check
  - failure-monitor: {slow_ms: 700}
  - recovery: {max_recoveries: 3, retry_attempts: 1}
  - recorder: {video_fps: 10}
  - harvester
```

## Built-in plugins

| plugin | what it does |
|---|---|
| `failure-monitor` | why and when an attempt failed, first cause in time: `dropped` (the gripper never opened), `released_off_target`, `grasped_air`, `not_reached`, `collision`, `fell_off_table`, `stalled`, `slow_model` (model latency), `camera_failure`, `timeout`; live verdicts mid-attempt |
| `recovery` | the one control plugin. On a live verdict: arms home (`yam_env.home_path`) + `reset_episode()` + same prompt; grasped air: open, lift 5 cm, continue; `camera_failure`: stop; then retry the scene (`retry_attempts`) |
| `harvester` | every attempt as LeRobot v2.1 (state/action: right arm, right gripper, left arm, left gripper; videos overhead / wrist_l / wrist_r) |
| `success-check` | object in the bowl / on the plate with the gripper released → success; the attempt ends there |
| `recorder` | MP4 per camera per attempt; video time = simulated time, so `ep003_top.mp4#t=9.3` is the moment |
| `camera-check` | flags a black, uniform or frozen camera before the next model call; recovery stops the robot |

Examples of external plugins in `plugins-external/`: `gripper-check` (observe) and `safety` (modify_actions: a joint
speed cap and a workspace box — it may only make a chunk stricter). Recaps, the run summary and the report are part
of **History**, made by the core for every run.

Every threshold is a plugin setting: on the Plugins page, or `- failure-monitor: {reach_m: 0.05}`.

## Write a plugin

Every plugin follows [docs/PLUGIN_STANDARD.md](docs/PLUGIN_STANDARD.md) — and cannot do otherwise: a plugin that
breaks it does not import, and the error says the rule, the file:line and the fix.

```bash
dreamkit new hand-distance --kind observe      # or modify / control: a package that already passes everything
dreamkit check dreamkit-hand-distance          # manifest, Params (descriptions, units), docstring, names
dreamkit test dreamkit-hand-distance           # replay tests: no simulator, no model, no credits
dreamkit bench dreamkit-hand-distance          # ms/step on all 12 bundled fixtures, budget 1 ms
dreamkit install ./dreamkit-hand-distance      # check + test + bench first, then pip install; refused otherwise
```

```python
from pydantic import BaseModel, Field
from dreamkit import Manifest, Plugin


class Params(BaseModel):
    closed_cmd: float = Field(0.3, ge=0, le=1, description="Gripper command below which it is shut",
                              json_schema_extra={"unit": ""})
    settle_s: float = Field(0.6, ge=0, description="How long the fingers get to close",
                            json_schema_extra={"unit": "s"})


class GripperCheck(Plugin):
    """Catches a gripper closed on nothing and a grip that loosens mid-carry."""

    name = "gripper-check"
    Params = Params
    manifest = Manifest(group="diagnosis", icon="Hand", capability="observe",
                        hooks=["on_episode_start", "on_step"], step_rate_hz=10)

    def on_episode_start(self, ctx):
        self.closed_at = None

    def on_step(self, ctx):
        if ctx.world.grip("right") < 0.04:                      # ctx.world: pos, tcp, grip, contacts, table_z
            ctx.emit("warn", "right gripper closed on nothing")   # checked: level, form, units, 1/s
            ctx.ui.mark(ctx.t, "closed on nothing", "warn")      # the panel's timeline
```

What a plugin may do depends on its capability: `observe` reads (and reports failures with `ctx.verdict`),
`modify_actions` may only make the chunk stricter (the manager rejects anything faster or outside joint limits),
`control` may stop, retry or drive the arms — and only one control plugin (`recovery`) is on at a time. Files go to
`run_dir/<plugin>/` through `ctx.files`, slow work through `ctx.defer`, the UI through `ctx.ui.*`. Every hook is timed;
5 errors in a row or a plugin over its 1 ms/step budget three times is switched off, and the panel says why.

Fixtures for tests: `dreamkit.testing.replay("dropped_mid_carry", plugins=["failure-monitor"])`; record new ones from
the simulator with `dreamkit record my_case --mock --behavior grasped_air`.
`dreamkit bench failure-monitor --accuracy` scores the verdicts on the whole fixture library (12/12 today).

## Rules we keep

The robot is Dream Labs' as shipped: dreamkit never edits the simulator's files, the arm model or the cameras, and
every chunk goes through `dream_control.validate_chunk` in the loop after the plugins. Plugins may only tighten
actions; the example loop only moves the fruit (scene randomisation).

## How it works

```
 your control loop (sim or robot)                       the panel — dreamkit ui (FastAPI :8800 + React)
 ─────────────────────────────────                      ────────────────────────────────────────────────
 client = dreamkit.wrap(DreamClient)                     Live     watches the run being written
   │ get_action()  step()  episode()                     Plugins  dreamkit.yaml + each plugin's status
   ▼                                                     History  videos, verdicts, recaps, datasets
 PluginManager ── order, one control plugin,                 ▲
   │              budget, errors, scoped ctx                 │ reads files only — never your process
   ▼                                                         │
 plugins ── failure-monitor → recovery → harvester …         │
   │        (ctx.world, ctx.ui, ctx.files, ctx.defer)        │
   ▼                                                         │
 runs/<id>/  run.json  events.jsonl  live/  ep003/  history/ ┘
```

1. **wrap** — `dreamkit.wrap(dream.DreamClient())` is the same client; `get_action()` runs `before_model` / `after_model`
   around the model call, `step()` runs `on_step` after each executed row, `episode()` opens and closes an attempt.
2. **manager** — loads plugins by the standard (requires, conflicts, one control plugin), gives each a context scoped to
   its capability, times every hook and switches off a plugin that keeps failing or runs over budget.
3. **plugins** — measure, decide, act (only `recovery` acts), save an attempt's files in `runs/<id>/ep003/`; the
   harvester adds successful attempts to `datasets/harvest-<task>/`.
4. **runs/** — the run folder is the only interface: events, live frames, a heartbeat, results, videos.
5. **panel** — follows that folder, so your loop never depends on it: open or close it any time, over SSH too.

## Develop the panel

```bash
cd ui && npm install
npm run dev        # http://localhost:5173, proxies the API to `dreamkit ui` on :8800
npm run build      # writes dreamkit/ui_dist/, which `dreamkit ui` serves
```
