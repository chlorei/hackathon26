# dreamkit plugin standard (v1.4)

Every plugin — built-in or from GitHub — follows this document. `dreamkit check <plugin>` enforces the rules
marked **[check]**. If code and this document disagree, this document wins.

---

## 1. Layout

A plugin is a Python package. Built-ins live in `dreamkit/plugins/<module>/`, external ones in their own repo:

```
dreamkit-<name>/
├── pyproject.toml          # package + [tool.dreamkit] manifest + entry point
├── README.md               # what it does, params table, example events
├── <module>/
│   ├── __init__.py         # exports the plugin class only
│   ├── plugin.py           # Params + Plugin class
│   └── ui/index.html       # optional: the plugin's own page in the panel (§7a)
└── tests/
    └── test_<module>.py    # replay tests (section 9)
```

- `name`: kebab-case, 1–3 words, a noun phrase: `recovery`, `failure-monitor`, `gripper-check`. **[check]**
- module: the same in snake_case: `failure_monitor`. **[check]**
- One plugin per package. No shared mutable globals between plugins.

---

## 2. Manifest  **[check]**

```toml
[project]
name = "dreamkit-gripper-check"
version = "0.1.0"                      # semver
dependencies = ["dreamkit>=0.1,<0.2"]

[project.entry-points."dreamkit.plugins"]
gripper-check = "gripper_check:GripperCheck"

[tool.dreamkit]
api = "1"                              # plugin API version this plugin targets
group = "diagnosis"                    # core | diagnosis | safety | output
icon = "Hand"                          # a lucide-react icon name
capability = "observe"                 # observe | modify_actions | control
hooks = ["on_step", "on_episode_end"]  # only the hooks the class implements
step_rate_hz = 5                       # 1–20; how often on_step runs
needs_frames = false                   # true only if it reads camera images
needs = ["robot"]                      # "robot" (any robot) or ["robot", "world"] (simulator only)
requires = ["failure-monitor"]         # plugins that must run before this one
conflicts = []                         # plugins that must not be on at the same time
ui = "ui/index.html"                   # optional: a page shown in the panel, relative to the module folder (§7a)
```

Built-ins declare the same fields as class attributes (`manifest = Manifest(...)`); the manager reads both forms the
same way.

---

## 3. Capabilities

| capability | may | may not |
|---|---|---|
| `observe` | read anything, `ctx.emit`, `ctx.ui.*`, write `ctx.result` keys it owns | change actions, stop or restart |
| `modify_actions` | everything above + make `ctx.actions` **stricter** (clip, slow, zero) | make actions larger, faster or outside joint limits; stop |
| `control` | everything in `observe` + `ctx.stop()`, `ctx.request_retry()`, `ctx.override()`, `ctx.reset_policy()` | — |

- At most **one** `control` plugin is on at a time. The manager refuses the second and the UI says
  `conflicts with <name>`. **[check]** The built-in one is `recovery` (it also retries a scene and stops on
  `camera_failure`). Any plugin reports a failure as it happens with `ctx.verdict(reason, at_s, message)`; the control
  plugin decides what to do. Success ends an attempt in the core: once `success-check` writes `success = True`.
- `modify_actions` plugins run in config order, as a chain, last before the robot.
- No plugin ever changes the robot model, the cameras, Dream Labs' simulator files, or bypasses `validate_chunk`.

---

## 4. Class skeleton

```python
from pydantic import BaseModel, Field
from dreamkit import Plugin, Manifest


class Params(BaseModel):
    min_lift_m: float = Field(0.02, ge=0, le=0.2, description="How high the object must rise to count as lifted",
                              json_schema_extra={"unit": "m"})
    window_s: float = Field(1.0, gt=0, description="How long the grip must stay closed",
                            json_schema_extra={"unit": "s"})
    reasons: list[str] = Field(["dropped"], description="Failure reasons that trigger this plugin",
                               json_schema_extra={"choices": ["dropped", "released_off_target", "not_reached"]})


class GripperCheck(Plugin):
    """Catches a gripper closed on nothing and a grip that loosens mid-carry."""   # = UI description

    name = "gripper-check"
    Params = Params
    manifest = Manifest(group="diagnosis", icon="Hand", capability="observe",
                        hooks=["on_episode_start", "on_step", "on_episode_end"], step_rate_hz=5)

    def on_episode_start(self, ctx):
        self.closed_at = None            # all per-episode state is reset here

    def on_step(self, ctx):
        ...

    def on_episode_end(self, ctx):
        ...
```

Rules:
- Params is a pydantic model. **Every** field has `description`; physical values have `unit`; enumerations have
  `choices`. The settings form is generated from this — no hand-made UI. **[check]**
- Param names: snake_case. A physical value has a unit and the matching suffix: `_m`, `_cm`, `_s`, `_ms`, `_hz`,
  `_deg`, `_rad`, `_rad_s`, `_n` (newtons), `_fps`. A dimensionless number (a count, a fraction, a 0…1 gripper
  command) declares `unit=""` and carries no suffix. Every number declares its unit. **[check]**
- No magic numbers in hook bodies: thresholds live in Params.
- The class docstring is the one-line UI description: starts with a verb, one sentence, ≤ 90 characters, ends with a
  period. **[check]**
- `self.params` is the validated Params instance. Plugins never read `dreamkit.yaml` themselves.

---

## 5. Hooks

Order in one episode:
`on_run_start → on_episode_start → (before_model → after_model → on_step × N)… → on_episode_end → on_run_end`

| hook | rate | typical use |
|---|---|---|
| `on_run_start` / `on_run_end` | once | open/close files, summaries |
| `on_episode_start` | per attempt | reset state |
| `before_model` | per model call | check inputs, stop before sending |
| `after_model` | per model call | inspect / tighten actions, latency |
| `on_step` | `step_rate_hz` | track positions, contacts, grip |
| `on_episode_end` | per attempt | verdicts, files |

Performance  **[check at runtime]**
- The control step is 50 ms. All plugins together get **5 ms per step**; one plugin should stay under **1 ms**.
- Anything slow (disk, video, network, reports, datasets) goes to `ctx.defer(fn, *args)` — a background worker.
- Read camera frames only with `needs_frames = true`; frames are not rendered otherwise.
- No `time.sleep`, no network calls, no subprocesses inside a hook.

Failure behaviour
- Never raise on purpose. To end an attempt use `ctx.stop(reason)` (control only).
- The manager catches any exception, records it against the plugin, and turns the plugin off after
  **5 errors in a row** or **3 runs over budget**, with the reason shown in the UI.

---

## 6. Reading the robot and the world

Two levels. Use the helpers, not raw MuJoCo indexing, so every plugin measures the same thing the same way.

**`ctx.robot` — always there** (a real robot reports it; so does the simulator):

```python
ctx.robot.joints("right")           # measured joint angles, radians (6)
ctx.robot.grip("right")             # measured opening, 0 closed … 1 open
ctx.robot.command("right")          # the row executed this step (6 joints + gripper)
ctx.robot.grip_command("right")     # its gripper value
ctx.robot.tracking_error("right")   # max |commanded − measured| joint, radians
ctx.robot.command_speed("right")    # how far the command moved since the last step
ctx.robot.speed("right")            # measured joint speed, rad/s
ctx.robot.last_timing               # the model's last round trip
ctx.frames                          # cameras (needs_frames = true)
```

**`ctx.world` — only in the simulator** (`client.step(env, ...)`; `ctx.world.available` is False otherwise):

```python
ctx.world.pos("apple")              # np.array([x, y, z]) in metres, world frame (Z up)
ctx.world.tcp("right")              # tool point of a hand
ctx.world.contacts()                # [(body_a, body_b), …]; contacts(with_force=True) adds force_n
ctx.world.table_z                   # table height
ctx.target                          # parsed task: {"object": "banana", "target": "plate"}
ctx.t                               # simulated seconds since the attempt started
```

The loop says which it has: `client.step(env, images=..., command=...)` in the simulator, or
`client.step(state={"left": …, "right": …}, images=..., command=(left, right))` on a robot. A plugin declares
`needs = ["robot"]` or `["robot", "world"]`; the manager switches a world plugin off on a robot and the panel marks it
**sim only**. Prefer `["robot"]`: a plugin that works from the robot's own signals works everywhere.

Without a world, success cannot be measured: `success-check` leaves the outcome unknown, the panel asks a person for
✓ / ✗ (or the loop calls `client.label(True)`), and `harvester` keeps the attempt in `unlabeled/` until then. Verdicts
say what they rest on: `ctx.verdict(..., source="robot")` or `source="world"`.

If a plugin needs something the helpers lack, add it to `ctx.robot` / `ctx.world` in the core, not inside the plugin.

---

## 7. Events, verdicts, UI

Events — `ctx.emit(level, message, **data)`  **[check]**
- `level`: `info` | `success` | `warn` | `error`.
- `message`: lowercase sentence fragment, no trailing period, object and time in it:
  `released banana off target at 9.3 s`, `arms home, asking again (2/3)`, `round trip 820 ms, slow`.
- `data`: snake_case keys with unit suffixes: `{"t_s": 9.3, "dist_cm": 12.1}`.
- At most **one event per second** per plugin on `on_step`; aggregate instead of spamming.

Verdicts — `ctx.result`
- Only `success-check` writes `ctx.result["success"]`; only `failure-monitor` writes `reason`, `at_s`, `details`.
- Other plugins add keys under their own name: `ctx.result["gripper-check"] = {...}`.
- Failure reasons are snake_case codes from the shared list in `dreamkit/reasons.py`
  (`dropped`, `released_off_target`, `grasped_air`, `not_reached`, `collision`, `fell_off_table`,
  `stalled`, `slow_model`, `camera_failure`, `timeout`; and `stopped`, `model_error`, `error` for attempts cut short).
  New codes are added there, with a human label.
- `data` keys: snake_case; a float is a measurement and needs a unit suffix (`t_s`, `dist_m`, `rtt_ms`); strings,
  ints and booleans do not.

UI — standard blocks first; a plugin that needs more ships one page in the panel's style (§7a), never custom React
```python
ctx.ui.metric("grip force", 12.3, unit="N")
ctx.ui.mark(t_s=9.3, label="released off target", level="error")   # timeline marker
ctx.ui.series("hand_to_object", t_s, value, unit="cm")              # chart line
ctx.ui.badge("recovered 2×", level="success")
```

---

## 7a. Plugin UI  **[check]**

A plugin may ship **one page** for what the standard blocks cannot show (a grid of frames, a table, downloads).
Declare it in the manifest: `ui = "ui/index.html"`, a path inside the plugin's module folder, so it ships with the
package.

Where it shows
- In Live and History, the plugin's lines in the feed get a `›`; a click opens a side panel with the page.
- The page runs in `<iframe sandbox="allow-scripts allow-downloads">`: no same-origin, so it cannot read the panel,
  its cookies or other plugins' pages. The panel passes `?run_id=<id>&attempt=<n>` (`attempt` is the episode number).
- The server serves it at `/plugin-ui/<name>/` with the theme already linked; files next to `index.html` are served
  from the same folder.

Data
- Only the plugin's own files, written in `run_dir/<plugin>/` (§8), read with `fetch("/media/<run_id>/<plugin>/…")`.
  Downloads are plain links to the same files (`<a href="/media/…" download>`).
- No network beyond `/media/` and `/theme/`; no API calls; the page never changes anything.

Style — the panel's, only
- The only stylesheet is the shared theme, `/theme/dreamkit-theme.css` (the panel's CSS variables and its base
  components: `.dk-card`, `.dk-title`, `.dk-pill`, `.dk-chip`, `.dk-dot`, `.dk-bar`, …). The server links it; the
  page adds layout CSS of its own if it needs to.
- Colours only through the theme's variables (`var(--dk-ok)`, …). **[check]** no `#hex`, `rgb()`, `hsl()` or named
  colours in CSS, in `style=` attributes or in script strings.
- No fonts of its own (`@font-face`, a `font-family` that is not a theme variable). **[check]**
- No external anything: no `http://`, `https://` or `//host` URLs, no CDN, no CSS/JS framework; scripts and styles
  inline or as files next to `index.html`. **[check]**

---

## 8. Files

- Write only inside `ctx.run_dir / "<plugin-name>/"`. Never outside the run folder.
- File names: snake_case, episode-numbered: `ep03_overhead.mp4`, `ep03_trace.json`.
- Large writes go through `ctx.defer`.

---

## 9. Tests  **[check]**

Each plugin ships replay tests: a recorded run is played back without the simulator, the model or credits.

```python
from dreamkit.testing import replay

def test_flags_release_off_target():
    result = replay("tests/fixtures/banana_off_target", plugins=["gripper-check"])
    assert result.events.any(plugin="gripper-check", level="error")
    assert result.budget_ms("gripper-check") < 1.0
```

- At least one passing case and one failing case.
- `dreamkit test <plugin>` runs them; `dreamkit test <plugin> --replay runs/<id>` runs on any recorded run.

---

## 10. Plugin Development Kit (PDK)

Everything a developer needs to write, debug and ship a plugin without a robot, without the simulator open and
without spending credits. Ships inside `dreamkit` (`pip install dreamkit` is enough).

### 10.1 CLI

| command | what it does |
|---|---|
| `dreamkit new <name> [--kind observe\|modify\|control]` | scaffolds a package from the template for that capability: manifest, Params, class, README, tests, a fixture |
| `dreamkit dev <path> [--replay runs/<id>]` | runs the plugin on a recorded run in a loop and **reloads on every save**; prints events, verdicts and ms/step |
| `dreamkit check <path> [--readme]` | validates everything marked **[check]**; `--readme` regenerates the params table |
| `dreamkit test <path> [--replay runs/<id>]` | runs the plugin's tests, or the plugin on any recorded run |
| `dreamkit bench <path>` | runs the plugin on all bundled fixtures and reports ms/step (avg, p95, max) against the 1 ms budget |
| `dreamkit record [--mock] [--task ...]` | records a run from the simulator into a new fixture (to reproduce a bug or add a case) |
| `dreamkit publish <path>` | runs check + test + bench, then prints the `registry.yaml` entry and the PR steps |

### 10.2 Python SDK — `dreamkit.sdk`

```python
from dreamkit.sdk import Plugin, Manifest, Params, Field   # base class + manifest + pydantic helpers
from dreamkit.sdk import reasons                            # shared failure codes and labels
from dreamkit.sdk.testing import replay, fake_ctx, fixtures # tests without the simulator
```

- `fake_ctx(**overrides)` — a `Context` with a scripted world (positions, grip, contacts over time) for unit tests of
  one hook, no recorded run needed.
- `replay(fixture, plugins=[...])` — plays a recorded run through the plugins and returns `events`, `result`, per-hook
  timings and UI blocks.
- `fixtures` — the bundled library of recorded runs (10.3), by name.
- Type hints on everything (`Context`, `World`, `Event`, `Result`), so the editor autocompletes `ctx.world.…`.

### 10.3 Fixture library

Small recorded runs bundled with dreamkit (state, positions, contacts, events; camera frames at low resolution), one
per situation, so every plugin can be tested on the same cases:

`success_apple_bowl`, `dropped_mid_carry`, `released_off_target`, `grasped_air`, `not_reached`, `collision_bowl`,
`fell_off_table`, `stalled`, `slow_model`, `camera_black`, `camera_frozen`.

Each fixture carries its expected verdict, so `failure-monitor` accuracy is measured on the whole library with one
command: `dreamkit bench failure-monitor --accuracy`.

### 10.4 Templates

`dreamkit new` has one template per capability, each a working, tested plugin to edit:
- `observe` — tracks a value on `on_step`, emits a warning, adds a timeline mark and a metric;
- `modify` — clamps joint speed in `after_model`, with a test that actions only get stricter;
- `control` — stops the attempt on a condition, with the `conflicts` field filled in.

### 10.5 Panel dev mode

`dreamkit ui --dev` adds a **Developer** tab:
- event inspector: every event with plugin, hook, time, data;
- per-plugin timing chart (ms/step over the run) and error list with tracebacks;
- the UI blocks a plugin produced (`ctx.ui.*`) shown as they will look;
- a **Replay** button on any run in History: re-runs the current plugin code on that run.

### 10.6 Docs

`docs/` in the repo: this standard, a 10-minute tutorial (`dreamkit new` → edit → `dev` → `test` → `publish`),
the `Context` / `World` reference generated from type hints, and the reasons list.

---

## 11. How the standard is enforced

Principle: **the right way is the easy way, the wrong way does not run.** Nobody has to remember this document;
the SDK, the manager and the installer refuse what breaks it, with a message that says how to fix it.

| layer | what it enforces | how |
|---|---|---|
| **Base class** | manifest, Params, docstring, names | `Plugin.__init_subclass__` validates the class when it is defined; a bad plugin fails at import with `PluginStandardError: <rule>, see PLUGIN_STANDARD.md §N` |
| **Scoped context** | capabilities | each plugin gets its own `ctx` built from its manifest: `observe` gets no `stop()` / `request_retry()`, and `ctx.actions` arrays are read-only (`writeable=False`); `modify_actions` gets a copy, and the manager rejects the change if any value got larger or left joint limits; undeclared hooks are never called |
| **Typed helpers** | events, verdicts, files, UI | `ctx.emit` checks level, unit suffixes in `data`, rate (1/s on `on_step`); `reason` must come from `reasons.py`; `ctx.files` only opens paths inside `run_dir/<plugin>/`; UI through `ctx.ui.*`, or one page under §7a |
| **Manager at runtime** | budget, errors, conflicts | measures every hook; over budget → warning, then off; 5 errors in a row → off; a second `control` plugin is not loaded; reasons shown in the panel |
| **Installer** | nothing unchecked gets in | `dreamkit install` runs `check`, the plugin's tests and `bench` on the fixture library before enabling; any failure → uninstall + the log in the panel |
| **Template repo** | the same tooling everywhere | `dreamkit new` adds `ruff` + `mypy` config, a pre-commit hook running `dreamkit check`, and a GitHub Action running `check` + `test` + `bench` on every push; the registry accepts a PR only with that Action green |
| **Panel** | the same look for every plugin | the UI is generated from the manifest (group, icon, description) and the Params JSON schema; a plugin's own page (§7a) gets the shared theme from the server, runs sandboxed, and `dreamkit check` refuses external URLs, own fonts and colours outside the theme's variables |

Error messages always name the rule, the file and line when known, and the fix:
`gripper_check/plugin.py:14 — Params.window has no unit suffix (§4). Rename to window_s and add unit="s".`

---

## 12. Before publishing — checklist

- [ ] `dreamkit check <path>` passes (manifest, naming, Params descriptions/units, docstring).
- [ ] `dreamkit test <path>` passes, budget under 1 ms per step.
- [ ] README: what it does, params table (generated by `dreamkit check --readme`), one example event.
- [ ] Capability is the smallest that works (`observe` if in doubt).
- [ ] A page of its own (§7a) only if the standard blocks cannot show it; it passes `dreamkit check`.
- [ ] Added to `registry.yaml` by pull request: name, repo, one-line description, group, icon.

---

## Changelog

**v1.1** (decisions taken while implementing v1)
- Names may be one word (`recovery`, `harvester`, `recorder`, …): 1–3 words.
- Unit suffixes only for physical values; dimensionless numbers declare `unit=""`.
- One control plugin: `recovery`. The old `retry` plugin is part of it (`retry_attempts`, `retry_reasons`).
  `camera-check` is `observe` and reports `camera_failure` through `ctx.verdict`; `recovery` stops the robot.
- `ctx.verdict()` (any capability) is the live failure channel; `ctx.override()` / `ctx.reset_policy()` are control-only.
- Demo faults (a black or frozen camera) are injected by the core (`faults:` in dreamkit.yaml), never by a plugin.
- `contacts(with_force=True)` returns `(a, b, force_n)`.
- Bundled fixtures: `collision_table` (the mock hits the table, not the bowl) instead of `collision_bowl`; also
  `released_near_target`.

**v1.4** (plugin pages)
- A plugin may ship one page in the panel (§7a): manifest `ui`, sandboxed iframe (`allow-scripts allow-downloads`),
  data only from `run_dir/<plugin>/` via `/media`, style only from the shared `/theme/dreamkit-theme.css`.
  `dreamkit check` refuses external URLs, own fonts and colours outside the theme's variables.
- The core keeps the Live heartbeat going while it waits for `ctx.defer` jobs, so a plugin waiting for a model call
  does not make Live take the run for finished.

**v1.3** (real robots)
- `ctx.robot` (always) and `ctx.world` (simulator only); `client.step(state=…)` without an env; manifest `needs`.
- failure-monitor judges grasped_air, dropped, collision, stalled, slow_model, camera_failure from robot signals
  when there is no world; verdicts carry `source`. Unknown outcomes are labelled by a person (or `client.label()`).
- reason `marked_failed` for a ✗ given by a person.

**v1.2** (cleanup)
- `missed_target` is folded into `released_off_target` (details say whether it was a near miss).
- Built-ins: failure-monitor (with model latency), recovery, harvester, success-check, recorder, camera-check.
  `latency-monitor` merged into failure-monitor; recaps and the report are part of History (core);
  `safety` and `gripper-check` are external examples in plugins-external/.
- dreamkit.yaml holds plugins and their settings only (+ sim_path, runs_dir).
