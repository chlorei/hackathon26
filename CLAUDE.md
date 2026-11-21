# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

dreamkit is a plugin library and monitor that runs **inside the user's own Dream-1 control loop** (Detect → Recover →
Learn). The same repo ships a second package, **dreamgym** (`dreamgym/`), which instead *drives* the simulator itself
(Arena, Compare, Prompt Tuner, fine-tune, Teleop); see `dreamgym/README.md`.

It depends on Dream Labs' repo checked out **side by side** at `../dreamlabs` (here: the sibling folder in this
workspace). The MuJoCo simulator is `../dreamlabs/dream_library/examples/yam_mujoco`, found via `sim_path` in
`dreamkit.yaml` (tests use `DREAMKIT_SIM`, dreamgym uses `DREAMGYM_SIM_DIR`). `../dreamlabs` has its own CLAUDE.md
and is Copybara-exported upstream code — do not edit it.

## Commands

Setup (Python 3.11–3.13, uv):
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install ../dreamlabs/dream_library/dream_client -e ".[dev]"
cp dreamkit.example.yaml dreamkit.yaml        # dreamkit.yaml is gitignored
```

Tests (no API key, no credits; tests needing the simulator skip if it isn't found):
```bash
pytest -q
pytest tests/plugins/test_recovery.py -q
pytest tests/test_kit.py::test_name -q
dreamkit check --all                           # every plugin against docs/PLUGIN_STANDARD.md
dreamkit test <plugin> | dreamkit bench <plugin> [--accuracy]
```

Run things:
```bash
python examples/sim_loop.py --mock --episodes 2   # reference integration; --mock = scripted failures, free
python examples/sim_loop.py --mock --no-world     # robot-only mode (no ctx.world)
dreamkit ui                                       # panel on :8800
dreamgym serve                                    # DreamGym app on :8765
```
Anything without `--mock` calls Dream-1 and **spends credits** (key in `~/.dreamlabs/credentials` or `DREAM_API_KEY`).
Don't run those unless asked.

Panel (React + Vite + Tailwind, `ui/`):
```bash
cd ui && npm install
npm run dev     # :5173, proxies API to `dreamkit ui` on :8800
npm run build   # writes ../dreamkit/ui_dist/ (served by `dreamkit ui`, shipped in the wheel) — rebuild after UI changes
npm run lint    # oxlint
```
Python style: ruff, line length 119.

## Architecture

```
user loop → dreamkit.wrap(DreamClient) (kit.py) → PluginManager (manager.py) → plugins → runs/<id>/ ← panel (server.py + ui/)
```

- **kit.py** — `wrap()` returns the same client; `get_action()` runs `before_model`/`after_model` around the model
  call, `step()` runs `on_step`, `episode()` opens/closes an attempt. `step(env=...)` = simulator mode;
  `step(state=...)` = real-robot mode.
- **attach.py + _boot/sitecustomize.py** — `dreamkit run -- <cmd>` / `dreamkit sim`: an overlay that patches
  `dream.DreamClient` and `yam_env.YamEnv.step` in an unmodified program (activated by `DREAMKIT_ATTACH=1`).
- **manager.py** — discovers plugins via the `dreamkit.plugins` entry point (built-ins register the same way in
  `pyproject.toml`), enforces order/`requires`/`conflicts`/**exactly one control plugin**, gives each a context scoped
  to its capability, times every hook. A plugin error never stops the loop; 5 errors in a row or 3 overruns of the
  1 ms/step budget switches it off.
- **plugin.py / standard.py** — the plugin API (`Plugin`, `Manifest`, `Params` pydantic model, `PluginContext`).
  `standard.py` encodes the rules of `docs/PLUGIN_STANDARD.md`; `Plugin.__init_subclass__` raises on a violation, so a
  non-conforming plugin does not even import. **If code and PLUGIN_STANDARD.md disagree, the document wins.**
- **Capabilities**: `observe` (read, `ctx.verdict`), `modify_actions` (may only make a chunk *stricter*; the manager
  rejects faster/out-of-limit chunks), `control` (stop/retry/drive — only `recovery`).
- **Context**: `ctx.robot` always (joints, gripper, command vs measured, cameras, latency); `ctx.world`
  (world.py: positions, contacts) only in the simulator — plugins with `needs: [robot, world]` are off on a real
  robot. Files via `ctx.files` (→ `runs/<id>/<plugin>/`), slow work via `ctx.defer`, UI via `ctx.ui.*`.
- **Run folder is the only interface** to the panel (storage.py, events.py, live.py): `run.json`, `events.jsonl`,
  `live/` frames + `live.json` heartbeat, per-plugin dirs, `history/`. `server.py` (FastAPI) only reads these and
  edits `dreamkit.yaml`; it never starts a run or sends prompts. Keep the loop independent of the panel.
- **mock.py** — `MockClient`, a scripted stand-in for `DreamClient` that injects one failure per attempt using the
  simulator's IK. Its numbers validate mechanisms only, never Dream-1.
- **pdk.py / template.py / testing.py** — `dreamkit check|test|bench|new|install|publish`; replay fixtures in
  `dreamkit/fixtures/` (`dreamkit.testing.replay("dropped_mid_carry", plugins=[...])`, record new ones with
  `dreamkit record`; `tools/make_fixtures.py`). Plugin tests are replay tests — no simulator, no model.
- Built-in plugins: `dreamkit/plugins/`. External examples (own pyproject, installed separately):
  `plugins-external/`. Public catalogue: `registry.yaml`.

## Rules to keep

- Never edit Dream Labs' simulator files, arm model or cameras; only import and call them (`sim.py`). The example
  loop only moves objects (scene randomisation).
- Every chunk still goes through `dream_control.validate_chunk` in the loop, after plugins.
- Plugins may only tighten actions; only one control plugin at a time.
- Real-robot mode must say `unknown` where it can't know (e.g. success), not guess.

`../PLUGINS_SPEC.md` (Russian, workspace root) is a per-plugin design spec of existing and planned plugins.
