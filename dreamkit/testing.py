"""Plugin tests without the simulator, the model or credits (PLUGIN_STANDARD.md §9, §10).

    from dreamkit.testing import replay
    result = replay("dropped_mid_carry", plugins=["failure-monitor"])
    assert result.result["reason"] == "dropped"
    assert result.budget_ms("failure-monitor") < 1.0

A fixture is one recorded attempt: the robot's state, every tracked body, tool points, grips, contacts, low-res camera
frames, the model's replies and timings. `dreamkit record` makes new ones; a library ships in dreamkit/fixtures/.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .world import SIDES, ReplayWorld

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TRACKED = ("apple", "orange", "banana", "water_bottle", "bowl", "plate")
FRAME_SIZE = (64, 36)


# --------------------------------------------------------------------------------------------------------------
# Recording


class FixtureRecorder:
    """Kit(record=path) feeds this; it keeps the first attempt of the run as a fixture."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.active = False
        self.done = False
        self.meta: dict[str, Any] = {}

    def begin(self, ctx, info: dict[str, Any]) -> None:
        if self.done:
            return
        self.active = True
        self.rows: dict[str, list[Any]] = {k: [] for k in ("t", "cmd_left", "cmd_right", "state_left", "state_right",
                                                           "tcp_left", "tcp_right", "grip_left", "grip_right")}
        self.pos: dict[str, list[Any]] = {b: [] for b in TRACKED if ctx.world.available and ctx.world.has(b)}
        self.contacts: list[list[Any]] = []
        self.frames: dict[str, list[np.ndarray]] = {}
        self.frame_steps: list[int] = []
        self.calls: list[dict[str, Any]] = []
        limits = {s: ctx.world.joint_limits(s).tolist() for s in SIDES} if ctx.world.available else {}
        bodies = ctx.world.bodies() if ctx.world.available else []
        self.meta = {"task": ctx.task, "limits": limits, "bodies": bodies,
                     "info": {k: v for k, v in info.items() if isinstance(v, (int, float, str, list, dict))}}
        self._sample(ctx, step0=True)

    def _sample(self, ctx, step0: bool = False) -> None:
        w = ctx.world
        nan7 = [float("nan")] * 7
        self.rows["t"].append(ctx.t)
        cmd = ctx.command
        self.rows["cmd_left"].append(list(map(float, cmd[0])) if cmd is not None else nan7)
        self.rows["cmd_right"].append(list(map(float, cmd[1])) if cmd is not None else nan7)
        for s in SIDES:
            self.rows[f"state_{s}"].append(list(map(float, ctx.state[s])) if s in (ctx.state or {}) else nan7)
            self.rows[f"tcp_{s}"].append(w.tcp(s).tolist() if w.available else [float("nan")] * 3)
            self.rows[f"grip_{s}"].append(w.grip(s) if w.available else float("nan"))
        for b in self.pos:
            self.pos[b].append(w.pos(b).tolist())
        self.contacts.append([[a, b, round(f, 2)] for a, b, f in (w.contacts(with_force=True) if w.available else [])
                              if f > 0.2])
        if ctx.frames_new or step0:
            if ctx.frames:
                self.frame_steps.append(len(self.rows["t"]) - 1)
                for cam, rgb in ctx.frames.items():
                    small = np.asarray(Image.fromarray(rgb).resize(FRAME_SIZE, Image.BILINEAR))
                    self.frames.setdefault(cam, []).append(small)

    def step(self, ctx) -> None:
        if self.active:
            self._sample(ctx)

    def model_call(self, ctx, reply: dict[str, Any]) -> None:
        if self.active:
            self.calls.append({"step": len(self.rows["t"]) - 1, "timing": ctx.timing,
                               "reply": {k: np.asarray(v, dtype=np.float32).round(5).tolist() for k, v in reply.items()
                                         if k in ("q_l_chunk", "q_r_chunk")}})

    def end(self, ctx, result: dict[str, Any]) -> None:
        if not self.active:
            return
        self.active, self.done = False, True
        out = self.path
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        arrays = {k: np.asarray(v, dtype=np.float32) for k, v in self.rows.items()}
        arrays.update({f"pos_{b}": np.asarray(v, dtype=np.float32) for b, v in self.pos.items()})
        arrays.update({f"frames_{c}": np.asarray(v, dtype=np.uint8) for c, v in self.frames.items()})
        arrays["frame_steps"] = np.asarray(self.frame_steps, dtype=np.int32)
        np.savez_compressed(out / "steps.npz", **arrays)
        (out / "contacts.json").write_text(json.dumps(self.contacts, separators=(",", ":")))
        (out / "calls.json").write_text(json.dumps(self.calls, separators=(",", ":")))
        expected = {"success": result.get("success"), "reason": result.get("reason")}
        (out / "fixture.json").write_text(json.dumps({**self.meta, "name": out.name, "steps": len(self.rows["t"]),
                                                      "expected": expected}, indent=2))

    def close(self, ctx) -> None:
        pass


# --------------------------------------------------------------------------------------------------------------
# Playing back


class Fixture:
    def __init__(self, path: str | Path):
        p = Path(path)
        if not p.exists() and (FIXTURES_DIR / str(path)).exists():
            p = FIXTURES_DIR / str(path)
        if not (p / "fixture.json").is_file():
            raise FileNotFoundError(f"no fixture at {path} (bundled: {', '.join(fixtures())})")
        self.path = p
        self.meta = json.loads((p / "fixture.json").read_text())
        z = np.load(p / "steps.npz")
        self.arrays = {k: z[k] for k in z.files}
        self.contacts = json.loads((p / "contacts.json").read_text())
        self.calls = json.loads((p / "calls.json").read_text())
        self.name = self.meta.get("name", p.name)
        self.task = self.meta["task"]
        self.expected = self.meta.get("expected", {})
        self.steps = int(self.meta["steps"])

    def world(self) -> ReplayWorld:
        a = self.arrays
        rec = {
            "pos": {k[4:]: a[k] for k in a if k.startswith("pos_")},
            "tcp": {s: a[f"tcp_{s}"] for s in SIDES},
            "grip": {s: a[f"grip_{s}"] for s in SIDES},
            "contacts": self.contacts,
            "bodies": self.meta.get("bodies", []),
        }
        limits = {s: np.asarray(v) for s, v in self.meta.get("limits", {}).items()}
        return ReplayWorld(rec, limits)

    def frames_at(self, i: int) -> dict[str, np.ndarray] | None:
        steps = list(self.arrays.get("frame_steps", []))
        if i not in steps:
            return None
        k = steps.index(i)
        return {key[7:]: self.arrays[key][k] for key in self.arrays if key.startswith("frames_")}


def fixtures() -> list[str]:
    return sorted(p.name for p in FIXTURES_DIR.iterdir() if (p / "fixture.json").is_file()) if FIXTURES_DIR.is_dir() else []


class ReplayEnv:
    """Stands in for the simulator: state() as recorded, no physics, no MuJoCo."""

    def __init__(self, fx: Fixture, world: bool = True):
        self.fx = fx
        self.i = 0
        self._world = fx.world() if world else None
        if world:
            self.dreamkit_world = self._world

    def state(self) -> dict[str, np.ndarray]:
        a = self.fx.arrays
        return {s: np.nan_to_num(a[f"state_{s}"][self.i]).astype(float) for s in SIDES}

    def at(self, i: int) -> None:
        self.i = i
        if self._world is not None:
            self._world.at(i)


class ReplayClient:
    mock = True

    def __init__(self, fx: Fixture):
        self.fx = fx
        self.k = 0
        self.last_timing: dict[str, Any] | None = None

    def policy(self, config):
        pass

    def set_observation(self, *a, **k):
        pass

    def get_action(self, **kw):
        call = self.fx.calls[min(self.k, len(self.fx.calls) - 1)]
        self.k += 1
        self.last_timing = call.get("timing") or {"mock": True}
        return {k: np.asarray(v, dtype=float).tolist() for k, v in call["reply"].items()}

    def reset_episode(self, recovery: bool = False):
        pass


class Events(list):
    def any(self, **match: Any) -> bool:
        return any(all(e.get(k) == v for k, v in match.items()) for e in self)

    def of(self, plugin: str) -> list[dict[str, Any]]:
        return [e for e in self if e.get("plugin") == plugin]


@dataclass
class ReplayResult:
    fixture: str
    result: dict[str, Any]
    events: Events
    stats: dict[str, Any]
    refused: dict[str, str]
    run_dir: Path
    expected: dict[str, Any] = field(default_factory=dict)

    def budget_ms(self, plugin: str) -> float:
        return float(self.stats.get(plugin, {}).get("ms_per_step", 0.0))

    def errors(self, plugin: str | None = None) -> list[dict[str, Any]]:
        """The plugin's own failures (a hook raised, the standard was broken) — not error-level verdicts."""
        return [e for e in self.events if e.get("level") == "error" and "hook" in (e.get("data") or {})
                and (plugin is None or e.get("plugin") == plugin)]


def replay(fixture: str | Path, plugins: list[Any], keep: bool = False, extra_dirs: list[Path] | None = None,
           world: bool = True) -> ReplayResult:
    """Plays one recorded attempt through these plugins (names, or {name: params}) and returns what they did.
    world=False plays it as a real robot would report it: state, gripper, commands, cameras — no positions/contacts."""
    from .kit import Kit  # noqa: PLC0415

    fx = Fixture(fixture)
    env = ReplayEnv(fx, world=world)
    tmp = Path(tempfile.mkdtemp(prefix="dreamkit-replay-"))
    specs = [p if isinstance(p, (str, dict)) else str(p) for p in plugins]
    kit = Kit(ReplayClient(fx), config=None, console=False, data_dir=str(tmp), plugins=specs, evaluation=True,
              plugin_dirs=[Path(d) for d in extra_dirs or []])
    call_steps = [c["step"] for c in fx.calls]
    with kit.episode(fx.task, env=env, images=fx.frames_at(0)):
        env.at(0)
        i = 0
        while i < fx.steps - 1:
            if i in call_steps or i == 0:
                kit.get_action()
            i += 1
            env.at(i)
            a = fx.arrays
            cmd = (a["cmd_left"][i], a["cmd_right"][i])
            cmd = None if np.isnan(cmd[0][0]) else (cmd[0].astype(float), cmd[1].astype(float))
            kit.step(env, images=fx.frames_at(i), command=cmd)
    kit.close()
    events = Events(e for e in kit.emitter.records if e.get("type") == "event")
    result = kit.episodes[0] if kit.episodes else {}
    out = ReplayResult(fx.name, result, events, kit.manager.stats_dict(), dict(kit.manager.refused), kit.run_dir,
                       fx.expected)
    if not keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def fake_ctx(plugin, **overrides: Any):
    """A scoped context with a scripted world for unit tests of one hook: fake_ctx(MyPlugin(), t=3.0, state=...)."""
    from .plugin import Context, PluginContext  # noqa: PLC0415

    core = Context(env=None, task=overrides.pop("task", "put apple into the bowl"), model="test", run_id="test",
                   run_dir=Path(tempfile.mkdtemp(prefix="dreamkit-fake-")))
    for k, v in overrides.items():
        setattr(core, k, v)
    if "datasets_dir" not in overrides:  # beside runs/, as in a real data folder: <data>/runs/<id>, <data>/datasets
        core.datasets_dir = Path(core.run_dir).parent.parent / "datasets"
    hook = overrides.get("hook", "on_step")
    return PluginContext(core, plugin, lambda: hook)
