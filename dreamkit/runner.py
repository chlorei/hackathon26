"""The control loop with hooks: dream_control.main's loop, headless, with plugins around every stage.

`python -m dreamkit.runner --config dreamkit.yaml [--mock] [--json-events]`. SIGINT stops cleanly (the current attempt
ends as `stopped`, the run is saved); a second SIGINT aborts.
"""

from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import config as config_mod
from . import scene, sim, storage
from .events import Emitter
from .manager import PluginManager
from .plugin import Context, EndEpisode, StopEpisode

STEP_S = 0.05


class _Interrupt:
    def __init__(self):
        self.requested = False

    def __call__(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True


def _summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    for e in episodes:
        if e.get("success") is False:
            reasons[e.get("reason") or "unknown"] = reasons.get(e.get("reason") or "unknown", 0) + 1
    return {
        "attempts": len(episodes),
        "successes": sum(1 for e in episodes if e.get("success")),
        "scenes": len({e.get("scene") for e in episodes}),
        "scene_successes": len({e.get("scene") for e in episodes if e.get("success")}),
        "reasons": reasons,
    }


class Runner:
    def __init__(self, cfg: config_mod.Config, base: Path, *, mock: bool, json_events: bool, run_id: str | None = None,
                 console: bool = True):
        self.cfg = cfg
        self.base = base
        self.mock = mock
        self.interrupt = _Interrupt()
        self.emitter = Emitter(json_events=json_events, console=console)
        self.run_id = run_id or storage.new_run_id()
        runs_root = Path(cfg.runs_dir).expanduser()
        if not runs_root.is_absolute():
            runs_root = base / runs_root
        self.run_dir = storage.run_dir(runs_root, self.run_id)
        self.emitter.open_file(self.run_dir / "events.jsonl")
        self.info: dict[str, Any] = {}

    # -- bookkeeping --------------------------------------------------------------------------------------------
    def _save(self, ctx: Context | None, status: str) -> None:
        episodes = ctx.episodes if ctx else []
        self.info.update(status=status, episodes=episodes, summary=_summary(episodes), updated=time.time())
        storage.write_json(self.run_dir / "run.json", self.info)

    # -- the run -------------------------------------------------------------------------------------------------
    def run(self) -> int:
        cfg = self.cfg
        signal.signal(signal.SIGINT, self.interrupt)
        try:
            signal.signal(signal.SIGTERM, self.interrupt)
        except ValueError:  # pragma: no cover - not the main thread
            pass
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
        self.info = {
            "id": self.run_id,
            "task": cfg.task,
            "model": "mock" if self.mock else cfg.model,
            "mock": self.mock,
            "episodes_planned": cfg.episodes,
            "seconds": cfg.seconds,
            "seed": cfg.seed,
            "started": time.time(),
            "plugins": [p.name for p in cfg.plugins if p.enabled],
        }
        self._save(None, "starting")

        yam_env, dc = sim.use(cfg.sim_dir(self.base))
        manager = PluginManager(cfg.plugins, extra_dirs=[self.base / "plugins"])
        if self.mock:
            from .policy import MockPolicy

            policy = MockPolicy(importlib.import_module("mock_pickups"), cfg.mock.latency_ms, cfg.mock.jitter_ms,
                                seed=cfg.seed)
        else:
            from .policy import DreamPolicy

            policy = DreamPolicy(dc, cfg.model)

        env = yam_env.YamEnv()
        ctx = Context(env=env, task=cfg.task, model=self.info["model"], run_id=self.run_id, run_dir=self.run_dir,
                      config=cfg, emitter=self.emitter, seed=cfg.seed)
        self.emitter.emit("run_start", run_id=self.run_id, task=cfg.task, model=ctx.model, mock=self.mock,
                          episodes=cfg.episodes, plugins=manager.names, run_dir=str(self.run_dir))
        for name in manager.missing:
            ctx.emit("error", f"plugin '{name}' is not installed (dreamkit list / dreamkit install {name})")
        for line in manager.failed + manager.problems:
            ctx.emit("error", f"plugin failed to load: {line}")
        self._save(ctx, "running")
        status = "finished"
        try:
            manager.call("on_run_start", ctx)
            scene_index = 0
            attempt = 1
            while scene_index < cfg.episodes and not self.interrupt.requested:
                ctx.episode += 1
                ctx.scene, ctx.attempt = scene_index, attempt
                ctx.seed = cfg.seed + scene_index
                behavior = cfg.mock.behaviors[(ctx.episode - 1) % len(cfg.mock.behaviors)] if self.mock else None
                self._episode(ctx, env, dc, policy, manager, behavior)
                self._save(ctx, "running")
                if ctx._retry and not self.interrupt.requested:
                    attempt += 1
                else:
                    scene_index += 1
                    attempt = 1
            if self.interrupt.requested:
                status = "stopped"
            ctx._plugin = "dreamkit"
            try:
                manager.call("on_run_end", ctx)
            except (StopEpisode, EndEpisode):
                pass
        except KeyboardInterrupt:
            status = "aborted"
        except Exception as error:
            status = "error"
            ctx._plugin = "dreamkit"
            ctx.emit("error", f"run failed: {type(error).__name__}: {error}", traceback=traceback.format_exc())
        finally:
            try:
                policy.close()
            except Exception:
                pass
            env.close()
            summary = _summary(ctx.episodes)
            self.info["finished"] = time.time()
            self._save(ctx, status)
            self.emitter.emit("run_end", run_id=self.run_id, status=status, run_dir=str(self.run_dir), **summary)
            self.emitter.close()
        return 0 if status in ("finished", "stopped") else 1

    def _episode(self, ctx: Context, env, dc, policy, manager: PluginManager, behavior: str | None) -> None:
        cfg = self.cfg
        env.reset()
        positions = scene.randomize(env, cfg.randomize.objects, cfg.randomize.xy_jitter, ctx.seed)
        # Fill the camera windows the way dream_control.main does: hold the arms, keep a frame every 4th step.
        windows = {name: deque(maxlen=dc.FRAMES) for name in dc.VIDEO_ROLES}
        steps = 0
        images: dict[str, np.ndarray] = {}
        while any(len(w) < dc.FRAMES for w in windows.values()):
            state = env.state()
            env.step(state["left"], state["right"])
            images = dc.capture(env, windows, keep=steps % dc.FRAME_EVERY_STEPS == 0)
            steps += 1

        ctx.step, ctx.t = 0, 0.0
        ctx.result = {}
        ctx.actions = ctx.command = ctx.timing = None
        ctx._retry = False
        ctx.frames, ctx.frames_new = dict(images), True
        ctx.state = env.state()
        ctx.store.setdefault("episode", {})
        ctx.store["episode"] = {"positions": positions}
        ctx.episode_dir.mkdir(parents=True, exist_ok=True)
        self.emitter.emit("episode_start", episode=ctx.episode, scene=ctx.scene, attempt=ctx.attempt, seed=ctx.seed,
                          task=ctx.task, behavior=behavior, positions=positions)
        started = time.time()
        requests = 0
        ended_by = None
        try:
            manager.call("on_episode_start", ctx)
            policy.begin_episode(env, windows, ctx.task, steps, behavior=behavior)
            self.emitter.frames(ctx.frames, ctx.episode, ctx.t, force=True)
            while ctx.t < cfg.seconds - 1e-9:
                if self.interrupt.requested:
                    raise StopEpisode("stopped by user", "dreamkit")
                manager.call("before_model", ctx)
                reply = policy.get_action(env, windows, ctx.task, steps)  # physics holds while we wait
                requests += 1
                ctx.timing = policy.timing
                adjustments: list[str] = []
                left, right = dc.validate_chunk(env, reply, cfg.actions_per_chunk, verbose=False,
                                                adjustments=adjustments)
                if adjustments:
                    ctx.store["episode"].setdefault("limit_adjustments", 0)
                    ctx.store["episode"]["limit_adjustments"] += len(adjustments)
                ctx.actions = (left, right)
                manager.call("after_model", ctx)
                # Plugins may only tighten the chunk: run whatever they left through validate_chunk again.
                left, right = ctx.actions
                if len(left) == 0 or len(right) == 0:  # a plugin dropped every row: hold for one step instead
                    state = env.state()
                    left, right = state["left"][None, :], state["right"][None, :]
                left, right = dc.validate_chunk(env, {"q_l_chunk": left, "q_r_chunk": right}, cfg.actions_per_chunk,
                                                verbose=False)
                executed = 0
                for l, r in zip(left, right):
                    if self.interrupt.requested or ctx.t >= cfg.seconds - 1e-9:
                        break
                    tick = time.monotonic()
                    ctx.command = (l, r)
                    env.step(l, r)
                    executed += 1
                    kept = steps % dc.FRAME_EVERY_STEPS == 0
                    ctx.frames_new = kept or steps % cfg.render_every == 0
                    if ctx.frames_new:
                        ctx.frames = dict(dc.capture(env, windows, keep=kept))
                    if kept:
                        policy.observe(env, windows, ctx.task, steps)
                    steps += 1
                    ctx.step += 1
                    ctx.t = ctx.step * STEP_S
                    ctx.state = env.state()
                    manager.call("on_step", ctx)
                    if ctx.frames_new:
                        self.emitter.frames(ctx.frames, ctx.episode, ctx.t)
                    if cfg.realtime:
                        time.sleep(max(0.0, STEP_S - (time.monotonic() - tick)))
                policy.advance(executed)
            ended_by = "timeout"
        except EndEpisode as end:
            ended_by = end.reason
        except StopEpisode as stop:
            ended_by = "stopped"
            ctx.result.setdefault("success", False)
            ctx.result.update(reason="stopped" if stop.plugin == "dreamkit" else "stopped_by_plugin",
                              at=round(ctx.t, 2), details={"message": stop.reason, "plugin": stop.plugin})
            ctx._plugin = stop.plugin or "dreamkit"
            ctx.emit("error", f"attempt stopped: {stop.reason}")
            ctx._plugin = "dreamkit"
        except Exception as error:
            ended_by = "error"
            name = type(error).__name__
            ctx.result.update(success=False, reason="model_error" if name.startswith("Dream") else "error",
                              at=round(ctx.t, 2), details={"message": f"{name}: {error}"})
            ctx.emit("error", f"attempt failed: {name}: {error}")
        ctx.result.setdefault("ended_by", ended_by)
        try:
            manager.call("on_episode_end", ctx)
        except (StopEpisode, EndEpisode):
            pass
        try:
            policy.end_episode()
        except Exception:
            pass
        result = {
            "episode": ctx.episode,
            "scene": ctx.scene,
            "attempt": ctx.attempt,
            "seed": ctx.seed,
            "task": ctx.task,
            "behavior": behavior,
            "success": ctx.result.get("success"),
            "reason": None if ctx.result.get("success") else ctx.result.get("reason", "unknown"),
            "duration_s": round(ctx.t, 2),
            "requests": requests,
            "wall_s": round(time.time() - started, 2),
            "retry_requested": ctx._retry,
            **{k: v for k, v in ctx.result.items() if k not in ("success", "reason")},
        }
        videos = sorted(p.name for p in ctx.episode_dir.glob("*.mp4"))
        result["videos"] = [f"ep{ctx.episode}/{v}" for v in videos]
        ctx.episodes.append(json.loads(json.dumps(result, default=lambda v: np.asarray(v).tolist())))
        self.emitter.emit("episode_end", episode=ctx.episode, scene=ctx.scene, attempt=ctx.attempt,
                          duration_s=ctx.t, result=ctx.episodes[-1])


def run(config_path: str | None, *, mock: bool = False, json_events: bool = False, run_id: str | None = None,
        **overrides: Any) -> int:
    cfg, base = config_mod.load(config_path, **overrides)
    return Runner(cfg, base, mock=mock, json_events=json_events, run_id=run_id).run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dreamkit.runner", description=__doc__)
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("--mock", action="store_true", help="scripted policy instead of Dream-1: no API calls")
    parser.add_argument("--json-events", action="store_true", help="JSON lines on stdout instead of the console")
    parser.add_argument("--headless", action="store_true", help="accepted for compatibility; always headless")
    parser.add_argument("--run-id")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--task")
    parser.add_argument("--model")
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args(argv)
    if args.config is None and Path("dreamkit.yaml").is_file():
        args.config = "dreamkit.yaml"
    try:
        return run(args.config, mock=args.mock, json_events=args.json_events, run_id=args.run_id,
                   episodes=args.episodes, task=args.task, model=args.model, seconds=args.seconds)
    except (FileNotFoundError, ValueError) as error:
        print(f"dreamkit: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
