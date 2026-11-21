"""dreamkit.wrap(): plugins inside the user's own control loop.

    client = dreamkit.wrap(dream.DreamClient(), config="dreamkit.yaml")
    client.policy({...})
    with client.episode("put apple into the bowl", env=env):
        while client.t < 60:
            client.set_observation(frames, observation)
            reply = client.get_action()          # before_model -> model -> after_model
            for left, right in rows:
                env.step(left, right)
                client.step(env, images=env.images(), command=(left, right))
    client.close()
"""

from __future__ import annotations

import atexit
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import config as config_mod
from . import storage
from .events import Emitter
from .live import LiveWriter
from .manager import PluginManager
from .plugin import Context, EndEpisode, StopEpisode

STEP_S = 0.05


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


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class Kit:
    """A DreamClient with dreamkit's plugins around it. Everything not overridden passes straight through."""

    def __init__(self, client: Any, config: str | Path | None = "dreamkit.yaml", *, console: bool = True,
                 run_id: str | None = None, **overrides: Any):
        path = Path(config) if config is not None else None
        if path is not None and not path.is_file():
            path = None  # no config file: built-in defaults, no plugins
        cfg, base = config_mod.load(path, **overrides)
        if cfg.light:
            cfg = cfg.model_copy(update={"render_every": 4, "stream_fps": min(cfg.stream_fps, 5)})
        self.config = cfg
        self.client = client
        self.manager = PluginManager(cfg.plugins, extra_dirs=[base / "plugins"])
        root = Path(cfg.runs_dir).expanduser()
        root = root if root.is_absolute() else base / root
        self.run_id = run_id or storage.new_run_id()
        self.run_dir = storage.run_dir(root, self.run_id)
        self.emitter = Emitter(json_events=False, console=console)
        self.emitter.open_file(self.run_dir / "events.jsonl")
        self.live = LiveWriter(self.run_dir, fps=cfg.stream_fps)
        self.ctx = Context(env=None, task="", model="", run_id=self.run_id, run_dir=self.run_dir, config=cfg,
                           emitter=self.emitter, seed=cfg.seed)
        self.ctx.reply = None
        self._in_episode = False
        self._closed = False
        self._scene = 0
        self._next_is_retry = False
        self._episode_started = 0.0
        self._requests = 0
        self.info: dict[str, Any] = {
            "id": self.run_id, "task": "", "model": "", "mock": bool(getattr(client, "mock", False)),
            "started": time.time(), "plugins": self.manager.names, "status": "running", "light": cfg.light,
        }
        self._save()
        self.emitter.emit("run_start", run_id=self.run_id, plugins=self.manager.names, mock=self.info["mock"])
        for name in self.manager.missing:
            self.ctx.emit("error", f"plugin '{name}' is not installed (dreamkit install {name})")
        for line in self.manager.failed + self.manager.problems:
            self.ctx.emit("error", f"plugin failed to load: {line}")
        self.manager.call("on_run_start", self.ctx)
        self._beat(force=True)
        atexit.register(self.close)

    # -- pass-through -----------------------------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def __enter__(self) -> "Kit":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(status="stopped" if exc_type is KeyboardInterrupt else "error" if exc_type else "finished")

    @property
    def t(self) -> float:
        """Seconds into the current attempt (steps x 50 ms)."""
        return self.ctx.t

    @property
    def retry_requested(self) -> bool:
        """True when a plugin (retry) asked to run the last attempt's scene again."""
        return self._next_is_retry

    @property
    def episodes(self) -> list[dict[str, Any]]:
        return self.ctx.episodes

    def policy(self, config: dict[str, Any]) -> Any:
        self.ctx.model = str(config.get("checkpoint", ""))
        self.info["model"] = "mock" if self.info["mock"] else self.ctx.model
        self._save()
        return self.client.policy(config)

    def set_observation(self, frames: Any, observation: dict[str, Any], **kwargs: Any) -> Any:
        prompt = observation.get("prompt") if isinstance(observation, dict) else None
        if prompt and not self.ctx.task:
            self.ctx.task = prompt
        return self.client.set_observation(frames, observation, **kwargs)

    def get_action(self, **kwargs: Any) -> dict[str, Any]:
        """before_model -> the model -> after_model. Raises StopEpisode / EndEpisode when a plugin ends the attempt
        (caught by `with client.episode(...)`)."""
        if not self._in_episode:
            self._begin(self.ctx.task, self.ctx.env, None)
        ctx = self.ctx
        self.manager.call("before_model", ctx)
        reply = self.client.get_action(**kwargs)
        self._requests += 1
        ctx.timing = getattr(self.client, "last_timing", None)
        ctx.reply = reply
        yam = isinstance(reply, dict) and "q_l_chunk" in reply and "q_r_chunk" in reply
        ctx.actions = (np.array(reply["q_l_chunk"], dtype=float), np.array(reply["q_r_chunk"], dtype=float)) if yam else None
        before = None if ctx.actions is None else (ctx.actions[0].copy(), ctx.actions[1].copy())
        self.manager.call("after_model", ctx)
        if yam and ctx.actions is not None and before is not None:
            left, right = (np.asarray(a, dtype=float) for a in ctx.actions)
            if left.shape != before[0].shape or not np.array_equal(left, before[0]) or not np.array_equal(right, before[1]):
                if len(left) == 0:  # a plugin dropped every row: hold where the arms are
                    state = ctx.state or {}
                    left = np.asarray(state.get("left", before[0][:1]))[None, :] if "left" in state else before[0][:1]
                    right = np.asarray(state.get("right", before[1][:1]))[None, :] if "right" in state else before[1][:1]
                reply = {**reply, "q_l_chunk": left.tolist(), "q_r_chunk": right.tolist()}
        self._beat()
        return reply

    def reset_episode(self) -> Any:
        fn = getattr(self.client, "reset_episode", None)
        return fn() if fn else None

    # -- the robot's side ---------------------------------------------------------------------------------------
    def step(self, env: Any = None, *, images: dict[str, np.ndarray] | None = None,
             command: tuple[Any, Any] | None = None) -> None:
        """Call after every executed action row (20 Hz). Runs on_step; may raise StopEpisode / EndEpisode."""
        ctx = self.ctx
        if not self._in_episode:
            self._begin(ctx.task, env, images)
        if env is not None:
            ctx.env = env
        ctx.step += 1
        ctx.t = ctx.step * STEP_S
        state = getattr(ctx.env, "state", None)
        if callable(state):
            ctx.state = state()
        ctx.command = command
        ctx.frames_new = images is not None
        if images is not None:
            ctx.frames = dict(images)
        self.manager.call("on_step", ctx)
        if ctx.frames_new:
            self.live.frames(ctx.frames)
        self._beat()

    @contextmanager
    def episode(self, task: str | None = None, env: Any = None, images: dict[str, np.ndarray] | None = None,
                **info: Any) -> Iterator["Kit"]:
        """One attempt. A plugin's stop (camera-check) or done (success-check) ends the block quietly."""
        self._begin(task or self.ctx.task, env, images, **info)
        status = None
        try:
            yield self
        except EndEpisode as end:
            status = end.reason
        except StopEpisode as stop:
            self._stopped(stop)
        except KeyboardInterrupt:
            self._stopped(StopEpisode("stopped by user", "dreamkit"))
            self._end()
            raise
        except Exception as error:
            name = type(error).__name__
            self.ctx.result.update(success=False, reason="model_error" if name.startswith("Dream") else "error",
                                   at=round(self.ctx.t, 2), details={"message": f"{name}: {error}"})
            self.ctx.emit("error", f"attempt failed: {name}: {error}")
        self.ctx.result.setdefault("ended_by", status or "loop")
        self._end()

    # -- internals ----------------------------------------------------------------------------------------------
    def _begin(self, task: str, env: Any, images: dict[str, np.ndarray] | None, **info: Any) -> None:
        if self._in_episode:
            self._end()
        ctx = self.ctx
        if not self._next_is_retry and ctx.episode > 0:
            self._scene += 1
        ctx.attempt = ctx.attempt + 1 if self._next_is_retry else 1
        ctx.scene = self._scene
        ctx.episode += 1
        ctx.task = task or ctx.task
        self.info["task"] = self.info["task"] or ctx.task
        ctx.env = env if env is not None else ctx.env
        ctx.step, ctx.t = 0, 0.0
        ctx.result, ctx.actions, ctx.command, ctx.timing, ctx.reply = {}, None, None, None, None
        ctx._retry, ctx._telemetry = False, {}
        state = getattr(ctx.env, "state", None)
        ctx.state = state() if callable(state) else {}
        if images is None and hasattr(ctx.env, "images"):
            try:
                images = ctx.env.images()
            except Exception:
                images = None
        ctx.frames, ctx.frames_new = (dict(images) if images else {}), images is not None
        ctx.store["episode"] = dict(info)
        ctx.episode_dir.mkdir(parents=True, exist_ok=True)
        self._in_episode = True
        self._next_is_retry = False
        self._episode_started = time.time()
        self._requests = 0
        self.emitter.emit("episode_start", episode=ctx.episode, scene=ctx.scene, attempt=ctx.attempt, task=ctx.task,
                          **_plain(info))
        try:
            self.manager.call("on_episode_start", ctx)
        except (StopEpisode, EndEpisode) as early:
            if isinstance(early, StopEpisode):
                self._stopped(early)
        if ctx.frames:
            self.live.frames(ctx.frames, force=True)
        self._beat(force=True)

    def _stopped(self, stop: StopEpisode) -> None:
        ctx = self.ctx
        ctx.result.setdefault("success", False)
        ctx.result.update(reason="stopped" if stop.plugin == "dreamkit" else "stopped_by_plugin", at=round(ctx.t, 2),
                          details={"message": stop.reason, "plugin": stop.plugin})
        ctx._plugin = stop.plugin or "dreamkit"
        ctx.emit("error", f"attempt stopped: {stop.reason}")
        ctx._plugin = "dreamkit"

    def _end(self) -> None:
        if not self._in_episode:
            return
        self._in_episode = False
        ctx = self.ctx
        try:
            self.manager.call("on_episode_end", ctx)
        except (StopEpisode, EndEpisode):
            pass
        try:
            self.reset_episode()
        except Exception:
            pass
        r = ctx.result
        result = {
            "episode": ctx.episode, "scene": ctx.scene, "attempt": ctx.attempt, "task": ctx.task,
            "success": r.get("success"),
            "reason": None if r.get("success") else r.get("reason", "unknown"),
            "duration_s": round(ctx.t, 2), "requests": self._requests,
            "wall_s": round(time.time() - self._episode_started, 2), "retry_requested": ctx._retry,
            **{k: v for k, v in r.items() if k not in ("success", "reason")},
            **{k: v for k, v in ctx.store.get("episode", {}).items() if k not in r},
        }
        result["videos"] = sorted(f"ep{ctx.episode}/{p.name}" for p in ctx.episode_dir.glob("*.mp4"))
        ctx.episodes.append(_plain(result))
        self._next_is_retry = ctx._retry
        self.emitter.emit("episode_end", episode=ctx.episode, scene=ctx.scene, attempt=ctx.attempt,
                          duration_s=ctx.t, result=ctx.episodes[-1])
        self._save()
        self._beat(force=True)

    def _beat(self, force: bool = False) -> None:
        ctx = self.ctx
        self.live.beat({"run_id": self.run_id, "episode": ctx.episode, "scene": ctx.scene, "attempt": ctx.attempt,
                        "task": ctx.task, "t": round(ctx.t, 2), "in_episode": self._in_episode,
                        "status": self.info["status"]}, ctx._telemetry, ctx.t if self._in_episode else None, force)

    def _save(self) -> None:
        self.info.update(episodes=self.ctx.episodes, summary=_summary(self.ctx.episodes), updated=time.time())
        storage.write_json(self.run_dir / "run.json", self.info)

    def close(self, status: str = "finished") -> None:
        """Ends the run: on_run_end (report, recap), run.json finished. Safe to call twice."""
        if self._closed:
            return
        if self._in_episode:
            self._end()
        self._closed = True
        try:
            self.manager.call("on_run_end", self.ctx)
        except (StopEpisode, EndEpisode):
            pass
        self.info["status"] = status
        self.info["finished"] = time.time()
        self._save()
        self._beat(force=True)
        self.emitter.emit("run_end", run_id=self.run_id, status=status, **_summary(self.ctx.episodes))
        self.emitter.close()
        fn = getattr(self.client, "close", None)
        if fn:
            try:
                fn()
            except Exception:
                pass


def wrap(client: Any, config: str | Path | None = "dreamkit.yaml", **overrides: Any) -> Kit:
    """Puts dreamkit's plugins around a dream.DreamClient (or anything with policy / set_observation / get_action)."""
    return Kit(client, config, **overrides)
