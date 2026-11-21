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
import queue
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import config as config_mod
from . import cost, history, reasons, storage
from .events import Emitter
from .live import LiveWriter
from .manager import PluginManager
from .plugin import Context, EndEpisode, StopEpisode
from .world import World

STEP_S = 0.05


def _summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    reasons_count: dict[str, int] = {}
    for e in episodes:
        if e.get("success") is False:
            reasons_count[e.get("reason") or "unknown"] = reasons_count.get(e.get("reason") or "unknown", 0) + 1
    return {
        "attempts": len(episodes),
        "successes": sum(1 for e in episodes if e.get("success")),
        "scenes": len({e.get("scene") for e in episodes}),
        "scene_successes": len({e.get("scene") for e in episodes if e.get("success")}),
        "reasons": reasons_count,
    }


def _world(env: Any) -> World:
    """The env's own world (a replay brings one), else the simulator's."""
    return getattr(env, "dreamkit_world", None) or World(env)


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


class Deferred:
    """ctx.defer's worker: one background thread, jobs in order, errors charged to the plugin that sent them."""

    def __init__(self, on_error):
        self.jobs: queue.Queue = queue.Queue()
        self.on_error = on_error
        self.thread = threading.Thread(target=self._run, name="dreamkit-defer", daemon=True)
        self.thread.start()

    def __call__(self, plugin: str, fn, args, kwargs) -> None:
        self.jobs.put((plugin, fn, args, kwargs))

    def _run(self) -> None:
        while True:
            plugin, fn, args, kwargs = self.jobs.get()
            try:
                if fn is not None:
                    fn(*args, **kwargs)
            except Exception as error:  # noqa: BLE001 - reported, never raised into the loop
                self.on_error(plugin, error)
            finally:
                self.jobs.task_done()

    def flush(self, timeout: float = 120.0, heartbeat=None, every_s: float = 1.0) -> bool:
        """Waits for the queued jobs. `heartbeat` runs every `every_s` meanwhile, so the panel's Live page does not
        take a loop that waits for a plugin's model call (§5) for a finished one."""
        done = threading.Event()
        self.jobs.put(("dreamkit", done.set, (), {}))
        deadline = time.monotonic() + timeout
        while not done.wait(min(every_s, max(0.0, deadline - time.monotonic()))):
            if time.monotonic() >= deadline:
                return False
            if heartbeat is not None:
                heartbeat()
        return True


class Kit:
    """A DreamClient with dreamkit's plugins around it. Everything not overridden passes straight through."""

    def __init__(self, client: Any, config: str | Path | None = "dreamkit.yaml", *, console: bool = True,
                 run_id: str | None = None, record: str | Path | None = None,
                 plugin_dirs: list[Path] | None = None, **overrides: Any):
        path = Path(config) if config is not None else None
        if path is not None and not path.is_file():
            path = None  # no config file: built-in defaults, no plugins
        cfg, base = config_mod.load(path, **overrides)
        if cfg.light:
            cfg = cfg.model_copy(update={"render_every": 4, "stream_fps": min(cfg.stream_fps, 5)})
        self.config = cfg
        self.client = client
        self.manager = PluginManager(cfg.plugins, extra_dirs=[base / "plugins", *(plugin_dirs or [])])
        root = Path(cfg.runs_dir).expanduser()
        root = root if root.is_absolute() else base / root
        self.run_id = run_id or storage.new_run_id()
        self.run_dir = storage.run_dir(root, self.run_id)
        self.emitter = Emitter(json_events=False, console=console)
        self.emitter.open_file(self.run_dir / "events.jsonl")
        self.live = LiveWriter(self.run_dir, fps=cfg.stream_fps)
        self.ctx = Context(env=None, task="", model="", run_id=self.run_id, run_dir=self.run_dir, config=cfg,
                           emitter=self.emitter)
        self.ctx._reset_policy = self._reset_policy
        self.deferred = Deferred(self._deferred_error)
        self.ctx._defer = self.deferred
        self.recorder = None
        if record is not None:
            from .testing import FixtureRecorder

            self.recorder = FixtureRecorder(Path(record))
        self._in_episode = False
        self._closed = False
        self._scene = 0
        self._next_is_retry = False
        self._episode_started = 0.0
        self._requests = 0
        self._run_calls = 0  # real model calls in this run: what it costs (dreamkit.cost)
        self._frozen: dict[str, np.ndarray] = {}
        self._watch_checked, self._watch_seen = 0.0, False
        self.info: dict[str, Any] = {
            "mode": None,
            "id": self.run_id, "task": "", "model": "", "mock": bool(getattr(client, "mock", False)),
            "started": time.time(), "plugins": self.manager.names, "control": self.manager.control,
            "refused": self.manager.refused, "status": "running", "light": cfg.light, "standard": "1",
            "cost_usd": 0.0,
        }
        self.ctx.mock = self.info["mock"]
        self._save()
        self.emitter.emit("run_start", run_id=self.run_id, plugins=self.manager.names, mock=self.info["mock"])
        for name in self.manager.missing:
            self.ctx.event("dreamkit", "error", f"plugin {name} is not installed (dreamkit install {name})")
        for name, why in self.manager.refused.items():
            self.ctx.event("dreamkit", "warn", f"plugin {name} not loaded: {why}")
        for line in self.manager.failed + self.manager.problems:
            self.ctx.event("dreamkit", "error", f"plugin failed to load: {line}")
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

    def _price(self) -> float:
        """Dollars per robot-hour of this run's model (dreamkit.cost); 0 for a mock."""
        return cost.price_per_hour(self.ctx.model, self.info["mock"], self.config.prices)

    @property
    def retry_requested(self) -> bool:
        """True when the control plugin asked to run the last attempt's scene again."""
        return self._next_is_retry

    @property
    def episodes(self) -> list[dict[str, Any]]:
        return self.ctx.episodes

    @property
    def wants_frames(self) -> bool:
        """Render cameras this step? Only if a plugin needs frames, or someone watches the Live page."""
        return self.manager.needs_frames or self._watched() or self.recorder is not None

    def _watched(self) -> bool:
        """Someone has the cameras open in Live (the panel touches live/.watching); looked up twice a second."""
        now = time.monotonic()
        if now - self._watch_checked > 0.5:
            self._watch_checked = now
            watcher = self.run_dir / "live" / ".watching"
            try:
                self._watch_seen = time.time() - watcher.stat().st_mtime < 3.0
            except OSError:
                self._watch_seen = False
        return self._watch_seen

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
        """before_model -> the model (or the control plugin's rows) -> after_model."""
        if not self._in_episode:
            self._begin(self.ctx.task, self.ctx.env, None)
        ctx = self.ctx
        self.manager.call("before_model", ctx)
        if ctx._override is not None:  # the control plugin answers instead of the model
            left, right, by = ctx._override
            ctx._override = None
            reply = {"q_l_chunk": left.tolist(), "q_r_chunk": right.tolist()}
            ctx.timing = None
            ctx.store["overridden_by"] = by
        else:
            reply = self.client.get_action(**kwargs)
            self._requests += 1
            self._run_calls += 1
            ctx._telemetry["spent_usd"] = round(cost.usd(self._run_calls, self._price()), 4)  # Live: money as it goes
            ctx.timing = getattr(self.client, "last_timing", None)
            ctx.store.pop("overridden_by", None)
            if self.recorder is not None:
                self.recorder.model_call(ctx, reply)
        ctx.reply = reply
        yam = isinstance(reply, dict) and "q_l_chunk" in reply and "q_r_chunk" in reply
        ctx.actions = (np.array(reply["q_l_chunk"], dtype=float), np.array(reply["q_r_chunk"], dtype=float)) if yam else None
        before = None if ctx.actions is None else (ctx.actions[0].copy(), ctx.actions[1].copy())
        self.manager.call("after_model", ctx)
        if yam and ctx.actions is not None and before is not None:
            left, right = ctx.actions
            if left.shape != before[0].shape or not np.array_equal(left, before[0]) or not np.array_equal(right, before[1]):
                if len(left) == 0:  # a plugin dropped every row: hold where the arms are
                    st = ctx.state or {}
                    left = np.asarray(st["left"])[None, :] if "left" in st else before[0][:1]
                    right = np.asarray(st["right"])[None, :] if "right" in st else before[1][:1]
                reply = {**reply, "q_l_chunk": np.asarray(left).tolist(), "q_r_chunk": np.asarray(right).tolist()}
        self._beat()
        return reply

    def reset_episode(self) -> Any:
        fn = getattr(self.client, "reset_episode", None)
        return fn() if fn else None

    def _reset_policy(self) -> None:
        """A new model session mid-attempt (recovery): the scene and the attempt go on."""
        client = self.client
        if getattr(client, "mock", False):
            client.reset_episode(recovery=True)
        elif hasattr(client, "reset_episode"):
            client.reset_episode()

    # -- the robot's side ---------------------------------------------------------------------------------------
    def step(self, env: Any = None, *, state: dict[str, Any] | None = None,
             images: dict[str, np.ndarray] | None = None, command: tuple[Any, Any] | None = None) -> None:
        """Call after every executed action row (20 Hz). Runs on_step; may end the attempt (inside episode()).

        Simulator: step(env, images=..., command=...) — the plugins get ctx.robot and ctx.world.
        Real robot: step(state={"left": 7 values, "right": 7 values}, images=..., command=(left, right)) — no world;
        plugins that need it go quiet ("sim only")."""
        ctx = self.ctx
        if not self._in_episode:
            self._begin(ctx.task, env, images, state=state)
        if env is not None and env is not ctx.env:
            ctx.env = env
            ctx.world = _world(env)
            self._announce_world()
        ctx.step += 1
        ctx.t = ctx.step * STEP_S
        if state is None:
            read = getattr(ctx.env, "state", None)
            state = read() if callable(read) else None
        if state is not None:
            ctx.state = {k: np.asarray(v, dtype=float) for k, v in state.items()}
        ctx.command = command
        ctx.robot.update(ctx.state, command)
        ctx.robot.last_timing = ctx.timing
        ctx.frames_new = images is not None
        if images is not None:
            ctx.frames = self._faults(dict(images))
        if self.recorder is not None:
            self.recorder.step(ctx)
        self.manager.call("on_step", ctx)
        if ctx.frames_new and self._watched():  # JPEGs for Live only while its cameras are open
            self.live.frames(ctx.frames)
        self._beat()
        if ctx.result.get("success") is True:  # success-check said so: the attempt is done
            raise EndEpisode("success", "success-check")

    def label(self, success: bool, episode: int | None = None) -> None:
        """The outcome, from your loop (a sensor, an operator): for a robot without a world, success is not measured.
        During an attempt it sets this attempt's outcome; afterwards, the given (or the last) attempt's."""
        if self._in_episode and episode in (None, self.ctx.episode):
            self.ctx.result["success"] = bool(success)
            self.ctx.result["labeled_by"] = "loop"
            return
        ep = next((e for e in reversed(self.ctx.episodes) if episode in (None, e["episode"])), None)
        if ep is not None:
            ep["success"], ep["labeled_by"] = bool(success), "loop"
            if success:
                ep["reason"] = None
            self._save()

    def _announce_world(self) -> None:
        off = self.manager.set_world(self.ctx.world.available)
        mode = "simulator: robot + world" if self.ctx.world.available else "robot only (no world)"
        if mode != self.info.get("mode"):
            self.info["mode"] = mode
            self.ctx.event("dreamkit", "info", f"mode: {mode}" + (f"; sim only, off here: {', '.join(off)}" if off else ""))

    def _faults(self, frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        f = self.config.faults
        if (f.camera_black or f.camera_frozen) and self.ctx.t >= f.at_s and (f.episode is None or f.episode == self.ctx.episode):
            if f.camera_black in frames:
                frames[f.camera_black] = (frames[f.camera_black] * 0.03).astype(np.uint8)
            if f.camera_frozen in frames:
                frozen = self._frozen.setdefault(f.camera_frozen, frames[f.camera_frozen].copy())
                frames[f.camera_frozen] = frozen
        return frames

    @contextmanager
    def episode(self, task: str | None = None, env: Any = None, images: dict[str, np.ndarray] | None = None,
                state: dict[str, Any] | None = None, **info: Any) -> Iterator["Kit"]:
        """One attempt. A stop by the control plugin, or success, ends the block quietly.
        Give env= in the simulator, or state= (and images=) on a real robot."""
        self._begin(task or self.ctx.task, env, images, state=state, **info)
        try:
            yield self
        except EndEpisode as end:
            self.ctx.result.setdefault("ended_by", end.reason)
        except StopEpisode as stop:
            self._stopped(stop)
        except KeyboardInterrupt:
            self._stopped(StopEpisode("stopped", "dreamkit", "stopped by the user"))
            self._end()
            raise
        except Exception as error:
            name = type(error).__name__
            self.ctx.result.update(success=False, reason="model_error" if name.startswith("Dream") else "error",
                                   at_s=round(self.ctx.t, 2), details={"message": f"{name}: {error}"})
            self.ctx.event("dreamkit", "error", f"attempt failed: {name}: {error}")
        self.ctx.result.setdefault("ended_by", "loop")
        self._end()

    # -- internals ----------------------------------------------------------------------------------------------
    def _begin(self, task: str, env: Any, images: dict[str, np.ndarray] | None, state: dict[str, Any] | None = None,
               **info: Any) -> None:
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
        if env is not None:
            ctx.env = env
        ctx.world = _world(ctx.env)
        ctx.step, ctx.t = 0, 0.0
        ctx.result, ctx.actions, ctx.command, ctx.timing, ctx.reply = {}, None, None, None, None
        ctx._retry, ctx._telemetry, ctx._override, ctx._ui = False, {}, None, {}
        ctx._telemetry["spent_usd"] = round(cost.usd(self._run_calls, self._price()), 4)
        self._frozen = {}
        if state is None:
            read = getattr(ctx.env, "state", None)
            state = read() if callable(read) else None
        ctx.state = {k: np.asarray(v, dtype=float) for k, v in state.items()} if state else {}
        ctx.robot.reset(ctx.state)
        self._announce_world()
        if images is None and hasattr(ctx.env, "images") and self.wants_frames:
            try:
                images = ctx.env.images()
            except Exception:
                images = None
        ctx.frames, ctx.frames_new = (self._faults(dict(images)) if images else {}), images is not None
        ctx.store["episode"] = dict(info)
        ctx.store.pop("verdict", None)  # live verdicts and recovery segments belong to one attempt
        ctx.store["verdicts"] = []
        ctx.store["segment"] = 0
        self._in_episode = True
        self._next_is_retry = False
        self._episode_started = time.time()
        self._requests = 0
        self.emitter.emit("episode_start", episode=ctx.episode, scene=ctx.scene, attempt=ctx.attempt, task=ctx.task,
                          **_plain(info))
        if self.recorder is not None:
            self.recorder.begin(ctx, info)
        try:
            self.manager.call("on_episode_start", ctx)
        except (StopEpisode, EndEpisode) as early:
            if isinstance(early, StopEpisode):
                self._stopped(early)
        if ctx.frames and self._watched():
            self.live.frames(ctx.frames, force=True)
        self._beat(force=True)

    def _stopped(self, stop: StopEpisode) -> None:
        ctx = self.ctx
        ctx.result.setdefault("success", False)
        reason = stop.reason if reasons.is_reason(stop.reason) else "error"
        ctx.result.update(reason=reason, at_s=round(ctx.t, 2), stopped_by=stop.plugin,
                          details={"message": stop.message, "plugin": stop.plugin})
        ctx.event(stop.plugin or "dreamkit", "error", f"attempt stopped: {stop.message}", reason=reason)

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
        if r.get("success") is not None:
            r["success"] = bool(r["success"])  # a numpy bool from a plugin would fail every `is False`
        if r.get("success") is None and r.get("reason") not in (None, "timeout"):
            r["success"] = False  # nobody judged success, but a failure was found: that settles it
        ctx.result["recap"] = history.episode_recap(ctx, self.run_dir, self.deferred)  # after the recorder closed
        self.deferred.flush(heartbeat=lambda: self._beat(force=True))  # videos, traces, datasets, the recap clip
        # and plugins' model calls are done before the attempt is saved; the heartbeat keeps Live connected meanwhile
        r = ctx.result
        if r.get("success") is not True and not r.get("reason"):  # no failure-monitor: the first live verdict, if any
            first = (ctx.store.get("verdicts") or [None])[0]
            r["success"] = False if first or r.get("success") is False else r.get("success")
            if first:
                r.update(reason=first["reason"], at_s=first["at_s"], message=first["message"])
        result = {
            "episode": ctx.episode, "scene": ctx.scene, "attempt": ctx.attempt, "task": ctx.task,
            "success": r.get("success"),
            "reason": r.get("reason", "timeout") if r.get("success") is False else None,  # unknown outcome: no reason
            "duration_s": round(ctx.t, 2), "requests": self._requests,
            "cost_usd": round(cost.usd(self._requests, self._price()), 4),
            "wall_s": round(time.time() - self._episode_started, 2), "retry_requested": ctx._retry,
            **{k: v for k, v in r.items() if k not in ("success", "reason")},
            **{k: v for k, v in ctx.store.get("episode", {}).items() if k not in r},
            "ui": ctx._ui,
            "verdicts": ctx.store.get("verdicts", []),
        }
        result["videos"] = sorted(
            str(p.relative_to(self.run_dir)) for p in (self.run_dir / "recorder").glob(f"ep{ctx.episode:03d}_*.mp4"))
        ctx.episodes.append(_plain(result))
        self.info["cost_usd"] = round(cost.usd(self._run_calls, self._price()), 4)
        self._next_is_retry = ctx._retry
        if self.recorder is not None:
            self.recorder.end(ctx, ctx.episodes[-1])
        self.emitter.emit("episode_end", episode=ctx.episode, scene=ctx.scene, attempt=ctx.attempt,
                          duration_s=ctx.t, result=ctx.episodes[-1])
        self._save()
        self._beat(force=True)

    def _deferred_error(self, plugin: str, error: Exception) -> None:
        stats = self.manager.stats.get(plugin)
        if stats is not None:
            stats.errors += 1
            stats.last_error = f"ctx.defer job failed: {type(error).__name__}: {error}"
        tb = traceback.extract_tb(error.__traceback__)
        where = f"{Path(tb[-1].filename).name}:{tb[-1].lineno}" if tb else ""
        self.ctx.event(plugin, "error", f"background job failed: {type(error).__name__}: {error}", where=where)

    def _beat(self, force: bool = False) -> None:
        ctx = self.ctx
        self.live.beat({"run_id": self.run_id, "episode": ctx.episode, "scene": ctx.scene, "attempt": ctx.attempt,
                        "task": ctx.task, "t": round(ctx.t, 2), "in_episode": self._in_episode,
                        "status": self.info["status"], "badges": {k: v.get("badges", []) for k, v in ctx._ui.items()},
                        "metrics": {k: v.get("metrics", {}) for k, v in ctx._ui.items()}},
                       ctx._telemetry, ctx.t if self._in_episode else None, force)

    def _save(self) -> None:
        self.info.update(episodes=self.ctx.episodes, summary=_summary(self.ctx.episodes), updated=time.time(),
                         plugin_stats=self.manager.stats_dict(), step_budget_over=self.manager.step_ms_over)
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
        self.deferred.flush(heartbeat=lambda: self._beat(force=True))
        self.info["status"] = status
        self.info["finished"] = time.time()
        try:
            self.summary = history.write_run(self.run_dir, self.info, self.ctx.episodes)
            self.ctx.event("dreamkit", "success" if any(e.get("success") for e in self.ctx.episodes) else "info",
                           self.summary["text"][0].lower() + self.summary["text"][1:].rstrip("."))
        except Exception as error:  # the run is saved either way
            self.ctx.event("dreamkit", "error", f"history summary failed: {error}")
        self._save()
        self._beat(force=True)
        if self.recorder is not None:
            self.recorder.close(self.ctx)
        self.emitter.emit("run_end", run_id=self.run_id, status=status, run_dir=str(self.run_dir),
                          **_summary(self.ctx.episodes))
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
