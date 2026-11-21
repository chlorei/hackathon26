"""dreamkit as an overlay on a program it did not write: `dreamkit run -- <command>`.

The program (Dream Labs' own simulator, a station script, anything on `dream.DreamClient`) runs unchanged.
When it imports `dream`, its `DreamClient` gains dreamkit's plugins; when it imports Dream Labs' `yam_env`, every
`YamEnv.step()` becomes a dreamkit step with the simulator's world. Without `yam_env` (a real robot) the steps come
from the observations the program sends, robot data only.

    one policy() ... close()  = one attempt
    get_action()              = before_model -> model (or recovery's rows) -> after_model
    YamEnv.step(left, right)  = on_step with ctx.world  (simulator)
    set_observation(...)      = on_step from the observation's state (robot, when no YamEnv)

When a plugin ends the attempt (success, a camera failure, recovery giving up), the program's next get_action()
raises DreamkitStop, a RuntimeError: the program stops that task the way it stops on any error, and keeps running.
Nothing here edits the program's files; Dream Labs' modules are only wrapped in memory, in that process.
"""

from __future__ import annotations

import base64
import importlib.abc
import importlib.machinery
import io
import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

SIDES = ("left", "right")


class DreamkitStop(RuntimeError):
    """A plugin ended the attempt; the program stops this task (and stays up for the next one)."""


class _Session:
    """One dreamkit run per process, created on the first policy(); one attempt per policy() ... close()."""

    def __init__(self) -> None:
        self.kit: Any = None
        self.env: Any = None  # the last YamEnv the program built
        self.active = False  # inside an attempt
        self.pending = False  # policy() opened, the attempt starts with the first observation
        self.ended: str | None = None  # why dreamkit ended the attempt (raised at the next get_action)
        self.frames: dict[str, np.ndarray] | None = None
        self.lock = threading.RLock()
        self.mock: Any = None
        self.attempts = 0

    # -- the run ---------------------------------------------------------------------------------------------------
    def kit_for(self, raw: Any) -> Any:
        if self.kit is None:
            from .kit import Kit

            config = os.environ.get("DREAMKIT_CONFIG") or "dreamkit.yaml"
            self.kit = Kit(raw, config, console=False)
            where = os.environ.get("DREAMKIT_PANEL", "http://127.0.0.1:8800")
            print(f"  dreamkit: run {self.kit.run_id} · plugins {', '.join(self.kit.manager.names) or 'none'} · {where}",
                  flush=True)
        else:
            self.kit.client = raw
        return self.kit

    def begin(self, raw: Any, config: dict[str, Any]) -> None:
        with self.lock:
            kit = self.kit_for(raw)
            if self.active and not self.fresh():
                self.end(final=True)
            kit.policy(config)  # model name for the run; opens the real (or mock) session
            self.ended = None
            if not self.active:  # the attempt starts with the first observation: where the task (prompt) arrives
                self.pending = True

    def fresh(self) -> bool:
        """The attempt has not asked the model yet (a session reopened now is the same attempt)."""
        return self.kit is not None and self.kit._requests == 0

    def start(self, task: str = "") -> None:
        with self.lock:
            if self.active and self.fresh() and task and task != self.kit.ctx.task:
                self.kit.ctx.task = task  # reopened before any step, now with this task
                return
            if not self.pending or self.kit is None:
                return
            self.pending = False
            self.kit.ctx.task = task
            self.kit.info["task"] = self.kit.info["task"] or task
            self.kit._begin(task, self.env, None)
            self.active, self.attempts = True, self.attempts + 1

    def end(self, why: str = "loop", final: bool = False) -> None:
        """Ends the attempt. A close() before the first model call is not an end: the program is reopening its session (Dream
        Labs' sim does that for a model that reads hand poses), and the attempt goes on in the new one."""
        with self.lock:
            kit = self.kit
            if why == "loop" and not final and self.active and self.fresh():
                return
            self.pending = False
            if kit is None or not self.active:
                return
            self.active = False
            kit.ctx.result.setdefault("ended_by", why)
            kit._end()
            ep = kit.ctx.episodes[-1] if kit.ctx.episodes else {}
            ok = ep.get("success")
            mark = "✓ success" if ok else "✗ " + str(ep.get("reason") or "failed") if ok is False else "? outcome unknown"
            at = f" at {ep['at_s']:.1f} s" if ok is False and ep.get("at_s") is not None else ""
            print(f"  dreamkit: attempt {ep.get('episode', '?')} {mark}{at} · details in History", flush=True)

    # -- steps -------------------------------------------------------------------------------------------------------
    def step(self, **kw: Any) -> None:
        from .plugin import EndEpisode, StopEpisode

        with self.lock:
            if not self.active or self.kit is None:
                return
            kit = self.kit
            try:
                kit.step(**kw)
            except EndEpisode as end:
                self.ended = "success" if end.reason == "success" else end.reason
                self.end(end.reason)
            except StopEpisode as stop:
                kit._stopped(stop)
                self.ended = stop.message
                self.end(stop.plugin or "dreamkit")
            except Exception as error:  # dreamkit must never break the program it watches
                kit.ctx.event("dreamkit", "error", f"overlay step failed: {type(error).__name__}: {error}")


SESSION = _Session()


# -- the client ---------------------------------------------------------------------------------------------------------
def _state_from(observation: Any) -> dict[str, np.ndarray] | None:
    """Joint state from an observation in the model's keys (state.left_arm …) or the station's (state.<slot>…)."""
    if not isinstance(observation, dict):
        return None
    out = {}
    for side in SIDES:
        arm, grip = observation.get(f"state.{side}_arm"), observation.get(f"state.{side}_gripper")
        if arm is None:
            slot = next((k.split(".")[1] for k in observation if k.startswith("state.") and k.endswith(".joint_pos")
                         and side in k), None)
            if slot:
                arm, grip = observation.get(f"state.{slot}.joint_pos"), observation.get(f"state.{slot}.gripper_pos")
        if arm is None or grip is None:
            return None
        out[side] = np.r_[np.asarray(arm, dtype=float)[:6], np.asarray(grip, dtype=float).ravel()[:1]]
    return out


def _images_from(frames: Any) -> dict[str, np.ndarray] | None:
    """Camera images from set_observation's frames: arrays, or base64 JPEG strings (Dream Labs' simulator)."""
    if not isinstance(frames, dict):
        return None
    from PIL import Image

    out = {}
    for key, value in frames.items():
        name = key.rsplit(".", 1)[-1]
        if isinstance(value, np.ndarray):
            out[name] = value
        elif isinstance(value, (str, bytes)):
            try:
                raw = base64.b64decode(value) if isinstance(value, str) else value
                out[name] = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            except Exception:
                continue
    return out or None


def _wrap_client(cls: type) -> type:
    """A DreamClient whose policy / set_observation / get_action / close go through dreamkit."""

    class _Raw:
        """The client as the model sees it (no dreamkit), for the Kit to call."""

        def __init__(self, owner: Any):
            self._owner = owner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._owner, name)

        def policy(self, config: dict[str, Any]) -> Any:
            return cls.policy(self._owner, config)

        def set_observation(self, frames: Any, observation: Any, **kw: Any) -> Any:
            return cls.set_observation(self._owner, frames, observation, **kw)

        def get_action(self, **kw: Any) -> Any:
            return cls.get_action(self._owner, **kw)

        def reset_episode(self, **kw: Any) -> Any:
            return cls.reset_episode(self._owner)

        @property
        def last_timing(self) -> Any:
            return getattr(self._owner, "last_timing", None)

    class DreamClient(cls):  # type: ignore[misc, valid-type]
        __doc__ = cls.__doc__

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._dk_mock = os.environ.get("DREAMKIT_MOCK")
            if not self._dk_mock:
                super().__init__(*args, **kwargs)

        def _dk_raw(self) -> Any:
            if not self._dk_mock:
                return _Raw(self)
            if SESSION.mock is None:  # scripted failures, no model calls (dreamkit's mock, in Dream Labs' scene)
                from .mock import MockClient

                if SESSION.env is None:
                    raise RuntimeError("dreamkit --mock needs Dream Labs' simulator (YamEnv) in this program")
                SESSION.mock = MockClient(SESSION.env, behaviors=self._dk_mock.split(","))
            elif SESSION.attempts and not (SESSION.active and SESSION.fresh()):
                SESSION.mock.reset_episode()  # the next scripted behaviour (not for a session merely reopened)
            return SESSION.mock

        def policy(self, config: dict[str, Any]) -> Any:
            SESSION.begin(self._dk_raw(), config)

        def set_observation(self, frames: Any, observation: Any, **kw: Any) -> Any:
            kit = SESSION.kit
            if kit is None:
                return super().set_observation(frames, observation, **kw)
            prompt = observation.get("prompt") if isinstance(observation, dict) else None
            SESSION.start(str(prompt or ""))
            out = kit.set_observation(frames, observation, **kw)
            if SESSION.env is None and SESSION.active:  # no simulator here: the observation is the robot's step
                SESSION.step(state=_state_from(observation), images=_images_from(frames))
            return out

        def get_action(self, **kw: Any) -> Any:
            if SESSION.ended:
                why, SESSION.ended = SESSION.ended, None
                raise DreamkitStop(f"dreamkit finished the task: {why}")
            SESSION.start()  # a program that asks before observing
            if SESSION.kit is None or not SESSION.active:
                return super().get_action(**kw)
            from .plugin import EndEpisode, StopEpisode

            kit = SESSION.kit
            try:
                return kit.get_action(**kw)
            except EndEpisode as end:
                SESSION.end(end.reason)
                raise DreamkitStop(f"dreamkit ended the attempt: {end.reason}") from None
            except StopEpisode as stop:  # recovery gave up, or a camera failed, before asking the model
                kit._stopped(stop)
                SESSION.end(stop.plugin or "dreamkit")
                raise DreamkitStop(f"dreamkit ended the attempt: {stop.message}") from None

        def close(self) -> Any:
            SESSION.end()
            if not self._dk_mock:
                return super().close()
            return None

        @property
        def last_timing(self) -> Any:
            if self._dk_mock:
                return getattr(SESSION.mock, "last_timing", None)
            return super().last_timing

    DreamClient.__name__ = DreamClient.__qualname__ = cls.__name__
    DreamClient.__module__ = cls.__module__
    return DreamClient


def _patch_dream(module: Any) -> None:
    if getattr(module, "_dreamkit", False):
        return
    module.DreamClient = _wrap_client(module.DreamClient)
    module._dreamkit = True


def _patch_yam_env(module: Any) -> None:
    cls = getattr(module, "YamEnv", None)
    if cls is None or getattr(cls, "_dreamkit", False):
        return
    init, step, images = cls.__init__, cls.step, cls.images

    def __init__(self: Any, *a: Any, **k: Any) -> None:
        init(self, *a, **k)
        SESSION.env = self

    def step_(self: Any, left: Any, right: Any) -> Any:
        out = step(self, left, right)
        if SESSION.active and self is SESSION.env:
            frames, SESSION.frames = SESSION.frames, None
            SESSION.step(env=self, images=frames, command=(np.asarray(left, float), np.asarray(right, float)))
        return out

    def images_(self: Any) -> Any:
        out = images(self)
        if self is SESSION.env:
            SESSION.frames = out  # handed to the plugins with the next step
        return out

    cls.__init__, cls.step, cls.images = __init__, step_, images_
    cls._dreamkit = True


_PATCHES = {"dream": _patch_dream, "yam_env": _patch_yam_env}


class _Finder(importlib.abc.MetaPathFinder):
    """Patches `dream` and `yam_env` right after they are imported, wherever they come from."""

    def find_spec(self, name: str, path: Any, target: Any = None) -> Any:
        if name not in _PATCHES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        loader, patch = spec.loader, _PATCHES[name]

        class _Loader(importlib.abc.Loader):
            def create_module(self, spec: Any) -> Any:
                return loader.create_module(spec)

            def exec_module(self, module: Any) -> None:
                loader.exec_module(module)
                try:
                    patch(module)
                except Exception as error:  # never break the import itself
                    print(f"  dreamkit: could not attach to {name}: {error}", file=sys.stderr)

        spec.loader = _Loader()
        return spec


def install() -> None:
    """Called at interpreter start (sitecustomize) when DREAMKIT_ATTACH=1."""
    for name, patch in _PATCHES.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    import atexit

    atexit.register(lambda: SESSION.kit and (SESSION.end(final=True), SESSION.kit.close()))


def boot_dir() -> Path:
    return Path(__file__).resolve().parent / "_boot"
