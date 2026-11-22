"""The plugin API (PLUGIN_STANDARD.md). A plugin subclasses Plugin, declares a Manifest and a Params model, and gets
its own PluginContext in every hook: what it may do is decided by its capability, not by good manners."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import numpy as np
from pydantic import BaseModel, ValidationError

from . import reasons as reasons_mod
from .standard import HOOKS, UNIT_SUFFIX, Manifest, NoParams, PluginStandardError, Problem, check_class
from .task import parse
from .world import Robot, World

if TYPE_CHECKING:  # pragma: no cover
    from .events import Emitter

LEVELS = ("info", "success", "warn", "error")


class StopEpisode(Exception):
    """Raised by ctx.stop() (control plugins only): the attempt ends as a failure with this reason code."""

    def __init__(self, reason: str, plugin: str | None = None, message: str = ""):
        super().__init__(message or reason)
        self.reason = reason
        self.plugin = plugin
        self.message = message or reasons_mod.label(reason)


class EndEpisode(Exception):
    """The attempt is over early, normally (the task is done)."""

    def __init__(self, reason: str, plugin: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.plugin = plugin


class Plugin:
    """Base class: subclasses are checked against the standard when they are defined."""

    name: ClassVar[str] = "base"
    manifest: ClassVar[Manifest] = Manifest()
    Params: ClassVar[type[BaseModel]] = NoParams

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None  # the module body running the `class` statement
        if caller is not None:
            cls._dk_origin = (caller.f_code.co_filename, caller.f_lineno)
        problems = check_class(cls)
        if problems:
            raise PluginStandardError(problems)

    def __init__(self, **params: Any):
        try:
            self.params = self.Params(**params)
        except ValidationError as error:
            raise PluginStandardError([Problem("§4", f"{self.name} settings", f"invalid settings: {error}",
                                               "Fix the values in dreamkit.yaml (see the plugin's Params).")]) from None

    @classmethod
    def description(cls) -> str:
        return (cls.__doc__ or "").strip().split("\n\n")[0].replace("\n", " ").strip()

    def on_run_start(self, ctx: PluginContext) -> None: ...
    def on_episode_start(self, ctx: PluginContext) -> None: ...
    def before_model(self, ctx: PluginContext) -> None: ...
    def after_model(self, ctx: PluginContext) -> None: ...
    def on_step(self, ctx: PluginContext) -> None: ...
    def on_episode_end(self, ctx: PluginContext) -> None: ...
    def on_run_end(self, ctx: PluginContext) -> None: ...


# --------------------------------------------------------------------------------------------------------------
# The core's own context: one per run, owned by the Kit. Plugins never see it directly.


@dataclass
class Context:
    env: Any
    task: str
    model: str
    run_id: str
    run_dir: Path | None = None
    config: Any = None
    datasets_dir: Path | None = None  # <data_dir>/datasets: where datasets go (the harvester's, DreamGym's)
    ## What the model's time has cost, kept by the core after every call (dreamkit.cost): plugins only read it.
    cost: dict[str, Any] = field(default_factory=lambda: {"price_per_hour": 0.0, "run_calls": 0, "attempt_calls": 0,
                                                           "run_usd": 0.0, "attempt_usd": 0.0, "dream_price_per_hour": 0.0})
    episode: int = 0
    scene: int = 0
    attempt: int = 1
    seed: int = 0
    step: int = 0
    t: float = 0.0
    frames: dict[str, np.ndarray] = field(default_factory=dict)
    frames_new: bool = False
    state: dict[str, np.ndarray] = field(default_factory=dict)
    reply: dict[str, Any] | None = None
    actions: tuple[np.ndarray, np.ndarray] | None = None
    command: tuple[np.ndarray, np.ndarray] | None = None
    timing: dict[str, Any] | None = None
    result: dict[str, Any] = field(default_factory=dict)
    store: dict[str, Any] = field(default_factory=dict)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    mock: bool = False  # a scripted client: no model, nothing billed
    emitter: Emitter | None = None
    world: World = field(default_factory=lambda: World(None))
    robot: Robot = field(default_factory=Robot)
    _retry: bool = False
    _stop_run: dict[str, Any] | None = None  # set by ctx.stop_run(): {"by", "reason"}; the loop starts no new attempt
    _telemetry: dict[str, float] = field(default_factory=dict)
    _override: tuple[Any, Any, str] | None = None
    _reset_policy: Any = None
    _defer: Any = None  # set by the Kit: fn(plugin, callable, args, kwargs)
    _ui: dict[str, dict[str, Any]] = field(default_factory=dict)  # per plugin, this attempt

    @property
    def target(self) -> dict[str, str | None]:
        obj, target = parse(self.task or "")
        return {"object": obj, "target": target}

    @property
    def episode_dir(self) -> Path | None:
        """runs/<run>/ep003/: the files of this attempt (camera videos, recap.mp4, trace.json, result.json)."""
        if self.run_dir is None or self.episode < 1:
            return None
        d = self.run_dir / f"ep{self.episode:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def event(self, plugin: str, level: str, message: str, **data: Any) -> None:
        if self.emitter is not None:
            t_s = data.pop("t_s", round(self.t, 2))  # a plugin may date its event itself
            self.emitter.event(plugin, level, message, episode=self.episode, t_s=t_s, **data)

    def verdict(self, plugin: str, reason: str, at_s: float, message: str, **details: Any) -> dict[str, Any]:
        """The live verdict channel: what went wrong, mid-attempt. control plugins (recovery) act on it."""
        v = {"reason": reason, "at_s": round(float(at_s), 2), "message": message, "by": plugin,
             "segment": self.store.get("segment", 0), **details}
        self.store.setdefault("verdicts", []).append(v)
        self.store["verdict"] = v
        return v


# --------------------------------------------------------------------------------------------------------------
# What a plugin gets: a view of the core context scoped by its manifest.

## ctx.result keys that belong to one plugin each (§7); everything else lives under the plugin's own name.
RESULT_OWNERS = {"success": "success-check", "reason": "failure-monitor", "at_s": "failure-monitor",
                 "details": "failure-monitor", "message": "failure-monitor", "candidates": "failure-monitor",
                 "timeline": "failure-monitor"}
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_FLOAT_SUFFIXES = tuple(UNIT_SUFFIX.values()) + ("_frac", "_pct", "_x", "_y", "_z")


def _readonly(a: Any) -> np.ndarray:
    arr = np.array(a, dtype=float, copy=True)
    arr.setflags(write=False)
    return arr


def _refuse(section: str, where: str, what: str, fix: str) -> PluginStandardError:
    return PluginStandardError([Problem(section, where, what, fix)])


class ScopedResult:
    """ctx.result for one plugin: read everything, write only the keys it owns (§7)."""

    def __init__(self, core: "Context", plugin: str):
        self._core, self._plugin = core, plugin

    @property
    def _data(self) -> dict[str, Any]:
        return self._core.result  # a new dict every attempt: always read it from the core

    def _check(self, key: str) -> None:
        owner = RESULT_OWNERS.get(key)
        if key == self._plugin or owner == self._plugin:
            return
        raise _refuse("§7", f"{self._plugin}: ctx.result[{key!r}]",
                      f"{self._plugin} may not write {key!r}" + (f" (only {owner} writes it)" if owner else ""),
                      f'Write your own data under ctx.result["{self._plugin}"].')

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __setitem__(self, key: str, value: Any) -> None:
        self._check(key)
        if key == "reason" and value is not None and not reasons_mod.is_reason(value):
            raise _refuse("§7", f"{self._plugin}: ctx.result['reason']", f"{value!r} is not in dreamkit/reasons.py",
                          "Use a code from dreamkit.reasons.REASONS or add one there.")
        self._data[key] = value

    def setdefault(self, key: str, value: Any) -> Any:
        if key not in self._data:
            self[key] = value
        return self._data[key]

    def update(self, other: dict[str, Any] | None = None, **kw: Any) -> None:
        for k, v in {**(other or {}), **kw}.items():
            self[k] = v

    def items(self):
        return dict(self._data).items()

    def copy(self) -> dict[str, Any]:
        return dict(self._data)


class UI:
    """§7: the only way a plugin shows something in the panel."""

    def __init__(self, core: Context, plugin: str):
        self._core, self._plugin = core, plugin

    def _blocks(self) -> dict[str, Any]:
        return self._core._ui.setdefault(self._plugin, {"metrics": {}, "marks": [], "badges": []})

    def metric(self, name: str, value: float, unit: str = "") -> None:
        self._blocks()["metrics"][name] = {"value": float(value), "unit": unit}

    def mark(self, t_s: float, label: str, level: str = "info") -> None:
        if level not in LEVELS:
            raise _refuse("§7", f"{self._plugin}: ctx.ui.mark", f"level {level!r}", f"Use one of {', '.join(LEVELS)}.")
        mark = {"t_s": round(float(t_s), 2), "label": label, "level": level}
        self._blocks()["marks"].append(mark)
        if self._core.emitter is not None:
            self._core.emitter.emit("ui", plugin=self._plugin, episode=self._core.episode, kind="mark", **mark)

    def series(self, name: str, t_s: float, value: float, unit: str = "") -> None:
        self._core._telemetry[name] = float(value)

    def badge(self, text: str, level: str = "info") -> None:
        badges = self._blocks()["badges"]
        head = text.split(" ")[0]
        badges[:] = [b for b in badges if b["text"].split(" ")[0] != head] + [{"text": text, "level": level}]


class Files:
    """§8: a plugin's own files (the data its page reads) live in run_dir/<plugin>/; files of an attempt go to
    ctx.episode_dir."""

    _NAME = re.compile(r"^[a-z0-9][a-z0-9_.\-]*$")

    def __init__(self, core: Context, plugin: str):
        self._core, self._plugin = core, plugin

    @property
    def dir(self) -> Path:
        if self._core.run_dir is None:
            raise _refuse("§8", self._plugin, "no run folder", "Files need a run (dreamkit.wrap()).")
        d = self._core.run_dir / self._plugin
        d.mkdir(parents=True, exist_ok=True)
        return d

    def path(self, name: str) -> Path:
        """`ep003_trace.json`, or a sub-path like `success/ep003/meta/info.json` — always inside run_dir/<plugin>/."""
        parts = Path(name).parts
        if not parts or Path(name).is_absolute() or any(p == ".." or not self._NAME.match(p) for p in parts):
            raise _refuse("§8", f"{self._plugin}: ctx.files.path({name!r})",
                          "file names are lowercase snake_case inside run_dir/<plugin>/", 'Use names like "ep003_trace.json".')
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def episode_name(self, suffix: str) -> str:
        return f"ep{self._core.episode:03d}_{suffix}"

    def relative(self, path: Path) -> str:
        """A file of this plugin as the run folder sees it (what the panel links to)."""
        return str(path.relative_to(self._core.run_dir))

    def open(self, name: str, mode: str = "w", **kw: Any):
        return self.path(name).open(mode, **kw)


class PluginContext:
    """What a hook receives. Capability decides what it can change (§3, §11)."""

    def __init__(self, core: Context, plugin: Plugin, hook_name: Callable[[], str]):
        self._core = core
        self._plugin = plugin
        self._name = plugin.name
        self._cap = plugin.manifest.capability
        self._hook = hook_name
        self.ui = UI(core, self._name)
        self.files = Files(core, self._name)
        self.result = ScopedResult(core, self._name)
        self._last_step_event = -1e9
        self.suppressed_events = 0
        self._actions_copy: tuple[np.ndarray, np.ndarray] | None = None
        self._actions_set = False

    # -- read-only views of the core ------------------------------------------------------------------------------
    env = property(lambda self: self._core.env)
    task = property(lambda self: self._core.task)
    model = property(lambda self: self._core.model)
    mock = property(lambda self: self._core.mock)
    run_id = property(lambda self: self._core.run_id)
    ## §8: this attempt's folder, runs/<run>/ep003/: the files of the attempt (videos, trace.json ...), shared by all
    episode_dir = property(lambda self: self._core.episode_dir)
    ## the money so far (the core counts every billed model call): price_per_hour, run_calls, attempt_calls, run_usd,
    ## attempt_usd, dream_price_per_hour (what Dream-1 would bill, e.g. to try a budget on a mock). Read-only.
    cost = property(lambda self: dict(self._core.cost))
    ## <data_dir>/datasets/: datasets live here, not inside a run (the harvester appends to one)
    datasets_dir = property(lambda self: self._core.datasets_dir)
    run_dir = property(lambda self: self._core.run_dir)
    config = property(lambda self: self._core.config)
    episode = property(lambda self: self._core.episode)
    scene = property(lambda self: self._core.scene)
    attempt = property(lambda self: self._core.attempt)
    seed = property(lambda self: self._core.seed)
    step = property(lambda self: self._core.step)
    t = property(lambda self: self._core.t)
    state = property(lambda self: self._core.state)
    command = property(lambda self: self._core.command)
    timing = property(lambda self: self._core.timing)
    reply = property(lambda self: self._core.reply)
    store = property(lambda self: self._core.store)  # shared between plugins, by agreement
    episodes = property(lambda self: list(self._core.episodes))
    world = property(lambda self: self._core.world)
    robot = property(lambda self: self._core.robot)
    target = property(lambda self: self._core.target)
    frames_new = property(lambda self: self._core.frames_new and self._plugin.manifest.needs_frames)

    @property
    def frames(self) -> dict[str, np.ndarray]:
        if not self._plugin.manifest.needs_frames:
            raise _refuse("§5", f"{self._name}: ctx.frames", "reads camera frames without needs_frames",
                          "Set needs_frames=True in the manifest.")
        return self._core.frames

    # -- actions: read-only for observe/control, a copy for modify_actions ----------------------------------------
    @property
    def actions(self) -> tuple[np.ndarray, np.ndarray] | None:
        a = self._core.actions
        if a is None:
            return None
        if self._actions_copy is None:
            if self._cap == "modify_actions":
                self._actions_copy = (np.array(a[0], dtype=float), np.array(a[1], dtype=float))
            else:
                self._actions_copy = (_readonly(a[0]), _readonly(a[1]))
        return self._actions_copy

    @actions.setter
    def actions(self, value: tuple[Any, Any]) -> None:
        if self._cap != "modify_actions":
            raise _refuse("§3", f"{self._name}: ctx.actions = ...", f"a {self._cap} plugin may not change actions",
                          'Declare capability="modify_actions" (it may only make them stricter).')
        self._actions_copy = (np.array(value[0], dtype=float), np.array(value[1], dtype=float))
        self._actions_set = True

    def _begin_hook(self) -> None:
        self._actions_copy = None
        self._actions_set = False

    # -- reporting -------------------------------------------------------------------------------------------------
    def emit(self, level: str, message: str, **data: Any) -> None:
        """§7: one event in the feed. Checked: level, message form, data keys, rate (1/s from on_step)."""
        where = f"{self._name}: ctx.emit"
        if level not in LEVELS:
            raise _refuse("§7", where, f"level {level!r}", f"Use one of {', '.join(LEVELS)}.")
        if not message or message[0].isupper() or message.endswith("."):
            raise _refuse("§7", where, f"message {message!r} is not a lowercase fragment without a trailing period",
                          "Write e.g. 'dropped apple at 9.3 s'.")
        for key, value in data.items():
            if not _KEY_RE.match(key):
                raise _refuse("§7", where, f"data key {key!r} is not snake_case",
                              "Use snake_case keys with unit suffixes: t_s, dist_cm.")
            if isinstance(value, float) and not key.endswith(_FLOAT_SUFFIXES):
                raise _refuse("§7", where, f"data key {key!r} holds a measurement without a unit suffix",
                              f"Rename it with its unit: {key}_s, {key}_m, {key}_ms …")
            if key == "reason" and value is not None and not reasons_mod.is_reason(value):
                raise _refuse("§7", where, f"reason {value!r} is not in dreamkit/reasons.py",
                              "Use a code from dreamkit.reasons.REASONS.")
        if self._hook() == "on_step":
            now = self._core.t
            if now - self._last_step_event < 1.0:
                self.suppressed_events += 1
                return
            self._last_step_event = now
        self._core.event(self._name, level, message, **data)

    def verdict(self, reason: str, at_s: float, message: str, **details: Any) -> None:
        """Report a failure as it happens (any capability). The control plugin may act on it; failure-monitor writes
        the attempt's final reason. Say what it rests on: source="robot" (state, gripper, command) or "world"."""
        if not reasons_mod.is_reason(reason):
            raise _refuse("§7", f"{self._name}: ctx.verdict", f"{reason!r} is not a reason code",
                          "Use a code from dreamkit.reasons.REASONS.")
        v = self._core.verdict(self._name, reason, at_s, message, **details)
        self._core.event(self._name, "warn", f"verdict: {message} at {v['at_s']:.1f} s", reason=reason,
                         at_s=v["at_s"], kind="verdict")

    def defer(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """§5: slow work (disk, video, datasets, reports) on the background worker, in order."""
        if self._core._defer is None:
            fn(*args, **kwargs)
        else:
            self._core._defer(self._name, fn, args, kwargs)

    # -- control only ------------------------------------------------------------------------------------------------
    def _control(self, what: str) -> None:
        if self._cap != "control":
            raise _refuse("§3", f"{self._name}: ctx.{what}()", f"a {self._cap} plugin may not {what.replace('_', ' ')}",
                          "Report it with ctx.verdict(); the control plugin (recovery) acts on it.")

    def stop(self, reason: str, message: str = "") -> None:
        self._control("stop")
        if not reasons_mod.is_reason(reason):
            raise _refuse("§7", f"{self._name}: ctx.stop", f"{reason!r} is not a reason code",
                          "Use a code from dreamkit.reasons.REASONS.")
        raise StopEpisode(reason, self._name, message)

    def stop_run(self, reason: str, message: str = "") -> None:
        """Control only: after this attempt the loop starts no new one (client.stop_requested). Use ctx.stop()
        as well to end the attempt now."""
        self._control("stop_run")
        if not reasons_mod.is_reason(reason):
            raise _refuse("§7", f"{self._name}: ctx.stop_run", f"{reason!r} is not a reason code",
                          "Use a code from dreamkit.reasons.REASONS.")
        if self._core._stop_run is None:
            self._core._stop_run = {"by": self._name, "reason": reason, "message": message or reasons_mod.label(reason)}
            self._core.event(self._name, "error", f"run stopping: {message or reasons_mod.label(reason)}", reason=reason)

    def request_retry(self) -> None:
        self._control("request_retry")
        self._core._retry = True

    def override(self, left_rows: Any, right_rows: Any) -> None:
        """before_model: answer this get_action() with these rows instead of calling the model."""
        self._control("override")
        self._core._override = (np.asarray(left_rows, dtype=float), np.asarray(right_rows, dtype=float), self._name)

    def reset_policy(self) -> None:
        """A new model session (client.reset_episode()) without ending the attempt."""
        self._control("reset_policy")
        if self._core._reset_policy is not None:
            self._core._reset_policy()


__all__ = ["Plugin", "Manifest", "Context", "PluginContext", "StopEpisode", "EndEpisode", "LEVELS", "HOOKS",
           "PluginStandardError"]
