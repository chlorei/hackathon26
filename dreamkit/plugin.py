"""The plugin API: a Plugin subclass overrides any of the hooks and reads/writes the shared Context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .events import Emitter

HOOKS = (
    "on_run_start",
    "on_episode_start",
    "before_model",
    "after_model",
    "on_step",
    "on_episode_end",
    "on_run_end",
)

LEVELS = ("info", "warn", "error", "success")


class StopEpisode(Exception):
    """Raised by ctx.stop(): the current attempt ends as a failure with this reason."""

    def __init__(self, reason: str, plugin: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.plugin = plugin


class EndEpisode(Exception):
    """Raised by ctx.end_episode(): the attempt is over early, normally (e.g. the task is done)."""

    def __init__(self, reason: str, plugin: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.plugin = plugin


class Plugin:
    """Base class. Every hook is optional; `params` holds the plugin's config entry."""

    name: str = "base"
    description: str = ""
    ## Default parameters, shown and editable in the panel; config values override them.
    defaults: dict[str, Any] = {}

    def __init__(self, **params: Any):
        self.params = {**self.defaults, **params}

    def on_run_start(self, ctx: Context) -> None: ...
    def on_episode_start(self, ctx: Context) -> None: ...
    def before_model(self, ctx: Context) -> None: ...
    def after_model(self, ctx: Context) -> None: ...
    def on_step(self, ctx: Context) -> None: ...
    def on_episode_end(self, ctx: Context) -> None: ...
    def on_run_end(self, ctx: Context) -> None: ...


@dataclass
class Context:
    """Everything a hook sees. One Context lives for the whole run; per-attempt fields are reset by the runner."""

    env: Any
    task: str
    model: str
    run_id: str
    run_dir: Any = None  # pathlib.Path of runs/<id>
    config: Any = None
    episode: int = 0  # attempt counter across the run, 1-based once started
    scene: int = 0  # scene index (0-based); a retry keeps the scene
    attempt: int = 1  # attempt number on this scene
    seed: int = 0
    step: int = 0
    t: float = 0.0
    frames: dict[str, np.ndarray] = field(default_factory=dict)
    frames_new: bool = False  # True on steps where the cameras were rendered
    state: dict[str, np.ndarray] = field(default_factory=dict)
    actions: tuple[np.ndarray, np.ndarray] | None = None
    command: tuple[np.ndarray, np.ndarray] | None = None  # the row being executed this step (left, right)
    timing: dict[str, Any] | None = None
    result: dict[str, Any] = field(default_factory=dict)
    store: dict[str, Any] = field(default_factory=dict)
    episodes: list[dict[str, Any]] = field(default_factory=list)  # finished attempts' results
    emitter: Emitter | None = None
    _plugin: str = "dreamkit"
    _retry: bool = False

    @property
    def episode_dir(self):
        return self.run_dir / f"ep{self.episode}" if self.run_dir is not None else None

    def emit(self, level: str, message: str, **data: Any) -> None:
        """Adds an event to the feed, signed by the plugin whose hook is running."""
        if level not in LEVELS:
            level = "info"
        if self.emitter is not None:
            self.emitter.event(self._plugin, level, message, episode=self.episode, t=round(self.t, 2), **data)

    def stop(self, reason: str) -> None:
        """Stops the attempt as a failure."""
        raise StopEpisode(reason, self._plugin)

    def end_episode(self, reason: str = "done") -> None:
        """Ends the attempt early without marking it failed (success-check uses it)."""
        raise EndEpisode(reason, self._plugin)

    def request_retry(self) -> None:
        """Asks the runner to run this scene again after on_episode_end."""
        self._retry = True


PluginFactory = Callable[..., Plugin]
