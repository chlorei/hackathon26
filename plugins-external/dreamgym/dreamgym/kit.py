"""DreamGym drives the simulator through dreamkit.wrap (PLUGIN_STANDARD.md §13): every episode is a dreamkit attempt.

So the loop plugins of dreamkit.yaml run inside Arena, Compare, Record and Teleop exactly as in your own loop:
failure-monitor names failures, recovery fixes them from where the arms are, harvester and recorder save them. The
runs land in dreamkit's runs/ with source "gym" and show up in History with a Gym badge.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def config_path() -> str | None:
    """dreamkit.yaml: $DREAMKIT_CONFIG (the panel sets it for the runs it starts), else ./dreamkit.yaml."""
    env = os.environ.get("DREAMKIT_CONFIG")
    if env:
        return env
    return "dreamkit.yaml" if Path("dreamkit.yaml").is_file() else None


def paths() -> tuple[Path, Path]:
    """(runs, datasets) of the dreamkit data folder (dreamkit.yaml's data_dir): DreamGym keeps nothing of its own."""
    from dreamkit import config as config_mod

    path = config_path()
    cfg, base = config_mod.load(path if path and Path(path).is_file() else None)
    return cfg.runs_path(base), cfg.datasets_path(base)


def plugins_with(recovery: bool | None) -> list[Any] | None:
    """dreamkit.yaml's plugins, with recovery forced on or off (None: as configured)."""
    if recovery is None:
        return None
    from dreamkit import config as config_mod

    path = config_path()
    cfg, _ = config_mod.load(path if path and Path(path).is_file() else None)
    plugins: list[Any] = [p.model_dump() for p in cfg.plugins if p.name != "recovery"]
    if recovery:
        old = next((p for p in cfg.plugins if p.name == "recovery"), None)
        plugins.append({**(old.model_dump() if old else {"name": "recovery", "params": {}}), "enabled": True})
    return plugins


class PolicyClient:
    """A DreamGym policy seen as a DreamClient, so dreamkit.wrap stands around Dream-1 and the scripted expert alike.

    get_action() is the policy's next chunk; reset_episode() (what recovery asks for) starts the policy again from
    where the arms are now: a new model session for Dream-1, a new plan for the scripted expert."""

    mock = False

    def __init__(self, policy: Any):
        self.gym_policy = policy
        self.env: Any = None
        self.gctx: Any = None
        self.task: Any = None
        self.prompt = ""

    def bind(self, env: Any, gctx: Any, task: Any, prompt: str) -> None:
        self.env, self.gctx, self.task, self.prompt = env, gctx, task, prompt

    def policy(self, config: dict[str, Any]) -> None:  # the gym policy opens its own session
        return None

    def set_observation(self, frames: Any, observation: Any, **_: Any) -> None:  # the gym policy feeds itself
        return None

    def get_action(self, **_: Any) -> dict[str, Any]:
        chunk = self.gym_policy.next_chunk(self.env, self.gctx)
        return {"q_l_chunk": np.asarray(chunk.left).tolist(), "q_r_chunk": np.asarray(chunk.right).tolist()}

    @property
    def last_timing(self) -> Any:
        return getattr(getattr(self.gym_policy, "_client", None), "last_timing", None)

    def reset_episode(self, **_: Any) -> None:
        self.gym_policy.begin(self.env, self.task, self.prompt, self.gctx)

    def close(self) -> None:  # the arena closes the policy itself
        return None


def open_kit(client: Any, *, model: str, info: dict[str, Any], recovery: bool | None = None) -> Any:
    """A dreamkit run for DreamGym: source "gym", the variant in its info; quiet console (DreamGym prints its own)."""
    import dreamkit

    plugins = plugins_with(recovery)
    kit = dreamkit.wrap(client, config=config_path(), console=False, **({"plugins": plugins} if plugins is not None else {}))
    kit.ctx.model = model
    kit.info.update(model=model, source="gym", gym=info)
    kit._save()
    return kit
