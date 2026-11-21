"""dreamkit.yaml, validated with pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_SIM_PATH = "../dreamlabs/dream_library/examples/yam_mujoco"


class Randomize(BaseModel):
    objects: list[str] = []
    xy_jitter: float = Field(0.0, ge=0, le=0.1)


class Mock(BaseModel):
    """The scripted stand-in for the model (`--mock`): no API calls, no credits."""

    ## Behaviour per attempt, cycled: success, dropped, not_reached, grasped_air, missed_target, stalled, collision.
    behaviors: list[str] = ["success"]
    latency_ms: float = 250
    jitter_ms: float = 80


class PluginSpec(BaseModel):
    name: str
    enabled: bool = True
    params: dict[str, Any] = {}


class Config(BaseModel):
    sim_path: str = DEFAULT_SIM_PATH
    model: str = "dreamlabs/dream-1-beta"
    task: str = "put apple into the bowl"
    episodes: int = Field(1, ge=1, le=1000)
    seconds: float = Field(60, gt=0, le=3600)
    actions_per_chunk: int = Field(24, ge=1, le=48)
    seed: int = 42
    randomize: Randomize = Randomize()
    plugins: list[PluginSpec] = []
    runs_dir: str = "runs"
    mock: Mock = Mock()
    realtime: bool = False  # sleep so a 50 ms step takes 50 ms of wall time (nicer to watch)
    render_every: int = Field(2, ge=1, le=4)  # render cameras every N steps (2 = 10 fps); the model gets 5 fps
    registry_url: str = "https://raw.githubusercontent.com/dreamkit-hack/dreamkit/main/registry.yaml"

    @field_validator("plugins", mode="before")
    @classmethod
    def _plugins(cls, value: Any) -> list[dict[str, Any]]:
        """Accepts `- name`, `- name: {params}` and `- {name: x, enabled: false, params: {...}}`."""
        out = []
        for item in value or []:
            if isinstance(item, str):
                out.append({"name": item})
            elif isinstance(item, dict) and "name" in item:
                out.append(item)
            elif isinstance(item, dict) and len(item) == 1:
                (name, params), = item.items()
                params = dict(params or {})
                enabled = params.pop("enabled", True)
                out.append({"name": name, "enabled": enabled, "params": params})
            else:
                raise ValueError(f"bad plugin entry: {item!r}")
        return out

    def sim_dir(self, base: Path | None = None) -> Path:
        path = Path(self.sim_path).expanduser()
        if not path.is_absolute() and base is not None:
            path = (base / path).resolve()
        return path


def load(path: str | Path | None, **overrides: Any) -> tuple[Config, Path]:
    """Reads the config; returns it with the directory relative paths resolve against."""
    data: dict[str, Any] = {}
    base = Path.cwd()
    if path is not None:
        p = Path(path).expanduser()
        data = yaml.safe_load(p.read_text()) or {}
        base = p.resolve().parent
    data.update({k: v for k, v in overrides.items() if v is not None})
    return Config.model_validate(data), base


def dump_plugins(specs: list[PluginSpec]) -> list[Any]:
    """The compact YAML form of a plugin list."""
    out: list[Any] = []
    for s in specs:
        if s.enabled and not s.params:
            out.append(s.name)
        else:
            entry = dict(s.params)
            if not s.enabled:
                entry["enabled"] = False
            out.append({s.name: entry})
    return out
