"""dreamkit.yaml, validated with pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_SIM_PATH = "../dreamlabs/dream_library/examples/yam_mujoco"


class Faults(BaseModel):
    """Demo/test faults the core injects before any plugin sees the step (plugins never alter the world)."""

    camera_black: str | None = None  # a camera name that goes dark
    camera_frozen: str | None = None  # a camera name whose picture stops changing
    at_s: float = Field(5.0, ge=0)
    episode: int | None = None  # only in this attempt (1-based); None = every attempt


class PluginSpec(BaseModel):
    name: str
    enabled: bool = True
    params: dict[str, Any] = {}


class Config(BaseModel):
    """dreamkit.yaml: which plugins, their settings, where the simulator and the runs are. What to run (task,
    attempts, model, mock) belongs to the user's loop, not here (examples/sim_loop.py takes them as arguments)."""

    sim_path: str = DEFAULT_SIM_PATH
    runs_dir: str = "runs"
    plugins: list[PluginSpec] = []
    # advanced, rarely set by hand; examples/sim_loop.py passes faults and light through dreamkit.wrap(...)
    render_every: int = Field(2, ge=1, le=4)  # render cameras every N steps (2 = 10 fps); the model gets 5 fps
    stream_fps: float = Field(8, ge=1, le=20)  # live frames to the panel per second
    light: bool = False  # less CPU: cameras rendered at 5 fps, panel stream at 5 fps
    faults: Faults = Faults()
    registry_url: str = "https://raw.githubusercontent.com/chlorei/hackathon26/main/registry.yaml"

    @field_validator("plugins", mode="before")
    @classmethod
    def _plugins(cls, value: Any) -> list[dict[str, Any]]:
        """Accepts `- name`, `- name: {params}` and `- {name: x, enabled: false, params: {...}}`."""
        out = []
        for item in value or []:
            if isinstance(item, str):
                out.append({"name": item})
            elif isinstance(item, dict) and "name" in item:
                item = dict(item)
                item["params"] = {("on" if k is True else "off" if k is False else k): v
                                  for k, v in (item.get("params") or {}).items()}
                out.append(item)
            elif isinstance(item, dict) and len(item) == 1:
                (name, params), = item.items()
                # YAML 1.1 reads a bare `on:` key as True (retry's `on: [...]`): give it its name back.
                params = {("on" if k is True else "off" if k is False else k): v for k, v in (params or {}).items()}
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
