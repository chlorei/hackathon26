"""`dreamkit new <name> --kind observe|modify|control`: a plugin package that passes `dreamkit check`, `test` and
`bench` from the first minute, with the same tooling everywhere (PLUGIN_STANDARD.md §10.4, §11)."""

from __future__ import annotations

from pathlib import Path

PLUGIN = {
    "observe": '''"""{name}: tracks how far the hands are from the object and flags when they stay away too long."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from dreamkit import Manifest, Plugin
from dreamkit.plugin import PluginContext


class Params(BaseModel):
    far_m: float = Field(0.25, gt=0, le=2, description="A hand farther than this from the object is far away",
                         json_schema_extra={{"unit": "m"}})
    after_s: float = Field(8.0, gt=0, le=600, description="Warn when both hands stay far this long",
                           json_schema_extra={{"unit": "s"}})


class {cls}(Plugin):
    """Warns when both hands stay far from the object for too long."""

    name = "{name}"
    Params = Params
    manifest = Manifest(group="diagnosis", icon="Ruler", capability="observe",
                        hooks=["on_episode_start", "on_step", "on_episode_end"], step_rate_hz=5)

    def on_episode_start(self, ctx: PluginContext) -> None:
        self.far_since: float | None = None
        self.warned = False
        self.closest_m = float("inf")

    def on_step(self, ctx: PluginContext) -> None:
        obj = ctx.target["object"]
        if not ctx.world.available or obj is None:
            return
        dist = min(float(np.linalg.norm(ctx.world.tcp(s) - ctx.world.pos(obj))) for s in ("left", "right"))
        self.closest_m = min(self.closest_m, dist)
        ctx.ui.series("{name}_dist", ctx.t, dist, unit="m")
        if dist < self.params.far_m:
            self.far_since = None
            return
        self.far_since = ctx.t if self.far_since is None else self.far_since
        if not self.warned and ctx.t - self.far_since >= self.params.after_s:
            self.warned = True
            ctx.ui.mark(ctx.t, f"hands far from {{obj}}", "warn")
            ctx.emit("warn", f"hands far from {{obj}} for {{self.params.after_s:.0f}} s", dist_m=round(dist, 3))

    def on_episode_end(self, ctx: PluginContext) -> None:
        if np.isfinite(self.closest_m):
            ctx.ui.metric("closest approach", self.closest_m * 100, unit="cm")
            ctx.result["{name}"] = {{"closest_m": round(self.closest_m, 3), "warned": self.warned}}
''',
    "modify": '''"""{name}: slows every joint to a speed cap; it only ever makes the chunk stricter."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from dreamkit import Manifest, Plugin
from dreamkit.plugin import PluginContext


class Params(BaseModel):
    max_speed_rad_s: float = Field(2.0, gt=0, le=20, description="No joint moves faster than this",
                                   json_schema_extra={{"unit": "rad/s"}})


class {cls}(Plugin):
    """Clamps every joint's speed to a cap, slowing the chunk without changing where it goes."""

    name = "{name}"
    Params = Params
    manifest = Manifest(group="safety", icon="Gauge", capability="modify_actions", hooks=["after_model"])

    def after_model(self, ctx: PluginContext) -> None:
        if ctx.actions is None or not ctx.state:
            return
        cap = self.params.max_speed_rad_s * 0.05
        rows = []
        for i, side in enumerate(("left", "right")):
            out = ctx.actions[i].copy()
            prev = np.asarray(ctx.state[side][:6], dtype=float)
            for r in range(len(out)):
                out[r, :6] = prev + np.clip(out[r, :6] - prev, -cap, cap)
                prev = out[r, :6]
            rows.append(out)
        ctx.actions = (rows[0], rows[1])
''',
    "control": '''"""{name}: stops the attempt when the object is knocked off the table (the one control plugin)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from dreamkit import Manifest, Plugin
from dreamkit.plugin import PluginContext


class Params(BaseModel):
    below_m: float = Field(0.05, gt=0, le=1, description="This far below the table top, the object is off the table",
                           json_schema_extra={{"unit": "m"}})


class {cls}(Plugin):
    """Stops the attempt as soon as the object falls off the table."""

    name = "{name}"
    Params = Params
    manifest = Manifest(group="core", icon="OctagonX", capability="control", hooks=["on_step"], step_rate_hz=10,
                        conflicts=["recovery"])

    def on_step(self, ctx: PluginContext) -> None:
        obj = ctx.target["object"]
        if ctx.world.available and obj and ctx.world.pos(obj)[2] < ctx.world.table_z - self.params.below_m:
            ctx.emit("error", f"{{obj}} fell off the table at {{ctx.t:.1f}} s, stopping", t_s=round(ctx.t, 2))
            ctx.stop("fell_off_table", f"{{obj}} fell off the table")
''',
}

TESTS = {
    "observe": '''from dreamkit.testing import replay


def test_warns_when_the_hands_never_come_close():
    result = replay("not_reached", plugins=[{{"{name}": {{"after_s": 4.0, "far_m": 0.08}}}}])
    assert result.events.any(plugin="{name}", level="warn")
    assert result.budget_ms("{name}") < 1.0


def test_quiet_on_a_normal_pick():
    result = replay("success_apple_bowl", plugins=["{name}"])
    assert not result.events.of("{name}")
    assert not result.errors("{name}")
''',
    "modify": '''from dreamkit.testing import replay


def test_only_makes_actions_stricter():
    result = replay("success_apple_bowl", plugins=[{{"{name}": {{"max_speed_rad_s": 0.5}}}}])
    assert not result.errors("{name}")  # the manager would reject a change that adds speed
    assert result.budget_ms("{name}") < 1.0


def test_no_change_when_the_cap_is_generous():
    result = replay("dropped_mid_carry", plugins=[{{"{name}": {{"max_speed_rad_s": 20.0}}}}])
    assert not result.errors("{name}")
''',
    "control": '''from dreamkit.testing import replay


def test_stops_when_the_object_falls_off_the_table():
    result = replay("fell_off_table", plugins=["{name}"])
    assert result.result["reason"] == "fell_off_table"
    assert result.result["stopped_by"] == "{name}"


def test_keeps_going_on_a_normal_pick():
    result = replay("success_apple_bowl", plugins=["{name}"])
    assert result.result.get("stopped_by") is None
    assert result.budget_ms("{name}") < 1.0
''',
}

PYPROJECT = '''[project]
name = "dreamkit-{name}"
version = "0.1.0"
description = "{name}: a dreamkit plugin"
requires-python = ">=3.11"
dependencies = ["dreamkit>=0.3,<0.4"]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6", "mypy>=1.10", "pre-commit>=3"]

[project.entry-points."dreamkit.plugins"]
{name} = "{module}:{cls}"

[tool.dreamkit]
api = "1"
group = "{group}"
capability = "{capability}"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["{module}"]

[tool.ruff]
line-length = 119
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "N"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
files = ["{module}"]

[tool.pytest.ini_options]
testpaths = ["tests"]
'''

PRECOMMIT = '''# pre-commit install  →  every commit runs these
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: local
    hooks:
      - id: dreamkit-check
        name: dreamkit check (PLUGIN_STANDARD.md)
        entry: dreamkit check .
        language: system
        pass_filenames: false
'''

WORKFLOW = '''name: plugin
on: [push, pull_request]
jobs:
  standard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - name: install dreamkit and this plugin
        run: |
          pip install "dreamkit @ git+https://github.com/chlorei/hackathon26"
          pip install -e ".[dev]"
      - name: dreamkit check
        run: dreamkit check .
      - name: dreamkit test
        run: dreamkit test .
      - name: dreamkit bench (1 ms/step budget on the fixture library)
        run: dreamkit bench .
      - name: ruff
        run: ruff check . && ruff format --check .
      - name: mypy
        run: mypy
'''

README = '''# dreamkit-{name}

{description}

```bash
dreamkit install ./dreamkit-{name}      # runs check + tests + bench first
```

```yaml
plugins:
  - {name}
```

## Develop

```bash
pip install -e ".[dev]" && pre-commit install
dreamkit check .        # the standard (PLUGIN_STANDARD.md)
dreamkit test .         # replay tests, no simulator, no credits
dreamkit bench .        # ms/step on every bundled fixture
dreamkit publish .      # all three, then the registry.yaml entry
```

## Settings

<!-- params -->
<!-- /params -->
'''

GROUP = {"observe": ("diagnosis", "observe"), "modify": ("safety", "modify_actions"), "control": ("core", "control")}
DESCRIPTION = {
    "observe": "Warns when both hands stay far from the object for too long.",
    "modify": "Clamps every joint's speed to a cap, slowing the chunk without changing where it goes.",
    "control": "Stops the attempt as soon as the object falls off the table.",
}


def scaffold(name: str, kind: str, where: Path) -> Path:
    module = name.replace("-", "_")
    cls = "".join(part.capitalize() for part in name.split("-"))
    group, capability = GROUP[kind]
    root = where / f"dreamkit-{name}"
    (root / module).mkdir(parents=True)
    (root / "tests").mkdir()
    (root / ".github" / "workflows").mkdir(parents=True)
    fmt = {"name": name, "module": module, "cls": cls, "group": group, "capability": capability,
           "description": DESCRIPTION[kind]}
    (root / module / "__init__.py").write_text(f"from .plugin import {cls}\n\n__all__ = [\"{cls}\"]\n")
    (root / module / "plugin.py").write_text(PLUGIN[kind].format(**fmt))
    (root / "tests" / f"test_{module}.py").write_text(TESTS[kind].format(**fmt))
    (root / "pyproject.toml").write_text(PYPROJECT.format(**fmt))
    (root / ".pre-commit-config.yaml").write_text(PRECOMMIT)
    (root / ".github" / "workflows" / "plugin.yml").write_text(WORKFLOW)
    (root / "README.md").write_text(README.format(**fmt))
    (root / ".gitignore").write_text("__pycache__/\n*.egg-info/\n.venv/\n")
    return root
