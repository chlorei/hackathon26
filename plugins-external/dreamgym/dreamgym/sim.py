"""Finds Dream Labs' YAM MuJoCo example and imports its modules.

DreamGym does not copy the simulator: it reuses `yam_env.py` (the scene and physics), `dream_control.py`
(observation building, frame windows, chunk validation, opening a DreamClient) and `mock_pickups.py` (the IK
solver the scripted policy uses), so it stays in step with whatever Dream Labs ships.

Where the example is looked up, first match wins:
1. `DREAMGYM_SIM_DIR` (the folder that holds `yam_env.py`);
2. `sim_path` of $DREAMKIT_CONFIG or ./dreamkit.yaml (the same simulator dreamkit drives);
3. the `dream_examples` package that `pip install dreamlabs` installs (`dream_examples/yam_mujoco/`);
4. a `dreamlabs` checkout next to this repository (`../dreamlabs/dream_library/examples/yam_mujoco`).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterator

_MODULES: dict[str, ModuleType] = {}


def _candidates() -> Iterator[Path]:
    env = os.environ.get("DREAMGYM_SIM_DIR")
    if env:
        yield Path(env).expanduser()
    for config in filter(None, [os.environ.get("DREAMKIT_CONFIG"), "dreamkit.yaml"]):
        config = Path(config).expanduser()
        if not config.is_file():
            continue
        try:
            import yaml

            sim_path = (yaml.safe_load(config.read_text()) or {}).get("sim_path")
            if sim_path:
                yield (config.parent / Path(sim_path).expanduser()).resolve()
        except Exception:
            pass
    for package in ("dream_examples", "robots.examples"):
        try:
            module = importlib.import_module(package)
        except ImportError:
            continue
        for base in getattr(module, "__path__", []):
            yield Path(base) / "yam_mujoco"
    here = Path(__file__).resolve()
    for parent in here.parents[1:6]:  # a checkout of this repo: plugins-external/dreamgym/dreamgym/
        yield parent / "dreamlabs" / "dream_library" / "examples" / "yam_mujoco"


def sim_dir() -> Path:
    """The folder of the YAM MuJoCo example."""
    for path in _candidates():
        if (path / "yam_env.py").is_file():
            return path
    raise RuntimeError(
        "Dream YAM sim not found. Install the library (pip install dreamlabs) or clone "
        "github.com/dreamlabs-co/dreamlabs and set DREAMGYM_SIM_DIR=<checkout>/dream_library/examples/yam_mujoco"
    )


def load() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Imports the example's modules once.

    @return `(yam_env, dream_control, ik)`; the third has `solve()` (vendored from mock_pickups).
    """
    if not _MODULES:
        folder = str(sim_dir())
        if folder not in sys.path:
            sys.path.insert(0, folder)
        for name in ("yam_env", "dream_control"):
            _MODULES[name] = importlib.import_module(name)
        from . import ik  # mock_pickups' IK, vendored: the installed wheel does not ship mock_pickups.py

        _MODULES["mock_pickups"] = ik
    return _MODULES["yam_env"], _MODULES["dream_control"], _MODULES["mock_pickups"]


def make_env():
    """A fresh simulated YAM station (two arms, three cameras, fruit, bowl, plate)."""
    yam_env, _, _ = load()
    return yam_env.YamEnv()
