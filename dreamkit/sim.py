"""Imports Dream Labs' simulator from `sim_path` unchanged. Nothing here edits it: we only import and call."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

TABLE_TOP_Z = 0.76  # the oak table box: centre z 0.73, half-height 0.03
OBJECTS = ("apple", "orange", "banana", "water_bottle", "bowl", "plate")
FRUITS = ("apple", "orange", "banana")
TARGETS = ("bowl", "plate")
## The free joint of each movable object (the bottle's joint is named differently from its body).
FREE_JOINT = {"apple": "apple_free", "orange": "orange_free", "banana": "banana_free", "water_bottle": "bottle_free"}


def use(sim_path: Path) -> tuple[ModuleType, ModuleType]:
    """Puts the simulator folder on sys.path and imports its yam_env and dream_control.
    @return (yam_env, dream_control)
    """
    sim_path = Path(sim_path).expanduser().resolve()
    if not (sim_path / "yam_env.py").is_file():
        raise FileNotFoundError(
            f"Dream Labs simulator not found at {sim_path}. Set `sim_path` in dreamkit.yaml to "
            "<dreamlabs checkout>/dream_library/examples/yam_mujoco."
        )
    if str(sim_path) not in sys.path:
        sys.path.insert(0, str(sim_path))
    return importlib.import_module("yam_env"), importlib.import_module("dream_control")
