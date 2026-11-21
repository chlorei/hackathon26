import os
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SIM = Path(os.environ.get("DREAMKIT_SIM", ROOT.parent / "dreamlabs/dream_library/examples/yam_mujoco"))


@pytest.fixture
def make_config(tmp_path):
    """Writes a dreamkit.yaml for a fast mock run and returns its path."""
    if not (SIM / "yam_env.py").is_file():
        pytest.skip(f"Dream Labs simulator not found at {SIM} (set DREAMKIT_SIM)")

    def make(**overrides):
        data = {
            "sim_path": str(SIM),
            "task": "put apple into the bowl",
            "episodes": 1,
            "seconds": 20,
            "runs_dir": str(tmp_path / "runs"),
            "render_every": 2,
            "mock": {"behaviors": ["success"], "latency_ms": 0},
            "plugins": ["success-check", "failure-monitor"],
        }
        data.update(overrides)
        path = tmp_path / "dreamkit.yaml"
        path.write_text(yaml.safe_dump(data))
        return path

    return make
