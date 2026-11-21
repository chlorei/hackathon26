"""Records the bundled replay fixtures (dreamkit/fixtures/) from Dream Labs' simulator with the mock policy.

    python tools/make_fixtures.py            # all
    python tools/make_fixtures.py stalled    # one
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("sim_loop", ROOT / "examples" / "sim_loop.py")
sim_loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim_loop)

OBSERVE = ["success-check", "failure-monitor", "camera-check"]
## name -> (mock behaviour, seconds, extra config, extra args)
FIXTURES = {
    "success_apple_bowl": ("success", 20, {}, []),
    "dropped_mid_carry": ("dropped", 16, {}, []),
    "released_off_target": ("released_off_target", 16, {}, []),
    "released_near_target": ("released_near_target", 20, {}, []),
    "grasped_air": ("grasped_air", 12, {}, []),
    "not_reached": ("not_reached", 18, {}, []),
    "collision_table": ("collision", 10, {}, []),
    "fell_off_table": ("fell_off_table", 14, {}, []),
    "stalled": ("stalled", 12, {}, []),
    "slow_model": ("success", 20, {}, ["--fake-rtt-ms", "950"]),
    "camera_black": ("success", 8, {"faults": {"camera_black": "top", "at_s": 3}}, []),
    "camera_frozen": ("success", 10, {"faults": {"camera_frozen": "top", "at_s": 2},
                                       "plugins": ["success-check", "failure-monitor",
                                                   "camera-check"]}, []),
}


def main(names: list[str]) -> None:
    cfg = yaml.safe_load((ROOT / "dreamkit.example.yaml").read_text())
    sim = str((ROOT / cfg["sim_path"]).resolve())
    out = ROOT / "dreamkit" / "fixtures"
    for name in names or FIXTURES:
        behavior, seconds, extra, args = FIXTURES[name]
        with tempfile.TemporaryDirectory() as tmp:
            config = {"sim_path": sim, "runs_dir": tmp, "render_every": 4, "plugins": OBSERVE, **extra}
            path = Path(tmp) / "c.yaml"
            path.write_text(yaml.safe_dump(config))
            sim_loop.main(["--mock", "--fast", "--config", str(path), "--episodes", "1", "--seconds", str(seconds),
                           "--behaviors", behavior, "--jitter", "0.02", "--record", str(out / name), *args])
        print(f"fixture {name}: {sum(f.stat().st_size for f in (out / name).iterdir()) // 1024} KB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
