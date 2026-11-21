"""The same attempts judged twice: with the simulator's world, and as a real robot would report them (--no-world:
only state, gripper, commands, cameras). Prints the two verdicts side by side.

    python examples/world_vs_robot.py --mock          # scripted failures, free (the mock fails on purpose)
    python examples/world_vs_robot.py --episodes 5    # Dream-1 (uses credits, twice)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sim_loop", HERE / "sim_loop.py")
sim_loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim_loop)
sys.path.insert(0, str(HERE.parent))
from dreamkit import config as dk_config  # noqa: E402

PLUGINS = ["camera-check", "success-check", "failure-monitor"]  # no recovery: both modes see the same attempts
BEHAVIORS = "success,dropped,grasped_air,released_off_target,not_reached,stalled,collision,fell_off_table"


def verdict(e: dict) -> str:
    if e.get("success") is True:
        return "success"
    if e.get("success") is False:
        at = e.get("at_s")
        return f"{e.get('reason')}" + (f" @{at:.1f}s" if isinstance(at, (int, float)) else "")
    return "unknown (to label)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--behaviors", default=BEHAVIORS)
    parser.add_argument("--episodes", type=int, help="attempts (default: one per mock behaviour)")
    parser.add_argument("--seconds", type=float, default=16)
    parser.add_argument("--config", default="dreamkit.yaml")
    args = parser.parse_args(argv)
    episodes = args.episodes or len(args.behaviors.split(","))
    cfg, base = dk_config.load(args.config if Path(args.config).is_file() else None)
    rows: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for mode, extra in (("with world", []), ("robot only", ["--no-world"])):
            path = Path(tmp) / f"{mode.replace(' ', '_')}.yaml"
            path.write_text(yaml.safe_dump({"sim_path": str(cfg.sim_dir(base)), "runs_dir": tmp, "plugins": PLUGINS}))
            argv2 = ["--config", str(path), "--episodes", str(episodes), "--seconds", str(args.seconds), "--fast", *extra]
            if args.mock:
                argv2 += ["--mock", "--behaviors", args.behaviors]
            sim_loop.main(argv2)
            run = max((p for p in Path(tmp).iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
            rows[mode] = json.loads((run / "run.json").read_text())["episodes"]
    names = args.behaviors.split(",") if args.mock else [f"attempt {i + 1}" for i in range(episodes)]
    print(f"\n{'attempt':22} {'with world':28} robot only (state, gripper, commands)")
    for i, name in enumerate(names[:episodes]):
        w, r = rows["with world"][i], rows["robot only"][i]
        print(f"{name:22} {verdict(w):28} {verdict(r)}")
    if args.mock:
        print("(mock: the failures are scripted on purpose; this compares how each mode sees the same failure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
