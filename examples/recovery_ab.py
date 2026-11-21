"""Same scenes, same seeds, twice: without recovery and with it. Prints the two success rates.

    python examples/recovery_ab.py --mock --episodes 5        # scripted failures, free
    python examples/recovery_ab.py --episodes 10              # Dream-1 (uses credits: twice the episodes)

With --mock the "model" after a recovery is scripted too (it succeeds), so the mock numbers show that the mechanism
works, not how much it helps Dream-1. Only the real run answers that.
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

BASE = ["camera-check", "success-check", "failure-monitor", "recorder"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=50)
    parser.add_argument("--task", default="put apple into the bowl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true", help="scripted failures instead of Dream-1 (tests the mechanism only)")
    parser.add_argument("--behaviors", default="dropped,grasped_air,released_off_target,not_reached,success")
    parser.add_argument("--config", default="dreamkit.yaml", help="sim_path and runs_dir come from here")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args(argv)

    cfg, base = dk_config.load(args.config if Path(args.config).is_file() else None)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for label, plugins in (("without recovery", BASE),
                               ("with recovery", [*BASE, {"recovery": {"retry_attempts": 0}}])):
            path = Path(tmp) / f"{label.replace(' ', '_')}.yaml"
            runs = Path(cfg.runs_dir) if Path(cfg.runs_dir).is_absolute() else base / cfg.runs_dir
            path.write_text(yaml.safe_dump({"sim_path": str(cfg.sim_dir(base)), "runs_dir": str(runs),
                                            "plugins": plugins}))
            print(f"\n=== {label} ===", flush=True)
            sim_argv = ["--config", str(path), "--episodes", str(args.episodes), "--seconds", str(args.seconds),
                        "--task", args.task, "--seed", str(args.seed)]
            if args.mock:
                sim_argv += ["--mock", "--behaviors", args.behaviors]
            if args.fast:
                sim_argv.append("--fast")
            sim_loop.main(sim_argv)
            run = max(runs.iterdir(), key=lambda p: p.stat().st_mtime)
            eps = json.loads((run / "run.json").read_text())["episodes"]
            results[label] = (sum(1 for e in eps if e["success"]), len(eps), run.name,
                              sum((e.get("recovery") or {}).get("count", 0) for e in eps))
    print()
    for label, (ok, n, run, rec) in results.items():
        extra = f", {rec} recoveries" if rec else ""
        print(f"success {label}: {100 * ok / max(n, 1):.0f}% ({ok}/{n}{extra}) — run {run}")
    if args.mock:
        print("(mock: after a recovery the scripted arm succeeds by design; run without --mock for Dream-1's numbers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
