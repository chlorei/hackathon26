"""A/B test and prompt tuner: the same scenes (seeds) under several variants, ranked with proof (from DreamGym).

Each variant is one ordinary dreamkit run of examples/sim_loop.py (so the panel shows it live and in History) with
the `reliability` plugin on; every variant after the first is compared with the first one scene by scene.

    python examples/ab_test.py --mock --fast --prompts "put apple into the bowl" "place the apple into the bowl"
    python examples/ab_test.py --tune --episodes 10                       # prompt tuner: paraphrases of --task
    python examples/ab_test.py --models dreamlabs/dream-1-beta,dream-1-ft:abcd1234-apple --episodes 20   # base vs fine-tune

Variants = every model x every prompt; the first is the baseline. Without --mock every episode calls Dream-1
(credits): the script prints the worst-case cost and asks before it starts (--yes skips the question).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plugins-external" / "reliability"))  # the plugin, installed or not
from dreamkit import config as dk_config  # noqa: E402
from dreamkit.task import parse  # noqa: E402
from reliability.plugin import compare, summarize, wilson  # noqa: E402

BASE = ["camera-check", "success-check", "failure-monitor", "recorder"]
CALLS_PER_HOUR = 2250  # Dream-1 bills a robot-hour as 2250 calls of 1.6 s of actions
NAMES = {"apple": ["apple", "red apple"], "banana": ["banana", "yellow banana"], "orange": ["orange", "orange fruit"],
         "bowl": ["bowl", "ceramic bowl"], "plate": ["plate", "blue plate"]}


def load_sim_loop():
    spec = importlib.util.spec_from_file_location("sim_loop", HERE / "sim_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paraphrases(task: str, limit: int) -> list[str]:
    """The task first (the baseline), then rewordings that keep the object and the target success-check reads."""
    obj, target = parse(task)
    if not obj or not target:
        return [task]
    prep = "into" if target == "bowl" else "onto"
    o, t = NAMES.get(obj, [obj]), NAMES.get(target, [target])
    out = []
    for c in (task, f"pick up the {o[0]} and put it {prep} the {t[0]}", f"place the {o[-1]} {prep} the {t[-1]}",
              f"move the {o[0]} {prep} the {t[0]}", f"grasp the {o[-1]}, lift it, and release it {prep} the {t[0]}",
              f"carefully pick up the {o[0]} and gently place it {prep} the {t[0]}", f"{obj} {prep} {target}"):
        if c not in out:
            out.append(c)
    return out[:limit]


def price(model: str, mock: bool) -> float:
    return 0.0 if mock else (2.0 if "flash" in model else 3.0)


def main(argv: list[str] | None = None, sim_loop=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="put apple into the bowl", help="the baseline prompt")
    parser.add_argument("--prompts", nargs="*", default=None, help="prompts to compare (the first is the baseline)")
    parser.add_argument("--tune", action="store_true", help="prompt tuner: compare paraphrases of --task")
    parser.add_argument("--max-prompts", type=int, default=5)
    parser.add_argument("--models", default="dreamlabs/dream-1-beta", help="comma-separated; the first is the baseline")
    parser.add_argument("--episodes", type=int, default=10, help="scenes per variant (same seeds for every variant)")
    parser.add_argument("--seconds", type=float, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="dreamkit.yaml", help="sim_path and runs_dir come from here")
    parser.add_argument("--mock", action="store_true", help="scripted arm instead of Dream-1 (tests the mechanism)")
    parser.add_argument("--behaviors", default="dropped,success,released_off_target,success")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--yes", action="store_true", help="do not ask before spending credits")
    args = parser.parse_args(argv)

    prompts = args.prompts or (paraphrases(args.task, args.max_prompts) if args.tune else [args.task])
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    variants = [(m, p) for m in models for p in prompts]
    if len(variants) < 2:
        parser.error("give two prompts (--prompts a b, or --tune) or two models (--models a,b)")
    unparsed = [p for p in prompts if None in parse(p)]
    if unparsed:
        parser.error(f"success-check cannot read an object and a target from: {unparsed}")
    worst = sum(price(m, args.mock) * args.episodes * args.seconds / 1.6 / CALLS_PER_HOUR for m, _ in variants)
    print(f"{len(variants)} variants x {args.episodes} scenes; worst case ${worst:.2f} of credits")
    if worst > 0 and not args.yes and input("start? [y/N] ").strip().lower() != "y":
        return 1

    sim_loop = sim_loop or load_sim_loop()
    cfg, base = dk_config.load(args.config if Path(args.config).is_file() else None)
    runs = cfg.runs_path(base)
    runs.mkdir(parents=True, exist_ok=True)
    rows, first_dir = [], None
    with tempfile.TemporaryDirectory() as tmp:
        for i, (model, prompt) in enumerate(variants):
            label = (model.split("/")[-1] if len(models) > 1 else "") + (f" | {prompt}" if len(prompts) > 1 else "")
            label = label.strip(" |")
            reliability = {"price_per_hour": price(model, args.mock)}
            if first_dir is not None:
                reliability["baseline_run"] = str(first_dir)
            path = Path(tmp) / f"variant{i}.yaml"
            path.write_text(yaml.safe_dump({"sim_path": str(cfg.sim_dir(base)), "runs_dir": str(runs),
                                            "plugins": [*BASE, {"reliability": reliability}]}))
            print(f"\n=== {i + 1}/{len(variants)}: {label} ===", flush=True)
            before = {p.name for p in runs.iterdir() if p.is_dir()}
            sim_argv = ["--config", str(path), "--episodes", str(args.episodes), "--seconds", str(args.seconds),
                        "--task", prompt, "--model", model, "--seed", str(args.seed)]
            if args.mock:
                sim_argv += ["--mock", "--behaviors", args.behaviors]
            if args.fast:
                sim_argv.append("--fast")
            sim_loop.main(sim_argv)
            new = [p for p in runs.iterdir() if p.is_dir() and p.name not in before]
            run = max(new, key=lambda p: p.stat().st_mtime)
            first_dir = first_dir or run
            rows.append({"label": label, "model": model, "prompt": prompt, "run": run.name,
                         "episodes": json.loads((run / "run.json").read_text())["episodes"]})

    base_eps = rows[0]["episodes"]
    for r in rows:
        r["summary"] = summarize(r["episodes"], 1.96, price(r["model"], args.mock), CALLS_PER_HOUR)
        r["vs_baseline"] = None if r is rows[0] else compare(base_eps, r["episodes"], "first_attempt")
    print("\nrank  success              vs baseline            variant")
    for k, r in enumerate(sorted(rows, key=lambda r: -r["summary"]["rate"]), 1):
        s, c = r["summary"], r["vs_baseline"]
        lo, hi = wilson(s["successes"], s["attempts"])
        vs = "baseline" if c is None else (f"{100 * c['delta']:+.0f} pp, p={c['p_value']:.3f} "
                                           + ("(real)" if c["p_value"] < 0.05 else "(could be luck)"))
        print(f"{k:>4}  {s['successes']:>2}/{s['attempts']:<2} {100 * s['rate']:>3.0f}% "
              f"({100 * lo:.0f}-{100 * hi:.0f}%)  {vs:<22} {r['label']}  [run {r['run']}]")
    out = runs / f"ab-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps([{k: v for k, v in r.items() if k != "episodes"} for r in rows], indent=2))
    print(f"\n{out}")
    if args.mock:
        print("(mock: scripted failures; these numbers test the mechanism, not Dream-1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
