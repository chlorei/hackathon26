"""dreamgym: measure, compare and improve Dream-1 in the YAM sim.

  dreamgym tasks
  dreamgym models
  dreamgym arena   --task banana_plate --policy dream-1 --n 30
  dreamgym compare --task banana_plate --policy dream-1 --policy dream:dream-1-ft:xxxx --n 30
  dreamgym compare --task banana_plate --policy dream-1 --prompt "put banana on the plate" --prompt "place the banana onto the blue plate"
  dreamgym record  --task banana_plate --n 50 --out data/banana
  dreamgym record  --task banana_plate --n 20 --policy dream-1 --out data/harvest_banana   # Harvester
  dreamgym finetune data/banana --validate-only    |  dreamgym finetune data/banana  |  --status
  dreamgym report  runs/<folder>
  dreamgym serve                       # web dashboard on http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import sim
from .arena import format_summary, run_arena, slug
from .tasks import TASKS, get_task

## DreamGym keeps its own folders, apart from dreamkit's runs/ (ignored by git): gym/runs, gym/data.
RUNS = Path("gym") / "runs"
DATA = Path("gym") / "data"


def _task(args):
    return get_task(args.task, jitter_m=args.jitter, lighting=args.lighting)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--n", type=int, default=20, help="episodes (seeds seed..seed+n-1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=60.0, help="simulated seconds per episode")
    p.add_argument("--jitter", type=float, default=None, help="object position randomisation, m (0 = stock scene)")
    p.add_argument("--lighting", type=float, default=None, help="light randomisation, +- fraction")
    p.add_argument("--action-limit", type=int, default=48, help="rows of each chunk executed")
    p.add_argument("--gifs", choices=["none", "failures", "all"], default="failures")
    p.add_argument("--out", type=Path, default=None)


def cmd_tasks(args) -> int:
    for t in TASKS.values():
        print(f"{t.name:<14} {t.prompt!r}  (jitter {t.jitter_m} m, yaw +-{t.yaw_rad} rad, light +-{t.lighting})")
    return 0


def cmd_models(args) -> int:
    import dream

    with dream.DreamClient() as client:
        models = client.get_models()
    print(json.dumps(models, indent=2, default=str))
    return 0


def cmd_arena(args) -> int:
    from .policies import make_policy
    from .report import build

    task = _task(args)
    policy = make_policy(args.policy, action_limit=args.action_limit)
    out = args.out or RUNS / f"{time.strftime('%m%d-%H%M')}_{task.name}_{slug(args.policy)}"
    summary = run_arena(task, policy, n=args.n, out_dir=out, seed0=args.seed, prompt=args.prompt,
                        max_seconds=args.max_seconds, gifs=args.gifs, resume=not args.fresh)
    print("\n" + format_summary(summary))
    print(f"\nreport: {build(out)}")
    return 0


def cmd_compare(args) -> int:
    from .compare import Variant, format_compare, run_compare
    from .report import build

    task = _task(args)
    prompts = args.prompt or [None]
    variants = [Variant(p, q) for p in args.policy for q in prompts]
    if len(variants) < 2:
        print("compare needs two variants: two --policy, or two --prompt", file=sys.stderr)
        return 2
    out = args.out or RUNS / f"{time.strftime('%m%d-%H%M')}_compare_{task.name}"
    report = run_compare(task, variants, n=args.n, out_dir=out, seed0=args.seed, max_seconds=args.max_seconds,
                         action_limit=args.action_limit, gifs=args.gifs)
    print("\n" + format_compare(report))
    print(f"\nreport: {build(out)}")
    return 0


def cmd_record(args) -> int:
    """Scripted demos -> LeRobot v2.1 dataset; only successful episodes are kept."""
    from .lerobot_writer import LeRobotWriter
    from .policies import make_policy
    from .runner import run_episode

    task = _task(args)
    out = args.out or DATA / task.name
    writer = LeRobotWriter(out, fps=args.fps)
    policy = make_policy(args.policy)
    env = sim.make_env()
    kept, tried, seed = 0, 0, args.seed
    try:
        while kept < args.n and tried < args.max_attempts:
            writer.begin(task.prompt)
            result, _ = run_episode(env, task, policy, seed=seed, max_seconds=args.max_seconds,
                                    step_hook=writer.step_hook)
            ok = writer.end(keep=result.success)
            kept += ok
            tried += 1
            print(f"seed {seed}: {result.reason:<15} {'kept' if ok else 'dropped'}  ({kept}/{args.n} kept)")
            seed += 1
    finally:
        writer.end(keep=False)
        env.close()
    print(f"\n{kept} episodes, {writer.total_frames} frames at {args.fps} fps in {out}")
    print(f"next: dreamgym finetune {out} --validate-only")
    return 0 if kept else 1


def cmd_finetune(args) -> int:
    from . import finetune

    if args.validate_only:
        out = finetune.validate(args.dataset)
    elif args.status:
        out = finetune.status(args.dataset)
    else:
        out = finetune.start(args.dataset, args.name, args.suffix, args.steps)
    print(json.dumps(out, indent=2, default=str))
    return 0 if out.get("ok", True) else 1


def cmd_serve(args) -> int:
    from .web.server import serve

    serve(args.runs, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_report(args) -> int:
    from .report import build

    print(build(args.folder))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="dreamgym", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("tasks", help="list task plugins").set_defaults(fn=cmd_tasks)
    sub.add_parser("models", help="models your account may run").set_defaults(fn=cmd_models)

    p = sub.add_parser("arena", help="N episodes, success rate, failure GIFs")
    _common(p)
    p.add_argument("--policy", default="dream-1", help="dream-1 | dream:<model id> | scripted")
    p.add_argument("--prompt", default=None, help="override the task's prompt")
    p.add_argument("--fresh", action="store_true", help="ignore episodes already in --out")
    p.set_defaults(fn=cmd_arena)

    p = sub.add_parser("compare", help="variants on the same seeds + paired significance test")
    _common(p)
    p.add_argument("--policy", action="append", required=True, help="repeatable")
    p.add_argument("--prompt", action="append", default=None, help="repeatable; crossed with every --policy")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("record", help="scripted demos -> LeRobot dataset for fine-tuning")
    _common(p)
    p.add_argument("--policy", default="scripted:0.004", help="scripted[:noise_m]")
    p.add_argument("--fps", type=int, default=10, choices=[10, 5])
    p.add_argument("--max-attempts", type=int, default=200)
    p.set_defaults(fn=cmd_record)

    p = sub.add_parser("finetune", help="validate / upload + fine-tune / job status of a dataset folder")
    p.add_argument("dataset", type=Path)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--name", default=None)
    p.add_argument("--suffix", default="dreamgym")
    p.add_argument("--steps", type=int, default=None)
    p.set_defaults(fn=cmd_finetune)

    p = sub.add_parser("serve", help="web dashboard: every run, live progress, start runs from the browser")
    p.add_argument("--runs", type=Path, default=RUNS)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("report", help="rebuild report.html of a run or compare folder")
    p.add_argument("folder", type=Path)
    p.set_defaults(fn=cmd_report)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
