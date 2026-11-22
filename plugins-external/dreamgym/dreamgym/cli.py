"""dreamgym: measure, compare and improve Dream-1 in the YAM sim.

  dreamgym tasks
  dreamgym models
  dreamgym arena   --task banana_plate --policy dream-1 --n 30
  dreamgym compare --task banana_plate --policy dream-1 --policy dream:dream-1-ft:xxxx --n 30
  dreamgym compare --task banana_plate --policy dream-1 --prompt "put banana on the plate" --prompt "place the banana onto the blue plate"
  dreamgym record  --task banana_plate --n 50 --out data/banana
  dreamgym record  --task banana_plate --mix banana_plate:30,apple_bowl:30,orange_bowl:30 --vary-prompts --out data/fruit
  dreamgym record  --task banana_plate --n 20 --policy dream-1 --out data/harvest_banana   # Harvester
  dreamgym finetune data/banana --validate-only    |  dreamgym finetune data/banana  |  --status
  dreamgym report  runs/<folder>
  dreamgym serve                       # dreamkit's panel, Data tab (= dreamkit ui)

Also as `dreamkit dreamgym <command>` once installed as a dreamkit app plugin (dreamkit install plugins-external/dreamgym).
Every episode runs through dreamkit.wrap: the loop plugins of dreamkit.yaml run in it, History shows it (badge Gym).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from . import sim
from .arena import format_summary, run_arena, slug
from .tasks import TASKS, get_task

## DreamGym keeps its own folders, apart from dreamkit's runs/ (ignored by git): gym/runs, gym/data.
def _runs() -> Path:
    """Gym's evaluation folders live with dreamkit's runs: <data_dir>/runs/gym-<id>/."""
    from .kit import paths

    return paths()[0]


def _data() -> Path:
    """Datasets live in <data_dir>/datasets/, with the harvester's."""
    from .kit import paths

    return paths()[1]


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

    if args.recovery_ab:  # the same seeds with dreamkit's recovery off, then on; McNemar on the pairs
        args.policy, args.prompt = [args.policy], [args.prompt] if args.prompt else None
        return cmd_compare(args)
    task = _task(args)
    policy = make_policy(args.policy, action_limit=args.action_limit)
    out = args.out or _runs() / f"gym-{time.strftime('%m%d-%H%M')}_{task.name}_{slug(args.policy)}"
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
    if getattr(args, "recovery_ab", False):
        from .compare import with_recovery_ab

        variants = with_recovery_ab(variants)
    if len(variants) < 2:
        print("compare needs two variants: two --policy, two --prompt, or --recovery-ab", file=sys.stderr)
        return 2
    out = args.out or _runs() / f"gym-{time.strftime('%m%d-%H%M')}_compare_{task.name}"
    report = run_compare(task, variants, n=args.n, out_dir=out, seed0=args.seed, max_seconds=args.max_seconds,
                         action_limit=args.action_limit, gifs=args.gifs)
    print("\n" + format_compare(report))
    print(f"\nreport: {build(out)}")
    return 0


def _mix(args) -> list[tuple[str, int]]:
    """[(task, episodes to keep)] from --mix banana_plate:20,apple_bowl:20, else the one --task with --n."""
    if not args.mix:
        return [(args.task, args.n)]
    out = []
    for part in args.mix.split(","):
        name, _, n = part.strip().partition(":")
        if name not in TASKS:
            raise SystemExit(f"unknown task {name!r} in --mix")
        out.append((name, int(n or args.n)))
    return out


def cmd_record(args) -> int:
    """Episodes -> one LeRobot v2.1 dataset; only successful episodes are kept.

    With --mix the tasks take turns (one episode each, round robin), so the dataset stays balanced while it grows and
    a stopped recording still has every task in it. Progress is rewritten after every episode
    (<dataset>/.dreamgym/record.progress.json): the Data tab shows the dataset growing live.
    """
    import numpy as np

    from . import dataset
    from .lerobot_writer import LeRobotWriter
    from .policies import make_policy
    from .runner import run_episode
    from .tasks import prompt_variants

    plan = _mix(args)
    out = args.out or _data() / plan[0][0]
    writer = LeRobotWriter(out, fps=args.fps)
    policy = make_policy(args.policy)
    source = "scripted" if policy.name.startswith("scripted") else policy.name
    from .kit import PolicyClient, open_kit

    tasks = {name: _task(argparse.Namespace(task=name, jitter=args.jitter, lighting=args.lighting)) for name, _ in plan}
    rng = np.random.default_rng(args.seed)
    want = dict(plan)
    # attempts allowed per task: --attempts-per-success x what that task needs, else --max-attempts for each
    cap = {n: math.ceil(k * args.attempts_per_success) if args.attempts_per_success else args.max_attempts
           for n, k in want.items()}
    kept = {name: 0 for name in want}
    tried = {name: 0 for name in want}
    progress = {"plan": want, "kept": kept, "tried": tried, "source": source, "status": "running",
                "started": time.time(), "last": None, "current": None, "pid": os.getpid()}
    dataset.write_progress(out, progress)
    env = sim.make_env()
    kit = open_kit(PolicyClient(policy), model=policy.name,
                   info={"mode": "record", "tasks": list(want), "policy": policy.name, "dataset": str(out)})
    seed = args.seed
    status = "done"
    try:
        while True:
            todo = [n for n in want if kept[n] < want[n] and tried[n] < cap[n]]
            if not todo:
                break
            for name in todo:
                task = tasks[name]
                prompt = task.prompt
                if args.vary_prompts:
                    prompt = str(rng.choice(prompt_variants(task)))
                progress["current"] = name
                dataset.write_progress(out, progress)
                writer.begin(prompt)
                result, _ = run_episode(env, task, policy, seed=seed, prompt=prompt, max_seconds=args.max_seconds,
                                        step_hook=writer.step_hook, kit=kit)
                ok = writer.end(keep=result.success)
                if ok:
                    dataset.add_episode(out, writer, task=name, prompt=prompt, source=source, seed=seed,
                                        run=getattr(result, "run_id", None), episode=getattr(result, "episode", None))
                kept[name] += ok
                tried[name] += 1
                progress["last"] = {"task": name, "seed": seed, "reason": result.reason, "kept": bool(ok)}
                dataset.write_progress(out, progress)
                print(f"{name} seed {seed}: {result.reason:<15} {'kept' if ok else 'dropped'}  "
                      f"({sum(kept.values())}/{sum(want.values())} kept)")
                seed += 1
    except KeyboardInterrupt:
        status = "stopped"
    except Exception:
        status = "failed"
        raise
    finally:
        writer.end(keep=False)
        progress.update(status=status, current=None)
        dataset.write_progress(out, progress)
        kit.close()
        env.close()
    print(f"\n{sum(kept.values())} episodes kept, {writer.total_frames} frames at {args.fps} fps in {out}")
    if status == "stopped":
        print("recording stopped by the user")
        return 0  # a stop is not a failure, whatever was kept by then
    if not sum(kept.values()):
        print(f"Error: no episode succeeded in {sum(tried.values())} tries, so nothing was kept")
        return 1
    print(f"next: dreamgym finetune {out} --validate-only")
    return 0


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
    """DreamGym lives in dreamkit's panel now (the Data tab): start the panel and open the tab."""
    import subprocess
    import threading
    import webbrowser

    url = f"http://127.0.0.1:{args.port}/#/app/dreamgym"
    print(f"DreamGym is the Data tab of dreamkit's panel: {url}")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    return subprocess.call([sys.executable, "-m", "dreamkit.cli", "ui", "--port", str(args.port)])


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
    p.add_argument("--recovery-ab", action="store_true",
                   help="the same seeds with dreamkit's recovery off and on, McNemar on the pairs")
    p.set_defaults(fn=cmd_arena)

    p = sub.add_parser("compare", help="variants on the same seeds + paired significance test")
    _common(p)
    p.add_argument("--policy", action="append", required=True, help="repeatable")
    p.add_argument("--prompt", action="append", default=None, help="repeatable; crossed with every --policy")
    p.add_argument("--recovery-ab", action="store_true", help="every variant with dreamkit's recovery off and on")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("record", help="scripted demos -> LeRobot dataset for fine-tuning")
    _common(p)
    p.add_argument("--policy", default="scripted:0.004", help="scripted[:noise_m]")
    p.add_argument("--fps", type=int, default=10, choices=[10, 5])
    p.add_argument("--max-attempts", type=int, default=200, help="per task")
    p.add_argument("--attempts-per-success", type=float, default=None,
                   help="cap each task at this many attempts per episode it needs (instead of --max-attempts)")
    p.add_argument("--mix", default=None, help="several tasks into one dataset: banana_plate:20,apple_bowl:20")
    p.add_argument("--vary-prompts", action="store_true", help="each episode gets a paraphrase of the task's prompt")
    p.set_defaults(fn=cmd_record)

    p = sub.add_parser("finetune", help="validate / upload + fine-tune / job status of a dataset folder")
    p.add_argument("dataset", type=Path)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--name", default=None)
    p.add_argument("--suffix", default="dreamgym")
    p.add_argument("--steps", type=int, default=None)
    p.set_defaults(fn=cmd_finetune)

    p = sub.add_parser("serve", help="dreamkit's panel with the Data tab open (dreamkit ui)")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("report", help="rebuild report.html of a run or compare folder")
    p.add_argument("folder", type=Path)
    p.set_defaults(fn=cmd_report)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
