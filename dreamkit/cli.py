"""dreamkit sim | run | ui | list | search | install | new"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _config_path(arg: str | None) -> str | None:
    if arg:
        return arg
    return "dreamkit.yaml" if Path("dreamkit.yaml").is_file() else None


def cmd_ui(args) -> int:
    import socket

    from .server import serve

    with socket.socket() as s:  # before serving: uvicorn's own message is followed by our "panel: <url>" otherwise
        if s.connect_ex(("127.0.0.1", args.port)) == 0:
            print(f"port {args.port} is already in use (another dreamkit ui? it is at http://127.0.0.1:{args.port}); "
                  f"use another one: dreamkit ui --port {args.port + 1}", file=sys.stderr)
            return 1
    serve(_config_path(args.config) or "dreamkit.yaml", host=args.host, port=args.port)
    return 0


def _panel(config: Path, port: int = 8800) -> tuple[str, object]:
    """The panel's address; starts `dreamkit ui` beside the program when nothing answers there yet."""
    import socket
    import subprocess
    import time

    url = f"http://127.0.0.1:{port}"
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            return url, None
    proc = subprocess.Popen([sys.executable, "-m", "dreamkit.cli", "-c", str(config), "ui", "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.1)
    return url, proc


def cmd_run(args) -> int:
    """Runs any program on dream.DreamClient with dreamkit on top (dreamkit/attach.py); the program is not edited."""
    import os
    import subprocess
    import webbrowser

    from . import attach

    command = [c for c in args.command if c != "--"]
    if not command:
        print("usage: dreamkit run [--mock BEHAVIOURS] -- <command>   e.g. dreamkit run -- python yam_mujoco.py")
        return 2
    if command[0] in ("python", "python3"):  # this environment's Python (the one dreamkit is installed in)
        command = [sys.executable, *command[1:]]
    elif command[0].endswith(".py"):
        command = [sys.executable, *command]
    config = Path(_config_path(args.config) or "dreamkit.yaml").resolve()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(attach.boot_dir()), *filter(None, [env.get("PYTHONPATH")])])
    env["DREAMKIT_ATTACH"] = "1"
    env["DREAMKIT_CONFIG"] = str(config)
    if args.mock:
        env["DREAMKIT_MOCK"] = args.mock
        env.setdefault("DREAM_API_KEY", "dreamkit-mock")  # the program asks for a key; the mock never sends it
    panel = None
    if not args.no_panel:
        url, panel = _panel(config, args.port)
        env["DREAMKIT_PANEL"] = url
        print(f"dreamkit: monitor on {url}  (plugins from {config if config.is_file() else 'defaults'})")
        if panel is not None:
            webbrowser.open(url + "/#/live")
    try:
        return subprocess.call(command, env=env)
    except KeyboardInterrupt:
        return 130
    finally:
        if panel is not None:
            panel.terminate()


def cmd_sim(args) -> int:
    """Dream Labs' own simulator (yam_mujoco.py, unchanged) with dreamkit on top."""
    from . import config as config_mod

    script = None
    try:  # the installed dreamlabs wheel ships it as dream_examples/yam_mujoco
        import importlib.util

        spec = importlib.util.find_spec("dream_examples")
        if spec and spec.submodule_search_locations:
            cand = Path(list(spec.submodule_search_locations)[0]) / "yam_mujoco" / "yam_mujoco.py"
            script = cand if cand.is_file() else None
    except Exception:
        script = None
    if script is None:
        cfg, base = config_mod.load(_config_path(args.config))
        cand = cfg.sim_dir(base) / "yam_mujoco.py"
        script = cand if cand.is_file() else None
    if script is None:
        print("dreamkit: Dream Labs' simulator not found. Install it (pip install dreamlabs) or set sim_path in "
              "dreamkit.yaml to the folder with yam_mujoco.py")
        return 1
    args.command = [str(script), *args.extra]
    return cmd_run(args)


def cmd_list(args) -> int:
    from rich.console import Console
    from rich.table import Table

    from . import config as config_mod
    from .manager import describe, discover

    found, problems = discover([Path("plugins")])
    enabled = set()
    path = _config_path(args.config)
    if path:
        cfg, _ = config_mod.load(path)
        enabled = {p.name for p in cfg.plugins if p.enabled}
    table = Table(title="dreamkit plugins")
    for col in ("name", "on", "source", "description"):
        table.add_column(col)
    for p in describe(found):
        table.add_row(p["name"], "●" if p["name"] in enabled else "", p["source"], p["description"])
    console = Console()
    console.print(table)
    for line in problems:
        console.print(f"[red]failed to load: {line}[/red]")
    return 0


def cmd_search(args) -> int:
    from rich.console import Console

    from . import config as config_mod
    from . import registry

    path = _config_path(args.config)
    url = config_mod.load(path)[0].registry_url if path else None
    entries, source = registry.load(url)
    hits = registry.search(entries, args.word) if args.word else entries
    console = Console()
    console.print(f"[dim]catalogue: {source}[/dim]")
    for e in hits:
        console.print(f"[bold]{e['name']}[/bold]  {e.get('description', '')}\n    [dim]{e.get('repo', '')}[/dim]")
    if not hits:
        console.print("nothing found")
    return 0


def _print(lines, ok: bool) -> int:
    from rich.console import Console

    console = Console(highlight=False)
    for line in ([lines] if isinstance(lines, str) else lines):
        style = "green" if line.startswith("✓") else "red" if line.startswith("✗") else None
        console.print(line, style=style, markup=False)
    console.print("[green]passed[/green]" if ok else "[red]failed[/red]")
    return 0 if ok else 1


def cmd_check(args) -> int:
    from . import pdk

    if args.all:
        from .manager import discover

        found, problems = discover()
        ok = not problems
        lines = [f"✗ {p}" for p in problems]
        for name in sorted(found):
            passed, out = pdk.check(name)
            ok = ok and passed
            lines += out
        return _print(lines, ok)
    passed, out = pdk.check(args.target, readme=args.readme)
    return _print(out, passed)


def cmd_test(args) -> int:
    from . import pdk

    if args.target == "all":
        from .manager import discover

        found, _ = discover()
        ok, lines = True, []
        for name in sorted(found):
            passed, out = pdk.test(name)
            ok = ok and passed
            lines.append(f"{'✓' if passed else '✗'} {name}: {out.splitlines()[-1] if out else ''}")
        return _print(lines, ok)
    passed, out = pdk.test(args.target, args.replay)
    return _print(out, passed)


def cmd_bench(args) -> int:
    from . import pdk

    passed, lines = pdk.bench(args.target, accuracy=args.accuracy)
    return _print(lines, passed)


def cmd_install(args) -> int:
    from . import pdk

    passed, log = pdk.install(args.target)
    return _print(log.splitlines(), passed)


def cmd_publish(args) -> int:
    from . import pdk

    passed, out = pdk.publish(args.path)
    return _print(out.splitlines(), passed)


INIT_YAML = """\
# dreamkit.yaml: which plugins run around Dream-1, and their settings. What to run (task, attempts, model)
# is your loop's business. Change a plugin's settings on the panel's Plugins page (dreamkit ui) or here.
sim_path: ../dreamlabs/dream_library/examples/yam_mujoco   # Dream Labs' simulator, relative to this file
data_dir: dreamkit-data                                   # runs/ and datasets/: everything dreamkit saves
prices: {dream-1: 3.0, flash: 2.0}                         # dollars per robot-hour of the model (Live, History)
plugins:                 # the working set: what a real arm reports is enough for every one of them
  - camera-check       # a black, uniform or frozen camera, before the model is asked again
  - vision-judge       # was the task done: a vision model looks at the last frame (GEMINI_API_KEY or ANTHROPIC_API_KEY)
  - failure-monitor    # why an attempt failed, and at which second: the one final verdict
  - recovery           # the one control plugin: takes over a failing attempt (arms home, ask again)
  - recorder           # every camera of every attempt to MP4 (History plays it)
  - harvester          # successful attempts into datasets/harvest-<task>/ (fine-tuning data)
  - gripper-check      # live warnings from the gripper (closed on nothing, loosening grip)
  - budget-guard       # warns as the credit budget goes; with recovery it stops the run safely
"""


def cmd_init(args) -> int:
    """A dreamkit.yaml with a good default set of plugins; an existing one is never overwritten."""
    path = Path(args.path)
    if path.exists():
        print(f"{path} already exists: left as it is (edit it, or use the panel's Plugins page)")
        return 1
    from .manager import discover

    # an add-on that is not installed stays in the file commented out, with how to get it: listed, it would greet a
    # new user with "plugin … is not installed" on the first run
    found, _ = discover()
    lines, on, missing = [], [], []
    for line in INIT_YAML.splitlines():
        name = line.strip()[2:].split("#")[0].strip() if line.startswith("  - ") else None
        if name and name not in found:
            missing.append(name)
            line = f"  # - {name}{line[4 + len(name):]}  (add-on: dreamkit install {name})"
        elif name:
            on.append(name)
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}: {', '.join(on)}")
    print("next: client = dreamkit.wrap(dream.DreamClient()) in your loop, and dreamkit ui to watch it")
    if missing:
        print(f"add-ons left commented out until installed: dreamkit install {' '.join(missing)}; without a vision "
              "judge you mark each attempt done or not (History, or the loop's prompt)")
    return 0


def cmd_judge(argv: list[str]) -> int:
    """Judge again the attempts that went 'not judged': vision-judge's saved frames, verdicts back into the run."""
    try:
        from vision_judge.rejudge import main as rejudge  # an add-on  # noqa: PLC0415
    except ImportError:
        print("dreamkit judge needs the vision-judge add-on: dreamkit install vision-judge", file=sys.stderr)
        return 1
    return rejudge(argv)


def cmd_new(args) -> int:
    from .template import scaffold

    name = args.name.strip().lower()
    if not re.fullmatch(r"[a-z]+(-[a-z]+){0,2}", name):
        print("plugin names are 1-3 lowercase words joined by '-', e.g. gripper-check", file=sys.stderr)
        return 2
    if Path(f"dreamkit-{name}").exists():
        print(f"dreamkit-{name} already exists", file=sys.stderr)
        return 1
    root = scaffold(name, args.kind, Path.cwd())
    print(f"created {root.name}/ ({args.kind}) — it already passes the standard:")
    print(f"  dreamkit check {root.name} && dreamkit test {root.name} && dreamkit bench {root.name}")
    return 0


def cmd_record(args) -> int:
    import importlib.util

    loop = Path(__file__).resolve().parent.parent / "examples" / "sim_loop.py"
    if not loop.is_file():
        print("dreamkit record needs the dreamkit checkout (examples/sim_loop.py)", file=sys.stderr)
        return 2
    spec = importlib.util.spec_from_file_location("sim_loop", loop)
    sim_loop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim_loop)
    argv = ["--episodes", "1", "--seconds", str(args.seconds), "--task", args.task, "--record", args.out, "--fast"]
    if args.mock:
        argv += ["--mock", "--behaviors", args.behavior]
    if args.config:
        argv += ["--config", args.config]
    code = sim_loop.main(argv)
    print(f"fixture saved in {args.out}: dreamkit test <plugin> --replay {args.out}")
    return code


def _runs_root(config: str | None) -> Path:
    from . import config as config_mod

    cfg, base = config_mod.load(_config_path(config))
    return cfg.runs_path(base)


def _human(n: int | float | None) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def cmd_where(args) -> int:
    """Where dreamkit keeps everything, and how big it is."""
    from . import config as config_mod
    from . import datasets, storage

    cfg, base = config_mod.load(_config_path(args.config))
    runs, ds = cfg.runs_path(base), cfg.datasets_path(base)
    all_runs = storage.list_runs(runs)
    print(f"config    {Path(_config_path(args.config) or 'dreamkit.yaml').resolve()}")
    print(f"data      {cfg.data_path(base)}")
    print(f"  runs/     {len(all_runs)} runs, {_human(datasets.size_bytes(runs) if runs.is_dir() else 0)}"
          f"  ({sum(1 for r in all_runs if r.get('mock'))} mock)")
    print(f"  datasets/ {len(datasets.list_all(ds, with_size=False))} datasets, "
          f"{_human(datasets.size_bytes(ds) if ds.is_dir() else 0)}")
    old = [p for p in (base / "runs", base / "gym", base / "harvest") if p.exists() and p.resolve() != runs.resolve()]
    if cfg.runs_dir or old:
        print("older places still in use: " + ", ".join(str(p) for p in old or [runs]) + "  ->  dreamkit migrate")
    return 0


def cmd_datasets(args) -> int:
    """Every dataset with its source and status; `dreamkit datasets check <name>` runs validate_dataset on one."""
    from . import config as config_mod
    from . import datasets

    cfg, base = config_mod.load(_config_path(args.config))
    root = cfg.datasets_path(base)
    if args.datasets_cmd in ("train", "status"):  # the one fine-tune path (the Data tab calls the same functions)
        folder = root / args.name
        if not folder.is_dir():
            print(f"no dataset {args.name!r} in {root}")
            return 1
        try:
            if args.datasets_cmd == "train":
                ft = datasets.train(folder, suffix=args.suffix, steps=args.steps, name=args.upload_name,
                                    wait_s=args.wait_s)
            else:
                ft = datasets.refresh(folder)
        except Exception as error:  # no key, no network, a refused dataset: say it plainly
            print(f"{args.name}: {type(error).__name__}: {error}")
            return 1
        model = ft.get("model")
        print(f"{args.name}: {ft.get('stage') or 'not started'}" + (f", model {model}" if model else
              (f", job {ft.get('job_id')}" if ft.get("job_id") else "")))
        return 0
    if args.datasets_cmd == "check":
        folder = root / args.name
        if not folder.is_dir():
            print(f"no dataset {args.name!r} in {root}")
            return 1
        report = datasets.check(folder)
        if report.get("ok"):
            print(f"{args.name}: validate_dataset OK ({report.get('episodes', '?')} episodes, {report.get('frames', '?')} frames). "
                  "Nothing was uploaded.")
            return 0
        print(f"{args.name}: validate_dataset failed: {report.get('error')}")
        return 1
    rows = datasets.list_all(root)
    if not rows:
        print(f"no datasets yet in {root} (the harvester adds successful attempts as they end)")
        return 0
    width = max(len(r["name"]) for r in rows)
    for r in rows:
        extra = f"  model {r['model']}" if r.get("model") else (f"  {r['pending']} waiting for a label" if r["pending"] else "")
        print(f"{r['name']:<{width}}  {r['source']:<9} {r['episodes']:>4} episodes  {_human(r['size_bytes']):>8}  "
              f"{r['status']}{extra}")
    return 0


def cmd_migrate(args) -> int:
    from .migrate import migrate

    return migrate(_config_path(args.config) or "dreamkit.yaml", dry_run=args.dry_run)


def cmd_review(args) -> int:
    """Every failed attempt (real model only unless --mock) into a CSV to label by watching the video."""
    import csv

    from . import storage

    root = _runs_root(args.config)
    rows = []
    for run in storage.list_runs(root):
        if run.get("mock") and not args.mock:
            continue
        for e in run.get("episodes", []):
            if e.get("success") is False:
                top = next((v for v in e.get("videos", []) if v.endswith("top.mp4")), "")
                rows.append({"run": run["id"], "episode": e["episode"], "task": e.get("task"),
                             "predicted": e.get("reason"), "at": e.get("at"),
                             "video": str(root / run["id"] / top) + (f"#t={max(0, (e.get('at') or 1) - 1):.1f}" if top else ""),
                             "human": "", "human_at": "", "note": ""})
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["run", "episode", "predicted", "human"])
        writer.writeheader()
        writer.writerows(rows[: args.limit] if args.limit else rows)
    print(f"{min(len(rows), args.limit or len(rows))} failures -> {args.out}. Watch each video, fill `human` with the "
          f"real reason (same names) and optionally `human_at`, then: dreamkit accuracy {args.out}")
    return 0


def cmd_accuracy(args) -> int:
    import csv

    with open(args.labels) as f:
        rows = [r for r in csv.DictReader(f) if (r.get("human") or "").strip()]
    if not rows:
        print("no labelled rows (fill the `human` column)")
        return 1
    hits = [r for r in rows if r["predicted"].strip() == r["human"].strip()]
    timed = [r for r in hits if (r.get("human_at") or "").strip() and r.get("at")]
    close = [r for r in timed if abs(float(r["at"]) - float(r["human_at"])) <= args.tolerance]
    print(f"failure-monitor: {len(hits)}/{len(rows)} reasons correct ({100 * len(hits) / len(rows):.0f}%)")
    if timed:
        print(f"moment within ±{args.tolerance:.0f} s: {len(close)}/{len(timed)}")
    confusion: dict[tuple[str, str], int] = {}
    for r in rows:
        if r not in hits:
            key = (r["human"].strip(), r["predicted"].strip())
            confusion[key] = confusion.get(key, 0) + 1
    for (human, predicted), n in sorted(confusion.items(), key=lambda kv: -kv[1]):
        print(f"  {n}× was {human}, said {predicted}")
    return 0


def cmd_harvest(args) -> int:
    print("`dreamkit harvest export` is gone: the harvester adds every successful attempt to "
          "<data_dir>/datasets/harvest-<task>/ as it ends, so the dataset is ready.\n"
          "  dreamkit datasets              list them\n  dreamkit datasets check <name> run validate_dataset on one")
    return 1


_BUILTIN_COMMANDS = {"ui", "init", "judge", "sim", "run", "list", "search", "check", "test", "bench", "install", "publish", "new",
                     "record", "review", "accuracy", "harvest", "where", "datasets", "migrate"}


def _app_command(argv: list[str]) -> int | None:
    """`dreamkit <app> ...` -> the app plugin's own commands (§13); None when argv[0] is no app."""
    if not argv or argv[0].startswith("-"):
        return None
    from . import apps

    found, _ = apps.discover()
    for a in found.values():
        if argv[0] in a.aliases and a.commands is not None:
            return int(a.commands(argv[1:]) or 0)
    return None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["judge"]:  # its own options (vision-judge's rejudge)
        return cmd_judge(argv[1:])
    head = [a for a in argv if not a.startswith("-")][:1]
    if head and head[0] not in _BUILTIN_COMMANDS:
        code = _app_command(argv)
        if code is not None:
            return code
    parser = argparse.ArgumentParser(prog="dreamkit", description="Plugins and a monitor for Dream-1. In your loop: client = dreamkit.wrap(dream.DreamClient()); demo loop: python examples/sim_loop.py --mock")
    parser.add_argument("-c", "--config", help="dreamkit.yaml (default: ./dreamkit.yaml)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("judge", help="judge again the attempts that went 'not judged' (vision-judge add-on): "
                   "--run <id> or --all-unjudged")
    p = sub.add_parser("ui", help="web panel")
    p.add_argument("-c", "--config", dest="config", default=argparse.SUPPRESS)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8800)
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("sim", help="Dream Labs' simulator, unchanged, with dreamkit on top (type tasks in it)")
    p.add_argument("--mock", metavar="BEHAVIOURS", help="no model calls: scripted attempts, e.g. dropped,grasped_air,success")
    p.add_argument("--no-panel", action="store_true", help="do not start the monitor")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("extra", nargs=argparse.REMAINDER, help="passed to yam_mujoco.py")
    p.set_defaults(func=cmd_sim)

    p = sub.add_parser("run", help="any program on dream.DreamClient, unchanged, with dreamkit on top")
    p.add_argument("--mock", metavar="BEHAVIOURS", help="no model calls (simulator only): scripted attempts")
    p.add_argument("--no-panel", action="store_true", help="do not start the monitor")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("command", nargs=argparse.REMAINDER, help="-- python yam_mujoco.py | -- dreamcli ... | -- python my_loop.py")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("list", help="installed plugins")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("search", help="search the plugin catalogue")
    p.add_argument("word", nargs="?", default="")
    p.set_defaults(func=cmd_search)
    p = sub.add_parser("check", help="check a plugin against PLUGIN_STANDARD.md")
    p.add_argument("target", nargs="?", default=".", help="plugin name or source folder")
    p.add_argument("--all", action="store_true", help="every installed plugin")
    p.add_argument("--readme", action="store_true", help="regenerate the README params table")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("test", help="run a plugin's replay tests (no simulator, no credits)")
    p.add_argument("target", nargs="?", default=".", help="plugin name, source folder, or 'all'")
    p.add_argument("--replay", help="run the plugin on this fixture instead")
    p.set_defaults(func=cmd_test)
    p = sub.add_parser("bench", help="ms/step on every bundled fixture against the 1 ms budget")
    p.add_argument("target", nargs="?", default=".")
    p.add_argument("--accuracy", action="store_true", help="also score verdicts against each fixture's expected one")
    p.set_defaults(func=cmd_bench)
    p = sub.add_parser("install", help="check + test + bench a plugin, then install it (never unchecked)")
    p.add_argument("target", help="catalogue name, owner/repo, GitHub URL, a .py URL or a folder")
    p.set_defaults(func=cmd_install)
    p = sub.add_parser("publish", help="check + test + bench, then the registry.yaml entry")
    p.add_argument("path", nargs="?", default=".")
    p.set_defaults(func=cmd_publish)
    p = sub.add_parser("init", help="write a dreamkit.yaml with the default plugins (never overwrites one)")
    p.add_argument("path", nargs="?", default="dreamkit.yaml")
    p.set_defaults(func=cmd_init)
    p = sub.add_parser("new", help="scaffold a plugin package that already passes the standard")
    p.add_argument("name")
    p.add_argument("--kind", choices=["observe", "modify", "control"], default="observe")
    p.set_defaults(func=cmd_new)
    p = sub.add_parser("record", help="record a replay fixture from the simulator")
    p.add_argument("out", help="fixture folder to create")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--behavior", default="success")
    p.add_argument("--task", default="put apple into the bowl")
    p.add_argument("--seconds", type=float, default=20)
    p.set_defaults(func=cmd_record)
    p = sub.add_parser("review", help="export failed attempts to a CSV for labelling (accuracy check)")
    p.add_argument("--out", default="labels.csv")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--mock", action="store_true", help="include mock runs")
    p.set_defaults(func=cmd_review)
    p = sub.add_parser("accuracy", help="failure-monitor accuracy from a labelled CSV")
    p.add_argument("labels")
    p.add_argument("--tolerance", type=float, default=2.0, help="seconds")
    p.set_defaults(func=cmd_accuracy)
    p = sub.add_parser("harvest", help="(gone: see dreamkit datasets)")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_harvest)

    p = sub.add_parser("where", help="where dreamkit keeps runs and datasets, and how big they are")
    p.set_defaults(func=cmd_where)
    p = sub.add_parser("datasets", help="every dataset with its source and status; check <name>: validate_dataset")
    p.set_defaults(func=cmd_datasets, datasets_cmd="list")
    dsub = p.add_subparsers(dest="datasets_cmd")
    c = dsub.add_parser("check", help="Dream Labs' validate_dataset on a dataset (local, nothing uploaded)")
    c.add_argument("name")
    c.set_defaults(func=cmd_datasets)
    c = dsub.add_parser("train", help="check, upload and start a Dream-1 fine-tune (uses fine-tuning credits)")
    c.add_argument("name")
    c.add_argument("--suffix", default="dreamkit", help="the fine-tune is named dream-1-ft:<id>-<suffix>")
    c.add_argument("--steps", type=int, default=None, help="training steps (multiple of 100); default: the service's")
    c.add_argument("--upload-name", default=None, help="the dataset's name on the platform (default: its folder)")
    c.add_argument("--wait-s", type=float, default=None, help="wait this long for processing (default: until done)")
    c.set_defaults(func=cmd_datasets)
    c = dsub.add_parser("status", help="ask Dream Labs about the upload / job; records the model id when done")
    c.add_argument("name")
    c.set_defaults(func=cmd_datasets)
    p = sub.add_parser("migrate", help="move runs/, gym/ and harvest/ into the data folder (moves, never deletes)")
    p.add_argument("--dry-run", action="store_true", help="print what would move, change nothing")
    p.set_defaults(func=cmd_migrate)

    args = parser.parse_args(argv)
    if not hasattr(args, "config"):
        args.config = None
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
