"""dreamkit run | ui | list | search | install | new"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _config_path(arg: str | None) -> str | None:
    if arg:
        return arg
    return "dreamkit.yaml" if Path("dreamkit.yaml").is_file() else None


def cmd_ui(args) -> int:
    from .server import serve

    serve(_config_path(args.config) or "dreamkit.yaml", host=args.host, port=args.port)
    return 0


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
    root = Path(cfg.runs_dir).expanduser()
    return root if root.is_absolute() else base / root


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
    from rich.console import Console

    from . import harvest

    console = Console()
    root = _runs_root(args.config)
    runs = [root / r for r in args.runs] if args.runs else sorted(p for p in root.iterdir() if p.is_dir())
    missing = [r.name for r in runs if not r.is_dir()]
    if missing:
        console.print(f"[red]no such run: {', '.join(missing)}[/red]")
        return 1
    out = Path(args.out)
    copied = harvest.export(runs, out)
    if not copied:
        console.print("no successful episodes in these runs (is the harvester plugin on?)")
        return 1
    console.print(f"{len(copied)} successful episode{'s' if len(copied) != 1 else ''} -> {out}")
    if args.no_validate:
        return 0
    report = harvest.validate(out)
    if report.get("ok"):
        console.print(f"[green]validate_dataset: OK[/green] {json.dumps({k: v for k, v in report.items() if k != 'ok'})}")
        console.print("Ready for fine-tuning. Nothing was uploaded; when you decide to: "
                      "client.upload_dataset(path, name, embodiment)")
        return 0
    console.print(f"[red]validate_dataset: {report.get('error')}[/red]")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dreamkit", description="Plugins and a monitor for Dream-1. In your loop: client = dreamkit.wrap(dream.DreamClient()); demo loop: python examples/sim_loop.py --mock")
    parser.add_argument("-c", "--config", help="dreamkit.yaml (default: ./dreamkit.yaml)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ui", help="web panel")
    p.add_argument("-c", "--config", dest="config", default=argparse.SUPPRESS)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8800)
    p.set_defaults(func=cmd_ui)

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
    p = sub.add_parser("harvest", help="successful episodes -> one folder, checked with validate_dataset (no upload)")
    hsub = p.add_subparsers(dest="harvest_cmd", required=True)
    e = hsub.add_parser("export", help="gather and validate")
    e.add_argument("runs", nargs="*", help="run ids (default: all runs)")
    e.add_argument("--out", default="harvest")
    e.add_argument("--no-validate", action="store_true")
    e.set_defaults(func=cmd_harvest)

    args = parser.parse_args(argv)
    if not hasattr(args, "config"):
        args.config = None
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
