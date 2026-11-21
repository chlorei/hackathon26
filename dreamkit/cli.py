"""dreamkit run | ui | list | search | install | new"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _config_path(arg: str | None) -> str | None:
    if arg:
        return arg
    return "dreamkit.yaml" if Path("dreamkit.yaml").is_file() else None


def cmd_run(args) -> int:
    from .runner import run

    return run(_config_path(args.config), mock=args.mock, json_events=args.json_events, episodes=args.episodes,
               task=args.task, model=args.model, seconds=args.seconds)


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


def cmd_install(args) -> int:
    from rich.console import Console

    from . import config as config_mod
    from . import registry

    path = _config_path(args.config)
    url = config_mod.load(path)[0].registry_url if path else None
    entries, _ = registry.load(url)
    console = Console()
    spec = registry.resolve(args.target, entries)
    console.print(f"installing [bold]{spec}[/bold] …")
    result = registry.install(args.target, entries)
    if result.returncode != 0:
        console.print(result.stdout[-2000:] + result.stderr[-2000:])
        console.print("[red]install failed[/red]")
        return 1
    console.print("[green]installed[/green] — enable it in dreamkit.yaml under `plugins:` or in the panel")
    return 0


PLUGIN_TEMPLATE = '''"""{title}: a dreamkit plugin."""

from dreamkit import Context, Plugin


class {cls}(Plugin):
    """What it checks, in one line."""

    name = "{name}"
    description = "TODO: one line for the panel"
    defaults = {{"threshold": 0.5}}  # every number lives here, editable from dreamkit.yaml and the panel

    def on_episode_start(self, ctx: Context) -> None:
        self.count = 0

    def on_step(self, ctx: Context) -> None:
        # ctx.env.data (MuJoCo), ctx.state, ctx.frames, ctx.command, ctx.t ...
        self.count += 1

    def on_episode_end(self, ctx: Context) -> None:
        ctx.emit("info", f"{name} saw {{self.count}} steps")
'''

PYPROJECT_TEMPLATE = '''[project]
name = "dreamkit-{name}"
version = "0.1.0"
description = "{name}: a dreamkit plugin"
requires-python = ">=3.11"
dependencies = ["dreamkit"]

[project.entry-points."dreamkit.plugins"]
{name} = "{module}:{cls}"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
only-include = ["{module}.py"]
'''


def cmd_new(args) -> int:
    name = args.name.strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
        print("plugin names are lowercase words joined by '-', e.g. gripper-check", file=sys.stderr)
        return 2
    module = name.replace("-", "_")
    cls = "".join(part.capitalize() for part in name.split("-"))
    folder = Path(f"dreamkit-{name}")
    if folder.exists():
        print(f"{folder} already exists", file=sys.stderr)
        return 1
    folder.mkdir()
    (folder / f"{module}.py").write_text(PLUGIN_TEMPLATE.format(title=name, cls=cls, name=name))
    (folder / "pyproject.toml").write_text(PYPROJECT_TEMPLATE.format(name=name, module=module, cls=cls))
    (folder / "README.md").write_text(f"# dreamkit-{name}\n\n```bash\ndreamkit install ./dreamkit-{name}\n```\n\n"
                                      f"Then add `- {name}` under `plugins:` in dreamkit.yaml.\n")
    print(f"created {folder}/ — edit {module}.py, then: dreamkit install ./{folder}")
    print(f"(or drop {module}.py into ./plugins/ to use it without installing)")
    return 0


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dreamkit", description="Plugins around the Dream-1 control loop")
    parser.add_argument("-c", "--config", help="dreamkit.yaml (default: ./dreamkit.yaml)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run attempts in the terminal")
    p.add_argument("-c", "--config", dest="config", default=argparse.SUPPRESS)
    p.add_argument("--mock", action="store_true", help="scripted policy, no API calls (no credits)")
    p.add_argument("--episodes", type=int)
    p.add_argument("--task")
    p.add_argument("--model")
    p.add_argument("--seconds", type=float)
    p.add_argument("--json-events", action="store_true")
    p.set_defaults(func=cmd_run)

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
    p = sub.add_parser("install", help="install a plugin: catalogue name, GitHub URL or local folder")
    p.add_argument("target")
    p.set_defaults(func=cmd_install)
    p = sub.add_parser("review", help="export failed attempts to a CSV for labelling (accuracy check)")
    p.add_argument("--out", default="labels.csv")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--mock", action="store_true", help="include mock runs")
    p.set_defaults(func=cmd_review)
    p = sub.add_parser("accuracy", help="failure-monitor accuracy from a labelled CSV")
    p.add_argument("labels")
    p.add_argument("--tolerance", type=float, default=2.0, help="seconds")
    p.set_defaults(func=cmd_accuracy)
    p = sub.add_parser("new", help="scaffold a plugin package")
    p.add_argument("name")
    p.set_defaults(func=cmd_new)

    args = parser.parse_args(argv)
    if not hasattr(args, "config"):
        args.config = None
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
