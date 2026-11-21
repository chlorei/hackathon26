"""App plugins (PLUGIN_STANDARD.md §13): a page, an API and commands of their own, inside dreamkit.

A loop plugin runs inside your control loop, one hook at a time. An app plugin is a tool around the robot: DreamGym
(Arena, Compare, datasets). It gets a tab in the panel, its router under /apps/<name>/api, its page under
/apps/<name>/ and `dreamkit <name> ...` commands. It never drives the robot itself: episodes go through
dreamkit.wrap(), so the loop plugins of dreamkit.yaml run in them too.

    # yourapp/app.py
    from dreamkit.sdk import App, AppContext, Manifest

    def router(ctx: AppContext) -> APIRouter: ...
    app = App(name="yourapp", manifest=Manifest(kind="app", title="Your app", icon="Wrench", needs=["robot"]),
              router=router, page=Path(__file__).parent / "web", commands=cli.main)

    # pyproject.toml
    [project.entry-points."dreamkit.apps"]
    yourapp = "yourapp.app:app"
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Callable

from .standard import NAME_RE, Manifest, Problem

GROUP = "dreamkit.apps"
## Names an app may not take: they are dreamkit's own commands and routes.
RESERVED = {"ui", "sim", "run", "list", "search", "check", "test", "bench", "install", "publish", "new", "record",
            "review", "accuracy", "harvest", "api", "apps", "media", "ws", "live", "plugins", "history"}


@dataclass
class AppContext:
    """What the panel hands an app's router: where dreamkit's config and runs are, and whether there is a world."""

    config_path: Path  # dreamkit.yaml (it may not exist)
    base: Path  # the folder relative paths resolve against
    runs_dir: Path  # dreamkit's runs/ (History)
    world: bool  # Dream Labs' simulator is installed here
    config: Any = None


@dataclass
class App:
    """An app plugin: a manifest (kind="app"), a router factory, a static page folder, a CLI entry."""

    name: str
    manifest: Manifest
    router: Callable[[AppContext], Any] | None = None  # -> fastapi.APIRouter, mounted at /apps/<name>/api
    page: Path | None = None  # a folder with index.html, served at /apps/<name>/ (the tab's frame)
    commands: Callable[[list[str]], int] | None = None  # `dreamkit <name> <args>` -> commands(args)
    description: str = ""
    origin: str = field(default="", compare=False)

    @property
    def title(self) -> str:
        return self.manifest.title or self.name

    @property
    def aliases(self) -> list[str]:
        """`dreamkit <name>`, and `dreamkit <title>` when the title is one plain word (DreamGym: dreamkit gym)."""
        out = [self.name]
        t = (self.manifest.title or "").strip().lower()
        if t and t.isalpha() and t != self.name and t not in RESERVED:
            out.append(t)
        return out

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "title": self.title, "icon": self.manifest.icon, "needs": self.manifest.needs,
                "description": self.description, "page": self.page is not None, "api": self.router is not None,
                "commands": self.aliases if self.commands else []}


def check_app(app: Any, where: str = "") -> list[Problem]:
    """Every rule an app object can break on its own (§13)."""
    where = where or getattr(app, "origin", "") or repr(app)
    if not isinstance(app, App):
        return [Problem("§13", where, f"the entry point is a {type(app).__name__}, not dreamkit.sdk.App",
                        "Point it at an App(...) instance: yourapp = \"yourapp.app:app\".")]
    out: list[Problem] = []
    if not NAME_RE.match(app.name) or app.name in RESERVED:
        out.append(Problem("§13", where, f"app name {app.name!r} is not 1-3 kebab-case words or is reserved",
                           "Pick a name like \"dreamgym\" that is not a dreamkit command."))
    m = app.manifest
    if not isinstance(m, Manifest) or m.kind != "app":
        out.append(Problem("§13", where, "manifest.kind is not \"app\"", 'Use Manifest(kind="app", title=..., icon=...).'))
        return out
    if not m.title:
        out.append(Problem("§13", where, "manifest has no title (the tab's label)", 'Add title="Gym".'))
    if m.hooks:
        out.append(Problem("§13", where, "an app has no loop hooks", "Move per-step work into a loop plugin."))
    if app.router is not None and not callable(app.router):
        out.append(Problem("§13", where, "router is not a function of AppContext", "router=lambda ctx: APIRouter()"))
    if app.page is not None and not (Path(app.page) / "index.html").is_file():
        out.append(Problem("§13", where, f"page {app.page} has no index.html", "Point page= at the folder of index.html."))
    if app.page is not None:
        html = (Path(app.page) / "index.html").read_text(errors="ignore") if (Path(app.page) / "index.html").is_file() else ""
        if THEME_HREF not in html:
            out.append(Problem("§13", where, f"the page does not load the panel's theme ({THEME_HREF})",
                               f'Add <link rel="stylesheet" href="{THEME_HREF}"> and colour with var(--dk-…).'))
    if app.commands is not None and not callable(app.commands):
        out.append(Problem("§13", where, "commands is not callable", "commands=main, where main(argv) -> int."))
    if app.page is None and app.router is None and app.commands is None:
        out.append(Problem("§13", where, "the app has no page, no API and no commands", "Give it at least one."))
    return out


def _source_entries(src: Path) -> dict[str, str]:
    import tomllib

    pyproject = src / "pyproject.toml"
    if not pyproject.is_file():
        return {}
    data = tomllib.loads(pyproject.read_text())
    return dict(data.get("project", {}).get("entry-points", {}).get(GROUP, {}))


def _load(spec: str, origin: str) -> Any:
    module, _, attr = spec.partition(":")
    obj = importlib.import_module(module.strip())
    for part in attr.strip().split(".") if attr else []:
        obj = getattr(obj, part)
    if isinstance(obj, App):
        obj.origin = origin
    return obj


def discover(extra: list[Path] | None = None) -> tuple[dict[str, App], list[str]]:
    """Installed apps (entry points), then source folders (DREAMKIT_PLUGIN_PATH, `extra`): {name: App}, problems."""
    found: dict[str, App] = {}
    problems: list[str] = []
    sources = [Path(p) for p in os.environ.get("DREAMKIT_PLUGIN_PATH", "").split(os.pathsep) if p] + list(extra or [])
    items: list[tuple[str, str, str]] = []
    for src in sources:
        for name, spec in _source_entries(src).items():
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            items.append((name, spec, f"source {src}"))
    for ep in entry_points(group=GROUP):
        items.append((ep.name, ep.value, f"pip {ep.dist.name if ep.dist else ep.value}"))
    for name, spec, origin in items:
        if name in found:
            continue
        try:
            app = _load(spec, origin)
        except Exception as error:  # noqa: BLE001 - one broken app never takes the panel down
            problems.append(f"app {name} ({origin}): failed to import: {type(error).__name__}: {error}")
            continue
        bad = check_app(app, origin)
        if bad:
            problems.append(f"app {name} ({origin}) refused:\n  " + "\n  ".join(str(p) for p in bad))
            continue
        if app.name != name:
            problems.append(f"app {name} ({origin}): entry point name differs from App.name {app.name!r}")
            continue
        found[name] = app
    return found, problems


def forget(src: Path) -> None:
    """Drops what a check imported from a source folder (a temporary clone during install): its sys.path entry and its
    modules, so the installed copy is imported next time, not the deleted clone."""
    src = Path(src).resolve()
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != src]
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and src in Path(file).resolve().parents:
            del sys.modules[name]
    importlib.invalidate_caches()


def world_available(cfg: Any, base: Path) -> bool:
    """Dream Labs' simulator is here: sim_path has yam_env.py, or the dreamlabs wheel's dream_examples."""
    try:
        if cfg is not None and (cfg.sim_dir(base) / "yam_env.py").is_file():
            return True
    except Exception:  # noqa: BLE001
        pass
    import importlib.util

    try:
        return importlib.util.find_spec("dream_examples") is not None
    except (ImportError, ValueError):
        return False


## The theme every app page loads: the same file as plugin pages (§7a), served by the panel.
THEME_HREF = "/theme/dreamkit-theme.css"
