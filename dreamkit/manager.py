"""Finds plugins (built-in, pip entry points, local folders) and calls their hooks without letting one crash a run."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
import traceback
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from .config import PluginSpec
from .plugin import HOOKS, Context, EndEpisode, Plugin, StopEpisode

BUILTIN_MODULES = (
    "success_check",
    "failure_monitor",
    "camera_check",
    "retry",
    "latency_monitor",
    "recorder",
    "safety",
    "recap",
    "harvester",
    "recovery",
    "report",
)
LOCAL_DIRS = (Path("~/.dreamkit/plugins").expanduser(), Path("plugins"))


def _classes_in(module) -> list[type[Plugin]]:
    return [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, Plugin) and obj is not Plugin and obj.__module__ == module.__name__
    ]


def discover(extra_dirs: list[Path] | None = None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Every plugin that can be loaded, by name: {"cls": class, "source": builtin|pip|local, "origin": str}.

    Later sources override earlier ones with the same name: built-in < pip < local folders.
    @return (plugins, problems): problems are human lines for plugins that failed to import.
    """
    found: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for mod in BUILTIN_MODULES:
        try:
            module = importlib.import_module(f"dreamkit.plugins.{mod}")
            for cls in _classes_in(module):
                found[cls.name] = {"cls": cls, "source": "builtin", "origin": module.__name__}
        except Exception as error:  # pragma: no cover - a broken built-in is a bug, but never fatal
            problems.append(f"builtin {mod}: {error}")
    for ep in entry_points(group="dreamkit.plugins"):
        if ep.value.startswith("dreamkit.plugins."):
            continue  # our own built-ins, already loaded
        try:
            cls = ep.load()
            found[ep.name] = {"cls": cls, "source": "pip", "origin": f"{ep.dist.name if ep.dist else '?'}:{ep.value}"}
        except Exception as error:
            problems.append(f"pip plugin {ep.name}: {error}")
    for folder in [*LOCAL_DIRS, *(extra_dirs or [])]:
        if not folder.is_dir():
            continue
        for file in sorted(folder.glob("*.py")):
            if file.name.startswith("_"):
                continue
            name = f"dreamkit_local_{file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(name, file)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                for cls in _classes_in(module):
                    found[cls.name] = {"cls": cls, "source": "local", "origin": str(file)}
            except Exception as error:
                problems.append(f"local plugin {file}: {error}")
    return found, problems


def describe(found: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": entry["cls"].description or (entry["cls"].__doc__ or "").strip().split("\n")[0],
            "defaults": entry["cls"].defaults,
            "group": getattr(entry["cls"], "group", "other"),
            "source": entry["source"],
            "origin": entry["origin"],
        }
        for name, entry in sorted(found.items())
    ]


class PluginManager:
    """Owns the plugin instances of one run, in config order, and calls their hooks."""

    def __init__(self, specs: list[PluginSpec], extra_dirs: list[Path] | None = None):
        self.available, self.problems = discover(extra_dirs)
        self.plugins: list[Plugin] = []
        self.missing: list[str] = []
        self.failed: list[str] = []
        for spec in specs:
            if not spec.enabled:
                continue
            entry = self.available.get(spec.name)
            if entry is None:
                self.missing.append(spec.name)
                continue
            try:
                plugin = entry["cls"](**spec.params)
                plugin.name = spec.name
                self.plugins.append(plugin)
            except Exception as error:
                self.failed.append(f"{spec.name}: {error}")

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.plugins]

    def get(self, name: str) -> Plugin | None:
        return next((p for p in self.plugins if p.name == name), None)

    def call(self, hook: str, ctx: Context) -> None:
        """Runs one hook on every plugin in order.

        StopEpisode / EndEpisode propagate (after the remaining plugins are skipped); any other exception is turned
        into an `error` event from that plugin and the run goes on.
        """
        assert hook in HOOKS, hook
        for plugin in self.plugins:
            method = getattr(plugin, hook, None)
            if method is None or getattr(type(plugin), hook, None) is getattr(Plugin, hook):
                continue  # not overridden
            ctx._plugin = plugin.name
            try:
                method(ctx)
            except (StopEpisode, EndEpisode):
                raise
            except Exception as error:
                tb = traceback.extract_tb(error.__traceback__)
                where = f"{Path(tb[-1].filename).name}:{tb[-1].lineno}" if tb else ""
                ctx.emit("error", f"{hook} failed: {type(error).__name__}: {error}", where=where, hook=hook)
            finally:
                ctx._plugin = "dreamkit"
