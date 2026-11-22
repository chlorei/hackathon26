"""Finds plugins, loads them by the standard (order, conflicts, one control plugin) and runs their hooks with a
budget: an error never stops the user's loop, a plugin that keeps failing or runs slow is switched off, and why is
shown in the panel."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
import time
import traceback
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import numpy as np

from .config import PluginSpec
from .plugin import Context, EndEpisode, Plugin, PluginContext, StopEpisode
from .standard import HOOKS, PluginStandardError, Problem

BUILTIN_MODULES = ("success_check", "failure_monitor", "camera_check", "recovery", "recorder", "harvester", "budget_guard")
LOCAL_DIRS = (Path("~/.dreamkit/plugins").expanduser(), Path("plugins"))
GENERATED_DIR = "plugins-generated"  # the Create tab's checked packages, beside the config's plugins/ folder
## Hooks that run inside a control step and count against the budget (§5).
STEP_HOOKS = ("before_model", "after_model", "on_step")
STEP_BUDGET_MS = 5.0
PLUGIN_BUDGET_MS = 1.0
BUDGET_WINDOW = 100  # on_step calls per budget check
MAX_ERRORS_IN_A_ROW = 5
MAX_OVER_BUDGET = 3
MERGED = {
    "retry": "retry is part of recovery now (recovery: retry_attempts, retry_reasons)",
    "latency-monitor": "latency is part of failure-monitor now (slow_ms)",
    "recap": "recaps are part of History now (always on)",
    "report": "the report is part of History now (always on)",
    "safety": "safety is an external example now: dreamkit install plugins-external/safety",
}


def _classes_in(module) -> list[type[Plugin]]:
    return [obj for _, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, Plugin) and obj is not Plugin and obj.__module__.startswith(module.__name__)]


def discover(extra_dirs: list[Path] | None = None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Every plugin that imports, by name: {"cls", "source", "origin"}. A plugin that breaks the standard does not
    import (PluginStandardError) and is listed in problems with the rule and the fix."""
    found: dict[str, dict[str, Any]] = {}
    problems: list[str] = []

    def take(load, source: str, origin: str) -> None:
        try:
            for cls in load():
                found[cls.name] = {"cls": cls, "source": source, "origin": origin}
        except PluginStandardError as error:
            problems.append(f"{origin}: refused by the plugin standard:\n{error}")
        except Exception as error:
            problems.append(f"{origin}: {type(error).__name__}: {error}")

    for mod in BUILTIN_MODULES:
        name = f"dreamkit.plugins.{mod}"
        if importlib.util.find_spec(name) is None:
            continue
        take(lambda name=name: _classes_in(importlib.import_module(name)), "builtin", name)
    for ep in entry_points(group="dreamkit.plugins"):
        if ep.value.startswith("dreamkit.plugins."):
            continue
        dist = ep.dist.name if ep.dist else "?"
        take(lambda ep=ep: [ep.load()], "pip", f"{dist}:{ep.value}")
    # plugin packages not installed yet (dreamkit check / test / bench / install run on a source folder), and the
    # panel's Create tab's packages: plugins-generated/<name>/ next to the local plugins/ folder
    roots = [(Path(src), "source") for src in filter(None, os.environ.get("DREAMKIT_PLUGIN_PATH", "").split(os.pathsep))]
    roots += [(p, "generated") for d in extra_dirs or [] for p in sorted((d.parent / GENERATED_DIR).glob("*"))
              if p.is_dir()]
    for src, source in roots:
        root = src.resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        for pkg in sorted(p.parent for p in root.glob("*/__init__.py")):
            if pkg.name in ("tests",) or pkg.name.startswith("."):
                continue
            if source == "generated":  # may have been regenerated since: import it afresh
                for mod in [m for m in sys.modules if m == pkg.name or m.startswith(pkg.name + ".")]:
                    del sys.modules[mod]
            take(lambda pkg=pkg: _classes_in(importlib.import_module(pkg.name)), source, str(pkg))
    for folder in [*LOCAL_DIRS, *(extra_dirs or [])]:
        if not folder.is_dir():
            continue
        for file in sorted(folder.glob("*.py")):
            if file.name.startswith("_"):
                continue

            def load(file=file):
                modname = f"dreamkit_local_{file.stem}"
                spec = importlib.util.spec_from_file_location(modname, file)
                module = importlib.util.module_from_spec(spec)
                sys.modules[modname] = module
                spec.loader.exec_module(module)
                return _classes_in(module)

            take(load, "local", str(file))
    return found, problems


def describe(found: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for name, entry in sorted(found.items()):
        cls = entry["cls"]
        out.append({
            "name": name, "description": cls.description(), "source": entry["source"], "origin": entry["origin"],
            "manifest": cls.manifest.model_dump(), "group": cls.manifest.group, "icon": cls.manifest.icon,
            "needs": cls.manifest.needs,
            "capability": cls.manifest.capability, "schema": cls.Params.model_json_schema(),
            "defaults": cls.Params().model_dump(mode="json"),
        })
    return out


class Stats:
    """One plugin's timings and errors in this run."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.total_ms: dict[str, float] = {}
        self.max_ms: dict[str, float] = {}
        self.errors = 0
        self.errors_in_a_row = 0
        self.last_error: str | None = None
        self.window: list[float] = []
        self.over_budget = 0
        self.disabled: str | None = None
        self.suppressed_events = 0
        self.steps = 0
        self.step_samples: list[float] = []

    def record(self, hook: str, ms: float) -> None:
        self.calls[hook] = self.calls.get(hook, 0) + 1
        self.total_ms[hook] = self.total_ms.get(hook, 0.0) + ms
        self.max_ms[hook] = max(self.max_ms.get(hook, 0.0), ms)

    def as_dict(self) -> dict[str, Any]:
        step_ms = sum(self.total_ms.get(h, 0.0) for h in STEP_HOOKS) / self.steps if self.steps else 0.0
        samples = sorted(self.step_samples)
        p95 = samples[int(0.95 * (len(samples) - 1))] if samples else 0.0
        return {
            "ms_per_step": round(step_ms, 4),
            "on_step_p95_ms": round(p95, 4),
            "on_step_max_ms": round(samples[-1], 4) if samples else 0.0,
            "hooks": {h: {"calls": self.calls[h], "avg_ms": round(self.total_ms[h] / self.calls[h], 4),
                          "max_ms": round(self.max_ms[h], 3)} for h in self.calls},
            "errors": self.errors, "last_error": self.last_error, "disabled": self.disabled,
            "suppressed_events": self.suppressed_events,
        }


class PluginManager:
    """The plugins of one run, in load order, each with its own scoped context."""

    def __init__(self, specs: list[PluginSpec], extra_dirs: list[Path] | None = None):
        self.available, self.problems = discover(extra_dirs)
        self.plugins: list[Plugin] = []
        self.refused: dict[str, str] = {}
        self.missing: list[str] = []
        self.failed: list[str] = []
        self.stats: dict[str, Stats] = {}
        self._hook = "on_run_start"
        self._contexts: dict[str, PluginContext] = {}
        self._core: Context | None = None
        self.step_ms_over = 0
        self.world = True  # set per attempt by the Kit
        self.inactive: dict[str, str] = {}  # plugins off for lack of the world (a real robot)
        candidates: list[Plugin] = []
        for spec in specs:
            if not spec.enabled:
                continue
            if spec.name in MERGED and spec.name not in self.available:
                self.refused[spec.name] = MERGED[spec.name]
                continue
            entry = self.available.get(spec.name)
            if entry is None:
                self.missing.append(spec.name)
                continue
            try:
                candidates.append(entry["cls"](**spec.params))
            except PluginStandardError as error:
                self.failed.append(f"{spec.name}: {error}")
            except Exception as error:
                self.failed.append(f"{spec.name}: {type(error).__name__}: {error}")
        self.plugins = self._resolve(candidates)
        for p in self.plugins:
            self.stats[p.name] = Stats()

    # -- loading rules (§2, §3) ------------------------------------------------------------------------------------
    def _resolve(self, candidates: list[Plugin]) -> list[Plugin]:
        accepted: list[Plugin] = []
        control: str | None = None
        for p in candidates:
            m = p.manifest
            clash = next((q.name for q in accepted if q.name in m.conflicts or p.name in q.manifest.conflicts), None)
            if clash:
                self.refused[p.name] = f"conflicts with {clash}"
                continue
            if m.capability == "control":
                if control:
                    self.refused[p.name] = f"conflicts with {control} (only one control plugin may be on)"
                    continue
                control = p.name
            accepted.append(p)
        names = {p.name for p in accepted}
        changed = True
        while changed:  # drop plugins whose requirements are off, until stable
            changed = False
            for p in list(accepted):
                lacking = [r for r in p.manifest.requires if r not in names]
                if lacking:
                    self.refused[p.name] = f"requires {', '.join(lacking)}, which is off"
                    accepted.remove(p)
                    names.discard(p.name)
                    changed = True
        ordered: list[Plugin] = []
        placed: set[str] = set()

        def place(p: Plugin, path: tuple[str, ...] = ()) -> None:
            if p.name in placed or p.name in path:
                return
            for r in p.manifest.requires:
                place(next(q for q in accepted if q.name == r), path + (p.name,))
            placed.add(p.name)
            ordered.append(p)

        for p in accepted:  # config order, every requirement before the plugin that needs it
            place(p)
        return ordered

    def set_world(self, available: bool) -> list[str]:
        """Whether this attempt has a simulator world; returns the plugins that go quiet without it."""
        self.world = available
        off = [p.name for p in self.plugins if "world" in p.manifest.needs and not available]
        for name in off:
            self.inactive[name] = "needs the simulator world (sim only): off on this robot"
        for p in self.plugins:
            if available and p.name in self.inactive:
                del self.inactive[p.name]
        return off

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.plugins]

    @property
    def control(self) -> str | None:
        return next((p.name for p in self.plugins if p.manifest.capability == "control"), None)

    @property
    def needs_frames(self) -> bool:
        return any(p.manifest.needs_frames for p in self.plugins if not self.stats[p.name].disabled)

    def get(self, name: str) -> Plugin | None:
        return next((p for p in self.plugins if p.name == name), None)

    def context(self, core: Context, plugin: Plugin) -> PluginContext:
        if self._core is not core:
            self._core = core
            self._contexts = {}
        ctx = self._contexts.get(plugin.name)
        if ctx is None:
            ctx = PluginContext(core, plugin, lambda: self._hook)
            self._contexts[plugin.name] = ctx
        return ctx

    def stats_dict(self) -> dict[str, Any]:
        for name, ctx in self._contexts.items():
            if name in self.stats:
                self.stats[name].suppressed_events = ctx.suppressed_events
        return {name: {**s.as_dict(), "inactive": self.inactive.get(name)} for name, s in self.stats.items()}

    # -- running hooks ------------------------------------------------------------------------------------------------
    def call(self, hook: str, core: Context) -> float:
        """Runs one hook on every plugin that declared it; returns the milliseconds spent.

        StopEpisode / EndEpisode propagate; anything else becomes an error event from that plugin and the loop goes on.
        """
        assert hook in HOOKS, hook
        self._hook = hook
        spent = 0.0
        for plugin in self.plugins:
            stats = self.stats[plugin.name]
            m = plugin.manifest
            if hook == "on_step":
                stats.steps += 1
            if stats.disabled or hook not in m.hooks:
                continue  # undeclared hooks are never called (§11)
            if "world" in m.needs and not self.world:
                continue  # sim only, and this is a robot
            if hook == "on_step" and core.step % max(1, int(round(20 / m.step_rate_hz))):
                continue
            ctx = self.context(core, plugin)
            ctx._begin_hook()
            started = time.perf_counter()
            try:
                getattr(plugin, hook)(ctx)
                stats.errors_in_a_row = 0
            except (StopEpisode, EndEpisode):
                raise
            except Exception as error:
                self._error(core, plugin, stats, hook, error)
            finally:
                ms = (time.perf_counter() - started) * 1000
                stats.record(hook, ms)
                spent += ms
            if hook == "after_model" and m.capability == "modify_actions" and ctx._actions_set:
                self._apply_stricter(core, plugin, ctx)
            if hook == "on_step":
                self._budget(core, plugin, stats, ms)
                if len(stats.step_samples) < 20000:
                    stats.step_samples.append(ms)
        if hook in STEP_HOOKS and spent > STEP_BUDGET_MS:
            self.step_ms_over += 1
        return spent

    def _error(self, core: Context, plugin: Plugin, stats: Stats, hook: str, error: Exception) -> None:
        stats.errors += 1
        stats.errors_in_a_row += 1
        tb = traceback.extract_tb(error.__traceback__)
        where = f"{Path(tb[-1].filename).name}:{tb[-1].lineno}" if tb else ""
        kind = "breaks the plugin standard" if isinstance(error, PluginStandardError) else "failed"
        stats.last_error = f"{hook} {kind}: {type(error).__name__}: {error}"
        core.event(plugin.name, "error", f"{hook} {kind}: {error}", where=where, hook=hook)
        if stats.errors_in_a_row >= MAX_ERRORS_IN_A_ROW and not stats.disabled:
            stats.disabled = f"switched off after {MAX_ERRORS_IN_A_ROW} errors in a row: {stats.last_error}"
            core.event(plugin.name, "error", f"switched off after {MAX_ERRORS_IN_A_ROW} errors in a row", hook=hook)

    def _budget(self, core: Context, plugin: Plugin, stats: Stats, ms: float) -> None:
        stats.window.append(ms)
        if len(stats.window) < BUDGET_WINDOW:
            return
        avg = sum(stats.window) / len(stats.window)
        stats.window = []
        if avg <= PLUGIN_BUDGET_MS:
            stats.over_budget = 0
            return
        stats.over_budget += 1
        if stats.over_budget >= MAX_OVER_BUDGET:
            stats.disabled = (f"switched off: on_step averaged {avg:.2f} ms, over the {PLUGIN_BUDGET_MS} ms budget "
                              f"{MAX_OVER_BUDGET} times; move slow work to ctx.defer")
            core.event(plugin.name, "error", f"switched off: {avg:.1f} ms per step, budget {PLUGIN_BUDGET_MS} ms",
                       avg_ms=round(avg, 3))
        else:
            core.event(plugin.name, "warn", f"over budget: {avg:.1f} ms per step, budget {PLUGIN_BUDGET_MS} ms",
                       avg_ms=round(avg, 3))

    def _apply_stricter(self, core: Context, plugin: Plugin, ctx: PluginContext) -> None:
        """§3: a modify_actions plugin may clip, slow or drop rows — never add speed or leave joint limits."""
        old, new = core.actions, ctx._actions_copy
        if old is None or new is None:
            return
        problem = None
        state = core.state or {}
        for i, side in enumerate(("left", "right")):
            o, n = np.asarray(old[i], dtype=float), np.asarray(new[i], dtype=float)
            if n.ndim != 2 or n.shape[1] != o.shape[1] or len(n) > len(o):
                problem = f"{side} rows changed shape {o.shape} -> {n.shape}"
                break
            if not len(n):
                continue
            start = np.asarray(state.get(side, o[0]), dtype=float)[: o.shape[1]]
            step_old = np.abs(np.diff(np.vstack([start, o[: len(n)]]), axis=0))
            step_new = np.abs(np.diff(np.vstack([start, n]), axis=0))
            if np.any(step_new[:, :6].max(axis=0) > step_old[:, :6].max(axis=0) + 1e-6):
                problem = f"{side} arm: a joint moves faster than the model asked"
                break
            if core.world.available:
                lim = core.world.joint_limits(side)
                if np.any(n[:, :6] < lim[:, 0] - 1e-6) or np.any(n[:, :6] > lim[:, 1] + 1e-6):
                    problem = f"{side} arm: a row leaves the joint limits"
                    break
            if np.any(n[:, 6] < -1e-6) or np.any(n[:, 6] > 1 + 1e-6):
                problem = f"{side} gripper outside 0…1"
                break
        if problem:
            err = PluginStandardError([Problem("§3", f"{plugin.name}: after_model", f"change rejected: {problem}",
                                               "modify_actions may only make actions stricter (clip, slow, drop).")])
            self._error(core, plugin, self.stats[plugin.name], "after_model", err)
        else:
            core.actions = (new[0].copy(), new[1].copy())
