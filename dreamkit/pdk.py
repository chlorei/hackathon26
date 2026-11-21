"""The Plugin Development Kit behind `dreamkit check | test | bench | new | install | publish` (PLUGIN_STANDARD.md §10,
§11). Every command works on a plugin name (built-in or installed) or on a plugin's source folder."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from . import registry
from .standard import PluginStandardError, Problem, check_class, check_pyproject

ROOT = Path(__file__).resolve().parent.parent  # the dreamkit checkout (tests/plugins/ lives here)
BUILTIN_TESTS = ROOT / "tests" / "plugins"
LOG = Path("~/.dreamkit/install.log").expanduser()


@dataclass
class Target:
    name: str | None
    cls: type | None
    source: str  # builtin | pip | local | source
    src: Path | None = None  # a source folder (external package)
    problems: list[str] = field(default_factory=list)

    @property
    def module(self) -> str | None:
        return self.cls.__module__.split(".")[-2] if self.cls and self.source == "builtin" else None


@contextmanager
def plugin_path(src: Path | None) -> Iterator[None]:
    """Makes a source folder's plugin importable (not installed) for this process and its children."""
    old = os.environ.get("DREAMKIT_PLUGIN_PATH")
    if src is not None:
        os.environ["DREAMKIT_PLUGIN_PATH"] = os.pathsep.join(filter(None, [str(src), old]))
    try:
        yield
    finally:
        if src is not None:
            if old is None:
                os.environ.pop("DREAMKIT_PLUGIN_PATH", None)
            else:
                os.environ["DREAMKIT_PLUGIN_PATH"] = old


def resolve(target: str) -> Target:
    from .manager import discover  # noqa: PLC0415

    path = Path(target).expanduser()
    if path.is_dir() and (path / "pyproject.toml").is_file():
        src = path.resolve()
        with plugin_path(src):
            found, problems = discover()
        mine = {n: e for n, e in found.items() if e["source"] == "source" and str(src) in e["origin"]}
        own_problems = [p for p in problems if str(src) in p]
        if not mine:
            return Target(None, None, "source", src, own_problems or [f"{src}: no dreamkit Plugin found"])
        name, entry = next(iter(mine.items()))
        return Target(name, entry["cls"], "source", src, own_problems)
    found, problems = discover()
    if target in found:
        e = found[target]
        return Target(target, e["cls"], e["source"])
    hits = [p for p in problems if target.replace("-", "_") in p or target in p]
    return Target(None, None, "missing", None, hits or [f"no plugin named {target!r} (dreamkit list)"])


# -- check -----------------------------------------------------------------------------------------------------------
def check(target: str, readme: bool = False) -> tuple[bool, list[str]]:
    t = resolve(target)
    lines: list[str] = []
    if t.cls is None:
        return False, [f"✗ {p}" for p in t.problems]
    problems: list[Problem] = check_class(t.cls)
    if t.src is not None:
        problems += check_pyproject(t.src, t.cls)
    try:
        t.cls.Params()
    except Exception as error:  # noqa: BLE001
        problems.append(Problem("§4", t.name or "?", f"Params defaults do not validate: {error}", "Fix the defaults."))
    if t.source == "builtin" and not (BUILTIN_TESTS / f"test_{t.module}.py").is_file():
        problems.append(Problem("§9", t.name, f"no tests/plugins/test_{t.module}.py", "Add replay tests."))
    m = t.cls.manifest
    lines.append(f"{t.name} ({t.source}) — {t.cls.description()}")
    lines.append(f"  manifest: group={m.group} icon={m.icon} capability={m.capability} hooks={m.hooks} "
                 f"step_rate_hz={m.step_rate_hz:g} needs_frames={m.needs_frames}")
    if m.capability == "control":
        lines.append("  note: control plugin — only one may be on; the manager refuses a second (§3)")
    for p in problems:
        lines.append(f"✗ {p}")
    if not problems:
        lines.append("✓ name, manifest, hooks, Params (descriptions, units, suffixes), docstring" +
                     (", pyproject, README, tests" if t.src else ""))
    if readme and t.src is not None and not problems:
        _write_readme_table(t)
        lines.append("✓ README params table updated")
    return not problems, lines


def params_table(cls: type) -> str:
    rows = ["| param | default | unit | what |", "|---|---|---|---|"]
    defaults = cls.Params().model_dump(mode="json")
    for name, info in cls.Params.model_fields.items():
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        unit = extra.get("unit") or (" / ".join(extra["choices"]) if extra.get("choices") else "")
        rows.append(f"| `{name}` | `{defaults[name]}` | {unit} | {info.description} |")
    return "\n".join(rows)


def _write_readme_table(t: Target) -> None:
    readme = t.src / "README.md"
    text = readme.read_text() if readme.is_file() else f"# dreamkit-{t.name}\n"
    block = f"<!-- params -->\n{params_table(t.cls)}\n<!-- /params -->"
    if "<!-- params -->" in text:
        text = re.sub(r"<!-- params -->.*?<!-- /params -->", block, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n## Settings\n\n" + block + "\n"
    readme.write_text(text)


# -- test ------------------------------------------------------------------------------------------------------------
def test(target: str, replay_fixture: str | None = None) -> tuple[bool, str]:
    t = resolve(target)
    if t.cls is None:
        return False, "\n".join(t.problems)
    if replay_fixture:
        from .testing import replay  # noqa: PLC0415

        with plugin_path(t.src):
            r = replay(replay_fixture, plugins=[*t.cls.manifest.requires, t.name])
        out = [f"{t.name} on {r.fixture}: {r.budget_ms(t.name):.3f} ms/step"]
        out += [f"  {e['level']:<7} {e.get('t_s') or 0:6.2f}s {e['plugin']}: {e['message']}" for e in r.events]
        out.append(f"  result: success={r.result.get('success')} reason={r.result.get('reason')}")
        return not r.errors(t.name), "\n".join(out)
    tests = (t.src / "tests") if t.src else (BUILTIN_TESTS / f"test_{t.module}.py" if t.module else None)
    if tests is None or not tests.exists():
        return False, f"no tests for {t.name} (§9)"
    env = dict(os.environ)
    if t.src:
        env["DREAMKIT_PLUGIN_PATH"] = os.pathsep.join(filter(None, [str(t.src), env.get("DREAMKIT_PLUGIN_PATH")]))
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", str(tests), "-p", "no:cacheprovider"],
                          capture_output=True, text=True, env=env, cwd=str(t.src or ROOT))
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


# -- bench -----------------------------------------------------------------------------------------------------------
def bench(target: str, accuracy: bool = False) -> tuple[bool, list[str]]:
    from .testing import fixtures, replay  # noqa: PLC0415

    t = resolve(target)
    if t.cls is None:
        return False, t.problems
    names = fixtures()
    lines = [f"{'fixture':<22} {'ms/step':>8} {'p95':>7} {'max':>7}  errors" + ("  verdict" if accuracy else "")]
    ok, right, worst = True, 0, 0.0
    plugins = [*t.cls.manifest.requires, t.name]
    if accuracy:
        plugins = ["camera-check", "success-check", t.name]  # as the fixtures were recorded
    for fx in names:
        with plugin_path(t.src):
            r = replay(fx, plugins=plugins)
        s = r.stats.get(t.name, {})
        avg, p95, mx = s.get("ms_per_step", 0.0), s.get("on_step_p95_ms", 0.0), s.get("on_step_max_ms", 0.0)
        worst = max(worst, avg)
        errs = len(r.errors(t.name))
        ok = ok and errs == 0 and avg <= 1.0
        line = f"{fx:<22} {avg:8.3f} {p95:7.3f} {mx:7.2f}  {errs}"
        if accuracy:
            want = r.expected.get("reason") if r.expected.get("success") is False else None
            got = r.result.get("reason") if r.result.get("success") is False else None
            hit = want == got
            right += hit
            line += f"  {'✓' if hit else '✗'} expected {want or 'success'}, got {got or 'success'}"
        lines.append(line)
    lines.append(f"budget 1 ms/step: {'✓' if worst <= 1.0 else '✗'} worst average {worst:.3f} ms")
    if accuracy:
        lines.append(f"accuracy: {right}/{len(names)} fixtures")
        ok = ok and right == len(names)
    return ok, lines


# -- install (§11: nothing unchecked gets in) --------------------------------------------------------------------------
def _fetch(target: str) -> tuple[Path | None, Path | None, str]:
    """A local copy of the plugin's source: (folder, tmp dir to delete, how)."""
    entries, _ = registry.load(None)
    path = Path(target).expanduser()
    if path.exists():
        return path.resolve(), None, f"local {path}"
    spec = registry.resolve(target, entries)
    m = re.match(r"git\+(https?://[^@#]+?)(?:@([^#]+))?(?:#subdirectory=(.+))?$", spec)
    if not m:
        return None, None, f"cannot fetch {spec!r}: give a GitHub URL, owner/repo, a catalogue name or a folder"
    url, ref, sub = m.groups()
    tmp = Path(tempfile.mkdtemp(prefix="dreamkit-install-"))
    cmd = ["git", "clone", "--depth", "1", *(["--branch", ref] if ref else []), url, str(tmp / "repo")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, tmp, f"git clone failed: {proc.stderr.strip()[-500:]}"
    return (tmp / "repo" / sub if sub else tmp / "repo"), tmp, f"cloned {url}{'@' + ref if ref else ''}"


def install(target: str) -> tuple[bool, str]:
    """check + tests + bench on the source first; pip install only if all pass; uninstall if it then fails to load."""
    log: list[str] = [f"$ dreamkit install {target}   ({time.strftime('%Y-%m-%d %H:%M:%S')})"]
    ok = False
    if registry.is_single_file(target):
        result = registry.install_file(target)
        log.append(result.stdout + result.stderr)
        ok = result.returncode == 0
        if ok:
            from .manager import discover  # noqa: PLC0415

            _, problems = discover()
            bad = [p for p in problems if Path(registry.raw_url(target)).name in p]
            if bad:
                ok = False
                Path(result.stdout.split("saved ")[-1].strip()).unlink(missing_ok=True)
                log += ["✗ refused by the plugin standard, removed:", *bad]
        return ok, _save_log(log, ok)
    src, tmp, how = _fetch(target)
    log.append(how)
    try:
        if src is None:
            return False, _save_log(log, False)
        passed, lines = check(str(src))
        log += ["-- check", *lines]
        if not passed:
            return False, _save_log(log + ["✗ not installed: dreamkit check failed"], False)
        passed, out = test(str(src))
        log += ["-- tests", out]
        if not passed:
            return False, _save_log(log + ["✗ not installed: the plugin's tests failed"], False)
        passed, lines = bench(str(src))
        log += ["-- bench", *lines]
        if not passed:
            return False, _save_log(log + ["✗ not installed: over the 1 ms/step budget or errors on the fixtures"], False)
        proc = subprocess.run(registry.pip_command("install", str(src)), capture_output=True, text=True)
        log += ["-- pip install", (proc.stdout + proc.stderr).strip()[-1500:]]
        if proc.returncode != 0:
            return False, _save_log(log + ["✗ pip install failed"], False)
        name = resolve(str(src)).name
        from .manager import discover  # noqa: PLC0415

        found, problems = discover()
        if name not in found or found[name]["source"] != "pip":
            dist = _dist_name(src)
            subprocess.run(registry.pip_command("uninstall", *(["-y"] if "uv" not in registry.pip_command()[0] else []),
                                                dist), capture_output=True, text=True)
            return False, _save_log(log + [f"✗ installed but did not load ({problems}); rolled back"], False)
        ok = True
        log.append(f"✓ {name} installed and checked — switch it on in the panel or dreamkit.yaml")
        return True, _save_log(log, True)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def _dist_name(src: Path) -> str:
    import tomllib  # noqa: PLC0415

    return tomllib.loads((src / "pyproject.toml").read_text())["project"]["name"]


def _save_log(lines: list[str], ok: bool) -> str:
    text = "\n".join(lines)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(text + f"\n=> {'installed' if ok else 'refused'}\n\n")
    return text


# -- publish ---------------------------------------------------------------------------------------------------------
def publish(path: str) -> tuple[bool, str]:
    out = []
    for step, fn in (("check", lambda: check(path)), ("test", lambda: test(path)), ("bench", lambda: bench(path))):
        passed, res = fn()
        out += [f"-- {step}", res if isinstance(res, str) else "\n".join(res)]
        if not passed:
            return False, "\n".join(out + [f"✗ {step} failed: fix it before publishing"])
    t = resolve(path)
    entry = (f"- name: {t.name}\n  repo: https://github.com/<you>/dreamkit-{t.name}\n"
             f"  description: {t.cls.description()}\n  group: {t.cls.manifest.group}\n  icon: {t.cls.manifest.icon}")
    out += ["✓ ready to publish. Push the repo (its GitHub Action runs check + test + bench), then add to registry.yaml:",
            entry, "and open a pull request on github.com/chlorei/hackathon26."]
    return True, "\n".join(out)


__all__ = ["check", "test", "bench", "install", "publish", "resolve", "params_table", "PluginStandardError"]
