"""The panel's Create tab: a plugin described in words -> a package a model writes -> the same filter as
`dreamkit install` (check + test + bench) -> the catalogue (plugins-generated/<name>/) or a .zip.

What a model writes is not trusted. Before anything runs it must: be exactly the five files of a package, import only
from a short list, call nothing that reaches outside (open, eval, subprocess, sleep, …), be an observe plugin in the
diagnosis or safety group. Then check, test and bench run in a separate process with a time limit, with no API keys
in its environment. Problems go back to the model, three tries at most.
"""

from __future__ import annotations

import ast
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import llm, reasons

DOC = Path(__file__).resolve().parent.parent / "docs" / "PLUGIN_STANDARD.md"
EXAMPLE = Path(__file__).resolve().parent.parent / "plugins-external" / "gripper-check"
# the most reliable first (3.6-flash answered every time in 30-40 s); a model out of its daily quota answers 429 at
# once, a hung one is cut at CALL_TIMEOUT_S, and the next one is asked
GEMINI_MODELS = ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite",
                 "gemini-3.1-flash-lite"]
CLAUDE_MODEL = "claude-opus-5-5"  # when no Gemini model answers and ANTHROPIC_API_KEY is set
MAX_TRIES = 3  # a package that fails the filter goes back to the model with the problems this many times in all
MAX_REASKS = 2  # a reply that is not JSON is asked again (not counted as a try)
CALL_TIMEOUT_S, ANSWER_DEADLINE_S, STEP_TIMEOUT_S = 60.0, 240.0, 300.0  # one model call; all models for one answer
NAME = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+){0,2}$")
GROUPS = ("diagnosis", "safety")
SECTIONS = ("## 2.", "## 4.", "## 5.", "## 6.", "## 7. ", "## 8.", "## 9.")  # of PLUGIN_STANDARD.md, for the prompt

ALLOWED_IMPORTS = {"__future__", "math", "statistics", "collections", "dataclasses", "typing", "enum", "itertools",
                   "functools", "json", "re", "numpy", "pydantic", "dreamkit", "pytest"}
BANNED_CALLS = {"open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "breakpoint", "input",
                "getattr", "setattr", "delattr", "memoryview"}
BANNED_ATTRS = {"parent", "parents", "unlink", "rmdir", "rename", "symlink_to", "hardlink_to", "chmod", "chown", "home",
                "cwd", "system", "popen", "sleep", "fork", "spawn", "kill", "remove", "rmtree", "glob", "rglob",
                "iterdir", "expanduser", "environ", "getenv", "mkdir", "touch"}
ALLOWED_DUNDERS = {"__init__", "__name__", "__all__", "__version__"}


# -- the prompt -------------------------------------------------------------------------------------------------------
def _digest() -> str:
    """The sections of PLUGIN_STANDARD.md a plugin author needs, verbatim (so the prompt never drifts from the doc)."""
    out, keep = [], False
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            keep = line.startswith(SECTIONS)
        if keep:
            out.append(line)
    return "\n".join(out)


def strip_ui(text: str) -> str:
    """The example without its page: a generated plugin has none (the server gives it the standard one)."""
    return re.sub(r',\s*ui="ui/index\.html"', "", re.sub(r'\nui = "ui/index\.html"', "", text))


def _example() -> str:
    files = ["pyproject.toml", "gripper_check/__init__.py", "gripper_check/plugin.py",
             "tests/test_gripper_check.py", "README.md"]
    return json.dumps({"name": "gripper-check", "summary": "Catches a gripper closed on nothing and a grip that "
                       "loosens mid-carry.", "files": {f: strip_ui((EXAMPLE / f).read_text(encoding="utf-8"))
                                                       for f in files}}, indent=1)


def _fixtures() -> list[str]:
    from .testing import fixtures  # noqa: PLC0415

    return [Path(f).name if isinstance(f, (str, Path)) else str(f) for f in fixtures()]


def build_prompt(messages: list[dict[str, str]], files: dict[str, str] | None, problems: list[str] | None) -> str:
    wishes = "\n".join(f"- {m['text']}" for m in messages if m.get("role") == "user")
    parts = [
        "You write ONE plugin package for dreamkit, a plugin library that runs inside a robot's control loop. "
        "The user describes what it should watch; you return the whole package.",
        "## What the user wants (in order; later messages refine earlier ones)\n" + wishes,
        "## Hard rules (a package that breaks one is refused before it runs)\n"
        "- capability = \"observe\" (it only watches and reports); group = \"diagnosis\" or \"safety\".\n"
        "- needs = [\"robot\"] unless it truly needs object positions; then [\"robot\", \"world\"] (simulator only).\n"
        "- Imports only from: " + ", ".join(sorted(ALLOWED_IMPORTS)) + " (and the package itself).\n"
        "- No network, no subprocess, no time.sleep, no open(), no eval/exec, no getattr/setattr, no os/sys/pathlib, "
        "no dunder attributes. A file, if any, only via ctx.files.path(\"name.json\").write_text(...) "
        "(that is run_dir/<plugin>/).\n"
        "- The robot: holding an object keeps a gripper's measured opening well above 0 (an apple: about 0.7), so "
        "whether a gripper is shut or open comes from the command, ctx.robot.grip_command(side) (below 0.5 = "
        "closing), not ctx.robot.grip(side). Joint speeds in normal motion stay under about 1 rad/s.\n"
        "- on_step must stay well under 1 ms (no loops over history, no big arrays); step_rate_hz between 2 and 20.\n"
        "- Feed messages: lowercase fragments, no trailing period, with numbers (\"right hand at 0.42 m/s near "
        "the bowl\"); at most one per second per kind; add ctx.ui.mark(ctx.t, message, level) for the same moment. "
        "Whatever the plugin reports, live or at the end of an attempt, goes to the feed with ctx.emit: a badge or "
        "ctx.result alone is not seen in Live.\n"
        "- Params: every field has a description and json_schema_extra={\"unit\": ...}; names end in their unit "
        "(_m, _s, _m_s, _rad_s, _hz …) where they have one; sensible ge/le bounds.\n"
        "- pyproject.toml: name \"dreamkit-<name>\", dependencies [\"dreamkit\"], the entry point, and "
        "[tool.dreamkit] with the same group/icon/capability/hooks/step_rate_hz/needs as the class manifest.\n"
        "- tests/test_<module>.py: replay tests only (dreamkit.testing.replay), one test per fixture for "
        "success_apple_bowl, dropped_mid_carry and not_reached, each asserting exactly `not r.errors(\"<name>\")` "
        "and `r.budget_ms(\"<name>\") < 1.0`. Do not assert that an event fires: the fixtures are recordings and "
        "you cannot know what they contain.\n"
        "- README.md: what it watches, the feed lines it writes, its settings.\n"
        "- No `ui` in the manifest or pyproject: the panel gives every plugin a standard page of its own.",
        "## Replay fixtures available to tests\n" + ", ".join(_fixtures()),
        "## Failure codes (dreamkit.reasons), if the plugin names a failure\n" + ", ".join(reasons.FAILURES),
        "## The plugin standard (excerpts of PLUGIN_STANDARD.md)\n" + _digest(),
        "## A complete, passing example package, in the answer format\n" + _example(),
    ]
    if files:
        parts.append("## The current package (change it as the latest message asks; keep the rest)\n"
                     + json.dumps({"files": files}, indent=1))
    if problems:
        parts.append("## Your last package was refused. Fix exactly these problems:\n" + "\n".join(problems))
    parts.append("## Answer\nReturn ONLY one JSON object, no markdown: {\"name\": \"<kebab-case, 1-3 words>\", "
                 "\"summary\": \"<one sentence>\", \"files\": {\"pyproject.toml\": \"...\", \"<module>/__init__.py\": "
                 "\"...\", \"<module>/plugin.py\": \"...\", \"tests/test_<module>.py\": \"...\", \"README.md\": "
                 "\"...\"}} where <module> is the name in snake_case.")
    return "\n\n".join(parts)


def ask(prompt: str) -> tuple[str, str]:
    """(text, model). Tests replace it with a fake model."""
    llm.load_env()
    deadline = time.monotonic() + ANSWER_DEADLINE_S  # a hung model (504) must leave the next ones time to answer
    try:
        return llm.gemini([], prompt, GEMINI_MODELS, CALL_TIMEOUT_S, deadline)
    except llm.Unavailable:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise
        return llm.claude([], prompt, CLAUDE_MODEL, CALL_TIMEOUT_S, time.monotonic() + CALL_TIMEOUT_S), CLAUDE_MODEL


# -- what comes back --------------------------------------------------------------------------------------------------
def parse(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        raise ValueError('the answer must be a JSON object with "name", "summary" and "files"')
    return data


def validate(data: dict[str, Any], taken: set[str]) -> list[str]:
    """The package's shape and its code, before anything of it runs. [] when it may be checked."""
    name = str(data.get("name") or "")
    if not NAME.match(name):
        return [f"name {name!r} must be kebab-case, 1-3 words (e.g. speed-near-bowl)"]
    if name in taken:
        return [f"name {name!r} is taken by an installed plugin; choose another"]
    module = name.replace("-", "_")
    files: dict[str, Any] = data["files"]
    want = {"pyproject.toml", "README.md", f"{module}/__init__.py", f"{module}/plugin.py", f"tests/test_{module}.py"}
    problems = [f"missing file {f}" for f in sorted(want - set(files))]
    problems += [f"unexpected file {f} (only {', '.join(sorted(want))})" for f in sorted(set(files) - want)]
    problems += [f"{f} must be text under 60 kB" for f, c in files.items() if not isinstance(c, str) or len(c) > 60_000]
    if problems:
        return problems
    for f in sorted(want):
        if f.endswith(".py"):
            problems += _scan(f, files[f], module)
    try:
        tool = tomllib.loads(files["pyproject.toml"]).get("tool", {}).get("dreamkit", {})
    except tomllib.TOMLDecodeError as error:
        return problems + [f"pyproject.toml: {error}"]
    if tool.get("capability") != "observe":
        problems.append('pyproject.toml [tool.dreamkit] capability must be "observe"')
    if tool.get("group") not in GROUPS:
        problems.append(f"pyproject.toml [tool.dreamkit] group must be one of {', '.join(GROUPS)}")
    return problems


def _scan(path: str, code: str, module: str) -> list[str]:
    try:
        tree = ast.parse(code, filename=path)
    except SyntaxError as error:
        return [f"{path}:{error.lineno}: syntax error: {error.msg}"]
    out = []
    for node in ast.walk(tree):
        where = f"{path}:{getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.Import):
            out += [f"{where}: import {a.name} is not allowed" for a in node.names
                    if a.name.split(".")[0] not in ALLOWED_IMPORTS | {module}]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and (node.module or "").split(".")[0] not in ALLOWED_IMPORTS | {module}:
                out.append(f"{where}: from {node.module} import is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            out.append(f"{where}: {node.func.id}() is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS:
                out.append(f"{where}: .{node.attr} is not allowed")
            elif node.attr.startswith("__") and node.attr not in ALLOWED_DUNDERS:
                out.append(f"{where}: .{node.attr} is not allowed")
        elif isinstance(node, ast.Name) and node.id.startswith("__") and node.id not in ALLOWED_DUNDERS:
            out.append(f"{where}: {node.id} is not allowed")
        elif isinstance(node, ast.keyword) and node.arg == "capability":
            if not (isinstance(node.value, ast.Constant) and node.value.value == "observe"):
                out.append(f'{where}: capability must be "observe"')
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and re.search(r"https?://", node.value):
            out.append(f"{where}: no URLs")
    return out


# -- the filter, in a separate process ----------------------------------------------------------------------------------
def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY") and k != "DREAM_API_KEY"}
    env["COLUMNS"] = "240"  # one problem per line, not wrapped (what the model and the chat see)
    env.update(extra)
    return env


def _run(*args: str, cwd: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run([sys.executable, "-c", "import sys; from dreamkit.cli import main; sys.exit(main())",
                               *args], capture_output=True, text=True, cwd=str(cwd), env=_env(), timeout=STEP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {STEP_TIMEOUT_S:.0f} s"
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


_DESCRIBE = """
import json, sys
from dreamkit.pdk import resolve
t = resolve(sys.argv[1])
m = t.cls.manifest
print(json.dumps({"name": t.name, "description": t.cls.description(), "hooks": m.hooks, "group": m.group,
                  "icon": m.icon, "needs": m.needs, "step_rate_hz": m.step_rate_hz,
                  "params": t.cls.Params.model_json_schema(), "defaults": t.cls.Params().model_dump(mode="json")}))
"""


def _describe(package: Path) -> dict[str, Any]:
    proc = subprocess.run([sys.executable, "-c", _DESCRIBE, str(package)], capture_output=True, text=True,
                          cwd=str(package), env=_env(DREAMKIT_PLUGIN_PATH=str(package)), timeout=60)
    return json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 and proc.stdout.strip() else {}


# Three recordings people recognise: one that works, one dropped, one grasping air
TRYOUT_FIXTURES = ["success_apple_bowl", "dropped_mid_carry", "grasped_air"]
_TRYOUT = """
import json, sys
from dreamkit.testing import replay
name, out = sys.argv[1], []
for fx in sys.argv[2:]:
    r = replay(fx, [name])
    said = [{"t": e.get("t_s", e.get("t")), "level": e.get("level"), "message": e.get("message")}
            for e in r.events if e.get("plugin") == name]
    out.append({"fixture": fx, "success": bool(r.expected.get("success")), "said": said[:8], "more": max(0, len(said) - 8)})
print(json.dumps(out))
"""


def _tryout(package: Path, name: str) -> list[dict[str, Any]]:
    """What the plugin says on a few recorded attempts, before anyone reads its code. Never fails the job."""
    try:
        proc = subprocess.run([sys.executable, "-c", _TRYOUT, name, *TRYOUT_FIXTURES], capture_output=True, text=True,
                              cwd=str(package), env=_env(DREAMKIT_PLUGIN_PATH=str(package.resolve())), timeout=STEP_TIMEOUT_S)
        return json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 and proc.stdout.strip() else []
    except (subprocess.TimeoutExpired, ValueError):
        return []


def _tail(text: str, n: int = 25) -> str:
    return "\n".join(text.splitlines()[-n:])


# -- a job ------------------------------------------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    messages: list[dict[str, str]]
    status: str = "running"  # running | ready | failed
    steps: list[dict[str, Any]] = field(default_factory=list)
    name: str | None = None
    summary: str = ""
    files: dict[str, str] = field(default_factory=dict)
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    tries: int = 0
    error: str | None = None  # one sentence for the chat
    details: str = ""  # the checks' output behind it
    tryout: list[dict[str, Any]] = field(default_factory=list)  # what it says on recorded attempts
    package: Path | None = None

    def step(self, text: str) -> None:
        self.steps.append({"t": round(time.time(), 2), "text": text})

    def public(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k not in ("package", "messages")}


JOBS: dict[str, Job] = {}


def start(messages: list[dict[str, str]], files: dict[str, str] | None, taken: set[str]) -> Job:
    job = Job(uuid.uuid4().hex[:12], messages)
    JOBS[job.id] = job
    threading.Thread(target=run, args=(job, files, taken), name=f"create-{job.id}", daemon=True).start()
    return job


def run(job: Job, files: dict[str, str] | None, taken: set[str]) -> Job:
    problems: list[str] | None = None
    work = Path(tempfile.mkdtemp(prefix="dreamkit-create-"))
    try:
        for job.tries in range(1, MAX_TRIES + 1):
            job.step("generating" if job.tries == 1 else f"fixing: {problems[0][:160] if problems else ''}")
            job.checks = {}
            data = _answer(job, build_prompt(job.messages, files, problems))
            problems = validate(data, taken)
            if not problems:
                job.step("checking")
                package = _write(work / f"try{job.tries}" / data["name"], data["files"])
                problems = _checks(job, package)
                if not problems:
                    job.name, job.summary, job.files, job.package = (data["name"], str(data.get("summary", "")),
                                                                     data["files"], package)
                    job.info = _describe(package)
                    job.step("trying it on recorded attempts")
                    job.tryout = _tryout(package, data["name"])
                    job.status = "ready"
                    job.step("ready")
                    return job
            files = data["files"]  # the next try fixes this package
        first = (problems or ["unknown problem"])[0]
        job.status = "failed"
        job.error = (f"generated plugin failed {first.removeprefix('dreamkit ')} after {MAX_TRIES} tries"
                     if first.startswith("dreamkit ") else f"generated plugin was refused: {first}")[:2000]
        job.details = "\n".join(problems or [])[:4000]
        job.step(f"failed after {MAX_TRIES} tries")
    except Exception as error:  # noqa: BLE001 - shown in the chat, never raised into the server
        job.status, job.error = "failed", friendly(error)[:2000]
        job.step(f"failed: {job.error[:200]}")
    return job


def friendly(error: BaseException) -> str:
    """What went wrong, as the chat says it: the cause and what to do, not a traceback."""
    text = str(error)
    if isinstance(error, llm.Unavailable) and error.quota:
        return "Gemini quota exhausted for today (free tier: 20 requests a model a day), try later" + \
               ("" if os.getenv("ANTHROPIC_API_KEY") else " or add ANTHROPIC_API_KEY to .env to use Claude")
    if isinstance(error, llm.Unavailable):
        return f"Gemini did not answer in time (models busy or overloaded), try again in a minute ({text})"
    if "GEMINI_API_KEY" in text:
        return "no GEMINI_API_KEY: add it to .env in the project folder"
    if isinstance(error, ValueError) and "valid JSON" in text:
        return f"the model's answer was not a plugin package ({text})"
    return f"{type(error).__name__}: {text}"


def _answer(job: Job, prompt: str) -> dict[str, Any]:
    error = None
    for _ in range(1 + MAX_REASKS):
        text, job.model = ask(prompt if error is None else
                              f"{prompt}\n\nYour last reply was not valid: {error}. Return valid JSON only.")
        try:
            return parse(text)
        except (ValueError, TypeError) as e:  # JSONDecodeError is a ValueError
            error = str(e)[:300]
            job.step(f"reply was not valid JSON, asking again ({error[:80]})")
    raise ValueError(f"the model gave no valid JSON in {1 + MAX_REASKS} replies: {error}")


def _write(package: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        target = (package / rel).resolve()
        if package.resolve() not in target.parents:
            raise ValueError(f"{rel} is outside the package")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return package


def _checks(job: Job, package: Path) -> list[str]:
    for step in ("check", "test", "bench"):
        ok, out = _run(step, str(package), cwd=package)
        job.checks[step] = {"ok": ok, "out": _tail(out, 40)}
        if not ok:
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            first = next((ln for ln in lines if ln.startswith(("✗", "E ", "FAILED"))), lines[-1] if lines else "")
            return [f"dreamkit {step} failed: {first[:200]}", _tail(out)]
    return []


# -- what the user does with it -----------------------------------------------------------------------------------------
def add_to_catalog(job: Job, base: Path) -> Path:
    if job.status != "ready" or job.package is None or not job.name:
        raise ValueError("this plugin is not ready")
    dest = base / "plugins-generated" / job.name
    if dest.exists():
        shutil.rmtree(dest)  # the same generated plugin, regenerated
    shutil.copytree(job.package, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return dest


def zip_bytes(job: Job) -> bytes:
    if job.status != "ready" or not job.name:
        raise ValueError("this plugin is not ready")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, text in sorted(job.files.items()):
            z.writestr(f"{job.name}/{rel}", text)
    return buf.getvalue()
