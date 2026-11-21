"""The plugin catalogue (registry.yaml, read from GitHub raw or the local file) and installation through pip."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

LOCAL = Path(__file__).resolve().parent.parent / "registry.yaml"


def load(url: str | None = None, timeout: float = 4.0) -> tuple[list[dict[str, Any]], str]:
    """The catalogue entries and where they came from. Falls back to the local registry.yaml when offline."""
    if url:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                data = yaml.safe_load(response.read().decode()) or []
                return [e for e in data if isinstance(e, dict) and e.get("name")], url
        except Exception:
            pass
    for path in (Path("registry.yaml"), LOCAL):
        if path.is_file():
            data = yaml.safe_load(path.read_text()) or []
            return [e for e in data if isinstance(e, dict) and e.get("name")], str(path)
    return [], "none"


def search(entries: list[dict[str, Any]], word: str) -> list[dict[str, Any]]:
    word = word.lower()
    return [e for e in entries if word in e["name"].lower() or word in str(e.get("description", "")).lower()
            or any(word in str(t).lower() for t in e.get("tags", []))]


def pip_command(*args: str) -> list[str]:
    """pip for the interpreter running dreamkit; uv's when the venv has no pip of its own."""
    try:
        import pip  # noqa: F401

        return [sys.executable, "-m", "pip", *args]
    except ImportError:
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("neither pip nor uv is available to install plugins") from None
        return [uv, "pip", *args, "--python", sys.executable]


PLUGIN_DIR = Path("~/.dreamkit/plugins").expanduser()


def resolve(target: str, entries: list[dict[str, Any]]) -> str:
    """A catalogue name, `owner/repo`, a GitHub URL (optionally `#subdirectory=...`), a git+ URL or a local path ->
    what pip installs."""
    target = target.strip()
    entry = next((e for e in entries if e["name"] == target), None)
    if entry is not None:
        target = entry.get("pip") or entry["repo"]
    if Path(target).expanduser().exists():
        return str(Path(target).expanduser().resolve())
    if re.fullmatch(r"[\w.-]+/[\w.-]+", target):
        target = f"https://github.com/{target}"
    if target.startswith(("https://github.com/", "http://github.com/")):
        base, _, fragment = target.partition("#")
        base = base.removesuffix("/")
        m = re.match(r"(https?://github\.com/[^/]+/[^/]+?)(?:\.git)?/tree/([^/]+)/(.+)$", base)
        if m:  # a folder in a repo: https://github.com/o/r/tree/main/plugins/x
            return f"git+{m.group(1)}.git@{m.group(2)}#subdirectory={m.group(3)}"
        return "git+" + base + (f"#{fragment}" if fragment else "")
    return target


def is_single_file(target: str) -> bool:
    return target.strip().split("#")[0].endswith(".py") and target.strip().startswith(("http://", "https://"))


def raw_url(url: str) -> str:
    """github.com/<o>/<r>/blob/<ref>/<path>.py -> raw.githubusercontent.com/<o>/<r>/<ref>/<path>.py"""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/(.+)$", url)
    return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else url


def install_file(url: str) -> subprocess.CompletedProcess:
    """A single-file plugin: downloaded into ~/.dreamkit/plugins/, where dreamkit loads it from."""
    src = raw_url(url.strip())
    name = Path(src.split("?")[0]).name
    try:
        with urllib.request.urlopen(src, timeout=15) as response:
            code = response.read().decode()
        if "Plugin" not in code:
            raise ValueError("this file does not define a dreamkit Plugin")
        PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
        (PLUGIN_DIR / name).write_text(code)
        return subprocess.CompletedProcess([src], 0, f"saved {PLUGIN_DIR / name}\n", "")
    except Exception as error:
        return subprocess.CompletedProcess([src], 1, "", f"{src}: {error}\n")


def install(target: str, entries: list[dict[str, Any]]) -> subprocess.CompletedProcess:
    if is_single_file(target):
        return install_file(target)
    spec = resolve(target, entries)
    return subprocess.run(pip_command("install", spec), capture_output=True, text=True)


def uninstall(name: str, found: dict[str, dict[str, Any]]) -> subprocess.CompletedProcess:
    """Removes a pip or local plugin (never a built-in)."""
    entry = found.get(name)
    if entry is None:
        return subprocess.CompletedProcess([name], 1, "", f"{name} is not installed\n")
    if entry["source"] == "builtin":
        return subprocess.CompletedProcess([name], 1, "", f"{name} is built in; switch it off instead\n")
    if entry["source"] == "local":
        Path(entry["origin"]).unlink(missing_ok=True)
        return subprocess.CompletedProcess([name], 0, f"removed {entry['origin']}\n", "")
    dist = entry["origin"].split(":")[0]
    return subprocess.run(pip_command("uninstall", "-y", dist) if "uv" not in pip_command()[0]
                          else pip_command("uninstall", dist), capture_output=True, text=True)
