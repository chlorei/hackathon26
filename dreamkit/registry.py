"""The plugin catalogue (registry.yaml, read from GitHub raw or the local file) and installation through pip."""

from __future__ import annotations

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


def resolve(target: str, entries: list[dict[str, Any]]) -> str:
    """A catalogue name, a GitHub URL, a git+ URL or a local path -> what pip installs."""
    entry = next((e for e in entries if e["name"] == target), None)
    if entry is not None:
        target = entry.get("pip") or entry["repo"]
    if Path(target).expanduser().exists():
        return str(Path(target).expanduser().resolve())
    if target.startswith(("https://github.com/", "http://github.com/")):
        return "git+" + target.removesuffix("/")
    return target


def install(target: str, entries: list[dict[str, Any]]) -> subprocess.CompletedProcess:
    spec = resolve(target, entries)
    return subprocess.run(pip_command("install", spec), capture_output=True, text=True)
