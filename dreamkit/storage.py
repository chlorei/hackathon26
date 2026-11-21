"""The run folder: runs/<id>/{run.json, events.jsonl, config.yaml, ep<N>/<cam>.mp4, report.html}."""

from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def run_dir(root: Path, run_id: str) -> Path:
    path = root / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    """Atomic write, so a reader (the server) never sees half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def read_run(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads((path / "run.json").read_text())
    except (OSError, ValueError):
        return None


def list_runs(root: Path) -> list[dict[str, Any]]:
    out = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        if d.is_dir():
            info = read_run(d)
            if info is not None:
                out.append(info)
    return out


def read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    try:
        with (path / "events.jsonl").open() as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return events
