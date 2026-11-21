"""Events: one JSON object per line on stdout (--json-events, read by the server), events.jsonl, and a rich console."""

from __future__ import annotations

import base64
import io
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

STYLE = {"info": "cyan", "warn": "yellow", "error": "red", "success": "green"}
ICON = {"info": "·", "warn": "⚠", "error": "✗", "success": "✓"}


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return round(f, 4) if np.isfinite(f) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


class Emitter:
    """Fans every record out to stdout JSON (for the server), a file, memory and/or the console."""

    def __init__(self, json_events: bool = False, console: bool = True, frame_fps: float = 5.0):
        self.json_events = json_events
        self.records: list[dict[str, Any]] = []
        self._file = None
        self._lock = threading.Lock()
        self._console = None
        if console and not json_events:
            from rich.console import Console

            self._console = Console(highlight=False)
        self.frame_period = 1.0 / frame_fps if frame_fps > 0 else float("inf")
        self._last_frame = 0.0

    def open_file(self, path: Path) -> None:
        self._file = path.open("a", buffering=1)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _write(self, record: dict[str, Any], keep: bool = True) -> None:
        record = _plain(record)
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            if keep:
                self.records.append(record)
                if self._file is not None:
                    self._file.write(line + "\n")
            if self.json_events:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()

    def emit(self, type_: str, **payload: Any) -> None:
        self._write({"type": type_, "ts": round(time.time(), 3), **payload})
        if self._console is not None:
            self._print_other(type_, payload)

    def event(self, plugin: str, level: str, message: str, **data: Any) -> None:
        episode = data.pop("episode", None)
        t = data.pop("t", None)
        record = {
            "type": "event",
            "ts": round(time.time(), 3),
            "plugin": plugin,
            "level": level,
            "message": message,
            "episode": episode,
            "t": t,
            "data": data,
        }
        self._write(record)
        if self._console is not None:
            style = STYLE.get(level, "white")
            when = f"ep{episode} {t:6.2f}s" if episode is not None and t is not None else " " * 12
            self._console.print(
                f"[dim]{when}[/dim] [{style}]{ICON.get(level, '·')} {plugin:<16}[/{style}] {message}", markup=True
            )

    def frames(self, frames: dict[str, np.ndarray], episode: int, t: float, force: bool = False) -> None:
        """Live camera frames for the panel, throttled to frame_fps of wall time; only in --json-events mode."""
        if not self.json_events:
            return
        now = time.monotonic()
        if not force and now - self._last_frame < self.frame_period:
            return
        self._last_frame = now
        for cam, rgb in frames.items():
            buffer = io.BytesIO()
            Image.fromarray(rgb).resize((384, 216), Image.BILINEAR).save(buffer, format="JPEG", quality=70)
            self._write(
                {"type": "frame", "cam": cam, "episode": episode, "t": round(t, 2),
                 "jpeg": base64.b64encode(buffer.getvalue()).decode("ascii")},
                keep=False,
            )

    def _print_other(self, type_: str, payload: dict[str, Any]) -> None:
        c = self._console
        if type_ == "run_start":
            c.rule(f"[bold]dreamkit run {payload.get('run_id')}[/bold]")
            c.print(f"task: [bold]{payload.get('task')}[/bold]   model: {payload.get('model')}"
                    f"{'  [magenta](mock)[/magenta]' if payload.get('mock') else ''}")
            c.print(f"plugins: {', '.join(payload.get('plugins', []))}")
        elif type_ == "episode_start":
            c.rule(f"attempt {payload.get('episode')} · scene {payload.get('scene')} · try {payload.get('attempt')}",
                   style="dim")
        elif type_ == "episode_end":
            r = payload.get("result", {})
            ok = r.get("success")
            mark = "[green]✓ success[/green]" if ok else ("[red]✗ fail[/red]" if ok is False else "[dim]? unknown[/dim]")
            why = "" if ok else f"  reason: [bold]{r.get('reason')}[/bold]" + (
                f" at {r.get('at'):.1f}s" if isinstance(r.get("at"), (int, float)) else "")
            c.print(f"{mark}{why}  ({payload.get('duration_s', 0):.1f}s sim)")
        elif type_ == "run_end":
            c.rule(f"[bold]{payload.get('successes')}/{payload.get('attempts')} succeeded[/bold]"
                   f" · {payload.get('status')} · {payload.get('run_dir')}")
