"""What the panel's Live page reads: runs/<id>/live/<cam>.jpg (latest frames) and runs/<id>/live.json (heartbeat,
current attempt, telemetry). Files, not sockets: the user's loop never depends on the panel being up."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

HEARTBEAT_S = 0.5


class LiveWriter:
    def __init__(self, run_dir: Path, fps: float = 8.0, size: tuple[int, int] = (384, 216)):
        self.dir = run_dir / "live"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.status = run_dir / "live.json"
        self.period = 1.0 / max(fps, 0.5)
        self.size = size
        self._last_frames = 0.0
        self._last_beat = 0.0
        self.points: list[dict[str, Any]] = []  # telemetry since the last heartbeat

    def frames(self, frames: dict[str, np.ndarray], force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_frames < self.period:
            return
        self._last_frames = now
        for cam, rgb in frames.items():
            tmp = self.dir / f".{cam}.jpg"
            Image.fromarray(rgb).resize(self.size, Image.BILINEAR).save(tmp, format="JPEG", quality=72)
            os.replace(tmp, self.dir / f"{cam}.jpg")

    def beat(self, state: dict[str, Any], telemetry: dict[str, float] | None = None, t: float | None = None,
             force: bool = False) -> None:
        if telemetry and t is not None:
            if not self.points or t - self.points[-1]["t"] >= 0.15:
                self.points.append({"t": round(t, 2), "v": dict(telemetry)})
        now = time.monotonic()
        if not force and now - self._last_beat < HEARTBEAT_S:
            return
        self._last_beat = now
        record = {**state, "updated": time.time(), "points": self.points}
        self.points = []
        tmp = self.status.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, default=float))
        os.replace(tmp, self.status)
