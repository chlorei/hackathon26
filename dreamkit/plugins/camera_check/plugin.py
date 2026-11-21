"""camera-check: a model fed a dead camera acts blind — report it before the next model call (PLUGIN_STANDARD.md).

Observe-only: it calls a `camera_failure` verdict; the control plugin (recovery) stops the robot."""

from __future__ import annotations

import json

import numpy as np
from pydantic import BaseModel, Field

from dreamkit.plugin import Plugin, PluginContext
from dreamkit.standard import Manifest


class Params(BaseModel):
    min_mean: float = Field(10.0, ge=0, le=255, description="A picture darker than this on average is black",
                            json_schema_extra={"unit": ""})
    min_std: float = Field(3.0, ge=0, le=128, description="A picture with less contrast than this is uniform",
                           json_schema_extra={"unit": ""})
    frozen_calls: int = Field(3, ge=0, le=1000, description="Identical pictures across this many model calls while "
                              "the arms moved is frozen (0 = off)", json_schema_extra={"unit": ""})
    moved_rad: float = Field(0.02, gt=0, le=1, description="A joint moving this much between calls counts as motion",
                             json_schema_extra={"unit": "rad"})


class CameraCheck(Plugin):
    """Flags a black, uniform or frozen camera before the model is asked again."""

    name = "camera-check"
    Params = Params
    manifest = Manifest(group="safety", icon="Camera", capability="observe", needs_frames=True,
                        hooks=["on_episode_start", "before_model", "on_episode_end"], ui="ui/index.html")

    def on_episode_start(self, ctx: PluginContext) -> None:
        self.previous: dict[str, np.ndarray] = {}
        self.same: dict[str, int] = {}
        self.reported = False
        self.last_q: dict[str, np.ndarray] | None = None
        self.seen: dict[str, dict[str, float]] = {}  # per camera, for the page only (run_dir/camera-check/)
        self.shots: dict[str, np.ndarray] = {}  # the picture that failed, small, for the page only

    def before_model(self, ctx: PluginContext) -> None:
        if self.reported:
            return
        p = self.params
        problems = []
        q = {s: np.asarray(ctx.state[s][:6], dtype=float) for s in ("left", "right")} if ctx.state else None
        arm_moved = {s: q is not None and self.last_q is not None
                     and float(np.max(np.abs(q[s] - self.last_q[s]))) > p.moved_rad for s in ("left", "right")}
        self.last_q = q
        for cam, rgb in ctx.frames.items():
            mean, std = float(rgb.mean()), float(rgb.std())
            seen = self.seen.setdefault(cam, {"checks": 0, "min_mean": mean, "min_std": std, "max_same": 0})
            seen["checks"] += 1
            seen["min_mean"], seen["min_std"] = min(seen["min_mean"], mean), min(seen["min_std"], std)
            if mean < p.min_mean:
                problems.append(f"camera {cam} is black (mean {mean:.1f})")
            elif std < p.min_std:
                problems.append(f"camera {cam} shows a uniform picture (std {std:.1f})")
            if p.frozen_calls > 0:
                prev = self.previous.get(cam)
                if prev is not None and np.array_equal(prev, rgb):
                    # a wrist camera rides its own arm; any other camera sees both. Still arm, still picture: fine.
                    moved = arm_moved[cam] if cam in arm_moved else any(arm_moved.values())
                    if moved:
                        self.same[cam] = self.same.get(cam, 0) + 1
                else:
                    self.same[cam] = 0
                self.previous[cam] = rgb.copy()
                seen["max_same"] = max(seen["max_same"], self.same[cam])
                if self.same[cam] >= p.frozen_calls:
                    problems.append(f"camera {cam} is frozen ({self.same[cam]} identical pictures)")
        if problems:
            for cam, rgb in ctx.frames.items():
                if any(f"camera {cam} " in p for p in problems):
                    step = max(1, rgb.shape[1] // 320)
                    self.shots[cam] = np.array(rgb[::step, ::step])
            self.reported = True
            ctx.result["camera-check"] = {"problems": problems}
            ctx.ui.badge("camera failure", "error")
            ctx.verdict("camera_failure", ctx.t, "; ".join(problems), source="robot")  # cameras are the robot's

    def on_episode_end(self, ctx: PluginContext) -> None:
        if ctx.run_dir is None or not self.seen:
            return
        name = f"ep{ctx.episode:03d}_checks.json"  # what the page shows: how many pictures each camera passed
        shots = {cam: f"ep{ctx.episode:03d}_{cam}.jpg" for cam in self.shots}
        record = {"episode": ctx.episode, "cameras": {c: {k: round(v, 2) for k, v in d.items()} for c, d in self.seen.items()},
                  "problems": (ctx.result.get("camera-check") or {}).get("problems", []), "frames": shots}
        pictures = self.shots

        def write() -> None:
            from PIL import Image  # noqa: PLC0415 - only when a page record is written

            for cam, file in shots.items():
                Image.fromarray(np.ascontiguousarray(pictures[cam], dtype=np.uint8)).save(ctx.files.path(file), quality=85)
            ctx.files.path(name).write_text(json.dumps(record))

        ctx.defer(write)
