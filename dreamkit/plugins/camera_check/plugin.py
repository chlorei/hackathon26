"""camera-check: a model fed a dead camera acts blind — report it before the next model call (PLUGIN_STANDARD.md).

Observe-only: it calls a `camera_failure` verdict; the control plugin (recovery) stops the robot."""

from __future__ import annotations

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
                        hooks=["on_episode_start", "before_model"])

    def on_episode_start(self, ctx: PluginContext) -> None:
        self.previous: dict[str, np.ndarray] = {}
        self.same: dict[str, int] = {}
        self.reported = False
        self.last_q: dict[str, np.ndarray] | None = None

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
                if self.same[cam] >= p.frozen_calls:
                    problems.append(f"camera {cam} is frozen ({self.same[cam]} identical pictures)")
        if problems:
            self.reported = True
            ctx.result["camera-check"] = {"problems": problems}
            ctx.ui.badge("camera failure", "error")
            ctx.verdict("camera_failure", ctx.t, "; ".join(problems))
