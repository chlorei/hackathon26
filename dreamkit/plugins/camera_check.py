"""camera-check: stops the robot before asking the model when a camera is black, flat or frozen."""

from __future__ import annotations

import numpy as np

from ..plugin import Context, Plugin


class CameraCheck(Plugin):
    """A model fed a dead camera acts blind; better stop than guess."""

    name = "camera-check"
    group = "safety"
    description = "Stops the attempt when a camera is black, uniform or frozen"
    defaults = {
        "min_mean": 10.0,
        "min_std": 3.0,
        "frozen_calls": 0,  # stop when a camera is pixel-identical across this many model calls (0 = off;
        # in simulation a still scene renders identical frames, so leave it off unless the arm always moves)
        "simulate_failure": None,  # demo: a camera name (top/left/right) to black out
        "simulate_at_s": 4.0,  # ...from this second of the attempt
        "simulate_episode": None,  # ...only in this attempt (1-based); None = every attempt
    }

    def on_episode_start(self, ctx: Context) -> None:
        self.previous: dict[str, np.ndarray] = {}
        self.same: dict[str, int] = {}

    def _broken(self, ctx: Context) -> bool:
        cam = self.params["simulate_failure"]
        if not cam:
            return False
        ep = self.params["simulate_episode"]
        return ctx.t >= float(self.params["simulate_at_s"]) and (ep is None or int(ep) == ctx.episode)

    def on_step(self, ctx: Context) -> None:
        # The demo fault: the camera goes dark in what every later plugin, the panel and the recording see.
        cam = self.params["simulate_failure"]
        if self._broken(ctx) and cam in ctx.frames and ctx.frames_new:
            ctx.frames[cam] = (ctx.frames[cam] * 0.03).astype(np.uint8)

    def before_model(self, ctx: Context) -> None:
        problems = []
        for cam, rgb in ctx.frames.items():
            mean, std = float(rgb.mean()), float(rgb.std())
            if mean < self.params["min_mean"]:
                problems.append(f"camera {cam} is black (mean {mean:.1f})")
            elif std < self.params["min_std"]:
                problems.append(f"camera {cam} shows a uniform image (std {std:.1f})")
            limit = int(self.params["frozen_calls"] or 0)
            if limit > 0:
                prev = self.previous.get(cam)
                self.same[cam] = self.same.get(cam, 0) + 1 if prev is not None and np.array_equal(prev, rgb) else 0
                self.previous[cam] = rgb.copy()
                if self.same[cam] >= limit:
                    problems.append(f"camera {cam} is frozen ({self.same[cam]} identical frames)")
        if problems:
            ctx.result["camera"] = problems
            ctx.stop("; ".join(problems) + " — robot stopped before the next model call")
