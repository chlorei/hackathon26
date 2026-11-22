"""recorder: one MP4 per camera per attempt, in the attempt's folder runs/<run>/ep003/<cam>.mp4 (PLUGIN_STANDARD.md §8).

Video time = the attempt's simulated time, so `ep003_top.mp4#t=9.3` is the moment. Encoding runs on ctx.defer."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import av
import numpy as np
from pydantic import BaseModel, Field

from dreamkit.plugin import Plugin, PluginContext
from dreamkit.standard import Manifest


class Params(BaseModel):
    video_fps: int = Field(10, ge=1, le=20, description="Frames per second written to the video",
                           json_schema_extra={"unit": "fps"})
    cameras: list[str] = Field(["top", "left", "right"], description="Cameras to record",
                               json_schema_extra={"choices": ["top", "left", "right"]})
    crf: int = Field(28, ge=0, le=51, description="Quality: lower is better and larger", json_schema_extra={"unit": ""})
    preset: str = Field("ultrafast", description="Encoder speed: faster uses less CPU",
                        json_schema_extra={"choices": ["ultrafast", "veryfast", "fast", "medium"]})


class _Writer:
    def __init__(self, path, rgb: np.ndarray, crf: int, preset: str, fps: int):
        self.container = av.open(str(path), "w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.h, self.w = rgb.shape[0] - rgb.shape[0] % 2, rgb.shape[1] - rgb.shape[1] % 2
        self.stream.height, self.stream.width = self.h, self.w
        self.stream.pix_fmt = "yuv420p"
        self.stream.codec_context.time_base = Fraction(1, 1000)
        self.stream.options = {"preset": preset, "crf": str(crf)}
        self.last_pts = -1

    def add(self, rgb: np.ndarray, pts: int) -> None:
        if pts <= self.last_pts:
            return
        self.last_pts = pts
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb[: self.h, : self.w]), format="rgb24")
        frame.pts, frame.time_base = pts, Fraction(1, 1000)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


class Recorder(Plugin):
    """Records every camera of each attempt to MP4, timed in simulated seconds."""

    name = "recorder"
    Params = Params
    manifest = Manifest(group="output", icon="Film", capability="observe", needs_frames=True,
                        hooks=["on_episode_start", "on_step", "on_episode_end"], ui="ui/index.html")

    def on_episode_start(self, ctx: PluginContext) -> None:
        self.writers: dict[str, _Writer] = {}  # touched only on the defer worker
        self.next_t = 0.0
        self.paths: dict[str, Any] = {}
        self._frame(ctx, force=True)

    def _frame(self, ctx: PluginContext, force: bool = False) -> None:
        if not ctx.frames or (not force and ctx.t + 1e-6 < self.next_t):
            return
        self.next_t = ctx.t + 1.0 / self.params.video_fps - 1e-6
        pts = int(round(ctx.t * 1000))
        for cam in self.params.cameras:
            rgb = ctx.frames.get(cam)
            if rgb is None:
                continue
            if cam not in self.paths:
                self.paths[cam] = ctx.episode_dir / f"{cam}.mp4"  # §8: the attempt's files, runs/<run>/ep003/
            ctx.defer(self._write, cam, self.paths[cam], rgb.copy(), pts)

    def _write(self, cam: str, path, rgb: np.ndarray, pts: int) -> None:
        writer = self.writers.get(cam)
        if writer is None:
            writer = self.writers[cam] = _Writer(path, rgb, self.params.crf, self.params.preset, self.params.video_fps)
        writer.add(rgb, pts)

    def on_step(self, ctx: PluginContext) -> None:
        if ctx.frames_new:
            self._frame(ctx)

    def _close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers = {}

    def on_episode_end(self, ctx: PluginContext) -> None:
        ctx.defer(self._close)
        ctx.result["recorder"] = {"videos": [ctx.files.relative(p) for p in self.paths.values()]}
