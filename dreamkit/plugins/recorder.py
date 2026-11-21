"""recorder: one MP4 per camera per attempt, timestamped in simulated seconds so `#t=<at>` lands on the moment."""

from __future__ import annotations

from fractions import Fraction

import av
import numpy as np

from ..plugin import Context, Plugin


class Recorder(Plugin):
    """runs/<id>/ep<N>/<cam>.mp4 (H.264, browser-playable). Video time = the attempt's simulated time."""

    name = "recorder"
    group = "output"
    description = "Records every camera to MP4 per attempt (video time = sim time)"
    defaults = {"fps": 10, "cameras": ["top", "left", "right"], "crf": 28, "preset": "ultrafast"}

    def on_episode_start(self, ctx: Context) -> None:
        self.streams: dict[str, tuple] = {}
        self.last_pts: dict[str, int] = {}
        self.period = 1.0 / max(1, int(self.params["fps"]))
        self.next_t = 0.0
        self._write(ctx)

    def _open(self, ctx: Context, cam: str, rgb: np.ndarray):
        path = ctx.episode_dir / f"{cam}.mp4"
        container = av.open(str(path), "w", options={"movflags": "+faststart"})
        stream = container.add_stream("libx264", rate=int(self.params["fps"]))
        stream.height, stream.width = rgb.shape[0] - rgb.shape[0] % 2, rgb.shape[1] - rgb.shape[1] % 2
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 1000)
        stream.options = {"preset": str(self.params["preset"]), "crf": str(self.params["crf"])}
        self.streams[cam] = (container, stream)
        return container, stream

    def _write(self, ctx: Context) -> None:
        if ctx.t + 1e-6 < self.next_t:
            return
        self.next_t = ctx.t + self.period - 1e-6
        for cam in self.params["cameras"]:
            rgb = ctx.frames.get(cam)
            if rgb is None:
                continue
            container, stream = self.streams.get(cam) or self._open(ctx, cam, rgb)
            pts = int(round(ctx.t * 1000))
            if pts <= self.last_pts.get(cam, -1):
                continue
            self.last_pts[cam] = pts
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb[: stream.height, : stream.width]),
                                               format="rgb24")
            frame.pts = pts
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)

    def on_step(self, ctx: Context) -> None:
        if ctx.frames_new:
            self._write(ctx)

    def on_episode_end(self, ctx: Context) -> None:
        for cam, (container, stream) in self.streams.items():
            try:
                for packet in stream.encode():
                    container.mux(packet)
            finally:
                container.close()
        self.streams = {}
        if ctx.episode_dir is not None:
            ctx.result["videos"] = sorted(f"ep{ctx.episode}/{p.name}" for p in ctx.episode_dir.glob("*.mp4"))
