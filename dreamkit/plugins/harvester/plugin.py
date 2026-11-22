"""harvester — Learn: every successful attempt joins one LeRobot v2.1 dataset, <data_dir>/datasets/harvest-<task>/.

Nothing goes inside the run: the attempt is recorded in <data_dir>/datasets/.incoming/ while it runs (the layout of
Dream Labs' robots/rollout_episode.py: meta/, data/chunk-000/episode_000000.parquet,
videos/chunk-000/observation.images.<role>/episode_000000.mp4) and, when it succeeds, appended as the next episode of
the task's dataset (dreamkit.datasets.append_episode), whose dataset.json links it back to its run and attempt.
A failed attempt is dropped (its videos stay in the run). An attempt nobody could judge (a robot without a world)
waits in the dataset's .pending/ until it is marked ✓ (appended) or ✗ (dropped) in History. State and action follow
the YAM embodiment config: right arm, right gripper, left arm, left gripper. All encoding runs on ctx.defer."""

from __future__ import annotations

import json
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
from pydantic import BaseModel, Field

from dreamkit.plugin import Plugin, PluginContext
from dreamkit.standard import Manifest

SIDES = ("right", "left")  # the YAM embodiment config's order: right_arm [0,6], right_gripper [6,7], left_arm [7,13] ...
NAMES = [f"{side}_{part}" for side in SIDES for part in [*(f"joint_{i}" for i in range(6)), "gripper"]]
## The YAM embodiment config these datasets are written for (dream_control.YAM_EMBODIMENT).
YAM_CONFIG = {
    "state": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
    "action": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
    "video": {role: f"observation.images.{role}" for role in ("overhead", "wrist_l", "wrist_r")},
}
VIDEO_PATH = "videos/chunk-000/observation.images.{role}/episode_000000.mp4"
DATA_PATH = "data/chunk-000/episode_000000.parquet"


def _stats(values: np.ndarray, count: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    out = {k: v.tolist() for k, v in zip(("min", "max", "mean", "std"), (values.min(0), values.max(0), values.mean(0),
                                                                            values.std(0)))}
    return out | {"count": [count]}


def write_episode(folder: Path, task: str, fps: int, states: np.ndarray, actions: np.ndarray,
                  videos: dict[str, dict[str, Any]]) -> None:
    """The parquet and meta/ of a one-episode LeRobot v2.1 dataset whose videos are already in place."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = len(states)
    index = np.arange(n)
    columns = {
        "observation.state": states.astype(np.float32),
        "action": actions.astype(np.float32),
        "timestamp": (index / fps).astype(np.float32),
        "frame_index": index.astype(np.int64),
        "episode_index": np.zeros(n, dtype=np.int64),
        "index": index.astype(np.int64),
        "task_index": np.zeros(n, dtype=np.int64),
    }
    (folder / DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({k: pa.array(list(v) if v.ndim > 1 else v) for k, v in columns.items()}), folder / DATA_PATH)
    features: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    for name, values in columns.items():
        width = values.shape[1] if values.ndim > 1 else 1
        features[name] = {"dtype": str(values.dtype), "shape": [width],
                          "names": NAMES if name in ("observation.state", "action") else None}
        stats[name] = _stats(values, n)
    for role, video in videos.items():
        key = f"observation.images.{role}"
        features[key] = {
            "dtype": "video", "shape": [video["height"], video["width"], 3], "names": ["height", "width", "channels"],
            "info": {"video.height": video["height"], "video.width": video["width"], "video.channels": 3,
                     "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                     "video.fps": fps, "has_audio": False},
        }
        stats[key] = {k: v if k == "count" else [[[x]] for x in v] for k, v in video["stats"].items()}
    meta = folder / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    chunk = "chunk-{episode_chunk:03d}"
    (meta / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "robot_type": "yam", "total_episodes": 1, "total_frames": n, "total_tasks": 1,
        "total_videos": len(videos), "total_chunks": 1, "chunks_size": 1000, "fps": fps, "splits": {"train": "0:1"},
        "data_path": f"data/{chunk}/episode_{{episode_index:06d}}.parquet",
        "video_path": f"videos/{chunk}/{{video_key}}/episode_{{episode_index:06d}}.mp4",
        "features": features,
    }, indent=2))
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    (meta / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": [task], "length": n}) + "\n")
    (meta / "episodes_stats.jsonl").write_text(json.dumps({"episode_index": 0, "stats": stats}) + "\n")


class _Video:
    """One camera's MP4 at a constant frame rate (H.264, yuv420p), with LeRobot's sampled pixel statistics."""

    def __init__(self, path: Path, fps: int, rgb: np.ndarray, crf: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), "w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.height, self.width = rgb.shape[0] - rgb.shape[0] % 2, rgb.shape[1] - rgb.shape[1] % 2
        self.stream.height, self.stream.width = self.height, self.width
        self.stream.pix_fmt = "yuv420p"
        self.stream.codec_context.time_base = Fraction(1, fps)
        self.stream.options = {"preset": "veryfast", "crf": str(crf), "bf": "0"}
        self.frames = 0
        self.samples: list[np.ndarray] = []

    def add(self, rgb: np.ndarray) -> None:
        rgb = np.ascontiguousarray(rgb[: self.height, : self.width])
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = self.frames
        frame.time_base = self.stream.codec_context.time_base
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        if self.frames % 10 == 0:
            self.samples.append(rgb[::4, ::4].reshape(-1, 3) / 255.0)
        self.frames += 1

    def close(self) -> dict[str, Any]:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        pixels = np.concatenate(self.samples) if self.samples else np.zeros((1, 3))
        return {"height": self.height, "width": self.width, "stats": _stats(pixels, len(self.samples))}



class Params(BaseModel):
    dataset_fps: int = Field(10, ge=1, le=20, description="Rows and video frames per second in the dataset",
                             json_schema_extra={"unit": "fps"})
    cameras: dict[str, str] = Field({"top": "overhead", "left": "wrist_l", "right": "wrist_r"},
                                    description="Camera name to the video role the YAM embodiment config expects")
    dataset: str = Field("harvest-{task}", description="The dataset successful attempts join, in <data_dir>/datasets/; "
                         "{task} is the task in a few words")
    min_frames: int = Field(10, ge=1, le=10000, description="Shorter attempts are not saved",
                            json_schema_extra={"unit": ""})
    crf: int = Field(23, ge=0, le=51, description="Video quality: lower is better and larger",
                     json_schema_extra={"unit": ""})


class Harvester(Plugin):
    """Adds every successful attempt to a fine-tuning dataset (LeRobot v2.1), one per task."""

    name = "harvester"
    Params = Params
    manifest = Manifest(group="core", icon="Database", capability="observe", needs_frames=True,
                        hooks=["on_run_start", "on_episode_start", "on_step", "on_episode_end", "on_run_end"],
                        ui="ui/index.html")

    def on_run_start(self, ctx: PluginContext) -> None:
        self.added: dict[str, int] = {}  # dataset -> successful attempts added in this run
        self.pending = 0

    def on_episode_start(self, ctx: PluginContext) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.videos: dict[str, _Video] = {}  # touched only on the defer worker
        self.frames = 0
        self.next_t = 0.0
        self.ok = isinstance(ctx.state, dict) and all(s in ctx.state for s in SIDES) and ctx.datasets_dir is not None
        self.tmp = (ctx.datasets_dir / ".incoming" / f"{ctx.run_id}-ep{ctx.episode:03d}") if ctx.datasets_dir else None
        if not self.ok:
            ctx.emit("info", "needs env.state() with 'left' and 'right' (7 values each); nothing saved")
        elif self.tmp.exists():
            shutil.rmtree(self.tmp)

    def on_step(self, ctx: PluginContext) -> None:
        if not self.ok or not ctx.frames_new or ctx.t + 1e-6 < self.next_t:
            return
        cams = self.params.cameras
        if not all(cam in ctx.frames for cam in cams):
            return
        self.next_t = ctx.t + 1.0 / self.params.dataset_fps - 1e-6
        state = np.concatenate([np.asarray(ctx.state[s], dtype=float)[:7] for s in SIDES])
        if ctx.command is not None:
            left, right = ctx.command
            action = np.concatenate([np.asarray(right, dtype=float)[:7], np.asarray(left, dtype=float)[:7]])
        else:
            action = state
        self.states.append(state)
        self.actions.append(action)
        self.frames += 1
        for cam, role in cams.items():
            ctx.defer(self._add, role, self.tmp / VIDEO_PATH.format(role=role), ctx.frames[cam].copy())

    def _add(self, role: str, path: Path, rgb: np.ndarray) -> None:
        video = self.videos.get(role)
        if video is None:
            video = self.videos[role] = _Video(path, self.params.dataset_fps, rgb, self.params.crf)
        video.add(rgb)

    def on_episode_end(self, ctx: PluginContext) -> None:
        if not self.ok:
            return
        from dreamkit import datasets

        known = ctx.result.get("success") is not None
        success = ctx.result.get("success") is True
        keep = self.frames >= self.params.min_frames and (success or not known)
        name = datasets.harvest_name(ctx.task or "task", self.params.dataset)
        link = {"run": ctx.run_id, "episode": ctx.episode, "task": ctx.task, "model": ctx.model}
        ctx.defer(self._finish, ctx, keep, success, name, ctx.task, np.array(self.states), np.array(self.actions), link)

    def _finish(self, ctx: PluginContext, keep: bool, success: bool, name: str, task: str, states, actions,
                link: dict[str, Any]) -> None:
        from dreamkit import datasets

        videos = {role: v.close() for role, v in self.videos.items()}
        self.videos = {}
        if not keep:
            shutil.rmtree(self.tmp, ignore_errors=True)
            return
        write_episode(self.tmp, task, self.params.dataset_fps, states, actions, videos)
        root = ctx.datasets_dir / name
        if success:
            index = datasets.append_episode(root, self.tmp, source="harvester", **link)
            shutil.rmtree(self.tmp, ignore_errors=True)
            self.added[name] = self.added.get(name, 0) + 1
            ctx.result["harvester"] = {"dataset": name, "index": index, "frames": self.frames}
        else:  # nobody judged it: it waits for a ✓ or ✗ in History
            pending = root / ".pending" / self.tmp.name
            pending.parent.mkdir(parents=True, exist_ok=True)
            (self.tmp / "link.json").write_text(json.dumps(link, default=str))
            if pending.exists():
                shutil.rmtree(pending)
            shutil.move(str(self.tmp), pending)
            self.pending += 1
            ctx.result["harvester"] = {"dataset": name, "pending": pending.name, "frames": self.frames}

    def on_run_end(self, ctx: PluginContext) -> None:
        ctx.defer(self._summary, ctx)

    def _summary(self, ctx: PluginContext) -> None:
        if ctx.datasets_dir is not None:
            shutil.rmtree(ctx.datasets_dir / ".incoming", ignore_errors=True) if not any(
                (ctx.datasets_dir / ".incoming").glob("*")) else None
        total = sum(self.added.values())
        if total or self.pending:
            ctx.store["harvester"] = {"added": dict(self.added), "pending": self.pending}
            where = ", ".join(self.added) or "its dataset"
            ctx.emit("success" if total else "info",
                     f"{total} successful attempt{'s' if total != 1 else ''} added to {where}"
                     + (f"; {self.pending} waiting for a ✓ or ✗ in history" if self.pending else ""))
