"""harvester — Learn: every attempt as a LeRobot v2.1 dataset in run_dir/harvester/ (PLUGIN_STANDARD.md).

The layout repeats Dream Labs' robots package (robots/rollout_episode.py, not imported: it is a hardware package):
one LeRobot v2.1 dataset per attempt — meta/, data/chunk-000/episode_000000.parquet,
videos/chunk-000/observation.images.<role>/episode_000000.mp4 — plus dreamkit.json with the outcome. Successes go to
success/, failures to failed/ with the reason. State and action follow the YAM embodiment config: right arm, right
gripper, left arm, left gripper. All encoding runs on ctx.defer."""

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
    save_failed: bool = Field(True, description="Keep failed attempts too, under failed/, with the reason")
    min_frames: int = Field(10, ge=1, le=10000, description="Shorter attempts are not saved",
                            json_schema_extra={"unit": ""})
    crf: int = Field(23, ge=0, le=51, description="Video quality: lower is better and larger",
                     json_schema_extra={"unit": ""})


class Harvester(Plugin):
    """Saves every attempt as a LeRobot v2.1 dataset; successful ones are ready for fine-tuning."""

    name = "harvester"
    Params = Params
    manifest = Manifest(group="core", icon="Database", capability="observe", needs_frames=True,
                        hooks=["on_episode_start", "on_step", "on_episode_end", "on_run_end"], ui="ui/index.html")

    def on_episode_start(self, ctx: PluginContext) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.videos: dict[str, _Video] = {}  # touched only on the defer worker
        self.frames = 0
        self.next_t = 0.0
        self.ok = isinstance(ctx.state, dict) and all(s in ctx.state for s in SIDES)
        self.tmp = ctx.files.path(f"recording/ep{ctx.episode:03d}")
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
        outcome_known = ctx.result.get("success") is not None
        success = ctx.result.get("success") is True
        # no world and nobody judged it: kept as unlabeled/ until it is marked in History (or by client.label())
        split = "success" if success else "failed" if outcome_known else "unlabeled"
        target = ctx.files.path(f"{split}/ep{ctx.episode:03d}")
        outcome = {
            "run_id": ctx.run_id, "episode": ctx.episode, "scene": ctx.scene, "attempt": ctx.attempt,
            "task": ctx.task, "model": ctx.model, "success": success if outcome_known else None,
            "reason": None if success else ctx.result.get("reason"), "at_s": ctx.result.get("at_s"),
            "message": ctx.result.get("message"), "frames": self.frames,
        }
        keep = self.frames >= self.params.min_frames and (success or not outcome_known or self.params.save_failed)
        ctx.defer(self._finish, keep, ctx.task, np.array(self.states), np.array(self.actions), outcome, target)
        if keep:
            ctx.result["harvester"] = {"split": split, "path": ctx.files.relative(target), "frames": self.frames}

    def _finish(self, keep: bool, task: str, states, actions, outcome: dict[str, Any], target: Path) -> None:
        videos = {role: v.close() for role, v in self.videos.items()}
        self.videos = {}
        if not keep:
            shutil.rmtree(self.tmp, ignore_errors=True)
            return
        write_episode(self.tmp, task, self.params.dataset_fps, states, actions, videos)
        (self.tmp / "dreamkit.json").write_text(json.dumps(outcome, indent=2, default=str))
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(self.tmp), target)

    def on_run_end(self, ctx: PluginContext) -> None:
        ctx.defer(self._summary, ctx)

    def _summary(self, ctx: PluginContext) -> None:
        shutil.rmtree(ctx.files.dir / "recording", ignore_errors=True)
        ready = sorted((ctx.files.dir / "success").glob("ep*/meta/info.json"))
        failed = sorted((ctx.files.dir / "failed").glob("ep*/meta/info.json"))
        if ready or failed:
            ctx.store["harvester"] = {"success": len(ready), "failed": len(failed)}
            ctx.emit("success" if ready else "info",
                     f"{len(ready)} successful episode{'s' if len(ready) != 1 else ''} ready for fine-tuning"
                     f" ({len(failed)} failed kept for analysis)")
