"""Writes sim episodes as one LeRobot v2.1 dataset that Dream's fine-tuning accepts.

Layout (what dream.datasets.validate_dataset checks):
  meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl
  data/chunk-000/episode_000000.parquet          observation.state, action (14 floats), timestamp, indexes
  videos/chunk-000/observation.images.<role>/episode_000000.mp4   VP9, one per camera

Vector order follows the sim's YAM embodiment (dream_control.YAM_EMBODIMENT):
  right_arm [0:6], right_gripper [6:7], left_arm [7:13], left_gripper [13:14].
Video roles: overhead (sim camera `top`), wrist_l (`left`), wrist_r (`right`).

The sim steps at 20 Hz but training takes 30, 15, 10 or 5 fps only, so every second step is recorded (10 fps).
Video is encoded while the episode runs; a discarded episode (a failed demo) deletes its files.
"""

from __future__ import annotations

import json
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np

CAMERA_ROLES = {"top": "overhead", "left": "wrist_l", "right": "wrist_r"}
SIM_HZ = 20
VIDEO_OPTIONS = {"deadline": "realtime", "cpu-used": "6", "lag-in-frames": "0", "auto-alt-ref": "0", "crf": "32", "b:v": "0"}
## Training draws 48-frame chunks plus one, so shorter episodes are useless (and refused).
MIN_FRAMES = 49

STATE_NAMES = [f"right_joint{i}" for i in range(1, 7)] + ["right_gripper"] + \
              [f"left_joint{i}" for i in range(1, 7)] + ["left_gripper"]

EMBODIMENT_CONFIG = {
    "state": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
    "action": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
    "video": {role: f"observation.images.{role}" for role in CAMERA_ROLES.values()},
    "head_camera": "overhead",
}


def _vector(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.r_[right[:6], right[6:7], left[:6], left[6:7]].astype(np.float32)


class LeRobotWriter:
    def __init__(self, root: Path, fps: int = 10, size: tuple[int, int] = (640, 360)):
        if SIM_HZ % fps:
            raise ValueError("fps must divide 20 (10 or 5)")
        self.root = Path(root)
        self.fps, self.stride = fps, SIM_HZ // fps
        self.width, self.height = size
        self.tasks: dict[str, int] = {}
        self.episodes: list[dict] = []
        self.total_frames = 0
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        self._open = None
        self._resume()

    def _resume(self) -> None:
        """Appends to a dataset this writer (or an earlier run) already wrote, instead of overwriting episode 0."""
        meta = self.root / "meta"
        if not (meta / "info.json").is_file():
            return
        info = json.loads((meta / "info.json").read_text())
        if info.get("fps") != self.fps:
            raise ValueError(f"{self.root} was recorded at {info.get('fps')} fps, not {self.fps}")
        self.episodes = [json.loads(l) for l in (meta / "episodes.jsonl").read_text().splitlines() if l.strip()]
        for line in (meta / "tasks.jsonl").read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                self.tasks[t["task"]] = t["task_index"]
        self.total_frames = sum(e["length"] for e in self.episodes)

    @property
    def recording(self) -> bool:
        return self._open is not None

    @property
    def frames_in_episode(self) -> int:
        return len(self._open["state"]) if self._open else 0

    # --- one episode ------------------------------------------------------------------------------------------

    def begin(self, task: str) -> None:
        import av

        index = len(self.episodes)
        tmp = self.root / ".tmp" / f"episode_{index:06d}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        streams = {}
        for camera, role in CAMERA_ROLES.items():
            container = av.open(str(tmp / f"{role}.mp4"), mode="w")
            stream = container.add_stream("libvpx-vp9", rate=self.fps, options=VIDEO_OPTIONS)
            stream.width, stream.height, stream.pix_fmt = self.width, self.height, "yuv420p"
            stream.time_base = Fraction(1, self.fps)
            streams[camera] = (container, stream)
        self._open = {"task": task, "tmp": tmp, "streams": streams, "state": [], "action": []}

    def step_hook(self, env, left: np.ndarray, right: np.ndarray, step: int, images: dict | None = None) -> None:
        """Runner hook: called before each commanded step; records every `stride`-th one.

        @param images This step's camera images when the caller already rendered them (teleop), else rendered here.
        """
        if self._open is None or step % self.stride:
            return
        import av
        from PIL import Image

        state = env.state()
        self._open["state"].append(_vector(state["left"], state["right"]))
        self._open["action"].append(_vector(np.asarray(left), np.asarray(right)))
        pts = len(self._open["state"]) - 1
        for camera, rgb in (images if images is not None else env.images()).items():
            if (rgb.shape[1], rgb.shape[0]) != (self.width, self.height):
                rgb = np.asarray(Image.fromarray(rgb).resize((self.width, self.height), Image.BILINEAR))
            container, stream = self._open["streams"][camera]
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
            frame.pts = pts
            for packet in stream.encode(frame):
                container.mux(packet)

    def end(self, keep: bool) -> bool:
        """Closes the episode; keeps it only if `keep` and long enough. @return Whether it was kept."""
        ep, self._open = self._open, None
        if ep is None:
            return False
        for container, stream in ep["streams"].values():
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        n = len(ep["state"])
        if not keep or n < MIN_FRAMES:
            shutil.rmtree(ep["tmp"], ignore_errors=True)
            return False
        self._write_episode(ep, n)
        return True

    def _write_episode(self, ep: dict, n: int) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        index = len(self.episodes)
        task_index = self.tasks.setdefault(ep["task"], len(self.tasks))
        data_dir = self.root / "data" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "observation.state": pa.array([s.tolist() for s in ep["state"]], type=pa.list_(pa.float32())),
                "action": pa.array([a.tolist() for a in ep["action"]], type=pa.list_(pa.float32())),
                "timestamp": pa.array((np.arange(n) / self.fps).astype(np.float32)),
                "frame_index": pa.array(np.arange(n, dtype=np.int64)),
                "episode_index": pa.array(np.full(n, index, dtype=np.int64)),
                "index": pa.array(np.arange(self.total_frames, self.total_frames + n, dtype=np.int64)),
                "task_index": pa.array(np.full(n, task_index, dtype=np.int64)),
            }
        )
        pq.write_table(table, data_dir / f"episode_{index:06d}.parquet")
        for role in CAMERA_ROLES.values():
            video_dir = self.root / "videos" / "chunk-000" / f"observation.images.{role}"
            video_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ep["tmp"] / f"{role}.mp4"), video_dir / f"episode_{index:06d}.mp4")
        shutil.rmtree(ep["tmp"], ignore_errors=True)
        self.episodes.append({"episode_index": index, "tasks": [ep["task"]], "length": n})
        self.total_frames += n
        self._write_meta()

    # --- metadata ----------------------------------------------------------------------------------------------

    def _write_meta(self) -> None:
        video_feature = {
            "dtype": "video",
            "shape": [self.height, self.width, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.fps": self.fps, "video.codec": "vp9", "video.pix_fmt": "yuv420p",
                     "video.height": self.height, "video.width": self.width, "video.channels": 3,
                     "has_audio": False},
        }
        n = len(self.episodes)
        info = {
            "codebase_version": "v2.1",
            "robot_type": "yam",
            "total_episodes": n,
            "total_frames": self.total_frames,
            "total_tasks": len(self.tasks),
            "total_videos": n * len(CAMERA_ROLES),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {"train": f"0:{n}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [14], "names": STATE_NAMES},
                "action": {"dtype": "float32", "shape": [14], "names": STATE_NAMES},
                **{f"observation.images.{r}": video_feature for r in CAMERA_ROLES.values()},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
        }
        meta = self.root / "meta"
        (meta / "info.json").write_text(json.dumps(info, indent=2))
        (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in self.episodes))
        (meta / "tasks.jsonl").write_text(
            "".join(json.dumps({"task_index": i, "task": t}) + "\n" for t, i in sorted(self.tasks.items(), key=lambda x: x[1]))
        )
