"""Teleoperation in the browser: keys move a gripper, IK moves the joints, the sim runs at 20 Hz, demos are recorded.

One session at a time. Everything that touches MuJoCo (stepping, rendering, IK) runs on the session's own thread,
which owns the GL context; the web server only hands it key states and commands and reads back the latest JPEG.

Keys (KeyboardEvent.code): W/S or arrows up/down move away from / toward the robot, A/D or arrows left/right move
left/right as the top camera sees it, R/F up/down, Q/E turn the gripper, Space opens/closes it, Shift moves faster,
Tab switches arm. Directions follow yam_env: the top camera's left is +x, away from the arms is -y.
"""

from __future__ import annotations

import io
import queue
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from .. import sim
from ..core import STEP_S
from ..tasks import get_task

SPEED = 0.12  # m/s
FAST = 0.25
YAW_SPEED = 1.0  # rad/s
WORKSPACE = {"x": (-0.48, 0.48), "y": (-0.38, 0.30), "z": (0.775, 1.10)}
MOVES = {
    "KeyW": (0, -1), "ArrowUp": (0, -1), "KeyS": (0, 1), "ArrowDown": (0, 1),
    "KeyA": (1, 0), "ArrowLeft": (1, 0), "KeyD": (-1, 0), "ArrowRight": (-1, 0),
}


def _down(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]])


class TeleopClient:
    """Teleop has no model: dreamkit.wrap stands around this, so a person's recorded episodes are dreamkit attempts
    (the loop plugins watch them, History shows them). It never answers get_action: the keyboard drives."""

    mock = False
    last_timing = None

    def policy(self, config):
        return None

    def set_observation(self, *a, **k):
        return None

    def reset_episode(self, **_):
        return None

    def close(self):
        return None


class TeleopSession:
    def __init__(self, task: str, dataset: Path, seed: int = 0):
        self.task = get_task(task)
        self.dataset = Path(dataset)
        self.seed = seed
        self.keys: set[str] = set()
        self.commands: queue.Queue = queue.Queue()
        self.jpeg = b""
        self.state: dict = {"status": "starting", "task": task, "dataset": str(dataset)}
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="dreamgym-teleop")
        self.thread.start()

    # --- called from the web server ----------------------------------------------------------------------------
    def set_keys(self, keys: list[str]) -> None:
        self.keys = set(keys)

    def command(self, name: str) -> None:
        self.commands.put(name)

    def stop(self) -> None:
        self.commands.put("quit")

    # --- the sim thread ------------------------------------------------------------------------------------------
    def _run(self) -> None:
        try:
            self._loop()
        except Exception as exc:
            self.state.update(status="error", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-2000:])
        finally:
            self.running = False

    def _reset_scene(self, env, yam_env) -> None:
        import mujoco

        env.reset()
        self.task.reset(env, np.random.default_rng(self.seed))
        for _ in range(20):
            env.step(yam_env.HOME, yam_env.HOME)
        self.q = {s: env.state()[s][:6].copy() for s in ("left", "right")}
        self.grip = {"left": 1.0, "right": 1.0}
        self.grip_target = {"left": 1.0, "right": 1.0}
        self.yaw = {"left": 0.0, "right": 0.0}
        self.target = {}
        for s in ("left", "right"):
            p = env.data.site(f"{s}_tcp").xpos.copy()
            p[2] = max(p[2], 0.92)
            self.target[s] = p
        self.scratch = mujoco.MjData(env.model)
        self.seed += 1

    def _loop(self) -> None:
        from PIL import Image

        from ..lerobot_writer import LeRobotWriter
        from .. import ik

        from dreamkit.plugin import EndEpisode, StopEpisode

        from ..kit import open_kit

        yam_env, _, _ = sim.load()
        env = yam_env.YamEnv()
        kit = open_kit(TeleopClient(), model="teleop (keyboard)",
                       info={"mode": "teleop", "task": self.task.name, "dataset": str(self.dataset)})
        attempt = None  # the dreamkit attempt of the episode being recorded

        def finish(ok: bool | None, exc: BaseException | None = None) -> None:
            nonlocal attempt
            if attempt is None:
                return
            if ok:
                kit.label(True)
            cm, attempt = attempt, None
            cm.__exit__(type(exc) if exc else None, exc, exc.__traceback__ if exc else None)

        writer = None
        arm = "left"
        self._reset_scene(env, yam_env)
        tick, record_start, message = 0, None, "Ready: press Record, do the task, press Save."
        try:
            while self.running:
                t0 = time.monotonic()
                # commands from the page
                while not self.commands.empty():
                    cmd = self.commands.get_nowait()
                    if cmd == "quit":
                        self.running = False
                    elif cmd == "arm":
                        arm = "right" if arm == "left" else "left"
                    elif cmd == "grip":
                        self.grip_target[arm] = 0.0 if self.grip_target[arm] > 0.5 else 1.0
                    elif cmd == "record":
                        writer = writer or LeRobotWriter(self.dataset)
                        if not writer.recording:
                            writer.begin(self.task.prompt)
                            record_start, message = tick, "Recording…"
                            attempt = kit.episode(self.task.prompt, env=env, source="gym", teleop=True)
                            attempt.__enter__()
                    elif cmd in ("save", "discard") and writer is not None and writer.recording:
                        frames = writer.frames_in_episode
                        kept = writer.end(keep=cmd == "save")
                        finish(bool(self.task.success(env)) if cmd == "save" else None)
                        if cmd == "save":
                            message = f"Saved episode {len(writer.episodes)} ({frames} frames)." if kept else \
                                f"Not saved: {frames} frames, need at least 49 (about 5 s)."
                        else:
                            message = "Discarded."
                        record_start = None
                    elif cmd == "reset":
                        if writer is not None and writer.recording:
                            writer.end(keep=False)
                            record_start = None
                        finish(None)
                        self._reset_scene(env, yam_env)
                        message = "New scene."
                    elif cmd == "home":
                        for s in ("left", "right"):
                            self.target[s] = env.data.site(f"{s}_tcp").xpos.copy()
                if not self.running:
                    break
                # keys -> active gripper target
                keys = self.keys
                speed = (FAST if ("ShiftLeft" in keys or "ShiftRight" in keys) else SPEED) * STEP_S
                delta = np.zeros(3)
                for k, (dx, dy) in MOVES.items():
                    if k in keys:
                        delta[0] += dx
                        delta[1] += dy
                delta[2] += ("KeyR" in keys or "PageUp" in keys) - ("KeyF" in keys or "PageDown" in keys)
                self.target[arm] = self.target[arm] + delta * speed
                for i, axis in enumerate("xyz"):
                    lo, hi = WORKSPACE[axis]
                    self.target[arm][i] = float(np.clip(self.target[arm][i], lo, hi))
                self.yaw[arm] += (("KeyQ" in keys) - ("KeyE" in keys)) * YAW_SPEED * STEP_S
                # IK for both arms (the idle one holds its pose), gripper ramps
                rows = {}
                for s in ("left", "right"):
                    self.q[s] = ik.step_towards(env, s, self.q[s], self.target[s], _down(self.yaw[s]), scratch=self.scratch)
                    self.grip[s] += float(np.clip(self.grip_target[s] - self.grip[s], -0.1, 0.1))
                    rows[s] = np.r_[self.q[s], self.grip[s]]
                images = env.images() if tick % 2 == 0 else None
                if writer is not None and writer.recording:
                    writer.step_hook(env, rows["left"], rows["right"], tick, images=images)
                env.step(rows["left"], rows["right"])
                tick += 1
                if attempt is not None:  # the plugins see the recorded episode like any other attempt
                    try:
                        kit.step(env, images=images, command=(rows["left"], rows["right"]))
                    except (EndEpisode, StopEpisode) as end:
                        finish(None, end)
                        message = f"dreamkit: {getattr(end, 'message', None) or end.reason}"
                if images is not None:
                    top = Image.fromarray(images["top"])
                    canvas = Image.new("RGB", (640, 540))
                    canvas.paste(top, (0, 0))
                    canvas.paste(Image.fromarray(images["left"]).resize((320, 180)), (0, 360))
                    canvas.paste(Image.fromarray(images["right"]).resize((320, 180)), (320, 360))
                    buf = io.BytesIO()
                    canvas.save(buf, format="JPEG", quality=72)
                    self.jpeg = buf.getvalue()
                recording = writer is not None and writer.recording
                self.state = {
                    "status": "running", "task": self.task.name, "prompt": self.task.prompt,
                    "dataset": str(self.dataset), "arm": arm,
                    "gripper": "closed" if self.grip_target[arm] < 0.5 else "open",
                    "recording": recording,
                    "record_seconds": round((tick - record_start) * STEP_S, 1) if recording and record_start is not None else 0,
                    "frames": writer.frames_in_episode if recording else 0,
                    "episodes": len(writer.episodes) if writer else self._existing_episodes(),
                    "success": bool(self.task.success(env)), "placed": bool(self.task.placed(env)),
                    "target": [round(float(v), 3) for v in self.target[arm]],
                    "message": message,
                }
                time.sleep(max(0.0, STEP_S - (time.monotonic() - t0)))
        finally:
            if writer is not None and writer.recording:
                writer.end(keep=False)
            finish(None)
            kit.close()
            env.close()
            self.state = {**self.state, "status": "stopped", "recording": False}

    def _existing_episodes(self) -> int:
        meta = self.dataset / "meta" / "episodes.jsonl"
        return sum(1 for l in meta.read_text().splitlines() if l.strip()) if meta.is_file() else 0
