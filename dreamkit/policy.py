"""Where the action chunks come from: Dream-1 through Dream Labs' own dream_control functions, or a scripted mock.

Both answer get_action() with the model's reply shape, {"q_l_chunk": (48, 7), "q_r_chunk": (48, 7)}, so every chunk
goes through dream_control.validate_chunk in the runner either way.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

CHUNK = 48


class DreamPolicy:
    """Dream-1 via dream_control.open_policy / feed, as dream_control.main runs it."""

    mock = False

    def __init__(self, dc, model: str):
        self.dc = dc
        self.model = model
        settings = dc.settings()
        self.key = settings.get("DREAM_API_KEY", "")
        self.gateway = settings.get("DREAM_GATEWAY_URL", "https://api.dreamlabs.ai")
        if not self.key:
            raise RuntimeError("No Dream API key: set DREAM_API_KEY or ~/.dreamlabs/credentials (or use --mock).")
        self.poses = model in dc.POSE_MODELS
        self.client = None
        self.timing: dict[str, Any] | None = None

    def _open(self):
        self.client = self.dc.open_policy(self.dc.policy_config(self.model, self.gateway, self.poses), self.key)

    def begin_episode(self, env, windows, task: str, steps: int, behavior: str | None = None) -> None:
        if self.client is None:
            self._open()
        self.dc.feed(self.client, env, windows, task, self.model, steps, every=0, station=self.poses)

    def observe(self, env, windows, task: str, steps: int) -> None:
        self.dc.feed(self.client, env, windows, task, self.model, steps, station=self.poses)

    def get_action(self, env, windows, task: str, steps: int) -> dict[str, Any]:
        restarts = 0
        while True:
            try:
                reply = self.client.get_action()
                self.timing = self.client.last_timing
                return reply
            except Exception as error:
                self.timing = self.client.last_timing
                if type(error).__name__ == "DreamSessionExpired" and restarts < 3:
                    restarts += 1
                    continue
                if self.poses or not self.dc.needs_poses(error):
                    raise
                # Same fallback as dream_control.main: the model reads hand poses, reopen on the station keys.
                self.client.close()
                self.poses = True
                self._open()
                self.dc.feed(self.client, env, windows, task, self.model, steps, every=0, station=self.poses)

    def advance(self, rows: int) -> None:
        pass

    def end_episode(self) -> None:
        if self.client is not None:
            try:
                self.client.reset_episode()
            except Exception:
                self.client.close()
                self.client = None

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None


# ----------------------------------------------------------------------------------------------------------------
# Mock: a scripted pick-and-place planned with Dream Labs' own IK (mock_pickups.solve), with injectable failures.

## Grasp height of the tool point above each fruit's centre (from mock_pickups.run_trial).
GRASP_DZ = {"apple": 0.003, "orange": -0.009, "banana": 0.003}
BEHAVIORS = ("success", "dropped", "not_reached", "grasped_air", "missed_target", "stalled", "collision")


class MockPolicy:
    """Scripted stand-in for Dream-1: no network, no credits. `behavior` picks what goes wrong (or nothing)."""

    mock = True

    def __init__(self, sim_module_mock_pickups, latency_ms: float = 250, jitter_ms: float = 80, seed: int = 0):
        self.solve = sim_module_mock_pickups.solve
        self.latency_ms = latency_ms
        self.jitter_ms = jitter_ms
        self.rng = np.random.default_rng(seed)
        self.plan: list[tuple[np.ndarray, np.ndarray]] = []
        self.cursor = 0
        self.timing: dict[str, Any] | None = None
        self.behavior = "success"
        self.calls = 0

    # -- planning --------------------------------------------------------------------------------------------
    def _plan(self, env, obj: str | None, target: str | None, behavior: str) -> list[tuple[np.ndarray, np.ndarray]]:
        state = env.state()
        hold = {s: state[s].copy() for s in ("left", "right")}
        if obj is None or behavior == "idle":
            return [(hold["left"], hold["right"])]
        p = env.data.body(obj).xpos.copy()
        g = env.data.body(target).xpos.copy() if target else p + [0, -0.15, 0]
        sides = sorted(("left", "right"), key=lambda s: np.linalg.norm(env.data.site(f"{s}_tcp").xpos - p))
        yaws = [np.pi / 2, 0.0] if obj == "banana" else [0.0, np.pi / 2, np.pi / 4, -np.pi / 4]
        dz = GRASP_DZ.get(obj, 0.003)
        place_dz = 0.07 if target == "bowl" else 0.06

        place = g.copy()
        if behavior == "missed_target":
            away = place[:2] - p[:2]
            away = away / (np.linalg.norm(away) + 1e-9)
            place[:2] += away * 0.10  # overshoot past the target by 10 cm
        last_error: Exception | None = None
        for side in sides:
            for yaw in yaws:
                try:
                    return self._script(env, side, yaw, p, place, dz, place_dz, behavior, hold)
                except ValueError as error:
                    last_error = error
        raise RuntimeError(f"mock policy: no reachable grasp for {obj}: {last_error}")

    def _script(self, env, side, yaw, p, place, dz, place_dz, behavior, hold):
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]])
        other = "right" if side == "left" else "left"
        q = hold[side][:6].copy()
        rows: list[np.ndarray] = []

        def move(target, opening, n):
            nonlocal q
            end = self.solve(env, side, np.asarray(target, dtype=float), rot, q)
            start = q.copy()
            for a in np.linspace(0, 1, n):
                blend = a * a * (3 - 2 * a)
                rows.append(np.r_[start + (end - start) * blend, opening])
            q = end

        def grip(a, b, n):
            for o in np.linspace(a, b, n):
                rows.append(np.r_[q, o])

        def wait(opening, n):
            rows.extend(np.r_[q, opening] for _ in range(n))

        wait(1, 10)
        if behavior == "stalled":
            move(p + [0.0, 0.0, 0.16], 1, 40)  # starts, then freezes halfway there
            wait(1, 2000)
        elif behavior == "not_reached":
            off = p + [0.0, -0.09, 0.05]
            move(off + [0, 0, 0.08], 1, 40)
            move(off, 1, 30)
            grip(1, 0, 20)
            move(off + [0, 0, 0.1], 0, 40)
            wait(0, 400)
        elif behavior == "grasped_air":
            move(p + [0, 0, 0.12], 1, 40)
            move(p + [0, 0, 0.065], 1, 30)
            grip(1, 0, 20)
            move(p + [0, 0, 0.14], 0, 40)
            wait(0, 400)
        elif behavior == "collision":
            beside = p + [0.0, -0.10, 0.0]
            beside[2] = 0.76  # the table top
            move(beside + [0, 0, 0.10], 1, 40)
            move(beside + [0, 0, -0.035], 1, 30)  # drive the tool point 3.5 cm into the table
            wait(1, 400)
        else:
            move(p + [0, 0, 0.10], 1, 40)
            move(p + [0, 0, dz], 1, 40)
            grip(1, 0, 20)
            wait(0, 5)
            move(p + [0, 0, 0.10], 0, 40)
            if behavior == "dropped":
                mid = p + (place - p) * 0.3
                move([mid[0], mid[1], p[2] + 0.12], 0, 25)
                grip(0, 1, 6)  # lets go in mid-air, far from the target
                wait(1, 400)
            else:
                move([place[0], place[1], place[2] + 0.13], 0, 50)
                move([place[0], place[1], place[2] + place_dz], 0, 30)
                grip(0, 1, 15)
                move([place[0], place[1], place[2] + 0.13], 1, 30)
                wait(1, 400)
        hold_other = hold[other]
        return [(r, hold_other) if side == "left" else (hold_other, r) for r in rows]

    # -- the policy interface ----------------------------------------------------------------------------------
    def begin_episode(self, env, windows, task: str, steps: int, behavior: str | None = None) -> None:
        from .task import parse

        self.behavior = behavior or "success"
        obj, target = parse(task)
        self.plan = self._plan(env, obj, target, self.behavior)
        self.cursor = 0

    def observe(self, env, windows, task: str, steps: int) -> None:
        pass

    def get_action(self, env, windows, task: str, steps: int) -> dict[str, Any]:
        self.calls += 1
        rtt = max(0.0, self.latency_ms + self.rng.normal(0, self.jitter_ms)) if self.latency_ms > 0 else 0.0
        if rtt:
            time.sleep(rtt / 1000)
        rows = self.plan[self.cursor : self.cursor + CHUNK]
        rows += [self.plan[-1]] * (CHUNK - len(rows))
        self.timing = {
            "rtt_ms": round(rtt, 1),
            "serve_infer_ms": round(rtt * 0.6, 1),
            "network_ms": round(rtt * 0.4, 1),
            "mock": True,
        }
        return {
            "q_l_chunk": np.array([r[0] for r in rows]).tolist(),
            "q_r_chunk": np.array([r[1] for r in rows]).tolist(),
        }

    def advance(self, rows: int) -> None:
        self.cursor = min(self.cursor + rows, max(len(self.plan) - 1, 0))

    def end_episode(self) -> None:
        pass

    def close(self) -> None:
        pass
