"""MockClient: a scripted stand-in for dream.DreamClient on Dream Labs' simulator. Same calls (policy,
set_observation, get_action, last_timing, reset_episode, close), no network, no credits.

It plans a pick-and-place with the simulator's own IK (mock_pickups.solve) and injects one failure per attempt,
so every plugin can be built and tested for free. Phase lengths vary with the seed, so attempts differ.
"""

from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np

from .task import parse

CHUNK = 48
STEP_S = 0.05
GRASP_DZ = {"apple": 0.003, "orange": -0.009, "banana": 0.003}
## `dropped`: the grip loosens mid-carry to this opening (the fruit's width is ~0.7) without the policy opening the
## gripper, so the fruit slides out. `fell_off_table`: a violent wrist flick (joints, amplitude in rad, rows).
LOOSE_GRIP = 0.74
FLING_SHAKE = ([4], 1.57, 14)
BEHAVIORS = ("success", "dropped", "released_off_target", "released_near_target", "not_reached", "grasped_air", "stalled",
             "collision", "fell_off_table")


class MockClient:
    mock = True

    def __init__(self, env, behaviors: list[str] | None = None, seed: int = 0, latency_ms: float = 0.0,
                 recovery_behaviors: list[str] | None = None, fake_rtt_ms: float | None = None):
        self.env = env
        self.solve = importlib.import_module("mock_pickups").solve  # the simulator folder is on sys.path
        self.behaviors = list(behaviors or ["success"])
        ## What the "model" does after a recovery reset it mid-attempt (cycled); real Dream-1 decides for itself.
        self.recovery_behaviors = list(recovery_behaviors or ["success"])
        self.recoveries = 0
        self.after_recovery = False
        unknown = [b for b in self.behaviors + self.recovery_behaviors if b not in BEHAVIORS]
        if unknown:
            raise ValueError(f"unknown mock behaviour(s) {unknown}; choose from {', '.join(BEHAVIORS)}")
        self.seed = seed
        self.fake_rtt_ms = fake_rtt_ms  # report this round trip without waiting (the slow_model fixture)
        self.latency_ms = latency_ms
        self.attempt = 0
        self.plan: list[tuple[np.ndarray, np.ndarray]] | None = None
        self.t0 = 0.0
        self.task = ""
        self.last_timing: dict[str, Any] | None = None

    # -- DreamClient's surface -----------------------------------------------------------------------------------
    def policy(self, config: dict[str, Any]) -> None:
        self.config = config

    def set_observation(self, frames: Any, observation: dict[str, Any], **_: Any) -> None:
        self.task = observation.get("prompt", self.task) if isinstance(observation, dict) else self.task

    def get_action(self, **_: Any) -> dict[str, Any]:
        if self.plan is None:
            self.planned = True  # this attempt counts even if planning fails below
            if self.after_recovery:
                self.behavior = self.recovery_behaviors[(self.recoveries - 1) % len(self.recovery_behaviors)]
            else:
                self.behavior = self.behaviors[self.attempt % len(self.behaviors)]
            self.rng = np.random.default_rng(self.seed * 1000 + self.attempt * 10 + self.recoveries)
            self.plan = self._plan(self.behavior)
            self.t0 = float(self.env.data.time)
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        done = int(round((float(self.env.data.time) - self.t0) / STEP_S))
        rows = self.plan[done : done + CHUNK]
        rows += [self.plan[-1]] * (CHUNK - len(rows))
        self.last_timing = ({"rtt_ms": float(self.fake_rtt_ms + self.rng.normal(0, 60))} if self.fake_rtt_ms
                            else {"mock": True})  # no model was called: latency plugins say so, not "0 ms"
        return {"q_l_chunk": np.array([r[0] for r in rows]).tolist(), "q_r_chunk": np.array([r[1] for r in rows]).tolist()}

    def reset_episode(self, recovery: bool = False) -> None:
        """A new session. recovery=True: the same attempt goes on from where the arms are (dreamkit's recovery)."""
        if recovery:
            self.recoveries += 1
            self.after_recovery = True
        else:
            if self.plan is not None or self.after_recovery or getattr(self, "planned", False):
                self.attempt += 1
            self.planned = False
            self.after_recovery = False
        self.plan = None

    def close(self) -> None:
        pass

    # -- planning ------------------------------------------------------------------------------------------------
    def _n(self, base: int) -> int:
        """A phase length, ±30 % by the attempt's seed."""
        return max(4, int(round(base * self.rng.uniform(0.7, 1.3))))

    def _plan(self, behavior: str) -> list[tuple[np.ndarray, np.ndarray]]:
        env = self.env
        state = env.state()
        hold = {s: state[s].copy() for s in ("left", "right")}
        obj, target = parse(self.task)
        if obj is None:
            return [(hold["left"], hold["right"])]
        p = env.data.body(obj).xpos.copy()
        g = env.data.body(target).xpos.copy() if target else p + [0, -0.15, 0]
        sides = sorted(("left", "right"), key=lambda s: np.linalg.norm(env.data.site(f"{s}_tcp").xpos - p))
        yaws = [np.pi / 2, 0.0] if obj == "banana" else [0.0, np.pi / 2, np.pi / 4, -np.pi / 4]
        error: Exception | None = None
        for side in sides:
            place = g.copy()
            if behavior == "released_near_target":
                place[0] += 0.10 if side == "left" else -0.10
            for yaw in yaws:
                try:
                    return self._script(side, yaw, p, place, GRASP_DZ.get(obj, 0.003),
                                        0.07 if target == "bowl" else 0.06, behavior, hold)
                except ValueError as e:
                    error = e
        raise RuntimeError(f"mock: no reachable grasp for {obj}: {error}")

    def _script(self, side, yaw, p, place, dz, place_dz, behavior, hold):
        env = self.env
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]])
        other = "right" if side == "left" else "left"
        q = hold[side][:6].copy()
        ids = [env.model.joint(f"{side}_joint{i}").id for i in range(1, 7)]
        limits = env.model.jnt_range[ids]
        rows: list[np.ndarray] = []

        def move(target, opening, n):
            nonlocal q
            end = self.solve(env, side, np.asarray(target, dtype=float), rot, q)
            start = q.copy()
            for a in np.linspace(0, 1, self._n(n)):
                b = a * a * (3 - 2 * a)
                rows.append(np.r_[start + (end - start) * b, opening])
            q = end

        def grip(a, b, n):
            rows.extend(np.r_[q, o] for o in np.linspace(a, b, self._n(n)))

        def wait(opening, n):
            rows.extend(np.r_[q, opening] for _ in range(n))

        def shake(joints, amp, n, opening=0.0):
            for k in range(n):
                d = np.zeros(6)
                d[joints] = amp * (1 if k % 2 == 0 else -1)
                rows.append(np.r_[np.clip(q + d, limits[:, 0], limits[:, 1]), opening])

        wait(1, self._n(10))
        if behavior == "stalled":
            move(p + [0.0, 0.0, 0.10], 1, 40)  # starts towards the fruit, then freezes above it
            wait(1, 2400)
        elif behavior == "not_reached":
            off = p + [0.0, -0.09, 0.05]
            move(off + [0, 0, 0.08], 1, 40)
            move(off, 1, 30)
            grip(1, 0, 20)
            move(off + [0, 0, 0.1], 0, 40)
            wait(0, 2400)
        elif behavior == "grasped_air":
            move(p + [0, 0, 0.12], 1, 40)
            move(p + [0, 0, 0.065], 1, 30)
            grip(1, 0, 20)
            move(p + [0, 0, 0.14], 0, 40)
            wait(0, 2400)
        elif behavior == "collision":
            beside = p + [0.0, -0.10, 0.0]
            beside[2] = 0.76
            move(beside + [0, 0, 0.10], 1, 40)
            move(beside + [0, 0, -0.035], 1, 30)
            wait(1, 2400)
        else:
            move(p + [0, 0, 0.10], 1, 40)
            move(p + [0, 0, dz], 1, 40)
            grip(1, 0, 20)
            wait(0, 5)
            move(p + [0, 0, 0.12], 0, 40)
            if behavior == "dropped":
                mid = p + (place - p) * 0.35
                move([mid[0], mid[1], p[2] + 0.12], 0, 25)
                grip(0, LOOSE_GRIP, 6)  # the grip loosens mid-carry; the gripper is never opened
                wait(LOOSE_GRIP, 2400)
            elif behavior == "fell_off_table":
                wait(0, self._n(15))
                shake(*FLING_SHAKE)  # a violent wrist flick flings it off the table
                wait(0, 2400)
            elif behavior == "released_off_target":
                away = p[:2] - place[:2]
                away = away / (np.linalg.norm(away) + 1e-9)
                mid = p[:2] + away * 0.06
                move([mid[0], mid[1], p[2] + 0.12], 0, 25)
                grip(0, 1, 6)  # opens the gripper far from the target
                wait(1, 2400)
            else:
                move([place[0], place[1], place[2] + 0.13], 0, 50)
                move([place[0], place[1], place[2] + place_dz], 0, 30)
                grip(0, 1, 15)
                move([place[0], place[1], place[2] + 0.13], 1, 30)
                wait(1, 2400)
        hold_other = hold[other]
        return [(r, hold_other) if side == "left" else (hold_other, r) for r in rows]
