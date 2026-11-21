"""recovery: when failure-monitor calls a failure mid-attempt, fix it from where the robot is.

The scene is never reset (a real robot cannot teleport the apple back). Instead:
- dropped / released_off_target / not_reached / stalled: drive the arms home (yam_env.home_path), start a new
  session on the model (client.reset_episode()), and the user's loop asks again with the same prompt;
- grasped_air: open the gripper and lift that hand 5 cm, then let the model carry on (same session).
At most `max_recoveries` per attempt. Every action is an event in the feed.
"""

from __future__ import annotations

import importlib
from typing import Any

import mujoco
import numpy as np

from ..plugin import Context, Plugin

SIDES = ("left", "right")
STEP_S = 0.05


class Recovery(Plugin):
    """Detect → Recover: turns a failed attempt into another try without touching the scene."""

    name = "recovery"
    group = "control"
    description = "Fixes a failed attempt from where the robot is: arms home and ask again, or back off and continue"
    defaults = {
        "max_recoveries": 3,
        "on": ["dropped", "released_off_target", "not_reached", "stalled", "grasped_air"],
        "home_s": 2.0,
        "lift_m": 0.05,
        "lift_s": 1.0,
    }

    def on_episode_start(self, ctx: Context) -> None:
        self.used = 0
        self.handled: set[int] = set()
        self.queue: list[tuple[np.ndarray, np.ndarray]] = []
        self.queue_from_step = 0
        self.warned_out = False
        ctx.store["recoveries"] = 0

    def before_model(self, ctx: Context) -> None:
        # still driving rows of an earlier recovery? the loop may execute fewer rows per chunk than we planned
        if self.queue:
            done = ctx.step - self.queue_from_step
            self.queue = self.queue[done:]
            self.queue_from_step = ctx.step
            if self.queue:
                self._send(ctx)
                return
        verdict = ctx.store.get("verdict")
        if not verdict or verdict["segment"] in self.handled or verdict["reason"] not in self.params["on"]:
            return
        self.handled.add(verdict["segment"])
        if self.used >= int(self.params["max_recoveries"]):
            if not self.warned_out:
                self.warned_out = True
                ctx.emit("warn", f"recovery: {self.used} recoveries used, giving up on this attempt")
            return
        if not (hasattr(ctx.env, "data") and callable(getattr(ctx.env, "state", None))):
            ctx.emit("info", "recovery needs the simulator env in episode()/step()")
            return
        self.used += 1
        ctx.store["recoveries"] = self.used
        ctx.store["segment"] = ctx.store.get("segment", 0) + 1  # failure-monitor judges the next try on its own
        n = f"({self.used}/{self.params['max_recoveries']})"
        reason = verdict["reason"]
        if reason == "grasped_air":
            side = verdict.get("side") or self._closest_side(ctx)
            rows = self._lift(ctx, side)
            ctx.emit("info", f"recovery: grasped air — opening the {side} gripper and lifting it "
                             f"{self.params['lift_m'] * 100:.0f} cm, continuing {n}", reason=reason, action="lift")
        else:
            ctx.reset_policy()
            rows = self._home(ctx)
            ctx.emit("info", f"recovery: {reason.replace('_', ' ')} — arms home, asking again {n}", reason=reason,
                     action="home")
        self.queue = rows
        self.queue_from_step = ctx.step
        self._send(ctx)

    def _send(self, ctx: Context) -> None:
        rows = self.queue[:48]
        ctx.override([r[0] for r in rows], [r[1] for r in rows])

    def _closest_side(self, ctx: Context) -> str:
        return min(SIDES, key=lambda s: float(ctx.state[s][6]))

    def _home(self, ctx: Context) -> list[tuple[np.ndarray, np.ndarray]]:
        yam_env = importlib.import_module(type(ctx.env).__module__)  # Dream Labs' own home_path, file untouched
        path = yam_env.home_path(ctx.env.state(), seconds=float(self.params["home_s"]), hz=1 / STEP_S)
        return [(np.asarray(left, dtype=float), np.asarray(right, dtype=float)) for left, right in path]

    def _lift(self, ctx: Context, side: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """Joint targets that raise one tool point by lift_m, keeping its orientation (damped least squares)."""
        m, env = ctx.env.model, ctx.env
        scratch = mujoco.MjData(m)
        mujoco.mj_copyData(scratch, m, env.data)
        qadr = [m.joint(f"{side}_joint{i}").qposadr[0] for i in range(1, 7)]
        vadr = [m.joint(f"{side}_joint{i}").dofadr[0] for i in range(1, 7)]
        lim = np.array([m.joint(f"{side}_joint{i}").range for i in range(1, 7)])
        site = m.site(f"{side}_tcp").id
        mujoco.mj_kinematics(m, scratch)
        goal = scratch.site_xpos[site] + [0, 0, float(self.params["lift_m"])]
        rot0 = scratch.site_xmat[site].reshape(3, 3).copy()
        q = scratch.qpos[qadr].copy()
        for _ in range(60):
            scratch.qpos[qadr] = q
            mujoco.mj_kinematics(m, scratch)
            mujoco.mj_comPos(m, scratch)
            err = goal - scratch.site_xpos[site]
            r = rot0 @ scratch.site_xmat[site].reshape(3, 3).T
            w = 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
            if np.linalg.norm(err) < 0.002:
                break
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, scratch, jp, jr, site)
            J = np.vstack([jp[:, vadr], 0.3 * jr[:, vadr]])
            e = np.r_[err, 0.3 * w]
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), lim[:, 0], lim[:, 1])
        state = env.state()
        start = state[side][:6]
        steps = max(1, int(round(float(self.params["lift_s"]) / STEP_S)))
        other = "right" if side == "left" else "left"
        rows = []
        for i in range(1, steps + 1):
            a = i / steps
            a = a * a * (3 - 2 * a)
            this = np.r_[start + (q - start) * a, 1.0]  # gripper open: let go of nothing
            rows.append((this, state[other]) if side == "left" else (state[other], this))
        return rows

    def on_episode_end(self, ctx: Context) -> None:
        if self.used:
            ctx.result["recoveries"] = self.used
            outcome = "and then it worked" if ctx.result.get("success") else "and it still failed"
            ctx.emit("success" if ctx.result.get("success") else "info",
                     f"recovery: {self.used} recover{'y' if self.used == 1 else 'ies'} in this attempt, {outcome}")
