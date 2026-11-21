"""recovery — the one control plugin (PLUGIN_STANDARD.md §3). Detect → Recover.

On a live verdict, fix the attempt from where the robot is; the scene is never reset:
- dropped / released_off_target / not_reached / stalled: arms home (Dream Labs' yam_env.home_path), a new model
  session (client.reset_episode()), the user's loop asks again with the same prompt;
- grasped_air: open that gripper, lift it 5 cm, carry on (same session);
- camera_failure: stop the robot (a blind model must not move it).
When recoveries run out, the attempt may be retried on the same scene (what the old `retry` plugin did)."""

from __future__ import annotations

import importlib

import numpy as np
from pydantic import BaseModel, Field

from dreamkit.plugin import Plugin, PluginContext
from dreamkit.reasons import FAILURES
from dreamkit.standard import Manifest

SIDES = ("left", "right")
STEP_S = 0.05
CHOICES = {"choices": list(FAILURES)}


class Params(BaseModel):
    max_recoveries: int = Field(3, ge=0, le=20, description="Recoveries per attempt before giving up",
                                json_schema_extra={"unit": ""})
    reasons: list[str] = Field(["dropped", "released_off_target", "not_reached", "stalled", "grasped_air"],
                               description="Verdicts to recover from", json_schema_extra=CHOICES)
    stop_on: list[str] = Field(["camera_failure"], description="Verdicts that stop the robot at once",
                               json_schema_extra=CHOICES)
    home_s: float = Field(2.0, gt=0, le=10, description="How long driving the arms home takes",
                          json_schema_extra={"unit": "s"})
    lift_m: float = Field(0.05, gt=0, le=0.3, description="How far to lift a hand that grasped air",
                          json_schema_extra={"unit": "m"})
    lift_s: float = Field(1.0, gt=0, le=10, description="How long that lift takes", json_schema_extra={"unit": "s"})
    home_joints_rad: list[float] = Field([0.0] * 6, description="The home pose on a real robot (six joints); the "
                                         "simulator uses Dream Labs' own home path", json_schema_extra={"unit": "rad"})
    retry_attempts: int = Field(1, ge=0, le=10, description="Retries of a failed scene once recoveries are used up",
                                json_schema_extra={"unit": ""})
    retry_reasons: list[str] = Field(["dropped", "released_off_target", "grasped_air",
                                      "not_reached", "stalled"], description="Final reasons worth a retry",
                                     json_schema_extra=CHOICES)


class Recovery(Plugin):
    """Recovers a failing attempt from where the robot is, without resetting the scene."""

    name = "recovery"
    Params = Params
    manifest = Manifest(group="core", icon="LifeBuoy", capability="control",
                        hooks=["on_episode_start", "before_model", "on_episode_end"], ui="ui/index.html")

    def on_episode_start(self, ctx: PluginContext) -> None:
        self.used = 0
        self.handled: set[tuple[int, str]] = set()
        self.queue: list[tuple[np.ndarray, np.ndarray]] = []
        self.queue_from_step = 0
        self.gave_up = False
        self.actions: list[str] = []

    def before_model(self, ctx: PluginContext) -> None:
        if self.queue:  # still driving an earlier recovery; the loop may execute fewer rows per chunk
            self.queue = self.queue[ctx.step - self.queue_from_step:]
            self.queue_from_step = ctx.step
            if self.queue:
                self._send(ctx)
                return
        verdict = ctx.store.get("verdict")
        if not verdict:
            return
        key = (verdict["segment"], verdict["reason"])
        if key in self.handled:
            return
        self.handled.add(key)
        reason = verdict["reason"]
        if reason in self.params.stop_on:
            ctx.stop(reason, verdict.get("message", ""))
        if reason not in self.params.reasons:
            return
        if self.used >= self.params.max_recoveries:
            if not self.gave_up:
                self.gave_up = True
                ctx.emit("warn", f"{self.used} recoveries used, giving up on this attempt")
            return
        self.used += 1
        ctx.store["segment"] = ctx.store.get("segment", 0) + 1  # failure-monitor judges the next try on its own
        n = f"({self.used}/{self.params.max_recoveries})"
        if reason == "grasped_air" and ctx.world.available:
            side = verdict.get("side") or min(SIDES, key=lambda s: float(ctx.state[s][6]))
            rows = self._lift(ctx, side)
            self.actions.append("lift")
            ctx.emit("info", f"grasped air: opening the {side} gripper and lifting it {self.params.lift_m * 100:.0f} cm, "
                             f"continuing {n}", reason=reason, action="lift")
        elif reason == "grasped_air":  # a real robot: no IK here, so open the hand and ask again from where it is
            side = verdict.get("side") or min(SIDES, key=lambda s: float(ctx.state[s][6]))
            ctx.reset_policy()
            rows = self._open(ctx, side)
            self.actions.append("open")
            ctx.emit("info", f"grasped air: opening the {side} gripper, asking again {n}", reason=reason, action="open")
        else:
            ctx.reset_policy()
            rows = self._home(ctx)
            self.actions.append("home")
            ctx.emit("info", f"{reason.replace('_', ' ')}: arms home, asking again {n}", reason=reason, action="home")
        ctx.ui.badge(f"recovered {self.used}×", "success")
        # History's strip shows where the attempt was taken over (the feed line above says how)
        ctx.ui.mark(ctx.t, f"recovery {self.used}/{self.params.max_recoveries}: {self.actions[-1]} "
                           f"after {reason.replace('_', ' ')}", "info")
        self.queue, self.queue_from_step = rows, ctx.step
        self._send(ctx)

    def _send(self, ctx: PluginContext) -> None:
        rows = self.queue[:48]
        if rows:
            ctx.override([r[0] for r in rows], [r[1] for r in rows])

    def _hold(self, ctx: PluginContext, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        st = ctx.state
        return [(np.asarray(st["left"], dtype=float), np.asarray(st["right"], dtype=float))] * n

    def _home(self, ctx: PluginContext) -> list[tuple[np.ndarray, np.ndarray]]:
        try:  # in the simulator: Dream Labs' own home path, their file untouched
            yam_env = importlib.import_module(type(ctx.env).__module__)
            path = yam_env.home_path(ctx.env.state(), seconds=self.params.home_s, hz=1 / STEP_S)
            return [(np.asarray(a, dtype=float), np.asarray(b, dtype=float)) for a, b in path]
        except (AttributeError, ImportError):
            pass
        # any robot: ease from the measured state to the home pose (home_joints_rad, gripper open)
        state = ctx.robot.state
        if not state:
            return []
        home = np.r_[np.asarray(self.params.home_joints_rad, dtype=float), 1.0]
        steps = max(1, int(round(self.params.home_s / STEP_S)))
        rows = []
        for i in range(1, steps + 1):
            a = i / steps
            a = a * a * (3 - 2 * a)
            rows.append(tuple(state[s] + (home - state[s]) * a for s in SIDES))
        return rows

    def _open(self, ctx: PluginContext, side: str) -> list[tuple[np.ndarray, np.ndarray]]:
        st = ctx.robot.state
        rows = []
        for i in range(1, 11):
            this = st[side].copy()
            this[6] = st[side][6] + (1.0 - st[side][6]) * i / 10
            rows.append((this, st["right"]) if side == "left" else (st["left"], this))
        return rows

    def _lift(self, ctx: PluginContext, side: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """Joint targets that raise one tool point by lift_m, keeping its orientation (damped least squares)."""
        steps = max(1, int(round(self.params.lift_s / STEP_S)))
        env = ctx.env
        if not (hasattr(env, "model") and hasattr(env, "data")):
            return self._hold(ctx, steps)
        import mujoco  # noqa: PLC0415

        m = env.model
        scratch = mujoco.MjData(m)
        mujoco.mj_copyData(scratch, m, env.data)
        qadr = [m.joint(f"{side}_joint{i}").qposadr[0] for i in range(1, 7)]
        vadr = [m.joint(f"{side}_joint{i}").dofadr[0] for i in range(1, 7)]
        lim = np.array([m.joint(f"{side}_joint{i}").range for i in range(1, 7)])
        site = m.site(f"{side}_tcp").id
        mujoco.mj_kinematics(m, scratch)
        goal = scratch.site_xpos[site] + [0, 0, self.params.lift_m]
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
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[err, 0.3 * w]), lim[:, 0], lim[:, 1])
        state = env.state()
        start, other = state[side][:6], "right" if side == "left" else "left"
        rows = []
        for i in range(1, steps + 1):
            a = i / steps
            a = a * a * (3 - 2 * a)
            this = np.r_[start + (q - start) * a, 1.0]  # gripper open: let go of nothing
            rows.append((this, state[other]) if side == "left" else (state[other], this))
        return rows

    def on_episode_end(self, ctx: PluginContext) -> None:
        if self.used:
            worked = ctx.result.get("success") is True
            ctx.result["recovery"] = {"count": self.used, "actions": self.actions, "worked": worked}
            ctx.emit("success" if worked else "info", f"{self.used} recover{'y' if self.used == 1 else 'ies'} in this "
                     f"attempt, {'and then it worked' if worked else 'and it still failed'}")
        reason = ctx.result.get("reason")
        if ctx.result.get("success") is None and not reason:
            ctx.emit("info", "outcome unknown (no world): not retrying; mark it in History")
        if ctx.result.get("success") is False and reason in self.params.retry_reasons:
            if ctx.attempt - 1 < self.params.retry_attempts:
                ctx.request_retry()
                ctx.emit("info", f"retrying scene {ctx.scene + 1} after {reason.replace('_', ' ')} "
                                 f"({ctx.attempt}/{self.params.retry_attempts})")
