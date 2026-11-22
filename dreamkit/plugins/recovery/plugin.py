"""recovery — the one control plugin (PLUGIN_STANDARD.md §3). Detect → Recover.

On a live verdict, fix the attempt from where the robot is, with nothing but joint commands (as on a real arm);
the scene is never reset:
- dropped / released_early / collision / stalled: arms home (home_joints_rad), a new model session
  (client.reset_episode()), the user's loop asks again with the same prompt;
- grasped_air: open that gripper and ask again from where the arm is;
- camera_failure: stop the robot (a blind model must not move it).
When recoveries run out, the attempt may be retried on the same scene (what the old `retry` plugin did)."""

from __future__ import annotations

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
    reasons: list[str] = Field(["dropped", "released_early", "collision", "stalled", "grasped_air"],
                               description="Verdicts to recover from", json_schema_extra=CHOICES)
    stop_on: list[str] = Field(["camera_failure"], description="Verdicts that stop the robot at once",
                               json_schema_extra=CHOICES)
    stop_run_on: list[str] = Field(["budget_exhausted"], description="Verdicts that end the whole run safely: "
                                   "arms home, then no new attempt (or, if the verdict says so, after this one)",
                                   json_schema_extra={"choices": ["budget_exhausted", *FAILURES]})
    home_s: float = Field(2.0, gt=0, le=10, description="How long driving the arms home takes",
                          json_schema_extra={"unit": "s"})
    home_joints_rad: list[float] = Field([0.0] * 6, description="The home pose (six joints, each arm); the "
                                         "gripper opens on the way", json_schema_extra={"unit": "rad"})
    retry_attempts: int = Field(1, ge=0, le=10, description="Retries of a failed scene once recoveries are used up",
                                json_schema_extra={"unit": ""})
    stop_after_model_errors: int = Field(5, ge=0, le=100, description="End the run after this many attempts in a "
                                         "row fail with a model error (0: never)", json_schema_extra={"unit": ""})
    retry_reasons: list[str] = Field(["dropped", "released_early", "grasped_air", "collision", "stalled"],
                                     description="Final reasons worth a retry",
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
        self.halt: str | None = None  # stop_run_on: after the arms are home, end the attempt and the run

    def before_model(self, ctx: PluginContext) -> None:
        if self.queue:  # still driving an earlier recovery; the loop may execute fewer rows per chunk
            self.queue = self.queue[ctx.step - self.queue_from_step:]
            self.queue_from_step = ctx.step
            if self.queue:
                self._send(ctx)
                return
        if self.halt:  # the arms are home: end the attempt (the run was told to stop when the verdict came)
            reason, self.halt = self.halt, None
            ctx.stop(reason, "arms home, run stopped")
        verdict = ctx.store.get("verdict")
        if not verdict:
            return
        key = (verdict["segment"], verdict["reason"])
        if key in self.handled:
            return
        self.handled.add(key)
        reason = verdict["reason"]
        if reason in self.params.stop_run_on:  # the run must end: safely, arms home first unless told to finish
            if verdict.get("finish"):
                ctx.stop_run(reason, f"{verdict.get('message') or reason}: this attempt finishes, then the run stops")
                return
            ctx.stop_run(reason, f"{verdict.get('message') or reason}: arms home, no new attempt")  # whatever ends this one
            ctx.reset_policy()
            rows = self._home(ctx)
            self.actions.append("home")
            ctx.emit("warn", f"{reason.replace('_', ' ')}: arms home, then the run stops", reason=reason, action="home")
            ctx.ui.mark(ctx.t, f"{reason.replace('_', ' ')}: arms home, run stops", "error")
            self.halt = reason
            self.queue, self.queue_from_step = rows, ctx.step
            self._send(ctx)
            return
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
        if reason == "grasped_air":  # open the hand and ask again from where it is
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
        # ease from the measured state to the home pose (home_joints_rad, gripper open): joint commands only
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

    def on_episode_end(self, ctx: PluginContext) -> None:
        # the model failing attempt after attempt: more attempts only spin (the key, credits or network is the cause)
        streak = getattr(self, "model_errors", 0) + 1 if ctx.result.get("reason") == "model_error" else 0
        self.model_errors = streak
        if 0 < self.params.stop_after_model_errors <= streak:
            ctx.stop_run("model_error", f"{streak} attempts in a row ended in a model error: the run stops "
                         "(check the Dream API key, the credits or the network)")
            ctx.emit("error", f"{streak} attempts in a row ended in a model error: stopping the run")
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
