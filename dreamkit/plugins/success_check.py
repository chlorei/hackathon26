"""success-check: decides whether the task was done, from the simulator's physics."""

from __future__ import annotations

import numpy as np

from ..plugin import Context, Plugin
from ..task import parse


def in_target(obj_pos, target_pos, radius: float, below: float, above: float) -> bool:
    horizontal = float(np.linalg.norm(np.asarray(obj_pos[:2]) - np.asarray(target_pos[:2])))
    return horizontal < radius and target_pos[2] - below < obj_pos[2] < target_pos[2] + above


class SuccessCheck(Plugin):
    """Object in the bowl / on the plate, gripper open: success. Ends the attempt early once it holds for hold_s."""

    name = "success-check"
    description = "Detects task success: object inside the target zone with the gripper released"
    defaults = {
        "radius": {"bowl": 0.07, "plate": 0.09},
        "below_m": 0.01,
        "above_m": 0.12,
        "hold_s": 1.0,
        "open_above": 0.5,  # both grippers commanded at least this open
        "early_stop": True,
    }

    def on_episode_start(self, ctx: Context) -> None:
        self.obj, self.target = parse(ctx.task)
        self.since = None
        if self.obj is None or self.target is None:
            ctx.emit("warn", f"cannot read object/target from task '{ctx.task}'; success unknown")

    def check(self, ctx: Context) -> bool:
        if self.obj is None or self.target is None:
            return False
        radius = self.params["radius"]
        r = radius.get(self.target, 0.08) if isinstance(radius, dict) else float(radius)
        data = ctx.env.data
        return in_target(data.body(self.obj).xpos, data.body(self.target).xpos, r,
                         self.params["below_m"], self.params["above_m"])

    def _released(self, ctx: Context) -> bool:
        if ctx.command is None:
            return True
        return all(float(row[6]) >= self.params["open_above"] for row in ctx.command)

    def on_step(self, ctx: Context) -> None:
        if not self.params["early_stop"]:
            return
        if self.check(ctx) and self._released(ctx):
            if self.since is None:
                self.since = ctx.t
            elif ctx.t - self.since >= self.params["hold_s"]:
                self._set(ctx, True)
                ctx.end_episode("success")
        else:
            self.since = None

    def _set(self, ctx: Context, ok: bool) -> None:
        if "success" in ctx.result and ctx.result.get("reason") in ("stopped", "stopped_by_plugin", "model_error"):
            return  # the attempt was cut short; that reason stands
        ctx.result["success"] = ok
        if ok:
            ctx.result["reason"] = "success"
            ctx.result.setdefault("at", round(ctx.t, 2))
        pos = ctx.env.data.body(self.obj).xpos if self.obj else None
        if pos is not None:
            ctx.result["object_final"] = [round(float(v), 4) for v in pos]

    def on_episode_end(self, ctx: Context) -> None:
        if self.obj is None or self.target is None:
            return
        if ctx.result.get("success") is True:
            ctx.emit("success", f"{self.obj} is in the {self.target}")
            return
        ok = self.check(ctx)
        self._set(ctx, ok)
        if ok:
            ctx.emit("success", f"{self.obj} is in the {self.target}")
