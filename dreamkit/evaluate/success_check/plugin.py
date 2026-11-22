"""success-check: whether the task was done, from the simulator's physics — an evaluation tool (PLUGIN_STANDARD.md §6).

It reads the world (where the object is), which a real arm does not report, so it never runs in the working loop:
only dreamkit doctor --sim and the replay tests use it, as the right answer to measure against. In the working loop
success comes from vision-judge or a person."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from dreamkit.plugin import Plugin, PluginContext
from dreamkit.standard import Manifest


class Params(BaseModel):
    radius_bowl_m: float = Field(0.07, gt=0, le=0.3, description="How close to the bowl's centre counts as in the bowl",
                                 json_schema_extra={"unit": "m"})
    radius_plate_m: float = Field(0.09, gt=0, le=0.3, description="How close to the plate's centre counts as on it",
                                  json_schema_extra={"unit": "m"})
    below_m: float = Field(0.01, ge=0, le=0.1, description="How far below the target's base the object may sit",
                           json_schema_extra={"unit": "m"})
    above_m: float = Field(0.12, ge=0, le=0.5, description="How far above the target's base the object may sit",
                           json_schema_extra={"unit": "m"})
    hold_s: float = Field(1.0, ge=0, le=10, description="How long it must stay there, released, before it counts",
                          json_schema_extra={"unit": "s"})
    open_above: float = Field(0.5, ge=0, le=1, description="Gripper command above which the hand has let go",
                              json_schema_extra={"unit": ""})
    early_stop: bool = Field(True, description="End the attempt as soon as it succeeds")


class SuccessCheck(Plugin):
    """Detects success: the object rests in the target zone and the gripper has let go."""

    name = "success-check"
    Params = Params
    manifest = Manifest(group="evaluation", icon="CircleCheck", capability="observe",
                        hooks=["on_episode_start", "on_step", "on_episode_end"], step_rate_hz=20,
                        needs=["robot", "world"], ui="ui/index.html")

    def on_episode_start(self, ctx: PluginContext) -> None:
        self.obj, self.target = ctx.target["object"], ctx.target["target"]
        self.since: float | None = None
        if not ctx.world.available:
            self.obj = self.target = None  # no simulator: success is the user's to report
        elif self.obj is None or self.target is None:
            ctx.emit("warn", f"cannot read object and target from the task {ctx.task!r}; success unknown")

    def _inside(self, ctx: PluginContext) -> bool:
        if self.obj is None or self.target is None:
            return False
        p = self.params
        obj, tgt = ctx.world.pos(self.obj), ctx.world.pos(self.target)
        radius = p.radius_bowl_m if self.target == "bowl" else p.radius_plate_m
        return bool(float(np.linalg.norm(obj[:2] - tgt[:2])) < radius and tgt[2] - p.below_m < obj[2] < tgt[2] + p.above_m)

    def _released(self, ctx: PluginContext) -> bool:
        return ctx.command is None or all(float(row[6]) >= self.params.open_above for row in ctx.command)

    def on_step(self, ctx: PluginContext) -> None:
        if not self.params.early_stop or self.obj is None:
            return
        if self._inside(ctx) and self._released(ctx):
            self.since = ctx.t if self.since is None else self.since
            if ctx.t - self.since >= self.params.hold_s:
                ctx.result["success"] = True  # the core ends the attempt
                ctx.emit("success", f"{self.obj} is in the {self.target} at {ctx.t:.1f} s", t_s=round(ctx.t, 2))
        else:
            self.since = None

    def on_episode_end(self, ctx: PluginContext) -> None:
        if self.obj is None:
            return
        if ctx.result.get("success") is not True:
            ok = bool(self._inside(ctx))
            ctx.result["success"] = ok
            if ok:
                ctx.emit("success", f"{self.obj} is in the {self.target}")
        ctx.result["success-check"] = {"object_final_m": [round(float(v), 4) for v in ctx.world.pos(self.obj)]}
