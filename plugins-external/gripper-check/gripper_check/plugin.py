"""gripper-check: a dreamkit plugin that watches the grippers themselves, from the robot's own signals — commanded
vs measured opening — so it works on a real robot as well as in the simulator (PLUGIN_STANDARD.md).

It warns live and never calls a verdict: failure-monitor gives the one final verdict of an attempt and takes these
signals as evidence (ctx.result["gripper-check"]["signals"], written as they happen)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from dreamkit import Manifest, Plugin
from dreamkit.plugin import PluginContext

SIDES = ("left", "right")


class Params(BaseModel):
    closed_cmd: float = Field(0.3, ge=0, le=1, description="Gripper command below which the policy wants it shut",
                              json_schema_extra={"unit": ""})
    empty_below: float = Field(0.04, ge=0, le=1, description="Measured opening below this while shut means nothing "
                               "is between the fingers", json_schema_extra={"unit": ""})
    loosen_by: float = Field(0.08, ge=0, le=1, description="The fingers opening this much while still shut means the "
                             "grip is loosening", json_schema_extra={"unit": ""})
    settle_s: float = Field(0.6, ge=0, le=5, description="How long the fingers get to close before judging",
                            json_schema_extra={"unit": "s"})


class GripperCheck(Plugin):
    """Catches a gripper closed on nothing and a grip that loosens mid-carry."""

    name = "gripper-check"
    Params = Params
    manifest = Manifest(group="diagnosis", icon="Hand", capability="observe",
                        hooks=["on_episode_start", "on_step", "on_episode_end"], step_rate_hz=10, needs=["robot"],
                        ui="ui/index.html")

    def on_episode_start(self, ctx: PluginContext) -> None:
        self.closed_at = {s: None for s in SIDES}
        self.grip = {s: None for s in SIDES}
        self.flagged: dict[str, set[str]] = {s: set() for s in SIDES}
        self.events: list[dict] = []

    def on_step(self, ctx: PluginContext) -> None:
        if ctx.robot.command("left") is None or not ctx.robot.available:
            return
        p = self.params
        for side in SIDES:
            commanded = ctx.robot.grip_command(side)
            measured = ctx.robot.grip(side)
            if commanded >= p.closed_cmd:
                self.closed_at[side], self.grip[side] = None, None
                self.flagged[side].clear()
                continue
            if self.closed_at[side] is None:
                self.closed_at[side] = ctx.t
            if ctx.t - self.closed_at[side] < p.settle_s:
                continue
            if self.grip[side] is None:
                self.grip[side] = measured  # where the fingers stopped: the object's width
                if measured < p.empty_below and "empty" not in self.flagged[side]:
                    self._flag(ctx, side, "empty", f"{side} gripper closed on nothing")
            elif measured - self.grip[side] > p.loosen_by and "loose" not in self.flagged[side]:
                self._flag(ctx, side, "loose", f"{side} grip loosening: fingers opened "
                                               f"{100 * (measured - self.grip[side]):.0f}% while shut")

    def _flag(self, ctx: PluginContext, side: str, kind: str, message: str) -> None:
        self.flagged[side].add(kind)
        self.events.append({"t_s": round(ctx.t, 2), "side": side, "kind": kind})
        # evidence for failure-monitor, as it happens (a warning, not a verdict)
        ctx.result["gripper-check"] = {"signals": list(self.events), "events": list(self.events)}
        ctx.ui.mark(ctx.t, message, "warn")
        ctx.emit("warn", message, side=side, kind=kind)

    def on_episode_end(self, ctx: PluginContext) -> None:
        if self.events:
            ctx.ui.badge(f"gripper {len(self.events)}x", "warn")
