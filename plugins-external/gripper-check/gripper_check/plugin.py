"""gripper-check: a dreamkit plugin that watches the grippers themselves (PLUGIN_STANDARD.md)."""

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
                        hooks=["on_episode_start", "on_step", "on_episode_end"], step_rate_hz=10)

    def on_episode_start(self, ctx: PluginContext) -> None:
        self.closed_at = {s: None for s in SIDES}
        self.grip = {s: None for s in SIDES}
        self.flagged: dict[str, set[str]] = {s: set() for s in SIDES}
        self.events: list[dict] = []

    def on_step(self, ctx: PluginContext) -> None:
        if ctx.command is None or not ctx.world.available:
            return
        p = self.params
        for i, side in enumerate(SIDES):
            commanded = float(ctx.command[i][6])
            measured = ctx.world.grip(side)
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
        ctx.ui.mark(ctx.t, message, "warn")
        ctx.emit("warn", message, side=side, kind=kind)

    def on_episode_end(self, ctx: PluginContext) -> None:
        if self.events:
            ctx.result["gripper-check"] = {"events": self.events}
            ctx.ui.badge(f"gripper {len(self.events)}x", "warn")
