"""gripper-check: a dreamkit plugin that watches the grippers themselves."""

from dreamkit import Context, Plugin

SIDES = ("left", "right")


class GripperCheck(Plugin):
    """Catches a gripper that closes on nothing (fingers shut all the way) and a grip that loosens mid-carry."""

    name = "gripper-check"
    group = "diagnosis"
    description = "Catches a gripper closed on nothing and a grip that loosens mid-carry"
    defaults = {
        "closed_cmd": 0.3,  # commanded opening below this = the policy wants the gripper shut
        "empty_below": 0.04,  # measured opening below this while shut = nothing between the fingers
        "loosen_by": 0.08,  # measured opening growing this much while still commanded shut = losing the grip
        "settle_s": 0.6,  # fingers need this long to close before we judge
    }

    def on_episode_start(self, ctx: Context) -> None:
        self.closed_at = {s: None for s in SIDES}
        self.grip = {s: None for s in SIDES}
        self.flagged = {s: set() for s in SIDES}
        self.events = []

    def on_step(self, ctx: Context) -> None:
        if ctx.command is None:
            return
        p = self.params
        for i, side in enumerate(SIDES):
            commanded = float(ctx.command[i][6])
            measured = float(ctx.state[side][6])
            if commanded >= p["closed_cmd"]:
                self.closed_at[side], self.grip[side] = None, None
                self.flagged[side].clear()
                continue
            if self.closed_at[side] is None:
                self.closed_at[side] = ctx.t
            if ctx.t - self.closed_at[side] < p["settle_s"]:
                continue
            if self.grip[side] is None:
                self.grip[side] = measured  # how far the fingers stopped: the object's width
                if measured < p["empty_below"] and "empty" not in self.flagged[side]:
                    self.flagged[side].add("empty")
                    self._flag(ctx, side, "empty", f"{side} gripper closed on nothing")
            elif measured - self.grip[side] > p["loosen_by"] and "loose" not in self.flagged[side]:
                self.flagged[side].add("loose")
                self._flag(ctx, side, "loose",
                           f"{side} grip loosening: fingers opened {100 * (measured - self.grip[side]):.0f}% while closed")

    def _flag(self, ctx: Context, side: str, kind: str, message: str) -> None:
        self.events.append({"t": round(ctx.t, 2), "side": side, "kind": kind})
        ctx.emit("warn", message, side=side, kind=kind)

    def on_episode_end(self, ctx: Context) -> None:
        if self.events:
            ctx.result["gripper"] = self.events
