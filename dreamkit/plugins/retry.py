"""retry: runs the same scene again when an attempt fails for a reason worth retrying."""

from __future__ import annotations

from ..plugin import Context, Plugin


class Retry(Plugin):
    """Same scene (same seed) again, up to `attempts` more times, only for the listed reasons."""

    name = "retry"
    description = "Retries a failed attempt on the same scene for chosen failure reasons"
    defaults = {"attempts": 2, "on": ["not_reached", "grasped_air", "dropped", "missed_target"]}

    def on_episode_end(self, ctx: Context) -> None:
        if ctx.result.get("success") is not False:
            return
        reason = ctx.result.get("reason")
        on = self.params["on"]
        if on and reason not in on:
            return
        used = ctx.attempt - 1
        if used >= int(self.params["attempts"]):
            ctx.emit("warn", f"no retries left for this scene ({used} used)")
            return
        ctx.request_retry()
        ctx.emit("info", f"retrying scene {ctx.scene + 1} after '{reason}' (retry {used + 1}/{self.params['attempts']})")
