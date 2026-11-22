"""budget-guard: what a robot does when the model's credits run out mid-task — decided before they do.

It adds up what the run's model calls cost (dreamkit.cost, the same numbers Live and History show), warns in the feed
at the budget's thresholds and, at the stop threshold, calls a `budget_exhausted` verdict. It only watches (observe):
the run's one control plugin, recovery, acts on that verdict (its stop_run_on): arms home and the run ends, or, with
at_stop "finish", the attempt finishes first. Without recovery on it still warns, but nothing stops.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from dreamkit import cost
from dreamkit.plugin import Plugin, PluginContext
from dreamkit.standard import Manifest


class Params(BaseModel):
    budget_usd: float = Field(5.0, gt=0, le=100000, description="The most this run may spend on the model, in dollars",
                              json_schema_extra={"unit": ""})
    warn_pct: list[int] = Field([50, 80], description="Warn in the feed when this many percent of the budget are used",
                                json_schema_extra={"unit": ""})
    stop_pct: int = Field(95, ge=1, le=100, description="At this many percent of the budget the run stops safely",
                          json_schema_extra={"unit": ""})
    at_stop: str = Field("now", description="now: arms home and the run stops at once; finish: the attempt "
                         "finishes first", json_schema_extra={"choices": ["now", "finish"]})
    count_mock: bool = Field(False, description="Bill a mock run as Dream-1 (to try the stop without credits)",
                             json_schema_extra={"unit": ""})


class BudgetGuard(Plugin):
    """Warns as the model's credit budget is used and stops the run safely before it runs out."""

    name = "budget-guard"
    Params = Params
    manifest = Manifest(group="safety", icon="Wallet", capability="observe",
                        hooks=["on_run_start", "on_episode_start", "after_model", "on_episode_end"],
                        ui="ui/index.html")

    def on_run_start(self, ctx: PluginContext) -> None:
        self.warned: set[int] = set()
        self.stopped = False
        self.at_start = 0.0  # spent when the attempt began

    def _spent(self, ctx: PluginContext) -> float:
        """What the run has cost, as the core counts it (ctx.cost); a mock bills nothing unless count_mock asks to
        try the budget as if Dream-1 had been called."""
        c = ctx.cost
        if ctx.mock and self.params.count_mock:
            return c["run_calls"] * c["dream_price_per_hour"] / cost.CALLS_PER_HOUR
        return float(c["run_usd"])

    def _publish(self, ctx: PluginContext) -> None:
        ctx.ui.series("budget_usd", ctx.t, self.params.budget_usd)  # Live: "of $Y", the bar
        if ctx.mock and self.params.count_mock:  # the core bills a mock nothing; show what this test counts
            ctx.ui.series("spent_usd", ctx.t, round(self._spent(ctx), 4))

    def on_episode_start(self, ctx: PluginContext) -> None:
        self.at_start = self._spent(ctx)
        self._publish(ctx)

    def after_model(self, ctx: PluginContext) -> None:
        if "overridden_by" in ctx.store:  # rows a plugin sent instead of the model: not billed
            return
        self._publish(ctx)
        p = self.params
        spent, budget = self._spent(ctx), p.budget_usd
        used = 100 * spent / budget
        for pct in sorted(p.warn_pct):
            if used >= pct and pct not in self.warned and pct < p.stop_pct:
                self.warned.add(pct)
                message = f"budget {pct}% used: ${spent:.2f} of ${budget:.2f}"
                ctx.ui.mark(ctx.t, message, "warn")
                ctx.emit("warn", message, spent_frac=round(spent / budget, 4), pct=pct)
        if used >= p.stop_pct and not self.stopped:
            self.stopped = True
            how = "the attempt finishes, then the run stops" if p.at_stop == "finish" else "arms home, the run stops"
            message = f"budget {p.stop_pct}% used: ${spent:.2f} of ${budget:.2f}, {how}"
            ctx.ui.mark(ctx.t, message, "error")
            ctx.emit("error", message, spent_frac=round(spent / budget, 4), pct=p.stop_pct)
            ctx.verdict("budget_exhausted", ctx.t, f"budget {p.stop_pct}% used (${spent:.2f} of ${budget:.2f})",
                        source="robot", finish=p.at_stop == "finish")

    def on_episode_end(self, ctx: PluginContext) -> None:
        spent = self._spent(ctx)
        out: dict[str, Any] = {"spent_usd": round(spent, 4), "attempt_usd": round(spent - self.at_start, 4),
                               "budget_usd": self.params.budget_usd}
        if self.stopped:
            out["stopped"] = True
        ctx.result["budget-guard"] = out
