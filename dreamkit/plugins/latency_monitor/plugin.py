"""latency-monitor: above ~700 ms of round trip the arm waits long enough to hurt control (PLUGIN_STANDARD.md)."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from dreamkit.plugin import Plugin, PluginContext
from dreamkit.standard import Manifest


def _stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values)
    return {"mean_ms": round(float(a.mean()), 1), "p95_ms": round(float(np.percentile(a, 95)), 1),
            "max_ms": round(float(a.max()), 1), "calls": len(values)}


class Params(BaseModel):
    warn_ms: float = Field(700.0, gt=0, le=10000, description="Warn when one model round trip takes longer",
                           json_schema_extra={"unit": "ms"})


class LatencyMonitor(Plugin):
    """Warns on slow model replies and reports mean and p95 round trip."""

    name = "latency-monitor"
    Params = Params
    manifest = Manifest(group="safety", icon="Gauge", capability="observe",
                        hooks=["on_run_start", "on_episode_start", "after_model", "on_episode_end", "on_run_end"])

    def on_run_start(self, ctx: PluginContext) -> None:
        self.all: list[float] = []
        self.mock = False

    def on_episode_start(self, ctx: PluginContext) -> None:
        self.rtts: list[float] = []

    def after_model(self, ctx: PluginContext) -> None:
        timing = ctx.timing or {}
        if timing.get("mock"):
            self.mock = True
            return
        if timing.get("rtt_ms") is None:
            return
        rtt = float(timing["rtt_ms"])
        self.rtts.append(rtt)
        self.all.append(rtt)
        ctx.ui.series("rtt", ctx.t, rtt, unit="ms")
        if rtt > self.params.warn_ms:
            ctx.emit("warn", f"round trip {rtt:.0f} ms, slow", rtt_ms=rtt)

    def on_episode_end(self, ctx: PluginContext) -> None:
        if self.rtts:
            stats = _stats(self.rtts)
            ctx.result["latency-monitor"] = stats
            ctx.ui.metric("mean round trip", stats["mean_ms"], unit="ms")
        elif self.mock:
            ctx.result["latency-monitor"] = {"mock": True}

    def on_run_end(self, ctx: PluginContext) -> None:
        if self.all:
            s = _stats(self.all)
            ctx.store["latency-monitor"] = s
            ctx.emit("warn" if s["p95_ms"] > self.params.warn_ms else "info",
                     f"round trip over {s['calls']} calls: mean {s['mean_ms']:.0f} ms, p95 {s['p95_ms']:.0f} ms",
                     mean_ms=s["mean_ms"], p95_ms=s["p95_ms"])
        elif self.mock:
            ctx.store["latency-monitor"] = {"mock": True}
            ctx.emit("info", "mock: no model calls, no latency to report")
