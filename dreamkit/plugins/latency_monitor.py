"""latency-monitor: warns when Dream-1 answers slowly; mean and p95 per attempt and per run."""

from __future__ import annotations

import numpy as np

from ..plugin import Context, Plugin


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    a = np.asarray(values)
    return {"mean_ms": round(float(a.mean()), 1), "p95_ms": round(float(np.percentile(a, 95)), 1),
            "max_ms": round(float(a.max()), 1), "n": len(values)}


class LatencyMonitor(Plugin):
    """Above ~700 ms of round trip the arm waits long enough to hurt control."""

    name = "latency-monitor"
    description = "Warns on slow model round trips; reports mean and p95 latency"
    defaults = {"warn_ms": 700.0}

    def on_run_start(self, ctx: Context) -> None:
        ctx.store["latency_all"] = []

    def on_episode_start(self, ctx: Context) -> None:
        self.rtts: list[float] = []
        self.parts: dict[str, list[float]] = {}

    def after_model(self, ctx: Context) -> None:
        timing = ctx.timing or {}
        rtt = timing.get("rtt_ms")
        if rtt is None:
            return
        rtt = float(rtt)
        self.rtts.append(rtt)
        ctx.store.setdefault("latency_all", []).append(rtt)
        for key in ("serve_infer_ms", "network_ms"):
            if timing.get(key) is not None:
                self.parts.setdefault(key, []).append(float(timing[key]))
        if rtt > float(self.params["warn_ms"]):
            detail = ", ".join(f"{k.removesuffix('_ms')} {timing[k]:.0f} ms" for k in ("serve_infer_ms", "network_ms")
                               if timing.get(k) is not None)
            ctx.emit("warn", f"slow model reply: {rtt:.0f} ms" + (f" ({detail})" if detail else ""), rtt_ms=rtt)

    def on_episode_end(self, ctx: Context) -> None:
        stats = _stats(self.rtts)
        if stats:
            stats.update({k.removesuffix("_ms") + "_mean_ms": round(float(np.mean(v)), 1)
                          for k, v in self.parts.items() if v})
            ctx.result["latency"] = stats

    def on_run_end(self, ctx: Context) -> None:
        stats = _stats(ctx.store.get("latency_all", []))
        if stats:
            ctx.store["latency"] = stats
            level = "warn" if stats["p95_ms"] > float(self.params["warn_ms"]) else "info"
            ctx.emit(level, f"latency over {stats['n']} calls: mean {stats['mean_ms']:.0f} ms, "
                            f"p95 {stats['p95_ms']:.0f} ms", **stats)
