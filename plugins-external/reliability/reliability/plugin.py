"""reliability — how often does it really work, and is B really better than A (PLUGIN_STANDARD.md).

Turns the attempts of a run into numbers you can defend:
- the success rate with a Wilson confidence interval (3 of 3 is "somewhere between 44 % and 100 %", not "100 %");
- why the rest failed, how long successes took, model calls and the cost of one success;
- with `baseline_run`: a scene-by-scene comparison with an earlier run (same seed = same scenes), judged by McNemar's
  exact test, so "the fine-tune is better" is either proven or called what it is: could be luck.

Everything is computed from finished attempts (ctx.episodes), after success-check and failure-monitor have written
their verdicts, so the plugin does not depend on plugin order and costs nothing per step.
Ported from DreamGym's Arena and Compare (the stats module there).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from statistics import NormalDist
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from dreamkit.plugin import Plugin, PluginContext
from dreamkit.standard import Manifest

class Params(BaseModel):
    confidence: float = Field(0.95, ge=0.5, le=0.999, description="Confidence level of the success-rate interval",
                              json_schema_extra={"unit": ""})
    price_per_hour: float = Field(3.0, ge=0, le=100, description="Dollars per robot-hour of the model (Dream-1 3, "
                                  "flash 2); 0 for a mock run", json_schema_extra={"unit": ""})
    calls_per_hour: int = Field(2250, ge=1, description="Model calls billed as one robot-hour (1.6 s of actions each)",
                                json_schema_extra={"unit": ""})
    baseline_run: str = Field("", description="Run id or folder to compare with scene by scene; empty for none")
    pair_on: str = Field("first_attempt", description="Which outcome of a scene to compare",
                         json_schema_extra={"choices": ["first_attempt", "any_attempt"]})
    significance: float = Field(0.05, gt=0, lt=1, description="p-value below which a difference counts as real",
                                json_schema_extra={"unit": ""})


# --- statistics (dependency-free) ---------------------------------------------------------------------------------

def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: honest at small n, unlike p ± z·sqrt(p(1-p)/n)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant scenes (A won alone, B won alone)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p-value for two independent rates (used when the scenes do not pair)."""
    total_k, total = k1 + k2, n1 + n2
    if total == 0 or n1 == 0 or n2 == 0:
        return 1.0

    def prob(a: int) -> float:
        return math.comb(total_k, a) * math.comb(total - total_k, n1 - a) / math.comb(total, n1)

    observed = prob(k1)
    lo, hi = max(0, total_k - n2), min(total_k, n1)
    return min(1.0, sum(p for a in range(lo, hi + 1) if (p := prob(a)) <= observed * (1 + 1e-9)))


def _key(episode: dict[str, Any]) -> Any:
    """What makes two attempts the same scene: the seed the loop passed, else the scene number."""
    return episode.get("seed", episode.get("scene"))


def scene_outcomes(episodes: list[dict[str, Any]], pair_on: str) -> dict[Any, bool]:
    out: dict[Any, bool] = {}
    for e in episodes:
        key = _key(e)
        if pair_on == "first_attempt":
            if e.get("attempt", 1) == 1 and key not in out:
                out[key] = bool(e.get("success"))
        else:
            out[key] = out.get(key, False) or bool(e.get("success"))
    return out


def summarize(episodes: list[dict[str, Any]], z: float, price_per_hour: float, calls_per_hour: int) -> dict[str, Any]:
    n = len(episodes)
    k = sum(1 for e in episodes if e.get("success"))
    lo, hi = wilson(k, n, z)
    calls = sum(int(e.get("requests") or 0) for e in episodes)
    cost = calls * price_per_hour / calls_per_hour
    times = [float(e["duration_s"]) for e in episodes if e.get("success") and e.get("duration_s") is not None]
    reasons = Counter(e.get("reason") or "unknown" for e in episodes if not e.get("success"))
    return {
        "attempts": n, "successes": k, "rate": round(k / n, 4) if n else 0.0, "interval": [round(lo, 4), round(hi, 4)],
        "failures": dict(reasons.most_common()),
        "mean_time_to_success_s": round(sum(times) / len(times), 2) if times else None,
        "model_calls": calls, "cost_usd": round(cost, 4), "cost_per_success_usd": round(cost / k, 4) if k else None,
    }


def compare(base: list[dict[str, Any]], this: list[dict[str, Any]], pair_on: str) -> dict[str, Any]:
    a, b = scene_outcomes(base, pair_on), scene_outcomes(this, pair_on)
    shared = sorted(set(a) & set(b), key=str)
    if shared:
        only_a = sum(1 for s in shared if a[s] and not b[s])
        only_b = sum(1 for s in shared if b[s] and not a[s])
        ra, rb = sum(a[s] for s in shared) / len(shared), sum(b[s] for s in shared) / len(shared)
        return {"paired": True, "scenes": len(shared), "rate_base": round(ra, 4), "rate_this": round(rb, 4),
                "delta": round(rb - ra, 4), "only_base_won": only_a, "only_this_won": only_b,
                "p_value": round(mcnemar_exact(only_a, only_b), 4), "test": "mcnemar_exact"}
    ka, na, kb, nb = sum(a.values()), len(a), sum(b.values()), len(b)
    ra, rb = (ka / na if na else 0.0), (kb / nb if nb else 0.0)
    return {"paired": False, "scenes": 0, "rate_base": round(ra, 4), "rate_this": round(rb, 4),
            "delta": round(rb - ra, 4), "p_value": round(fisher_exact(ka, na, kb, nb), 4), "test": "fisher_exact"}


# --- the plugin ------------------------------------------------------------------------------------------------------

class Reliability(Plugin):
    """Measures the success rate with its confidence interval and proves A/B differences."""

    name = "reliability"
    Params = Params
    manifest = Manifest(group="output", icon="ChartNoAxesColumn", capability="observe",
                        hooks=["on_run_start", "on_episode_start", "on_run_end"], step_rate_hz=1)

    def on_run_start(self, ctx: PluginContext) -> None:
        self.z = NormalDist().inv_cdf(0.5 + self.params.confidence / 2)  # 0.95 -> 1.96
        self.baseline: list[dict[str, Any]] | None = None
        self.baseline_id = ""
        if self.params.baseline_run:
            path = Path(self.params.baseline_run).expanduser()
            if not path.is_absolute() and ctx.run_dir is not None:
                path = Path(ctx.run_dir).parent / self.params.baseline_run
            run = path / "run.json" if path.is_dir() else path
            try:
                data = json.loads(run.read_text())
                self.baseline, self.baseline_id = data.get("episodes", []), str(data.get("id", path.name))
                ctx.emit("info", f"comparing with run {self.baseline_id}, {len(self.baseline)} attempts",
                         attempts=len(self.baseline))
            except (OSError, ValueError) as error:
                ctx.emit("warn", f"baseline run {self.params.baseline_run} not readable: {type(error).__name__}")

    def _stats(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        return summarize(episodes, self.z, self.params.price_per_hour, self.params.calls_per_hour)

    def on_episode_start(self, ctx: PluginContext) -> None:
        done = ctx.episodes  # finished attempts, verdicts final
        if not done:
            return
        s = self._stats(done)
        lo, hi = s["interval"]
        ctx.ui.metric("success rate", 100 * s["rate"], unit="%")
        ctx.ui.badge(f"success {s['successes']}/{s['attempts']} · {100 * s['rate']:.0f}% "
                     f"({100 * lo:.0f}–{100 * hi:.0f}%)", level="info")

    def on_run_end(self, ctx: PluginContext) -> None:
        episodes = ctx.episodes
        if not episodes:
            return
        s = self._stats(episodes)
        lo, hi = s["interval"]
        level = "success" if s["successes"] else "warn"
        ctx.emit(level, f"success {s['successes']} of {s['attempts']} attempts, {100 * s['rate']:.0f}% "
                 f"({100 * self.params.confidence:.0f}% interval {100 * lo:.0f}–{100 * hi:.0f}%)",
                 successes=s["successes"], attempts=s["attempts"], rate_pct=round(100 * s["rate"], 1),
                 low_pct=round(100 * lo, 1), high_pct=round(100 * hi, 1),
                 cost_cents=round(100 * s["cost_usd"]), model_calls=s["model_calls"])
        out: dict[str, Any] = {"run": ctx.run_id, "model": ctx.model, "task": ctx.task,
                               "confidence": self.params.confidence, **s}
        if self.baseline is not None:
            c = compare(self.baseline, episodes, self.params.pair_on)
            real = c["p_value"] < self.params.significance
            verdict = ("better" if c["delta"] > 0 else "worse") if real else "not proven: could be luck"
            how = f"on {c['scenes']} shared scenes" if c["paired"] else "scenes do not pair, unpaired test"
            ctx.emit("success" if real and c["delta"] > 0 else ("error" if real else "info"),
                     f"vs run {self.baseline_id}: {100 * c['rate_this']:.0f}% vs {100 * c['rate_base']:.0f}% "
                     f"{how}, {100 * c['delta']:+.0f} pp, {verdict}",
                     delta_pct=round(100 * c["delta"], 1), p_value_frac=float(c["p_value"]),
                     scenes=c["scenes"], significant=real)
            out["baseline"] = {"run": self.baseline_id, **self._stats(self.baseline), "comparison": c,
                               "verdict": verdict}
        ctx.defer(self._write, ctx.files.path("summary.json"), out)

    @staticmethod
    def _write(path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, indent=2))
