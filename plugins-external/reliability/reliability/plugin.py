"""reliability — how often does it really work, and is B really better than A (PLUGIN_STANDARD.md).

Turns the attempts of a run into numbers you can defend:
- the success rate with a Wilson confidence interval (3 of 3 is "somewhere between 44 % and 100 %", not "100 %");
- why the rest failed, how long successes took, model calls and the cost of one success;
- with `baseline_run`: a scene-by-scene comparison with an earlier run (same seed = same scenes), judged by McNemar's
  exact test, so "the fine-tune is better" is either proven or called what it is: could be luck.

Everything is computed from finished attempts (ctx.episodes), after success-check and failure-monitor have written
their verdicts, so the plugin does not depend on plugin order and costs nothing per step. The numbers come from
dreamkit.stats (the same as History and the Data tab) and the money from what the core recorded per attempt
(dreamkit.cost): this plugin computes nothing of its own.
"""

from __future__ import annotations

import json
from statistics import NormalDist
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from dreamkit import stats
from dreamkit.plugin import Plugin, PluginContext
from dreamkit.standard import Manifest

class Params(BaseModel):
    confidence: float = Field(0.95, ge=0.5, le=0.999, description="Confidence level of the success-rate interval",
                              json_schema_extra={"unit": ""})
    baseline_run: str = Field("", description="Run id or folder to compare with scene by scene; empty for none")
    pair_on: str = Field("first_attempt", description="Which outcome of a scene to compare",
                         json_schema_extra={"choices": ["first_attempt", "any_attempt"]})
    significance: float = Field(0.05, gt=0, lt=1, description="p-value below which a difference counts as real",
                                json_schema_extra={"unit": ""})


# the one statistics module (History, Data, vision-judge use it too)
wilson, mcnemar_exact, fisher_exact = stats.wilson, stats.mcnemar_exact, stats.fisher_exact


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


def summarize(episodes: list[dict[str, Any]], z: float = stats.Z95) -> dict[str, Any]:
    """dreamkit.stats.summary (rate, interval, failures, cost from the core) plus the time a success took."""
    s = stats.summary(episodes, z)
    times = [float(e["duration_s"]) for e in episodes if e.get("success") and e.get("duration_s") is not None]
    return {**s, "rate": s["success_rate"], "interval": s["ci95"],  # the names this plugin's page and events use
            "mean_time_to_success_s": round(sum(times) / len(times), 2) if times else None}


def compare(base: list[dict[str, Any]], this: list[dict[str, Any]], pair_on: str) -> dict[str, Any]:
    a, b = scene_outcomes(base, pair_on), scene_outcomes(this, pair_on)
    shared = sorted(set(a) & set(b), key=str)
    if shared:
        p = stats.paired(a, b)
        return {"paired": True, "scenes": p["seeds"], "rate_base": p["rate_a"], "rate_this": p["rate_b"],
                "delta": p["delta"], "only_base_won": p["only_a_succeeded"], "only_this_won": p["only_b_succeeded"],
                "p_value": p["p_value"], "test": p["test"]}
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
                        hooks=["on_run_start", "on_episode_start", "on_run_end"], step_rate_hz=1,
                        ui="ui/index.html")

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
        return summarize(episodes, self.z)

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
