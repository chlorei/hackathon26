"""DreamGym's summaries of its evaluation rows, on dreamkit.stats (Wilson, McNemar, Fisher): one source of numbers."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


# the one statistics module: History, reliability and vision-judge use the same functions
from dreamkit.stats import fisher_exact, mcnemar_exact, wilson  # noqa: E402,F401


def summarize(results: Iterable[Mapping]) -> dict:
    """Success rate with its interval, failure breakdown, time and money per success."""
    rows = list(results)
    n = len(rows)
    k = sum(1 for r in rows if r["success"])
    cost = sum(r["cost_usd"] for r in rows)
    times = [r["seconds_to_success"] for r in rows if r["success"] and r["seconds_to_success"] is not None]
    lo, hi = wilson(k, n)
    return {
        "episodes": n,
        "successes": k,
        "success_rate": round(k / n, 4) if n else 0.0,
        "ci95": [round(lo, 4), round(hi, 4)],
        "failures": dict(Counter(r["reason"] for r in rows if not r["success"]).most_common()),
        "mean_seconds_to_success": round(sum(times) / len(times), 2) if times else None,
        "calls": sum(r["calls"] for r in rows),
        "cost_usd": round(cost, 4),
        "cost_per_success_usd": round(cost / k, 4) if k else None,
        "wall_minutes": round(sum(r["wall_seconds"] for r in rows) / 60, 1),
    }


def paired(a: Iterable[Mapping], b: Iterable[Mapping]) -> dict:
    """Compares two runs on the seeds both ran."""
    by_seed_a = {r["seed"]: r["success"] for r in a}
    by_seed_b = {r["seed"]: r["success"] for r in b}
    seeds = sorted(by_seed_a.keys() & by_seed_b.keys())
    only_a = sum(1 for s in seeds if by_seed_a[s] and not by_seed_b[s])
    only_b = sum(1 for s in seeds if by_seed_b[s] and not by_seed_a[s])
    ka = sum(by_seed_a[s] for s in seeds)
    kb = sum(by_seed_b[s] for s in seeds)
    n = len(seeds)
    return {
        "seeds": n,
        "rate_a": round(ka / n, 4) if n else 0.0,
        "rate_b": round(kb / n, 4) if n else 0.0,
        "delta": round((kb - ka) / n, 4) if n else 0.0,
        "only_a_succeeded": only_a,
        "only_b_succeeded": only_b,
        "p_value": round(mcnemar_exact(only_a, only_b), 4),
    }
