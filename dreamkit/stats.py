"""The one place dreamkit's numbers come from: how sure is "7 of 10", is A really better than B, what a success costs.

History, the Data tab (DreamGym), reliability and vision-judge all call these functions, so the same attempts give
the same numbers on every screen. Dependency-free (math only).

Which attempts count: an attempt whose outcome is known (success True or False) and whose task dreamkit could read
(task_readable is not False). Unknown outcomes (a robot without a world, not labelled yet) and invalid attempts are
reported apart, never in the rate.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

Z95 = 1.96


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a success rate (95% by default); honest at small n, unlike p ± z·sqrt(p(1-p)/n)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value for paired outcomes (the same scenes under A and B).

    only_a: scenes where A succeeded and B failed; only_b: the reverse. Pairing removes the scene's own difficulty,
    so it needs far fewer attempts than comparing two rates."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p-value for two independent rates k1/n1 vs k2/n2 (runs that are not paired)."""
    total_k, total = k1 + k2, n1 + n2
    if total == 0:
        return 1.0

    def prob(a: int) -> float:
        return math.comb(total_k, a) * math.comb(total - total_k, n1 - a) / math.comb(total, n1)

    observed = prob(k1)
    lo, hi = max(0, total_k - n2), min(total_k, n1)
    return min(1.0, sum(p for a in range(lo, hi + 1) if (p := prob(a)) <= observed * (1 + 1e-9)))


def rate(successes: int, n: int, z: float = Z95) -> dict[str, Any]:
    """{successes, attempts, success_rate, ci95}: the rate and its interval, rounded the same way everywhere."""
    lo, hi = wilson(successes, n, z)
    return {"successes": successes, "attempts": n, "success_rate": round(successes / n, 4) if n else 0.0,
            "ci95": [round(lo, 4), round(hi, 4)]}


def counts(e: Mapping[str, Any]) -> bool:
    """Whether an attempt counts in a rate: its outcome is known and its task could be read."""
    return e.get("success") is not None and e.get("task_readable") is not False


def _latency_ms(episodes: list[Mapping[str, Any]]) -> float | None:
    lat = [(e.get("failure-monitor") or {}).get("latency") or {} for e in episodes]
    lat = [x for x in lat if x.get("mean_ms") is not None and x.get("calls")]
    calls = sum(int(x["calls"]) for x in lat)
    return round(sum(float(x["mean_ms"]) * int(x["calls"]) for x in lat) / calls, 1) if calls else None


def summary(episodes: Iterable[Mapping[str, Any]], z: float = Z95) -> dict[str, Any]:
    """dreamkit attempts (run.json episodes) -> the numbers every screen shows.

    {attempts, successes, success_rate, ci95, unknown, invalid, failures {code: n}, top_reason, cost_usd,
     cost_per_success_usd, model_calls, mean_latency_ms}. Money is what the core recorded per attempt (cost_usd,
    dreamkit.cost); failure codes are the shared ones (old names mapped)."""
    from .reasons import code

    rows = list(episodes)
    judged = [e for e in rows if counts(e)]
    k = sum(1 for e in judged if e.get("success"))
    failures = Counter(code(e.get("reason")) or "unknown" for e in judged if e.get("success") is False)
    money = sum(float(e.get("cost_usd") or 0.0) for e in rows)
    return {
        **rate(k, len(judged), z),
        "unknown": sum(1 for e in rows if e.get("success") is None and e.get("task_readable") is not False),
        "invalid": sum(1 for e in rows if e.get("task_readable") is False),
        "failures": dict(failures.most_common()),
        "top_reason": failures.most_common(1)[0][0] if failures else None,
        "cost_usd": round(money, 4),
        "cost_per_success_usd": round(money / k, 4) if k else None,
        "model_calls": sum(int(e.get("requests") or 0) for e in rows),
        "mean_latency_ms": _latency_ms([e for e in judged if e.get("reason") == "slow_model"] or judged),
    }


def paired(a: Mapping[Any, bool], b: Mapping[Any, bool]) -> dict[str, Any]:
    """Two variants on the same scenes ({scene key: success}) -> rates on the shared scenes and McNemar's p."""
    keys = sorted(set(a) & set(b), key=str)
    only_a = sum(1 for s in keys if a[s] and not b[s])
    only_b = sum(1 for s in keys if b[s] and not a[s])
    ka, kb, n = sum(bool(a[s]) for s in keys), sum(bool(b[s]) for s in keys), len(keys)
    return {"seeds": n, "rate_a": round(ka / n, 4) if n else 0.0, "rate_b": round(kb / n, 4) if n else 0.0,
            "delta": round((kb - ka) / n, 4) if n else 0.0, "only_a_succeeded": only_a, "only_b_succeeded": only_b,
            "p_value": round(mcnemar_exact(only_a, only_b), 4), "test": "mcnemar_exact"}
