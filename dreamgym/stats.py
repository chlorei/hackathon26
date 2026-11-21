"""Small, dependency-free statistics: how sure is "7 of 10", and is A really better than B."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a success rate; honest at small n (unlike p +- 1.96 sqrt(p(1-p)/n))."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for paired outcomes.

    b: seeds where A succeeded and B failed; c: the reverse. Same seeds = same scenes, so pairing removes the
    scene's own difficulty from the comparison and needs far fewer episodes than comparing two rates.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p-value for two independent rates k1/n1 vs k2/n2 (unpaired runs)."""
    total_k, total = k1 + k2, n1 + n2
    if total == 0:
        return 1.0

    def prob(a: int) -> float:
        return math.comb(total_k, a) * math.comb(total - total_k, n1 - a) / math.comb(total, n1)

    observed = prob(k1)
    lo, hi = max(0, total_k - n2), min(total_k, n1)
    return min(1.0, sum(p for a in range(lo, hi + 1) if (p := prob(a)) <= observed * (1 + 1e-9)))


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
