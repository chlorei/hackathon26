from dreamgym.stats import fisher_exact, mcnemar_exact, paired, summarize, wilson


def test_wilson_known_values():
    lo, hi = wilson(7, 10)
    assert round(lo, 3) == 0.397 and round(hi, 3) == 0.892
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(0, 20)
    assert lo == 0.0 and 0.15 < hi < 0.17


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert round(mcnemar_exact(0, 8), 4) == 0.0078  # 8 discordant, all one way
    assert mcnemar_exact(3, 3) == 1.0


def test_fisher_symmetric_and_extreme():
    assert fisher_exact(5, 10, 5, 10) == 1.0
    assert fisher_exact(0, 10, 10, 10) < 1e-4


def _r(seed, ok, reason="success", cost=0.03, t=12.0):
    return {"seed": seed, "success": ok, "reason": reason if not ok else "success", "cost_usd": cost,
            "seconds_to_success": t if ok else None, "calls": 20, "wall_seconds": 60}


def test_summarize_and_paired():
    a = [_r(0, True), _r(1, False, "misplaced"), _r(2, False, "untouched"), _r(3, True)]
    b = [_r(0, True), _r(1, True), _r(2, True), _r(3, True)]
    s = summarize(a)
    assert s["successes"] == 2 and s["success_rate"] == 0.5
    assert s["failures"] == {"misplaced": 1, "untouched": 1}
    assert s["cost_per_success_usd"] == 0.06
    p = paired(a, b)
    assert p["seeds"] == 4 and p["only_b_succeeded"] == 2 and p["only_a_succeeded"] == 0
    assert p["delta"] == 0.5
