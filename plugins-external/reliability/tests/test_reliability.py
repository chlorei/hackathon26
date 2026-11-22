"""Replay tests: the numbers on one success and one failure, and the A/B comparison between two runs."""

import json
import sys
from pathlib import Path

import pytest

from dreamkit.testing import replay

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the package, installed or not
from reliability.plugin import compare, fisher_exact, mcnemar_exact, summarize, wilson  # noqa: E402

PLUGINS = ["success-check", "failure-monitor"]


def _summary(result):
    return json.loads((result.run_dir / "reliability" / "summary.json").read_text())


def test_counts_a_success_with_an_honest_interval():
    result = replay("success_apple_bowl", plugins=[*PLUGINS, "reliability"], keep=True)
    s = _summary(result)
    assert (s["successes"], s["attempts"]) == (1, 1)
    assert s["interval"][0] < 0.25 and s["interval"][1] == 1.0  # 1 of 1 is not "100 %"
    assert result.events.any(plugin="reliability", level="success")
    assert not result.errors("reliability")
    assert result.budget_ms("reliability") < 1.0


def test_counts_a_failure_and_its_reason():
    result = replay("dropped_mid_carry", plugins=[*PLUGINS, "reliability"], keep=True)
    s = _summary(result)
    assert s["successes"] == 0 and sum(s["failures"].values()) == 1
    assert result.events.any(plugin="reliability", level="warn")
    assert result.budget_ms("reliability") < 1.0


def test_compares_with_a_baseline_run():
    base = replay("dropped_mid_carry", plugins=[*PLUGINS, "reliability"], keep=True)
    this = replay("success_apple_bowl", keep=True,
                  plugins=[*PLUGINS, {"reliability": {"baseline_run": str(base.run_dir)}}])
    c = _summary(this)["baseline"]["comparison"]
    assert c["paired"] and c["scenes"] == 1
    assert (c["only_this_won"], c["only_base_won"]) == (1, 0)
    assert c["p_value"] == 1.0  # one scene proves nothing, and the plugin says so
    assert this.events.any(plugin="reliability", level="info")


def test_statistics():
    lo, hi = wilson(7, 10)
    assert (round(lo, 3), round(hi, 3)) == (0.397, 0.892)
    assert round(mcnemar_exact(0, 8), 4) == 0.0078
    assert fisher_exact(5, 10, 5, 10) == 1.0
    eps = [{"seed": i, "attempt": 1, "success": i < 2, "reason": None if i < 2 else "dropped", "requests": 25,
            "duration_s": 40.0} for i in range(4)]
    for e in eps:  # what the core records per attempt (dreamkit.cost): 25 calls of Dream-1
        e["cost_usd"] = 25 * 3.0 / 2250
    s = summarize(eps, 1.96)
    assert s["successes"] == 2 and s["failures"] == {"dropped": 2}
    assert s["cost_per_success_usd"] == pytest.approx(0.0667, abs=1e-4)
    better = [dict(e, success=True, reason=None) for e in eps]
    assert compare(eps, better, "first_attempt")["only_this_won"] == 2
