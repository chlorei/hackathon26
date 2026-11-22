"""budget-guard warns as the budget goes and, with recovery, stops the run safely (arms home first)."""

import json

from dreamkit.testing import replay

TINY = {"budget_usd": 0.005, "count_mock": True}  # a few model calls at Dream-1's price use it up


def test_at_the_limit_the_arms_go_home_and_the_run_stops():
    r = replay("success_apple_bowl", plugins=["failure-monitor", "recovery", {"budget-guard": TINY}], keep=True)
    mine = [e for e in r.events.of("budget-guard") if "pct" in e["data"]]  # besides the core's "verdict: …" line
    assert [e["data"]["pct"] for e in mine] == [50, 80, 95] and mine[-1]["level"] == "error"
    home = [e for e in r.events.of("recovery") if e["data"].get("action") == "home"]
    assert home and "the run stops" in home[0]["message"]  # recovery, the one control plugin, acts on the verdict
    assert r.result["success"] is False and r.result["reason"] == "budget_exhausted"
    assert r.result["budget-guard"]["stopped"] and r.result["budget-guard"]["spent_usd"] >= 0.95 * TINY["budget_usd"]
    run = json.loads((r.run_dir / "run.json").read_text())
    assert run["stop_requested"]["by"] == "recovery" and run["stop_requested"]["reason"] == "budget_exhausted"
    assert not r.errors("budget-guard") and r.budget_ms("budget-guard") < 1.0


def test_finish_lets_the_attempt_end_then_stops_the_run():
    r = replay("success_apple_bowl", plugins=["success-check", "recovery", {"budget-guard": {**TINY, "at_stop": "finish"}}], keep=True)
    assert r.result["success"] is True  # the attempt was not cut
    assert json.loads((r.run_dir / "run.json").read_text())["stop_requested"]["reason"] == "budget_exhausted"


def test_quiet_inside_the_budget_and_on_a_mock():
    r = replay("success_apple_bowl", plugins=["recovery", "budget-guard"])
    assert not r.events.of("budget-guard") and r.result["budget-guard"]["spent_usd"] == 0.0
