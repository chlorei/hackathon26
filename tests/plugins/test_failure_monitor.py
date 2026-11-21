import pytest

from dreamkit.testing import fixtures, replay


@pytest.mark.parametrize("fixture", [f for f in fixtures() if f not in ("success_apple_bowl", "slow_model")])
def test_names_the_recorded_failure(fixture):
    r = replay(fixture, plugins=["camera-check", "success-check", "failure-monitor"])
    assert r.result["success"] is False
    assert r.result["reason"] == r.expected["reason"], r.result.get("candidates")
    assert r.budget_ms("failure-monitor") < 1.0
    assert not r.errors("failure-monitor")


def test_quiet_on_success_with_a_timeline():
    r = replay("success_apple_bowl", plugins=["success-check", "failure-monitor"])
    assert r.result["success"] is True and "reason" not in r.result or r.result.get("reason") is None
    kinds = [m["kind"] for m in r.result["timeline"]]
    assert kinds[:3] == ["reached", "grasp", "lifted"]


def test_model_latency_is_part_of_failure_monitor():
    r = replay("slow_model", plugins=["failure-monitor"])
    assert r.events.any(plugin="failure-monitor", level="warn")
    assert r.result["failure-monitor"]["latency"]["mean_ms"] > 700


def test_live_verdict_mid_attempt():
    r = replay("dropped_mid_carry", plugins=["failure-monitor"])
    v = [e for e in r.events if e["data"].get("kind") == "verdict"]
    assert v and v[0]["data"]["reason"] == "dropped" and v[0]["t_s"] < 12
