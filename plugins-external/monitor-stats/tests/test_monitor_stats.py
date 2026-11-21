"""Replay tests with a fake vision model: no network, no credits, no simulator."""

import json
import re
import sys
from pathlib import Path

import pytest

from dreamkit.testing import replay

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the package, installed or not
from monitor_stats import judge as judge_mod  # noqa: E402
from monitor_stats.plugin import aggregate, wilson  # noqa: E402


def reply(prompt, success, reason=None, code=None):
    n = int(re.search(r"You get (\d+) frames", prompt).group(1))
    frames = [{"index": i, "status": "error" if not success and i == n - 1 else "ok", "comment": f"Frame {i}."}
              for i in range(n)]
    return json.dumps({"success": success, "score": 90 if success else 25, "failure_reason": reason,
                       "reason_code": code, "frames": frames})


@pytest.fixture
def model(monkeypatch):
    """Scripted answers: each item is a function prompt -> text, or an exception to raise."""
    calls, script = [], []

    def ask(images, prompt, provider, timeout_s, deadline):
        calls.append({"images": images, "prompt": prompt})
        step = script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(prompt)

    monkeypatch.setattr(judge_mod, "ask", ask)
    return calls, script


def test_judges_a_successful_attempt(model):
    calls, script = model
    script.append(lambda p: reply(p, True))
    result = replay("success_apple_bowl", plugins=["success-check", "monitor-stats"], keep=True)
    v = result.result["monitor-stats"]
    assert v["success"] is True and v["score"] == 90
    assert len(calls) == 1 and 2 <= len(calls[0]["images"]) <= 8
    assert all(img[:2] == b"\xff\xd8" for img in calls[0]["images"])  # JPEG
    assert "put apple into the bowl" in calls[0]["prompt"]
    assert v["frames"][0]["comment"] == "frame 0" and "t_s" in v["frames"][0]  # the panel's style, with times
    assert result.events.any(plugin="monitor-stats", level="success")
    assert not result.errors("monitor-stats")
    assert result.budget_ms("monitor-stats") < 1.0
    summary = json.loads((result.run_dir / "monitor-stats" / "summary.json").read_text())
    assert summary["overall"]["successes"] == 1
    assert summary["agreement"] == {"attempts": 1, "agree": 1, "rate": 1.0}  # same verdict as success-check


def test_judges_a_failed_attempt_with_reason_code(model):
    calls, script = model
    script.append(lambda p: reply(p, False, "Dropped the apple while carrying it.", "dropped"))
    result = replay("dropped_mid_carry", plugins=["monitor-stats"])
    v = result.result["monitor-stats"]
    assert v["success"] is False and v["reason"] == "dropped the apple while carrying it"
    assert v["reason_code"] == "dropped"
    ev = [e for e in result.events.of("monitor-stats") if e["level"] == "warn"]
    assert ev and ev[0]["data"]["reason"] == "dropped"
    assert not result.errors("monitor-stats")
    assert result.budget_ms("monitor-stats") < 1.0


def test_asks_again_after_a_bad_reply_and_drops_unknown_codes(model):
    calls, script = model
    script += [lambda p: "not json", lambda p: '{"success": "yes"}', lambda p: reply(p, False, "x", "made_up")]
    result = replay("dropped_mid_carry", plugins=["monitor-stats"])
    assert len(calls) == 3 and "rejected" in calls[1]["prompt"]
    assert result.result["monitor-stats"]["reason_code"] is None  # not in dreamkit/reasons.py


def test_model_down_does_not_break_the_run(model):
    calls, script = model
    script.append(RuntimeError("no Gemini model available (gemini-3.8-flash: 503)"))
    result = replay("success_apple_bowl", plugins=["monitor-stats"])
    assert result.result["monitor-stats"] == {"error": "no Gemini model available (gemini-3.8-flash: 503)"}
    assert result.events.any(plugin="monitor-stats", level="warn")
    assert not result.errors("monitor-stats")


def test_no_call_starts_after_the_deadline(monkeypatch):
    """dreamkit saves an attempt after 120 s; a verdict that late would be lost, so the judge gives up first."""
    import time

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    with pytest.raises(RuntimeError, match="out of time"):
        judge_mod.ask([b"x"], "p", "gemini", 30.0, time.monotonic() + 1.0)  # under MIN_CALL_S: no network


def _ep(task, judged, sim=None, code=None, score=None):
    return {"task": task, "success": sim, "monitor-stats": {
        "success": judged, "score": score if score is not None else (90 if judged else 20),
        "reason": None if judged else f"failed: {code}", "reason_code": code}}


def test_statistics_across_runs():
    eps = [_ep("cube", True, True), _ep("cube", True, True), _ep("cube", False, True, "dropped"),
           _ep("peg", False, False, "not_reached"), _ep("peg", False, False, "not_reached"), _ep("peg", True, False),
           {"task": "peg", "success": False, "monitor-stats": {"error": "503"}},  # not judged: not counted
           {"task": "cube", "success": True}]  # plugin was off: not counted
    s = aggregate(eps)
    assert s["overall"]["attempts"] == 6 and s["overall"]["successes"] == 3
    assert [t["task"] for t in s["by_task"]] == ["peg", "cube"]  # worst first
    assert s["reasons"][0]["code"] == "not_reached" and s["reasons"][0]["count"] == 2
    assert s["agreement"] == {"attempts": 6, "agree": 4, "rate": 0.6667}
    assert s["recommendation"]["task"] == "peg" and "not reached" in s["recommendation"]["text"]
    assert aggregate([])["recommendation"] is None
    lo, hi = wilson(7, 10)
    assert (round(lo, 3), round(hi, 3)) == (0.397, 0.892)  # the same numbers as reliability
