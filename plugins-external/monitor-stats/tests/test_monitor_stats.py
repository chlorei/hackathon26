"""Replay tests with a fake vision model: no network, no credits, no simulator."""

import io
import json
import os
import re
import sys
from pathlib import Path

import pytest
from PIL import Image

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
    assert v["success"] is True and v["score"] == 90 and v["model_s"] >= 0
    assert len(calls) == 1 and len(calls[0]["images"]) == min(10, len(calls[0]["images"])) >= 2
    assert all(img[:2] == b"\xff\xd8" for img in calls[0]["images"])  # JPEG
    assert "put apple into the bowl" in calls[0]["prompt"]
    assert v["frames"][0]["comment"] == "frame 0" and "t_s" in v["frames"][0]  # the panel's style, with times
    assert result.events.any(plugin="monitor-stats", level="info", message="judging attempt")
    assert result.events.any(plugin="monitor-stats", level="success")
    assert not result.errors("monitor-stats")
    assert result.budget_ms("monitor-stats") < 1.0
    # what the page reads: the judged frames as JPEGs (at most 512 px wide) and attempts.json
    folder = result.run_dir / "monitor-stats"
    [row] = json.loads((folder / "attempts.json").read_text())
    assert row["success"] is True and row["check"] is True and len(row["frames"]) >= 2
    jpg = (folder / row["frames"][0]["file"]).read_bytes()
    assert jpg[:2] == b"\xff\xd8" and Image.open(io.BytesIO(jpg)).width <= 512
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
    ev = [e for e in result.events.of("monitor-stats") if e["level"] == "error"]
    assert ev and ev[0]["message"].startswith("camera verdict: dropped") and ev[0]["data"]["reason"] == "dropped"
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
    script.append(RuntimeError("no Gemini model available (gemini-3.5-flash-lite: 503)"))
    result = replay("success_apple_bowl", plugins=["monitor-stats"])
    assert result.result["monitor-stats"] == {"error": "no Gemini model available (gemini-3.5-flash-lite: 503)"}
    assert result.events.any(plugin="monitor-stats", level="warn")
    assert not result.errors("monitor-stats")


def test_a_try_cut_short_by_recovery_gets_its_own_verdict(model):
    """The drop happens between frames of the whole attempt; judged per try, the try before the recovery shows it."""
    calls, script = model
    cut = lambda p: "sent back to try again" in p  # noqa: E731 - a try that ended in a recovery
    script += [lambda p: reply(p, False, "Dropped the apple while carrying it.", "dropped") if cut(p) else reply(p, True)
               for _ in range(5)]
    result = replay("dropped_mid_carry", plugins=["failure-monitor", "recovery", "monitor-stats"], keep=True)
    assert len(calls) >= 2 and all(cut(c["prompt"]) for c in calls[:-1]) and not cut(calls[-1]["prompt"])
    v = result.result["monitor-stats"]
    assert v["success"] is True and len(v["tries"]) == len(calls) - 1
    assert v["tries"][0]["try"] == 1 and v["tries"][0]["reason_code"] == "dropped"
    ev = [e for e in result.events.of("monitor-stats") if e["level"] == "error"]
    assert ev and ev[0]["message"].startswith("camera verdict: dropped") and "try 1" in ev[0]["message"]
    assert ev[0]["data"]["at_s"] is not None
    rows = json.loads((result.run_dir / "monitor-stats" / "attempts.json").read_text())
    n = len(calls)
    assert [(r["try"], r["final"]) for r in rows] == [(i + 1, i == n - 1) for i in range(n)]
    assert rows[0]["check"] is None and rows[-1]["check"] == result.result["success"]
    assert not result.errors("monitor-stats")


def test_gemini_quota_without_a_claude_key_is_reported_at_once(monkeypatch):
    def busy(*a):
        raise judge_mod.Unavailable("no Gemini model available (gemini-3.5-flash-lite: 429)", quota=True)

    monkeypatch.setattr(judge_mod, "load_env", lambda start=None: None)
    monkeypatch.setattr(judge_mod, "_ask_gemini", busy)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(judge_mod.Unavailable, match="^quota$"):
        judge_mod.ask([b"x"], "p", "gemini", 10.0, 0.0)


def test_claude_haiku_answers_when_gemini_cannot(monkeypatch):
    asked = []
    monkeypatch.setattr(judge_mod, "load_env", lambda start=None: None)
    monkeypatch.setattr(judge_mod, "_ask_gemini", lambda *a: (_ for _ in ()).throw(judge_mod.Unavailable("429", True)))
    monkeypatch.setattr(judge_mod, "_ask_claude", lambda *a: asked.append(a[-1]) or "{}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert judge_mod.ask([b"x"], "p", "gemini", 10.0, 0.0) == "{}"
    assert asked == ["claude-haiku-5-5"]


def test_no_call_starts_after_the_deadline(monkeypatch):
    """dreamkit saves an attempt after 120 s; a verdict that late would be lost, so the judge gives up first."""
    import time

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    with pytest.raises(RuntimeError, match="out of time"):
        judge_mod.ask([b"x"], "p", "gemini", 30.0, time.monotonic() + 1.0)  # under MIN_CALL_S: no network


def test_key_comes_from_the_nearest_env_file_without_overriding(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text('# keys\nexport GEMINI_API_KEY="from-file"\nANTHROPIC_API_KEY=set-before\n')
    (tmp_path / "sub").mkdir()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")
    judge_mod.load_env(tmp_path / "sub")
    assert os.environ["GEMINI_API_KEY"] == "from-file"
    assert os.environ["ANTHROPIC_API_KEY"] == "from-shell"  # the shell wins over the file


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
