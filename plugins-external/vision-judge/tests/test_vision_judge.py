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
from vision_judge import judge as judge_mod  # noqa: E402
from vision_judge.plugin import aggregate, wilson  # noqa: E402


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
    result = replay("success_apple_bowl", plugins=["success-check", "vision-judge"], keep=True)
    v = result.result["vision-judge"]
    assert v["success"] is True and v["score"] == 90 and v["model_s"] >= 0
    assert len(calls) == 1 and len(calls[0]["images"]) == 1  # the last frame: how the scene ended
    assert all(img[:2] == b"\xff\xd8" for img in calls[0]["images"])  # JPEG
    assert "put apple into the bowl" in calls[0]["prompt"]
    assert v["frames"][0]["comment"] == "frame 0" and "t_s" in v["frames"][0]  # the panel's style, with times
    assert result.events.any(plugin="vision-judge", level="info", message="judging attempt")
    assert result.events.any(plugin="vision-judge", level="success")
    assert not result.errors("vision-judge")
    assert result.budget_ms("vision-judge") < 1.0
    # what the page reads: the judged frames as JPEGs (at most 512 px wide) and attempts.json
    folder = result.run_dir / "vision-judge"
    [row] = json.loads((folder / "attempts.json").read_text())
    assert row["success"] is True and row["check"] is True and len(row["frames"]) == 1
    jpg = (folder / row["frames"][0]["file"]).read_bytes()
    assert jpg[:2] == b"\xff\xd8" and Image.open(io.BytesIO(jpg)).width <= 512
    summary = json.loads((result.run_dir / "vision-judge" / "summary.json").read_text())
    assert summary["overall"]["successes"] == 1
    assert summary["agreement"] == {"attempts": 1, "agree": 1, "rate": 1.0}  # same verdict as success-check


def test_judges_a_failed_attempt_with_reason_code(model):
    calls, script = model
    script.append(lambda p: reply(p, False, "Dropped the apple while carrying it.", "dropped"))
    result = replay("dropped_mid_carry", plugins=["vision-judge"])
    v = result.result["vision-judge"]
    assert v["success"] is False and v["reason"] == "dropped the apple while carrying it"
    assert v["reason_code"] == "dropped"
    ev = [e for e in result.events.of("vision-judge") if e["level"] == "warn"]  # a warning: the core decides success
    assert ev and ev[0]["message"].startswith("camera verdict: dropped") and ev[0]["data"]["reason"] == "dropped"
    assert not result.errors("vision-judge")
    assert result.budget_ms("vision-judge") < 1.0


def test_asks_again_after_a_bad_reply_and_drops_unknown_codes(model):
    calls, script = model
    script += [lambda p: "not json", lambda p: '{"success": "yes"}', lambda p: reply(p, False, "x", "made_up")]
    result = replay("dropped_mid_carry", plugins=["vision-judge"])
    assert len(calls) == 3 and "rejected" in calls[1]["prompt"]
    assert result.result["vision-judge"]["reason_code"] is None  # not in dreamkit/reasons.py


def test_model_down_does_not_break_the_run(model):
    calls, script = model
    script.append(RuntimeError("no Gemini model available (gemini-3.5-flash-lite: 503)"))
    result = replay("success_apple_bowl", plugins=["vision-judge"])
    assert result.result["vision-judge"] == {"error": "no Gemini model available (gemini-3.5-flash-lite: 503)"}
    assert result.events.any(plugin="vision-judge", level="warn")
    assert not result.errors("vision-judge")


def test_a_try_cut_short_by_recovery_gets_its_own_verdict(model):
    """The drop happens between frames of the whole attempt; judged per try, the try before the recovery shows it."""
    calls, script = model
    cut = lambda p: "sent back to try again" in p  # noqa: E731 - a try that ended in a recovery
    script += [lambda p: reply(p, False, "Dropped the apple while carrying it.", "dropped") if cut(p) else reply(p, True)
               for _ in range(5)]
    result = replay("dropped_mid_carry", plugins=["failure-monitor", "recovery", "vision-judge"], keep=True)
    assert len(calls) >= 2 and all(cut(c["prompt"]) for c in calls[:-1]) and not cut(calls[-1]["prompt"])
    v = result.result["vision-judge"]
    assert v["success"] is True and len(v["tries"]) == len(calls) - 1
    assert v["tries"][0]["try"] == 1 and v["tries"][0]["reason_code"] == "dropped"
    ev = [e for e in result.events.of("vision-judge") if e["level"] == "warn"]  # a warning: the core decides success
    assert ev and ev[0]["message"].startswith("camera verdict: dropped") and "try 1" in ev[0]["message"]
    assert ev[0]["data"]["at_s"] is not None
    rows = json.loads((result.run_dir / "vision-judge" / "attempts.json").read_text())
    n = len(calls)
    assert [(r["try"], r["final"]) for r in rows] == [(i + 1, i == n - 1) for i in range(n)]
    # no success-check in this replay: the physics check said nothing, the camera decided (who decides, §7b)
    assert rows[0]["check"] is None and rows[-1]["check"] is None and result.result["judged_by"] == "vision"
    assert not result.errors("vision-judge")


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
    return {"task": task, "success": sim, "vision-judge": {
        "success": judged, "score": score if score is not None else (90 if judged else 20),
        "reason": None if judged else f"failed: {code}", "reason_code": code}}


def test_statistics_across_runs():
    eps = [_ep("cube", True, True), _ep("cube", True, True), _ep("cube", False, True, "dropped"),
           _ep("peg", False, False, "not_reached"), _ep("peg", False, False, "not_reached"), _ep("peg", True, False),
           {"task": "peg", "success": False, "vision-judge": {"error": "503"}},  # not judged: not counted
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


@pytest.mark.parametrize("raw, want", [(True, True), ("true", True), ("Yes", True), (1, True), ("0", False),
                                       ("false", False), ("no.", False), (0, False)])
def test_a_yes_or_no_in_any_form_is_a_boolean(raw, want):
    v = judge_mod.validate({"success": raw, "score": "85%", "frames": [{"status": "ok"}]}, 1)
    assert v["success"] is want and v["score"] == 85


@pytest.mark.parametrize("bad", [{"success": "maybe", "score": 50}, {"success": True, "score": "high"},
                                 {"success": 2, "score": 50}, {"success": True, "score": 140}])
def test_what_cannot_be_read_stays_not_judged(bad):
    with pytest.raises(ValueError):
        judge_mod.validate({**bad, "frames": [{"status": "ok"}]}, 1)


def _unjudged_run(model):
    """A finished run whose one attempt went 'not judged' (the model was down), its frame saved."""
    calls, script = model
    script.append(RuntimeError("no Gemini model available (gemini-3.5-flash-lite: 503)"))
    result = replay("success_apple_bowl", plugins=["vision-judge"], keep=True)
    info = json.loads((result.run_dir / "run.json").read_text())
    assert info["episodes"][0]["success"] is None and "error" in info["episodes"][0]["vision-judge"]
    return result.run_dir


def test_judge_again_later_waits_out_a_503_and_writes_the_verdict_back(model, monkeypatch):
    from vision_judge import rejudge

    run_dir = _unjudged_run(model)
    calls, script = model
    monkeypatch.setattr(rejudge.time, "sleep", lambda s: None)
    script += [judge_mod.Unavailable("no Gemini model available (gemini-3.6-flash: 503)"), lambda p: reply(p, True)]
    said = []
    assert rejudge.rejudge_run(run_dir, "gemini", say=said.append) == (1, 0)
    assert any("waiting" in s for s in said)  # the 503: a pause, then the same frames again
    ep = json.loads((run_dir / "run.json").read_text())["episodes"][0]
    assert ep["vision-judge"]["success"] is True and ep["vision-judge"]["rejudged"] is True
    assert ep["success"] is True and ep["judged_by"] == "vision"  # nothing else had judged it
    assert json.loads((run_dir / "ep001" / "result.json").read_text())["success"] is True
    rows = json.loads((run_dir / "vision-judge" / "attempts.json").read_text())
    assert "error" not in rows[0] and rows[0]["rejudged"] is True
    assert rejudge.rejudge_run(run_dir, "gemini", say=said.append) == (0, 0)  # nothing left to judge


def test_judge_again_stops_at_once_on_a_429(model):
    from vision_judge import rejudge

    run_dir = _unjudged_run(model)
    calls, script = model
    script.append(judge_mod.Unavailable("quota", quota=True))
    with pytest.raises(rejudge.QuotaExhausted):
        rejudge.rejudge_run(run_dir, "gemini", say=lambda s: None)
    assert len(calls) == 2  # the run's own call, and one now: no retries on a quota
    assert json.loads((run_dir / "run.json").read_text())["episodes"][0]["success"] is None  # left as it was
