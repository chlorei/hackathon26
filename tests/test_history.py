"""History (recaps, summary, report) is made by the core for every run, not by plugins."""

import json

from dreamkit.testing import replay


def test_every_run_gets_a_recap_summary_and_report():
    r = replay("dropped_mid_carry", plugins=["failure-monitor", "recorder"], keep=True)
    assert "dropped" in r.result["recap"]["headline"]
    assert (r.run_dir / r.result["recap"]["video"]).stat().st_size > 500
    summary = json.loads((r.run_dir / "history" / "summary.json").read_text())
    assert summary["text"].startswith("0 of 1 attempts worked")
    assert "mock: scripted failures" in (r.run_dir / "history" / "report.html").read_text()


def test_a_success_is_told_too():
    r = replay("success_apple_bowl", plugins=["success-check"], keep=True)
    assert r.result["recap"]["headline"].startswith("Attempt 1: done in")
    assert r.result["recap"]["video"] is None  # no recorder, no clip
