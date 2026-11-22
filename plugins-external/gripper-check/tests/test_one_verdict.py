"""gripper-check warns, failure-monitor decides: one verdict per failure, the gripper's signals as its evidence."""

import sys
from pathlib import Path

import pytest

from dreamkit.testing import replay

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize("world", [True, False])
def test_grasped_air_is_one_verdict_with_the_gripper_as_evidence(world):
    r = replay("grasped_air", plugins=["success-check", "failure-monitor", "gripper-check"], world=world, keep=True)
    assert r.result["reason"] == "grasped_air"
    # the gripper warned (live), it did not decide
    assert r.events.any(plugin="gripper-check", level="warn")
    verdicts = r.result.get("verdicts") or []
    assert not [v for v in verdicts if v.get("by") == "gripper-check"]
    assert len([v for v in verdicts if v.get("reason") == "grasped_air"]) <= 1
    finals = [e for e in r.events if e["level"] == "error" and (e.get("data") or {}).get("reason")]
    assert len(finals) == 1 and finals[0]["plugin"] == "failure-monitor"
    evidence = r.result["details"].get("evidence") or []
    assert any(e["by"] == "gripper-check" and e["kind"] == "empty" for e in evidence)
