from dreamkit import harvest
from dreamkit.testing import replay


def test_a_successful_attempt_passes_validate_dataset():
    r = replay("success_apple_bowl", plugins=["success-check", "harvester"], keep=True)
    assert r.result["harvester"]["split"] == "success"
    eps = harvest.episodes(r.run_dir)
    assert len(eps) == 1
    report = harvest.validate(eps[0].parent)
    assert report["ok"], report
    assert r.budget_ms("harvester") < 1.0


def test_a_failed_attempt_goes_to_failed_with_its_reason():
    r = replay("dropped_mid_carry", plugins=["success-check", "failure-monitor", "harvester"], keep=True)
    assert r.result["harvester"]["split"] == "failed"
    assert len(harvest.episodes(r.run_dir, "failed")) == 1
