from dreamkit.testing import replay


def test_tells_a_failure_in_one_sentence_with_a_clip():
    r = replay("dropped_mid_carry", plugins=["failure-monitor", "recorder", "recap"], keep=True)
    recap = r.result["recap"]
    assert "dropped" in recap["headline"]
    assert (r.run_dir / recap["video"]).stat().st_size > 500


def test_tells_a_success():
    r = replay("success_apple_bowl", plugins=["success-check", "recap"])
    assert r.result["recap"]["headline"].startswith("Attempt 1: done in")
