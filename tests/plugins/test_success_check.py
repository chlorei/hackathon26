from dreamkit.testing import replay


def test_success_when_the_apple_rests_in_the_bowl():
    r = replay("success_apple_bowl", plugins=["success-check"])
    assert r.result["success"] is True and r.result["ended_by"] == "success"
    assert r.budget_ms("success-check") < 1.0


def test_no_success_when_the_apple_is_dropped():
    r = replay("dropped_mid_carry", plugins=["success-check"])
    assert r.result["success"] is False
    assert not r.errors("success-check")
