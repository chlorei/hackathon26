from dreamkit.testing import replay


def test_flags_a_black_camera_before_the_model_call():
    r = replay("camera_black", plugins=["camera-check"])
    verdicts = r.result["verdicts"]
    assert verdicts and verdicts[0]["reason"] == "camera_failure" and "black" in verdicts[0]["message"]
    assert r.budget_ms("camera-check") < 1.0


def test_quiet_with_working_cameras():
    r = replay("success_apple_bowl", plugins=["camera-check"])
    assert not r.result["verdicts"] and not r.errors("camera-check")
