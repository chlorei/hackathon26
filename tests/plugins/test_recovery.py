from dreamkit.testing import replay


def test_drives_home_and_asks_again_on_a_drop():
    r = replay("dropped_mid_carry", plugins=["failure-monitor", "recovery"])
    acts = [e["data"].get("action") for e in r.events.of("recovery") if e["data"].get("action")]
    assert acts and acts[0] == "home"
    assert r.result["recovery"]["count"] >= 1
    assert r.budget_ms("recovery") < 1.0


def test_stops_the_robot_on_a_camera_failure():
    r = replay("camera_black", plugins=["camera-check", "failure-monitor", "recovery"])
    assert r.result["reason"] == "camera_failure" and r.result["stopped_by"] == "recovery"


def test_does_nothing_on_success():
    r = replay("success_apple_bowl", plugins=["failure-monitor", "recovery"])
    assert "recovery" not in r.result and not r.errors("recovery")
