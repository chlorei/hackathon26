from dreamkit.testing import replay


def test_flags_a_black_camera_before_the_model_call():
    r = replay("camera_black", plugins=["camera-check"])
    verdicts = r.result["verdicts"]
    assert verdicts and verdicts[0]["reason"] == "camera_failure" and "black" in verdicts[0]["message"]
    assert r.budget_ms("camera-check") < 1.0


def test_keeps_what_it_checked_and_the_failing_picture_for_its_page():
    import json

    r = replay("camera_black", plugins=["camera-check"], keep=True)
    folder = r.run_dir / "camera-check"
    record = json.loads((folder / "ep001_checks.json").read_text())
    assert record["problems"] and all(c["checks"] >= 1 for c in record["cameras"].values())
    assert all((folder / f).read_bytes()[:2] == b"\xff\xd8" for f in record["frames"].values()) and record["frames"]
    assert set(r.result["camera-check"]) == {"problems"}  # ctx.result as before


def test_quiet_with_working_cameras():
    r = replay("success_apple_bowl", plugins=["camera-check"])
    assert not r.result["verdicts"] and not r.errors("camera-check")
