from dreamkit.testing import replay


def test_writes_one_video_per_camera():
    r = replay("success_apple_bowl", plugins=["recorder"], keep=True)
    videos = r.result["videos"]
    assert sorted(v.split("_")[-1] for v in videos) == ["left.mp4", "right.mp4", "top.mp4"]
    assert all((r.run_dir / v).stat().st_size > 500 for v in videos)
    assert r.budget_ms("recorder") < 1.0


def test_records_a_failed_attempt_too():
    r = replay("grasped_air", plugins=[{"recorder": {"cameras": ["top"]}}], keep=True)
    assert [v.split("/")[-1] for v in r.result["videos"]] == ["ep001_top.mp4"]
