from dreamkit.testing import replay


def test_writes_one_video_per_camera():
    r = replay("success_apple_bowl", plugins=["recorder"], keep=True)
    videos = r.result["videos"]
    assert sorted(videos) == ["ep001/left.mp4", "ep001/right.mp4", "ep001/top.mp4"]  # §8: the attempt's folder
    assert all((r.run_dir / v).stat().st_size > 500 for v in videos)
    assert r.budget_ms("recorder") < 1.0


def test_records_a_failed_attempt_too():
    r = replay("grasped_air", plugins=[{"recorder": {"cameras": ["top"]}}], keep=True)
    assert r.result["videos"] == ["ep001/top.mp4"]
