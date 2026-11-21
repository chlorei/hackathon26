"""Mock runs through Dream Labs' simulator: each scripted failure gets the right reason. No API calls."""

import json

import pytest

from dreamkit.runner import run


def _run(config_path):
    code = run(str(config_path), mock=True)
    runs = config_path.parent / "runs"
    (folder,) = list(runs.iterdir())
    return code, json.loads((folder / "run.json").read_text()), folder


@pytest.mark.parametrize("behavior, reason", [
    ("success", None),
    ("dropped", "dropped"),
    ("not_reached", "not_reached"),
    ("grasped_air", "grasped_air"),
    ("missed_target", "missed_target"),
    ("stalled", "stalled"),
    ("collision", "collision"),
])
def test_failure_monitor_names_the_scripted_failure(make_config, behavior, reason):
    code, info, _ = _run(make_config(mock={"behaviors": [behavior], "latency_ms": 0}))
    assert code == 0
    (episode,) = info["episodes"]
    if reason is None:
        assert episode["success"] is True
    else:
        assert episode["success"] is False
        assert episode["reason"] == reason, episode.get("candidates")
        assert 0 < episode["at"] < 20


def test_retry_then_success_with_video_and_report(make_config):
    code, info, folder = _run(make_config(
        mock={"behaviors": ["dropped", "success"], "latency_ms": 0},
        plugins=["success-check", "failure-monitor", {"retry": {"attempts": 1}}, "recorder", "report"],
    ))
    assert code == 0
    assert [(e["scene"], e["attempt"], e["success"]) for e in info["episodes"]] == [(0, 1, False), (0, 2, True)]
    assert (folder / "ep1" / "top.mp4").stat().st_size > 1000
    assert (folder / "report.html").is_file()
    assert info["summary"]["scene_successes"] == 1


def test_camera_failure_stops_the_robot(make_config):
    code, info, _ = _run(make_config(
        plugins=[{"camera-check": {"simulate_failure": "top", "simulate_at_s": 2.0}}, "success-check",
                 "failure-monitor"],
    ))
    (episode,) = info["episodes"]
    assert episode["reason"] == "stopped_by_plugin"
    assert "camera top is black" in episode["details"]["message"]
    assert episode["at"] < 5
