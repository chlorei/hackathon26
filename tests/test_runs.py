"""examples/sim_loop.py --mock on Dream Labs' simulator: each scripted failure gets the right reason. No API calls."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("sim_loop", ROOT / "examples" / "sim_loop.py")
sim_loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim_loop)


def _run(config_path, *args):
    code = sim_loop.main(["--mock", "--fast", "--config", str(config_path), "--seconds", "20", *args])
    (folder,) = list((config_path.parent / "runs").iterdir())
    return code, json.loads((folder / "run.json").read_text()), folder


@pytest.mark.parametrize("behavior, reason", [
    ("success", None),
    ("dropped", "dropped"),
    ("released_off_target", "released_off_target"),
    ("missed_target", "missed_target"),
    ("not_reached", "not_reached"),
    ("grasped_air", "grasped_air"),
    ("stalled", "stalled"),
    ("collision", "collision"),
])
def test_failure_monitor_names_the_scripted_failure(make_config, behavior, reason):
    code, info, _ = _run(make_config(), "--episodes", "1", "--behaviors", behavior)
    assert code == 0
    (episode,) = info["episodes"]
    if reason is None:
        assert episode["success"] is True
    else:
        assert episode["success"] is False
        assert episode["reason"] == reason, episode.get("candidates")
        assert 0 < episode["at"] < 20


def test_scenes_differ_and_retry_then_success_with_video_and_recap(make_config):
    code, info, folder = _run(make_config(
        plugins=["success-check", "failure-monitor", {"retry": {"attempts": 1}}, "latency-monitor", "recorder",
                 "recap", "report"],
    ), "--episodes", "2", "--behaviors", "dropped,success")
    assert code == 0
    eps = info["episodes"]
    assert [(e["scene"], e["attempt"], e["success"]) for e in eps] == [(0, 1, False), (0, 2, True), (1, 1, False), (1, 2, True)]
    assert eps[0]["positions"] != eps[2]["positions"]  # a new scene really moves the fruit
    assert eps[0]["positions"] == eps[1]["positions"]  # a retry keeps it
    assert len({e["duration_s"] for e in eps if e["success"]}) == 2  # mock phase lengths vary
    assert (folder / "ep1" / "top.mp4").stat().st_size > 1000
    assert (folder / "ep1" / "recap.mp4").stat().st_size > 1000
    assert eps[0]["latency"] == {"mock": True}
    recap = json.loads((folder / "recap.json").read_text())["text"]
    assert "Mock: no model calls" in recap and "Retries saved 2 scenes" in recap
    assert (folder / "report.html").is_file()


def test_camera_failure_stops_the_robot(make_config):
    code, info, _ = _run(make_config(
        plugins=[{"camera-check": {"simulate_failure": "top", "simulate_at_s": 2.0}}, "success-check",
                 "failure-monitor"],
    ), "--episodes", "1")
    (episode,) = info["episodes"]
    assert episode["reason"] == "stopped_by_plugin"
    assert "camera top is black" in episode["details"]["message"]
    assert episode["at"] < 5
