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
    # with the world: success-check (an evaluation tool) says whether the attempt worked, and the plugins may read it
    world = [] if "--no-world" in args else ["--with-world"]
    code = sim_loop.main(["--mock", "--fast", "--config", str(config_path), "--seconds", "20", *world, *args])
    (folder,) = list((config_path.parent / "runs").iterdir())
    return code, json.loads((folder / "run.json").read_text()), folder


# What a real arm's signals can tell apart (failure-monitor); where the object ended up (off target, never reached) only
# the world knows, so for those world-judge (the evaluation tool) names the failure and failure-monitor just sees one.
ROBOT_CAN_TELL = {"dropped", "grasped_air", "stalled", "collision"}


@pytest.mark.parametrize("behavior, reason", [
    ("success", None),
    ("dropped", "dropped"),
    ("released_off_target", "released_off_target"),
    ("released_near_target", "released_off_target"),
    ("not_reached", "not_reached"),
    ("grasped_air", "grasped_air"),
    ("stalled", "stalled"),
    ("collision", "collision"),
])
def test_failure_monitor_names_the_scripted_failure(make_config, behavior, reason):
    code, info, _ = _run(make_config(plugins=["success-check", "failure-monitor", "world-judge"]),
                         "--episodes", "1", "--behaviors", behavior)
    assert code == 0
    (episode,) = info["episodes"]
    if reason is None:
        assert episode["success"] is True
    else:
        assert episode["success"] is False
        # a hand that closes short of the object: never reached it, or grasped air next to it
        assert episode["world-judge"]["reason"] in ({reason, "grasped_air"} if reason == "not_reached" else {reason})
        if reason in ROBOT_CAN_TELL:
            assert episode["reason"] == reason, episode.get("candidates")
            assert 0 < episode["at_s"] < 20


def test_scenes_differ_and_retry_then_success_with_video_and_recap(make_config):
    code, info, folder = _run(make_config(
        plugins=["success-check", "failure-monitor", {"recovery": {"max_recoveries": 0, "retry_attempts": 1}},
                 "recorder"],
    ), "--episodes", "2", "--behaviors", "dropped,success")
    assert code == 0
    eps = info["episodes"]
    assert [(e["scene"], e["attempt"], e["success"]) for e in eps] == [(0, 1, False), (0, 2, True), (1, 1, False), (1, 2, True)]
    assert eps[0]["positions"] != eps[2]["positions"]  # a new scene really moves the fruit
    assert eps[0]["positions"] == eps[1]["positions"]  # a retry keeps it
    assert len({e["duration_s"] for e in eps if e["success"]}) == 2  # mock phase lengths vary
    assert (folder / "ep001" / "top.mp4").stat().st_size > 1000  # the attempt's own folder
    assert eps[0]["recap"]["video"] == "ep001/recap.mp4" and (folder / "ep001" / "recap.mp4").stat().st_size > 1000
    assert json.loads((folder / "ep001" / "result.json").read_text())["episode"] == 1
    assert eps[0]["failure-monitor"]["latency"] == {"mock": True}
    recap = json.loads((folder / "history" / "summary.json").read_text())["text"]
    assert "scripted on purpose" in recap and "Retries saved 2 scenes" in recap
    assert "mock: scripted failures" in (folder / "history" / "report.html").read_text()


def test_camera_failure_stops_the_robot(make_config):
    code, info, _ = _run(make_config(
        faults={"camera_black": "top", "at_s": 2.0},
        plugins=["camera-check", "success-check", "failure-monitor", "recovery"],
    ), "--episodes", "1")
    (episode,) = info["episodes"]
    assert episode["reason"] == "camera_failure" and episode["stopped_by"] == "recovery"
    assert "camera top is black" in episode["details"]["message"]
    assert episode["at_s"] < 5


def test_recovery_fixes_failures_mid_attempt_without_resetting_the_scene(make_config):
    code, info, folder = _run(make_config(
        plugins=["success-check", "failure-monitor", "recovery", "harvester"],
    ), "--episodes", "2", "--seconds", "45", "--behaviors", "dropped,grasped_air")  # failures the arm can tell
    eps = info["episodes"]
    assert [e["success"] for e in eps] == [True, True]
    assert [e["recovery"]["count"] for e in eps] == [1, 2]
    assert all(e["attempt"] == 1 for e in eps)  # one attempt each: no scene reset, no retry
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    actions = [e["data"].get("action") for e in events if e.get("plugin") == "recovery" and (e.get("data") or {}).get("action")]
    assert actions == ["home", "open", "home"]  # grasped air: open the gripper, then stalled: home + ask again


def test_harvested_episodes_pass_dream_labs_dataset_checks(make_config, tmp_path):
    from dreamkit import datasets

    code, info, folder = _run(make_config(plugins=["success-check", "failure-monitor", "harvester"]),
                              "--episodes", "2", "--behaviors", "success,dropped")
    assert not (folder / "harvester").exists()  # nothing harvested inside the run
    ok, failed = info["episodes"][0], info["episodes"][1]
    assert ok["harvester"]["dataset"] == "harvest-put-apple-into-the-bowl" and not failed.get("harvester")
    root = tmp_path / "datasets" / ok["harvester"]["dataset"]
    rec = datasets.describe(root)
    assert rec["episodes"] == 1 and rec["source"] == "harvester"
    assert datasets.find_episode(tmp_path / "datasets", folder.name, 1)["dataset"] == root.name
    report = datasets.check(root)
    assert report["ok"], report
    assert report["state_dim"] == 14 and report["action_dim"] == 14 and len(report["video_keys"]) == 3


def test_no_world_live_loop_finds_grasped_air_and_dropped(make_config):
    code, info, _ = _run(make_config(), "--no-world", "--episodes", "3", "--seconds", "16",
                         "--behaviors", "grasped_air,dropped,success")
    eps = info["episodes"]
    assert info["mode"] == "robot only (no world)"
    assert [e["reason"] for e in eps[:2]] == ["grasped_air", "dropped"]
    assert all(e["details"]["source"] == "robot" for e in eps[:2])
    assert eps[2]["success"] is None  # the robot cannot see the bowl: a person labels it


def test_a_tiny_budget_sends_the_arms_home_and_ends_the_run(make_config):
    """budget-guard with a few cents (a mock billed as Dream-1 for this test only): the run stops safely."""
    cfg = make_config(plugins=["success-check", "failure-monitor", "recovery",
                               {"budget-guard": {"budget_usd": 0.02, "count_mock": True}}])
    code, run, folder = _run(cfg, "--episodes", "4", "--behaviors", "success")
    assert run["stop_requested"]["by"] == "recovery" and run["stop_requested"]["reason"] == "budget_exhausted"
    assert len(run["episodes"]) < 4  # no new attempt after the stop
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    home = [e for e in events if e.get("plugin") == "recovery" and (e.get("data") or {}).get("action") == "home"
            and "run stops" in e.get("message", "")]
    assert home  # the arms were sent home before the run ended
    stop = [e for e in events if e.get("plugin") == "budget-guard" and (e.get("data") or {}).get("pct") == 95]
    assert stop and stop[0]["level"] == "error"
