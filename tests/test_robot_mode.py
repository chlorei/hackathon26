"""dreamkit on a real robot: no simulator world, only what the robot reports (state, gripper, commands, cameras)."""

import json

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

import dreamkit
from dreamkit.testing import replay

PLUGINS = ["camera-check", "success-check", "failure-monitor"]


@pytest.mark.parametrize("fixture, reason", [
    ("grasped_air", "grasped_air"),
    ("dropped_mid_carry", "dropped"),
    ("collision_table", "collision"),
    ("stalled", "stalled"),
    ("camera_black", "camera_failure"),
])
def test_failure_monitor_without_the_world(fixture, reason):
    r = replay(fixture, plugins=PLUGINS, world=False)
    assert r.result["success"] is False and r.result["reason"] == reason
    assert r.result["details"]["source"] == "robot"
    assert r.budget_ms("failure-monitor") < 1.0


@pytest.mark.parametrize("fixture", ["success_apple_bowl", "released_off_target"])
def test_what_the_robot_cannot_know_stays_unknown(fixture):
    r = replay(fixture, plugins=PLUGINS, world=False)
    assert r.result["success"] is None and not r.result.get("reason")


def test_robot_and_world_agree_where_the_robot_can_tell():
    for fixture in ("grasped_air", "dropped_mid_carry", "collision_table", "stalled"):
        w = replay(fixture, plugins=PLUGINS).result
        r = replay(fixture, plugins=PLUGINS, world=False).result
        assert w["reason"] == r["reason"], fixture
        assert abs(w["at_s"] - r["at_s"]) <= 0.5, fixture
        assert w["details"]["source"] == "world" if fixture != "stalled" else True


def test_recovery_works_without_the_simulator():
    r = replay("dropped_mid_carry", plugins=["failure-monitor", "recovery"], world=False)
    acts = [e["data"].get("action") for e in r.events.of("recovery") if e["data"].get("action")]
    assert acts and acts[0] == "home"


SIM_ONLY = '''
from dreamkit import Manifest, Plugin


class NeedsWorld(Plugin):
    """Counts steps, but only where there is a simulator world."""

    name = "needs-world"
    manifest = Manifest(group="diagnosis", icon="Plug", capability="observe", hooks=["on_step"], needs=["robot", "world"])

    def on_step(self, ctx):
        ctx.store["world_steps"] = ctx.store.get("world_steps", 0) + 1
'''


class Fake:
    def policy(self, config): pass
    def set_observation(self, *a, **k): pass
    def reset_episode(self): pass
    def get_action(self, **k):
        return {"q_l_chunk": np.zeros((48, 7)).tolist(), "q_r_chunk": np.zeros((48, 7)).tolist()}


def test_sim_only_plugins_are_off_on_a_robot(tmp_path):
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins" / "needs_world.py").write_text(SIM_ONLY)
    cfg = tmp_path / "dreamkit.yaml"
    cfg.write_text(yaml.safe_dump({"runs_dir": str(tmp_path / "runs"), "plugins": ["needs-world", "harvester"]}))
    client = dreamkit.wrap(Fake(), config=cfg, console=False)
    state = {"left": np.zeros(7), "right": np.zeros(7)}
    with client.episode("put apple into the bowl", state=state):
        client.get_action()
        for _ in range(5):
            client.step(state=state, command=(np.zeros(7), np.zeros(7)))
    client.close()
    assert "world_steps" not in client.ctx.store
    run = json.loads(next((tmp_path / "runs").glob("*/run.json")).read_text())
    assert run["mode"] == "robot only (no world)"
    assert "sim only" in run["plugin_stats"]["needs-world"]["inactive"]
    assert run["episodes"][0]["success"] is None  # nobody could judge it


def test_a_person_labels_an_unknown_attempt_and_the_dataset_follows(tmp_path):
    from dreamkit import datasets
    from dreamkit.server import create_app

    r = replay("success_apple_bowl", plugins=["success-check", "harvester"], world=False, keep=True)
    h = r.result["harvester"]
    assert h["pending"] and h["dataset"] == "harvest-put-apple-into-the-bowl"  # waits for a person
    run_dir = r.run_dir
    data = run_dir.parent.parent
    root = data / "datasets" / h["dataset"]
    assert (root / ".pending" / h["pending"]).is_dir() and datasets.describe(root)["episodes"] == 0
    cfg = tmp_path / "dreamkit.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": str(data), "plugins": []}))
    api = TestClient(create_app(cfg))
    resp = api.post(f"/api/runs/{run_dir.name}/episodes/1/label", json={"success": True})
    assert resp.status_code == 200, resp.text
    info = json.loads((run_dir / "run.json").read_text())
    assert info["episodes"][0]["success"] is True and info["episodes"][0]["labeled_by"] == "person"
    assert info["episodes"][0]["harvester"] == {"dataset": h["dataset"], "index": 0, "frames": h["frames"]}
    assert datasets.describe(root)["episodes"] == 1 and not (root / ".pending" / h["pending"]).exists()
    files = api.get(f"/api/runs/{run_dir.name}/episodes/1/files").json()
    assert files["dataset"]["dataset"] == h["dataset"] and any(f["name"] == "result.json" for f in files["files"])
    assert "1 of 1 attempts worked" in (run_dir / "history" / "summary.json").read_text()
