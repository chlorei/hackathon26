"""dreamkit on top of Dream Labs' own loop (dream_control.py, unchanged): `dreamkit run --mock ...`. No credits."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT.parent / "dreamlabs" / "dream_library" / "examples" / "yam_mujoco"


@pytest.mark.skipif(not (SIM / "dream_control.py").is_file(), reason="Dream Labs' simulator not checked out")
def test_overlay_on_dream_labs_loop_recovers_and_reports(tmp_path):
    cfg = tmp_path / "dreamkit.yaml"
    cfg.write_text(yaml.safe_dump({"runs_dir": str(tmp_path / "runs"),
                                   "plugins": ["success-check", "failure-monitor", "recovery"]}))
    out = subprocess.run([sys.executable, "-m", "dreamkit.cli", "-c", str(cfg), "run", "--no-panel", "--mock",
                          "grasped_air", "--", "python", str(SIM / "dream_control.py"), "--headless", "--seconds",
                          "45", "--quiet"], capture_output=True, text=True, timeout=600, cwd=tmp_path)
    assert "dreamkit: attempt 1 ✓ success" in out.stdout, out.stdout + out.stderr
    run = next((tmp_path / "runs").iterdir())
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    said = [(e.get("plugin"), e.get("message", "")) for e in events]
    assert any(p == "failure-monitor" and "grasped air" in m for p, m in said)
    assert any(p == "recovery" and "grasped air" in m for p, m in said)
    assert json.loads((run / "run.json").read_text())["episodes"][0]["success"] is True


class _FakeDream:
    """The surface dream.DreamClient has, without the network."""

    def __init__(self, *a, **k):
        self.calls = 0

    def policy(self, config):
        self.config = config

    def set_observation(self, frames, observation, *, timestamp=None):
        self.obs = observation

    def get_action(self, *, commanded=None):
        self.calls += 1
        return {"q_l_chunk": [[0.0] * 7] * 2, "q_r_chunk": [[0.0] * 7] * 2}

    def reset_episode(self):
        pass

    def close(self):
        pass

    @property
    def last_timing(self):
        return None


def _obs(task):
    return {"prompt": task, "state.left_arm": [0.0] * 6, "state.left_gripper": [1.0],
            "state.right_arm": [0.0] * 6, "state.right_gripper": [1.0]}


def test_session_reopened_before_any_step_is_one_attempt(tmp_path, monkeypatch):
    """Dream Labs' sim closes and reopens its session for a model that reads hand poses: still one attempt."""
    from dreamkit import attach

    cfg = tmp_path / "dreamkit.yaml"
    cfg.write_text(yaml.safe_dump({"runs_dir": str(tmp_path / "runs"), "plugins": []}))
    monkeypatch.setenv("DREAMKIT_CONFIG", str(cfg))
    monkeypatch.delenv("DREAMKIT_MOCK", raising=False)
    monkeypatch.setattr(attach, "SESSION", attach._Session())
    Client = attach._wrap_client(_FakeDream)

    first = Client()
    first.policy({"checkpoint": "m"})
    first.set_observation({}, _obs("put apple into the bowl"))
    first.close()  # reopened before any step
    second = Client()
    second.policy({"checkpoint": "m"})
    second.set_observation({}, _obs("put apple into the bowl"))
    for _ in range(3):
        second.get_action()
        second.set_observation({}, _obs("put apple into the bowl"))  # no YamEnv: observations are the robot's steps
    second.close()
    third = Client()  # the next task typed in the program: a new attempt
    third.policy({"checkpoint": "m"})
    third.set_observation({}, _obs("put banana on the plate"))
    third.get_action()
    third.set_observation({}, _obs("put banana on the plate"))
    third.close()
    kit = attach.SESSION.kit
    kit.close()
    eps = json.loads((kit.run_dir / "run.json").read_text())["episodes"]
    assert [e["episode"] for e in eps] == [1, 2]
    assert json.loads((kit.run_dir / "run.json").read_text())["task"] == "put apple into the bowl"
