"""Runner, Arena, Compare stats and the report, end to end on a fake sim (no MuJoCo, no network)."""

import json
import types

import numpy as np
import pytest

from dreamgym import sim
from dreamgym.core import Chunk


class FakeEnv:
    def __init__(self):
        self.steps = 0

    def reset(self):
        self.steps = 0

    def step(self, left, right):
        assert np.shape(left) == (7,) and np.shape(right) == (7,)
        self.steps += 1

    def state(self):
        return {"left": np.r_[np.zeros(6), 1.0], "right": np.r_[np.zeros(6), 1.0]}

    def images(self):
        return {name: np.zeros((360, 640, 3), np.uint8) for name in ("top", "left", "right")}

    def close(self):
        pass


@pytest.fixture(autouse=True)
def dreamkit_run(tmp_path, monkeypatch):
    """Episodes go through dreamkit.wrap: a dreamkit.yaml of our own, its runs in tmp (no plugins needed here)."""
    cfg = tmp_path / "dreamkit.yaml"
    cfg.write_text(f"runs_dir: {tmp_path / 'dreamkit-runs'}\nplugins: []\n")
    monkeypatch.setenv("DREAMKIT_CONFIG", str(cfg))


@pytest.fixture(autouse=True)
def fake_sim(monkeypatch):
    yam_env = types.SimpleNamespace(HOME=np.r_[np.zeros(6), 1.0], YamEnv=FakeEnv)

    def capture(env, windows, keep=True):
        images = env.images()
        if keep:
            for name in windows:
                windows[name].append("jpeg")
        return images

    def validate_chunk(env, actions, limit=48, *, verbose=True, adjustments=None):
        return (np.asarray(actions["q_l_chunk"], float)[:limit], np.asarray(actions["q_r_chunk"], float)[:limit])

    dc = types.SimpleNamespace(FRAMES=9, FRAME_EVERY_STEPS=4, VIDEO_ROLES={"top": 0, "left": 0, "right": 0},
                               capture=capture, validate_chunk=validate_chunk)
    monkeypatch.setattr(sim, "_MODULES", {"yam_env": yam_env, "dream_control": dc, "mock_pickups": None})
    monkeypatch.setattr("dreamgym.runner.scene.FREE_JOINTS", {})


class FakeTask:
    name, prompt = "fake", "do it"

    def reset(self, env, rng):
        self.ok_after = None if rng.integers(2) else 30  # half the seeds succeed

    def success(self, env):
        return self.ok_after is not None and env.steps > 56 + self.ok_after

    def failure_reason(self, env, start):
        return "released_off_target"


class FakePolicy:
    name, price_per_hour = "fake-policy", 3.0

    def begin(self, env, task, prompt, ctx):
        self.closed = False

    def next_chunk(self, env, ctx):
        ctx.calls += 1
        rows = np.tile(np.r_[np.zeros(6), 1.0], (48, 1))
        return Chunk(rows, rows)

    def on_frame(self, env, ctx):
        pass

    def close(self):
        self.closed = True


def test_episode_success_and_failure():
    from dreamgym.runner import run_episode

    reasons = set()
    for seed in range(6):
        result, frames = run_episode(FakeEnv(), FakeTask(), FakePolicy(), seed=seed, max_seconds=10, keep_frames=True)
        reasons.add(result.reason)
        assert result.calls >= 1 and frames
        if result.success:
            assert result.seconds_to_success is not None and result.sim_seconds < 10
        else:
            assert result.sim_seconds == pytest.approx(10, abs=0.05)
    assert reasons == {"success", "released_off_target"}


def test_arena_resume_and_report(tmp_path):
    from dreamgym.arena import load_results, run_arena
    from dreamgym.report import build

    summary = run_arena(FakeTask(), FakePolicy(), n=6, out_dir=tmp_path, max_seconds=10, env=FakeEnv(), log=lambda s: None)
    assert summary["episodes"] == 6 and 0 < summary["successes"] < 6
    assert summary["cost_usd"] > 0
    assert len(list((tmp_path / "gifs").glob("*.gif"))) == 6 - summary["successes"]
    run_arena(FakeTask(), FakePolicy(), n=8, out_dir=tmp_path, max_seconds=10, env=FakeEnv(), log=lambda s: None)
    assert sorted(r["seed"] for r in load_results(tmp_path)) == list(range(8))  # resumed, not re-run
    page = build(tmp_path).read_text()
    assert "Arena" in page and "seed" in page
    json.loads((tmp_path / "summary.json").read_text())
