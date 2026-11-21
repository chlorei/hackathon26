"""DreamGym as a dreamkit app plugin: the standard (§13), its API, and episodes that go through dreamkit.wrap.

The episode tests need Dream Labs' simulator (skipped without it) and never call the model: the scripted expert and
a policy that holds still. No credits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from dreamkit import reasons
from dreamkit.apps import AppContext, check_app

from dreamgym.app import app


def _sim_dir() -> Path | None:
    from dreamgym import sim

    try:
        return sim.sim_dir()
    except RuntimeError:
        return None


needs_sim = pytest.mark.skipif(_sim_dir() is None, reason="Dream Labs' simulator not installed")


@pytest.fixture()
def dreamkit_config(tmp_path, monkeypatch):
    cfg = tmp_path / "dreamkit.yaml"
    cfg.write_text(yaml.safe_dump({"runs_dir": str(tmp_path / "runs"),
                                   "plugins": ["success-check", "failure-monitor", "recovery"]}))
    monkeypatch.setenv("DREAMKIT_CONFIG", str(cfg))
    return cfg


def test_app_meets_the_standard():
    assert check_app(app) == []
    assert app.manifest.kind == "app" and app.title == "Gym" and "world" in app.manifest.needs
    assert "gym" in app.aliases


def test_failure_labels_are_dreamkit_codes():
    from dreamgym.runner import _label_error

    judged = {"fell_off_table", "still_held", "timeout", "not_reached", "released_off_target"}
    assert judged <= set(reasons.REASONS)
    assert _label_error(ValueError("x")) in reasons.REASONS
    for old, new in {"untouched": "not_reached", "misplaced": "released_off_target"}.items():
        assert reasons.code(old) == new


def test_recovery_ab_variants_on_the_same_seeds():
    from dreamgym.compare import Variant, with_recovery_ab

    vs = with_recovery_ab([Variant("dream-1")])
    assert [v.label for v in vs] == ["dream-1 | recovery off", "dream-1 | recovery on"]


def test_api_under_the_panel(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    ctx = AppContext(config_path=tmp_path / "dreamkit.yaml", base=tmp_path, runs_dir=tmp_path / "runs", world=True)
    web = FastAPI()
    web.include_router(app.router(ctx), prefix="/apps/dreamgym/api")
    client = TestClient(web)
    tasks = client.get("/apps/dreamgym/api/tasks").json()
    assert {t["name"] for t in tasks} >= {"apple_bowl", "banana_plate"}
    assert client.get("/apps/dreamgym/api/runs").json() == []
    bad = client.post("/apps/dreamgym/api/launch", json={"task": "no_such_task"})
    assert bad.status_code == 400 and "unknown task" in bad.json()["detail"]


class HoldStill:
    """A policy that never moves: the episode must fail, as not_reached."""

    name = "hold-still"
    price_per_hour = 0.0

    def begin(self, env, task, prompt, ctx):
        self.state = env.state()

    def next_chunk(self, env, ctx):
        from dreamgym.core import Chunk

        return Chunk(np.array([self.state["left"]] * 48), np.array([self.state["right"]] * 48))

    def on_frame(self, env, ctx):
        pass

    def close(self):
        pass


def _run_json(cfg: Path) -> dict:
    runs = sorted((cfg.parent / "runs").iterdir())
    return json.loads((runs[-1] / "run.json").read_text())


@needs_sim
def test_scripted_episode_runs_through_dreamkit(dreamkit_config):
    from dreamgym import sim
    from dreamgym.policies import make_policy
    from dreamgym.runner import run_episode
    from dreamgym.tasks import get_task

    env = sim.make_env()
    try:
        result, _ = run_episode(env, get_task("banana_plate"), make_policy("scripted"), seed=0, max_seconds=30)
    finally:
        env.close()
    assert result.success and result.reason == "success"
    run = _run_json(dreamkit_config)
    assert run["source"] == "gym" and run["id"] == result.run_id
    ep = run["episodes"][0]
    assert ep["success"] is True and ep["gym"]["seed"] == 0


@needs_sim
def test_failed_episode_is_named_with_a_shared_code(dreamkit_config):
    from dreamgym import sim
    from dreamgym.runner import run_episode
    from dreamgym.tasks import get_task

    env = sim.make_env()
    try:
        result, _ = run_episode(env, get_task("banana_plate"), HoldStill(), seed=0, max_seconds=4)
    finally:
        env.close()
    assert not result.success and result.reason == "not_reached"
    ep = _run_json(dreamkit_config)["episodes"][0]
    assert ep["success"] is False and ep["reason"] in reasons.REASONS
