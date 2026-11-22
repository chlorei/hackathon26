"""What the panel reads from /api/runs is always there, also for runs written by older dreamkit."""

import json

import yaml
from fastapi.testclient import TestClient

from dreamkit.server import create_app


def test_old_runs_get_every_field_the_panel_reads(tmp_path):
    runs = tmp_path / "runs"
    for rid, run in {
        "old-mock": {"id": "old-mock", "mock": True, "model": "mock", "status": "finished", "started": 1.0,
                     "episodes": [{"episode": 1, "task": "do it", "success": False, "reason": "camera_failure"}]},
        "real": {"id": "real", "model": "dreamlabs/dream-1-beta", "status": "finished", "started": 2.0, "task": "put apple into the bowl",
                 "episodes": [{"episode": 1, "task": "put apple into the bowl", "success": True, "duration_s": 12.5}]},
        "gym-scripted": {"id": "gym-scripted", "source": "gym", "model": "scripted:0.03 | recovery on", "status": "finished",
                         "started": 3.0, "episodes": []},
    }.items():
        (runs / rid).mkdir(parents=True)
        (runs / rid / "run.json").write_text(json.dumps(run))
    cfg = tmp_path / "dreamkit.yaml"
    cfg.write_text(yaml.safe_dump({"runs_dir": str(runs)}))
    client = TestClient(create_app(cfg))
    got = {r["id"]: r for r in client.get("/api/runs").json()["runs"]}
    assert got["real"]["kind"] == "dream" and got["real"]["source"] == "loop"
    assert got["old-mock"]["kind"] == "scripted" and got["gym-scripted"]["kind"] == "scripted"
    assert got["gym-scripted"]["source"] == "gym"
    ep = got["old-mock"]["episodes"][0]
    assert ep["task_readable"] is False and ep["duration_s"] == 0.0 and ep["videos"] == []
    assert got["real"]["episodes"][0]["task_readable"] is True
    for r in got.values():
        assert isinstance(r["seconds"], float) and isinstance(r["summary"], dict) and isinstance(r["task"], str)
    one = client.get("/api/runs/real").json()
    assert one["seconds"] == 12.5 and one["kind"] == "dream"
