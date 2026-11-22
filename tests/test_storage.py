"""One data folder: dreamkit migrate from the old places, dreamkit where / datasets, the storage API."""

import json
import shutil

import yaml
from fastapi.testclient import TestClient

from dreamkit import datasets
from dreamkit.cli import main as cli
from dreamkit.migrate import migrate
from dreamkit.server import create_app
from dreamkit.testing import replay


def _one_episode(dst):
    """A one-episode LeRobot dataset, as the old harvester wrote per attempt."""
    r = replay("success_apple_bowl", plugins=["success-check", "harvester"], keep=True)
    root = r.run_dir.parent.parent / "datasets" / r.result["harvester"]["dataset"]
    for part in ("meta", "data", "videos"):
        shutil.copytree(root / part, dst / part)
    return dst


def _old_layout(base):
    """runs/ beside the config with every old place in it, gym/data, gym/runs and harvest/."""
    (base / "dreamkit.yaml").write_text("sim_path: ../sim\nruns_dir: runs\nplugins:\n- recorder\n")
    run = base / "runs" / "r1"
    for rel in ("recorder/ep001_top.mp4", "history/ep001_recap.mp4", "failure-monitor/ep001_trace.json", "ep2/top.mp4"):
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_text(rel)
    ep = _one_episode(run / "harvester" / "success" / "ep001")
    (ep / "dreamkit.json").write_text(json.dumps({"task": "put apple into the bowl", "success": True}))
    (run / "harvester" / "failed" / "ep002").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"id": "r1", "task": "put apple into the bowl", "mock": True, "episodes": [
        {"episode": 1, "task": "put apple into the bowl", "videos": ["recorder/ep001_top.mp4"],
         "recap": {"video": "history/ep001_recap.mp4"}, "failure-monitor": {"trace": "failure-monitor/ep001_trace.json"},
         "harvester": {"split": "success", "path": "harvester/success/ep001"}},
        {"episode": 2, "task": "put apple into the bowl", "videos": ["ep2/top.mp4"]}]}))
    (run / "events.jsonl").write_text("")
    (base / "gym" / "data" / "fruit_mix" / "meta").mkdir(parents=True)
    (base / "gym" / "data" / "fruit_mix" / "meta" / "info.json").write_text(json.dumps({"total_episodes": 0}))
    (base / "gym" / "runs" / "1121_arena").mkdir(parents=True)
    (base / "gym" / "runs" / "1121_arena" / "episodes.jsonl").write_text("")
    _one_episode(base / "harvest" / "r0-ep003")


def _files(base):
    return sorted(p.name for p in base.rglob("*") if p.is_file() and p.suffix in (".mp4", ".json", ".jsonl", ".parquet"))


def test_migrate_moves_everything_into_one_data_folder(tmp_path):
    _old_layout(tmp_path)
    before = len(_files(tmp_path))
    said = []
    assert migrate(tmp_path / "dreamkit.yaml", say=said.append) == 0
    data = tmp_path / "dreamkit-data"
    run = data / "runs" / "r1"
    assert (run / "ep001" / "top.mp4").read_text() == "recorder/ep001_top.mp4"
    assert (run / "ep001" / "recap.mp4").is_file() and (run / "ep001" / "trace.json").is_file()
    assert (run / "ep002" / "top.mp4").read_text() == "ep2/top.mp4" and not (run / "ep2").exists()
    info = json.loads((run / "run.json").read_text())
    ep1 = info["episodes"][0]
    assert ep1["videos"] == ["ep001/top.mp4"] and ep1["recap"]["video"] == "ep001/recap.mp4"
    assert ep1["failure-monitor"]["trace"] == "ep001/trace.json" and info["episodes"][1]["videos"] == ["ep002/top.mp4"]
    assert ep1["harvester"] == {"dataset": "harvest-put-apple-into-the-bowl", "index": 0}
    assert json.loads((run / "ep001" / "result.json").read_text())["episode"] == 1
    harvest = data / "datasets" / "harvest-put-apple-into-the-bowl"
    assert datasets.describe(harvest)["episodes"] == 1 and datasets.check(harvest)["ok"]
    assert (run / ".migrated" / "harvester" / "failed" / "ep002").is_dir()  # kept, not deleted
    assert (data / "datasets" / "fruit_mix" / "dataset.json").is_file()
    assert (data / "runs" / "gym-1121_arena" / "episodes.jsonl").is_file()
    assert datasets.describe(data / "datasets" / "export")["episodes"] == 1
    assert (data / ".migrated" / "harvest" / "r0-ep003").is_dir()
    assert not (tmp_path / "runs").exists() and not (tmp_path / "gym").exists() and not (tmp_path / "harvest").exists()
    cfg = yaml.safe_load((tmp_path / "dreamkit.yaml").read_text())
    assert "runs_dir" not in cfg and cfg["data_dir"] == "dreamkit-data" and cfg["plugins"] == ["recorder"]
    assert len(_files(tmp_path)) >= before  # moved and appended copies: nothing lost
    assert any("-> dreamkit-data/runs/r1" in line for line in said)
    again = []
    migrate(tmp_path / "dreamkit.yaml", say=again.append)
    assert again[-1].startswith("0 moves")  # twice is harmless


def test_where_and_datasets_commands(tmp_path, capsys, monkeypatch):
    _old_layout(tmp_path)
    migrate(tmp_path / "dreamkit.yaml", say=lambda _: None)
    monkeypatch.chdir(tmp_path)
    assert cli(["where"]) == 0
    out = capsys.readouterr().out
    assert "dreamkit-data" in out and "1 runs" in out and "3 datasets" in out
    assert cli(["datasets"]) == 0
    out = capsys.readouterr().out
    assert "harvest-put-apple-into-the-bowl" in out and "harvester" in out and "collected" in out
    assert cli(["datasets", "check", "harvest-put-apple-into-the-bowl"]) == 0
    assert "validate_dataset OK" in capsys.readouterr().out
    assert cli(["datasets"]) == 0 and "checked" in capsys.readouterr().out
    assert cli(["harvest", "export"]) == 1 and "dreamkit datasets" in capsys.readouterr().out


def test_storage_api_and_deleting_mock_runs(tmp_path):
    _old_layout(tmp_path)
    migrate(tmp_path / "dreamkit.yaml", say=lambda _: None)
    api = TestClient(create_app(tmp_path / "dreamkit.yaml"))
    s = api.get("/api/storage").json()
    assert s["runs"] == 1 and s["mock_runs"] == 1 and s["datasets"] == 3 and s["datasets_bytes"] > 0
    listed = {d["name"]: d for d in api.get("/api/datasets").json()["datasets"]}
    assert listed["harvest-put-apple-into-the-bowl"]["source"] == "harvester"
    files = api.get("/api/runs/r1/episodes/1/files").json()
    assert {f["name"] for f in files["files"]} >= {"top.mp4", "recap.mp4", "trace.json", "result.json"}
    assert files["dataset"]["dataset"] == "harvest-put-apple-into-the-bowl"
    assert api.post("/api/storage/delete-mock-runs", json={}).status_code == 400  # needs the confirmation
    assert api.post("/api/storage/delete-mock-runs", json={"confirm": True}).json() == {"deleted": 1}
    assert api.get("/api/storage").json()["runs"] == 0
    assert datasets.describe(tmp_path / "dreamkit-data" / "datasets" / "harvest-put-apple-into-the-bowl")["episodes"] == 1
