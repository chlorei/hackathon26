"""fine-tune: checks the harvester's dataset of the task, trains only when asked, keeps the job in dataset.json."""

import json
import sys
import types
from pathlib import Path

import pytest

from dreamkit.testing import fake_ctx, replay

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the package, installed or not
from fine_tune.plugin import FineTune  # noqa: E402

TASK = "put apple into the bowl"


def _dataset(data: Path, episodes: int, task: str = TASK, frames: int = 120) -> Path:
    """What the harvester leaves: datasets/harvest-<task>/ with its meta and dataset.json."""
    from dreamkit import datasets

    root = data / "datasets" / datasets.harvest_name(task)
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": episodes, "total_frames": frames * episodes,
                                                         "fps": 10}))
    datasets.update(root, name=root.name, source="harvester",
                    episodes=[{"index": i, "run": "r0", "episode": i + 1, "task": task} for i in range(episodes)])
    return root


def _ctx(plugin, runs: Path, run_id: str, hook: str):
    (runs / run_id).mkdir(parents=True, exist_ok=True)
    return fake_ctx(plugin, run_dir=runs / run_id, run_id=run_id, task=TASK, hook=hook)


def _run(plugin, runs: Path, run_id: str) -> dict:
    plugin.on_run_start(_ctx(plugin, runs, run_id, "on_run_start"))
    plugin.on_run_end(_ctx(plugin, runs, run_id, "on_run_end"))  # no Kit: ctx.defer runs at once
    return json.loads((runs / run_id / "fine-tune" / "status.json").read_text())


def test_checks_the_task_dataset_and_records_it(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    root = _dataset(tmp_path, 2)
    _dataset(tmp_path, 4, task="put banana on the plate")  # another task's dataset is left alone
    monkeypatch.setattr("dreamkit.datasets.validate", lambda path, config=None: {"ok": True})
    status = _run(FineTune(min_episodes=5), runs, "r2")
    assert status["dataset"] == root.name and status["episodes"] == 2
    assert status["frames"] == 240 and status["validate"]["ok"]
    rec = json.loads((root / "dataset.json").read_text())
    assert rec["validate"]["ok"]  # the check is recorded on the dataset (the Data tab's status)
    assert not (runs / "r2" / "fine-tune" / "job.json").exists()  # auto_train off: nothing uploaded


class FakeClient:
    """DreamClient's dataset and fine-tune calls, recorded."""

    calls: list = []

    def get_embodiments(self):
        return []

    def add_embodiment(self, name, config):
        FakeClient.calls.append(("add_embodiment", name, config["head_camera"]))
        return types.SimpleNamespace(name=f"you/{name}")

    def upload_dataset(self, path, name, embodiment):
        FakeClient.calls.append(("upload", name, embodiment))
        return types.SimpleNamespace(id="ds_1", name=name, status="uploaded")

    def get_dataset_status(self, dataset_id):
        return types.SimpleNamespace(status="processed", error=None)

    def finetune(self, dataset, suffix, steps=None):
        FakeClient.calls.append(("finetune", dataset, suffix, steps))
        return {"id": "ft_1", "status": "queued"}

    def get_finetuning_status(self, job_id):
        return {"id": job_id, "status": "succeeded", "fine_tuned_model": "dream-1-ft:abcd1234-apple"}


def test_trains_when_asked_and_announces_the_model_next_run(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    root = _dataset(tmp_path, 3)
    monkeypatch.setattr("dreamkit.datasets.validate", lambda path, config=None: {"ok": True})
    monkeypatch.setitem(sys.modules, "dream", types.SimpleNamespace(DreamClient=FakeClient))
    FakeClient.calls = []
    _run(FineTune(min_episodes=3, auto_train=True, suffix="apple"), runs, "r1")
    job = json.loads((runs / "r1" / "fine-tune" / "job.json").read_text())
    assert job["stage"] == "training" and job["job_id"] == "ft_1"
    assert [c[0] for c in FakeClient.calls] == ["add_embodiment", "upload", "finetune"]
    assert FakeClient.calls[2][2] == "apple"
    # the next run asks about the job and carries the model id
    nxt = FineTune(min_episodes=3, auto_train=True)
    nxt.on_run_start(_ctx(nxt, runs, "r2", "on_run_start"))
    job2 = json.loads((runs / "r2" / "fine-tune" / "job.json").read_text())
    assert job2["stage"] == "done" and job2["model"] == "dream-1-ft:abcd1234-apple"
    rec = json.loads((root / "dataset.json").read_text())  # upload, job and model id live on the dataset
    assert rec["upload"]["dataset_id"] == "ds_1" and rec["finetune"]["model"] == "dream-1-ft:abcd1234-apple"
    from dreamkit import datasets

    assert datasets.describe(root)["status"] == "trained"


def test_not_enough_attempts_means_no_upload(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _dataset(tmp_path, 1)
    monkeypatch.setattr("dreamkit.datasets.validate", lambda path, config=None: {"ok": True})
    monkeypatch.setitem(sys.modules, "dream", types.SimpleNamespace(DreamClient=FakeClient))
    FakeClient.calls = []
    _run(FineTune(min_episodes=20, auto_train=True), runs, "r1")
    assert FakeClient.calls == []


def test_replay_runs_with_the_harvester():
    av = pytest.importorskip("av")
    if not hasattr(av, "open"):
        pytest.skip("PyAV is not installed")
    result = replay("success_apple_bowl", plugins=["success-check", "harvester", "fine-tune"], keep=True)
    assert not result.errors("fine-tune")
    status = json.loads((result.run_dir / "fine-tune" / "status.json").read_text())
    assert status["episodes"] >= 1
    assert result.budget_ms("fine-tune") < 1.0
