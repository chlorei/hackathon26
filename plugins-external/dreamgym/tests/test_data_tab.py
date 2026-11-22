"""The Data tab: recording several tasks into one dataset (round robin, live progress, where each episode came
from), the record command the page starts, removing episodes, and a page with nothing but the four steps.
No simulator, no network: a fake episode runner and a writer that skips video and parquet."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from dreamgym import cli, dataset, sim
from dreamgym.lerobot_writer import LeRobotWriter

PAGE = Path(__file__).resolve().parents[1] / "dreamgym" / "web" / "page" / "index.html"


class MetaOnlyWriter(LeRobotWriter):
    """Keeps episodes in meta/ only (no video, no parquet): enough for the bookkeeping around it."""

    def begin(self, task):
        self._open = {"task": task}

    def step_hook(self, *a, **k):
        pass

    def end(self, keep):
        ep, self._open = self._open, None
        if ep is None or not keep:
            return False
        self.tasks.setdefault(ep["task"], len(self.tasks))
        self.episodes.append({"episode_index": len(self.episodes), "tasks": [ep["task"]], "length": 60})
        self.total_frames += 60
        self._write_meta()
        return True


@pytest.fixture()
def fake_record(monkeypatch):
    calls = []

    def run_episode(env, task, policy, *, seed, prompt=None, **kw):
        calls.append((task.name, seed, prompt))
        ok = seed % 3 != 2  # every third attempt fails and is dropped
        return types.SimpleNamespace(success=ok, reason="success" if ok else "released_off_target"), []

    kit = types.SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("dreamgym.lerobot_writer.LeRobotWriter", MetaOnlyWriter)
    monkeypatch.setattr("dreamgym.runner.run_episode", run_episode)
    monkeypatch.setattr("dreamgym.policies.make_policy", lambda spec, **k: types.SimpleNamespace(name="scripted"))
    monkeypatch.setattr("dreamgym.kit.open_kit", lambda *a, **k: kit)
    monkeypatch.setattr(sim, "make_env", lambda: types.SimpleNamespace(close=lambda: None))
    return calls


def test_record_mix_takes_turns_and_reports_live(tmp_path, fake_record):
    out = tmp_path / "fruit"
    code = cli.main(["record", "--task", "banana_plate", "--mix", "banana_plate:2,apple_bowl:3", "--out", str(out),
                     "--vary-prompts", "--seed", "0"])
    assert code == 0
    assert [t for t, _, _ in fake_record[:4]] == ["banana_plate", "apple_bowl", "banana_plate", "apple_bowl"]
    progress = json.loads((out / ".dreamgym" / "record.progress.json").read_text())
    assert progress["kept"] == {"banana_plate": 2, "apple_bowl": 3} and progress["status"] == "done"
    d = dataset.describe(out)
    assert d["episodes"] == 5
    assert {t["task"]: t["episodes"] for t in d["balance"]} == {"banana_plate": 2, "apple_bowl": 3}
    assert all(r["source"] == "scripted" and r["seed"] is not None for r in d["rows"])
    assert any("aim for at least" in w for w in d["warnings"])  # 2 and 3 episodes are too few to learn from
    # a second recording appends to the same dataset
    cli.main(["record", "--task", "orange_bowl", "--n", "1", "--out", str(out), "--seed", "100"])
    assert dataset.describe(out)["episodes"] == 6


def test_page_record_command(tmp_path, monkeypatch):
    from dreamgym.web import server

    seen = {}
    monkeypatch.setattr(server, "_spawn_data_job", lambda data, name, kind, cmd: seen.update(cmd=cmd) or {"id": name})
    root = tmp_path / "gym" / "runs"
    out = server.launch_record(root, {"name": "fruit", "source": "scripted",
                                      "mix": {"banana_plate": 30, "apple_bowl": 20, "orange_bowl": 0}})
    cmd = seen["cmd"]
    assert out["name"] == "fruit"
    assert cmd[cmd.index("--mix") + 1] == "banana_plate:30,apple_bowl:20"  # zero means not this task
    assert "--vary-prompts" in cmd and cmd[cmd.index("--policy") + 1].startswith("scripted:")
    with pytest.raises(ValueError):
        server.launch_record(root, {"name": "Bad Name", "mix": {"banana_plate": 1}})
    with pytest.raises(ValueError):
        server.launch_record(root, {"name": "x", "mix": {}})


def test_remove_episodes_renumbers(tmp_path):
    pytest.importorskip("pyarrow")
    root = tmp_path / "ds"
    w = LeRobotWriter(root)
    for i in range(4):
        tmp = root / ".tmp" / f"e{i}"
        tmp.mkdir(parents=True)
        for role in ("overhead", "wrist_l", "wrist_r"):
            (tmp / f"{role}.mp4").write_bytes(f"video {i}".encode())
        n = 49 + i
        state = [np.zeros(14, np.float32)] * n
        w._write_episode({"task": "put banana on the plate", "tmp": tmp, "state": state, "action": state}, n)
        dataset.add_episode(root, w, task="banana_plate", source="scripted", seed=i)
    assert dataset.remove_episodes(root, [0, 2]) == 2
    import pyarrow.parquet as pq

    eps = [json.loads(l) for l in (root / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [e["episode_index"] for e in eps] == [0, 1] and [e["length"] for e in eps] == [50, 52]
    t1 = pq.read_table(root / "data" / "chunk-000" / "episode_000001.parquet").to_pydict()
    assert set(t1["episode_index"]) == {1} and t1["index"][0] == 50
    assert not (root / "data" / "chunk-000" / "episode_000002.parquet").exists()
    video = root / "videos" / "chunk-000" / "observation.images.overhead" / "episode_000001.mp4"
    assert video.read_bytes() == b"video 3"
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2 and info["total_frames"] == 102
    assert [r["seed"] for r in dataset.describe(root)["rows"]] == [3, 1]


def test_page_is_only_the_four_steps():
    page = PAGE.read_text()
    for step in ("Collect", "Check", "Fine-tune", "Compare"):
        assert f'"{step}"' in page
    for gone in ("New run", "Prompt Tuner", "Arena · one policy"):
        assert gone not in page
    assert '"/api/' not in page and "`/api/" not in page  # paths stay relative to the app's mount


def test_recording_started_elsewhere_is_seen(tmp_path):
    """A recording outlives a restart of the panel (its own session): the tab still sees it by its pid, refuses a
    second recording and removing episodes, and a recording that died without a last word reads "interrupted"."""
    import os
    import subprocess
    import sys

    from dreamgym.web import server

    root = tmp_path / "gym" / "runs"
    data = server.data_root(root)
    folder = data / "fruit"
    LeRobotWriter(folder)._write_meta()
    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        dataset.write_progress(folder, {"plan": {"banana_plate": 2}, "kept": {}, "tried": {}, "source": "scripted",
                                        "status": "running", "pid": alive.pid})
        d = server.dataset_detail(root, "fruit")
        assert d["recording"] and d["busy"] and d["progress"]["status"] == "running"
        with pytest.raises(ValueError, match="already recording"):
            server.launch_record(root, {"name": "fruit", "mix": {"banana_plate": 1}})
        with pytest.raises(ValueError, match="stop it first"):
            server.remove(root, "fruit", {"indices": [0]})
        server.dataset_action(root, "fruit", "stop", {})
        assert alive.wait(timeout=10) != 0  # SIGINT reached it
    finally:
        alive.kill()
    assert not server._pid_alive(alive.pid) or os.name == "nt"
    d = server.dataset_detail(root, "fruit")
    assert not d["recording"] and not d["busy"] and d["progress"]["status"] == "interrupted"


def test_evaluate_uses_scenes_not_in_the_dataset(tmp_path, monkeypatch):
    from dreamgym.web import server

    root = tmp_path / "gym" / "runs"
    folder = server.data_root(root) / "fruit"
    w = MetaOnlyWriter(folder)
    for seed in (0, 1, 1000, 1001):
        w.begin("put banana on the plate")
        w.end(keep=True)
        dataset.add_episode(folder, w, task="banana_plate", source="scripted", seed=seed)
    dataset.state_dir(folder).joinpath("finetune.json").write_text(json.dumps({"model": "dream-1-ft:x"}))
    seen = []
    monkeypatch.setattr(server, "launch", lambda root, body: seen.append(body) or {"id": "r"})
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    out = server.evaluate(root, "fruit", {"n": 20})
    assert seen[0]["seed"] == out["seed"] == server.EVAL_SEED0 > 1001  # not the training scenes 0.., 1000..
    dataset.add_episode(folder, w, task="banana_plate", source="scripted", seed=200_000)
    assert server.eval_seed0(folder) == 200_001


def test_new_episodes_make_the_format_check_stale(tmp_path):
    w = MetaOnlyWriter(tmp_path / "ds")
    (dataset.state_dir(tmp_path / "ds") / "validate.json").write_text('{"ok": true, "episodes": 0}')
    w.begin("put banana on the plate")
    w.end(keep=True)
    dataset.add_episode(tmp_path / "ds", w, task="banana_plate", source="scripted", seed=0)
    assert not (tmp_path / "ds" / ".dreamgym" / "validate.json").exists()


def test_attempts_are_capped_per_task(tmp_path, fake_record, monkeypatch):
    from dreamgym.web import server

    seen = {}
    monkeypatch.setattr(server, "_spawn_data_job", lambda data, name, kind, cmd: seen.update(cmd=cmd) or {"id": name})
    server.launch_record(tmp_path / "gym" / "runs", {"name": "m", "source": "dream-1",
                                                     "mix": {"banana_plate": 10, "apple_bowl": 2}})
    cmd = seen["cmd"]
    assert "--max-attempts" not in cmd and cmd[cmd.index("--attempts-per-success") + 1] == "5"
    # every third seed fails: apple_bowl (2 wanted, 3 attempts at 1.5 per success) stops at its own cap
    cli.main(["record", "--task", "banana_plate", "--mix", "banana_plate:4,apple_bowl:2", "--out", str(tmp_path / "o"),
              "--attempts-per-success", "1.5", "--seed", "0"])
    progress = json.loads((tmp_path / "o" / ".dreamgym" / "record.progress.json").read_text())
    assert progress["tried"]["apple_bowl"] <= 3 and progress["tried"]["banana_plate"] <= 6


def test_a_dataset_can_be_trained_again(tmp_path, monkeypatch):
    from dreamgym.web import server

    root = tmp_path / "gym" / "runs"
    folder = server.data_root(root) / "fruit"
    state = dataset.state_dir(folder)
    monkeypatch.setattr(server, "_spawn_data_job", lambda data, name, kind, cmd: {"id": kind})
    state.joinpath("finetune.json").write_text(json.dumps({"stage": "training", "job_id": "j1"}))
    with pytest.raises(ValueError, match="still going"):
        server.dataset_action(root, "fruit", "finetune", {})
    state.joinpath("finetune.json").write_text(json.dumps({"stage": "done", "job_id": "j1", "model": "m1"}))
    assert server.dataset_action(root, "fruit", "finetune", {})["id"] == "finetune"


def test_retrain_uploads_under_a_new_name_and_keeps_the_old_model(tmp_path, monkeypatch):
    from dreamgym import finetune

    folder = tmp_path / "fruit"
    finetune.state_dir(folder).joinpath("finetune.json").write_text(
        json.dumps({"stage": "done", "job_id": "j1", "model": "m1", "dataset_name": "fruit"}))
    uploaded = []
    client = types.SimpleNamespace(
        get_embodiments=lambda: [types.SimpleNamespace(name="me/" + finetune.EMBODIMENT)],
        upload_dataset=lambda path, name, emb: uploaded.append(name) or types.SimpleNamespace(id="d2", status="new", name=name),
        get_dataset_status=lambda i: types.SimpleNamespace(status="processed"),
        finetune=lambda name, suffix, steps=None: {"id": "j2"})
    monkeypatch.setattr(finetune, "_client", lambda: client)
    monkeypatch.setattr(finetune, "validate", lambda d: {"ok": True, "episodes": 1, "frames": 1, "fps": 10})
    record = finetune.start(folder, log=lambda *a: None)
    assert uploaded[0].startswith("fruit-") and record["job_id"] == "j2"
    assert record["previous"] == [{"model": "m1", "job_id": "j1", "dataset_name": "fruit", "started": None}]


def test_custom_tasks_typed_in_the_data_tab(tmp_path, monkeypatch):
    """A prompt of your own, with the object and the place (guessed from the words), saved in <data_dir>/dreamgym-tasks.json: listed,
    recordable (the record subprocess finds it through DREAMGYM_TASKS), forgotten again."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from dreamgym import tasks as T
    from dreamgym.web import server

    root = tmp_path / "gym" / "runs"
    app = FastAPI()
    app.include_router(server.make_router(root), prefix="/api")
    client = TestClient(app)
    try:
        assert client.get("/api/task-parts", params={"prompt": "drop the yellow banana in the bowl"}).json()["obj"] == "banana"
        out = client.post("/api/tasks", json={"prompt": "drop the yellow banana in the bowl"}).json()
        assert out["task"] == "banana_bowl"
        row = next(t for t in out["tasks"] if t["name"] == "banana_bowl")
        assert row == {**row, "obj": "banana", "target": "bowl", "custom": True, "prompt": "drop the yellow banana in the bowl"}
        assert json.loads((tmp_path / "gym" / T.TASKS_FILE).read_text())[0]["name"] == "banana_bowl"
        # explicit choices win over the words; the same task twice is one task; a second wording is a second task
        out = client.post("/api/tasks", json={"prompt": "move the fruit", "obj": "orange", "target": "plate"}).json()
        assert out["task"] == "orange_plate"
        assert client.post("/api/tasks", json={"prompt": "drop the yellow banana in the bowl"}).json()["task"] == "banana_bowl"
        assert client.post("/api/tasks", json={"prompt": "do a little dance"}).status_code == 400
        # the record command takes it, and its subprocess is told where the tasks are
        seen = {}
        monkeypatch.setattr(server, "_spawn_data_job", lambda data, name, kind, cmd: seen.update(cmd=cmd) or {"id": name})
        server.launch_record(root, {"name": "mine", "mix": {"banana_bowl": 3}})
        assert seen["cmd"][seen["cmd"].index("--mix") + 1] == "banana_bowl:3"
        assert server._env()["DREAMGYM_TASKS"] == str(tmp_path / "gym" / T.TASKS_FILE)
        assert T.prompt_variants(T.TASKS["banana_bowl"])[0] == "drop the yellow banana in the bowl"
        # built-in tasks cannot be forgotten; custom ones can
        assert client.delete("/api/tasks/apple_bowl").status_code == 400
        names = {t["name"] for t in client.delete("/api/tasks/banana_bowl").json()["tasks"]}
        assert "banana_bowl" not in names and "orange_plate" in names and "apple_bowl" in names
    finally:
        T.load_custom(tmp_path / "nothing.json")  # other tests see the built-in tasks only
        server.DREAMKIT.pop("tasks", None)
