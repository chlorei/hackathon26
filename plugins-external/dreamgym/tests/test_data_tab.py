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
