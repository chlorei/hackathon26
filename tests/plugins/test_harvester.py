"""harvester: successful attempts join <data_dir>/datasets/harvest-<task>/; nothing goes inside the run."""

import json

from dreamkit import datasets
from dreamkit.testing import replay


def test_a_successful_attempt_joins_the_task_dataset_and_passes_validate_dataset():
    r = replay("success_apple_bowl", plugins=["success-check", "harvester"], keep=True)
    h = r.result["harvester"]
    assert h["dataset"] == "harvest-put-apple-into-the-bowl" and h["index"] == 0
    assert not (r.run_dir / "harvester").exists()  # nothing inside the run
    root = r.run_dir.parent.parent / "datasets" / h["dataset"]
    rec = json.loads((root / "dataset.json").read_text())
    assert rec["source"] == "harvester" and rec["episodes"][0]["run"] == r.run_dir.name
    assert rec["episodes"][0]["episode"] == 1
    report = datasets.check(root)
    assert report["ok"], report
    assert datasets.describe(root)["status"] == "checked"
    assert r.budget_ms("harvester") < 1.0


def test_a_failed_attempt_is_not_kept():
    r = replay("dropped_mid_carry", plugins=["success-check", "failure-monitor", "harvester"], keep=True)
    assert "harvester" not in r.result or not r.result["harvester"]
    ds = r.run_dir.parent.parent / "datasets"
    assert datasets.list_all(ds) == []


def test_two_successes_append_as_episodes_0_and_1(tmp_path):
    a = replay("success_apple_bowl", plugins=["success-check", "harvester"], keep=True)
    root = a.run_dir.parent.parent / "datasets" / a.result["harvester"]["dataset"]
    from dreamkit.plugins.harvester.plugin import write_episode  # noqa: F401  (the one-episode layout it appends)

    second = tmp_path / "copy"
    # the dataset itself is a valid source: its first episode appended again becomes episode 1
    ep0 = tmp_path / "ep0"
    for part in ("meta", "data", "videos"):
        import shutil

        shutil.copytree(root / part, ep0 / part)
    n = datasets.append_episode(root, ep0, source="harvester", run="other", episode=7)
    assert n == 1
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2 and info["splits"] == {"train": "0:2"}
    assert (root / "data" / "chunk-000" / "episode_000001.parquet").is_file()
    assert datasets.find_episode(root.parent, "other", 7) == {"dataset": root.name, "index": 1}
    assert datasets.check(root)["ok"]
    assert not second.exists()
