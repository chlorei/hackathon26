"""One fine-tuning dataset folder as the Data tab sees it: episodes with where they came from, balance per task,
live recording progress, and removing episodes that should not be trained on.

The LeRobot files (meta/, data/, videos/) are what Dream's fine-tuning reads. Next to them, in the hidden
`.dreamgym/` folder that the check and the upload skip, DreamGym keeps:
  episodes.jsonl        one line per kept episode: task name, prompt, source, seed, length
  record.progress.json  the recording job's plan and counts, rewritten after every episode
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .lerobot_writer import CAMERA_ROLES, LeRobotWriter

## Fewer episodes than this per task and the fine-tune is unlikely to learn the task (a rule of thumb, not a limit).
MIN_PER_TASK = 20


def state_dir(root: Path) -> Path:
    d = Path(root) / ".dreamgym"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def add_episode(root: Path, writer: LeRobotWriter, **extra) -> None:
    """Notes where the episode the writer just kept came from (call right after writer.end() returned True)."""
    index = len(writer.episodes) - 1
    row = {"episode_index": index, "length": writer.episodes[-1]["length"], "created": time.time(), **extra}
    path = state_dir(root) / "episodes.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    (state_dir(root) / "validate.json").unlink(missing_ok=True)  # the last check was of a smaller dataset
    from dreamkit import datasets

    datasets.note_episode(Path(root), source=str(extra.get("source") or "gym"), run=extra.get("run"),
                          episode=extra.get("episode"), task=extra.get("task"), prompt=extra.get("prompt"))


def write_progress(root: Path, progress: dict) -> None:
    progress["updated"] = time.time()
    tmp = state_dir(root) / "record.progress.json.tmp"
    tmp.write_text(json.dumps(progress, indent=2))
    tmp.replace(state_dir(root) / "record.progress.json")


def _task_of(prompt: str, tasks: dict) -> str | None:
    """The task name of an episode recorded before DreamGym noted it: by prompt (or a paraphrase of it)."""
    from .tasks import prompt_variants

    for t in tasks.values():
        if prompt == t.prompt or prompt in prompt_variants(t):
            return t.name
    return None


def describe(root: Path) -> dict:
    """Everything the Data tab shows about one dataset, read from disk (cheap enough to poll every 2 s)."""
    from .tasks import TASKS

    root = Path(root)
    info = _read_json(root / "meta" / "info.json") or {}
    fps = info.get("fps") or 10
    episodes = _jsonl(root / "meta" / "episodes.jsonl")
    extras = {e["episode_index"]: e for e in _jsonl(root / ".dreamgym" / "episodes.jsonl")}
    # dataset.json (dreamkit.datasets) knows the source of episodes DreamGym did not record (the harvester's)
    rec = _read_json(root / "dataset.json") or {}
    for e in rec.get("episodes") or []:
        if "index" in e and e["index"] not in extras:
            extras[e["index"]] = {"source": rec.get("source"), "created": e.get("added"), "run": e.get("run"),
                                  "episode": e.get("episode")}
    rows = []
    for ep in episodes:
        i = ep["episode_index"]
        x = extras.get(i, {})
        prompt = (ep.get("tasks") or [""])[0]
        task = x.get("task") or _task_of(prompt, TASKS) or "?"
        rows.append({"index": i, "task": task, "prompt": prompt, "source": x.get("source", "?"), "seed": x.get("seed"),
                     "frames": ep["length"], "seconds": round(ep["length"] / fps, 1), "created": x.get("created")})
    per_task: dict[str, dict] = {}
    for r in rows:
        t = per_task.setdefault(r["task"], {"task": r["task"], "episodes": 0, "frames": 0, "prompts": set()})
        t["episodes"] += 1
        t["frames"] += r["frames"]
        t["prompts"].add(r["prompt"])
    balance = sorted(({**t, "prompts": len(t["prompts"])} for t in per_task.values()), key=lambda t: t["task"])
    frames = sum(r["frames"] for r in rows)
    warnings = []
    for t in balance:
        if t["episodes"] < MIN_PER_TASK:
            warnings.append(f"{t['task']}: {t['episodes']} episodes, aim for at least {MIN_PER_TASK}")
    if len(balance) > 1:
        lo, hi = min(t["episodes"] for t in balance), max(t["episodes"] for t in balance)
        if lo and hi / lo > 3:
            warnings.append(f"unbalanced: {hi} vs {lo} episodes between tasks; the model will favour the big one")
    progress = _read_json(root / ".dreamgym" / "record.progress.json")
    return {"name": root.name, "fps": info.get("fps"), "episodes": len(rows), "frames": frames,
            "minutes": round(frames / fps / 60, 1), "balance": balance, "warnings": warnings,
            "rows": list(reversed(rows)), "progress": progress}


def remove_episodes(root: Path, indices: list[int]) -> int:
    """Deletes episodes and renumbers the rest so the dataset stays a valid LeRobot v2.1 dataset.

    Files are renamed in increasing order; a kept episode only ever moves to a lower or equal number, whose file has
    already been moved or deleted, so nothing is overwritten.
    @return How many episodes were removed.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    drop = set(int(i) for i in indices)
    fps = (_read_json(root / "meta" / "info.json") or {}).get("fps", 10)
    writer = LeRobotWriter(root, fps=fps)  # reads meta/ (tasks, episodes)
    old = writer.episodes
    if not drop & {e["episode_index"] for e in old}:
        return 0
    data_dir = root / "data" / "chunk-000"
    video_dirs = [root / "videos" / "chunk-000" / f"observation.images.{role}" for role in CAMERA_ROLES.values()]
    for i in sorted(drop):
        (data_dir / f"episode_{i:06d}.parquet").unlink(missing_ok=True)
        for d in video_dirs:
            (d / f"episode_{i:06d}.mp4").unlink(missing_ok=True)
    kept, new_index, offset = [], {}, 0
    for ep in old:
        i = ep["episode_index"]
        if i in drop:
            continue
        j = len(kept)
        new_index[i] = j
        n = ep["length"]
        src = data_dir / f"episode_{i:06d}.parquet"
        table = pq.read_table(src)
        for name, values in (("episode_index", [j] * n), ("index", list(range(offset, offset + n)))):
            table = table.set_column(table.schema.get_field_index(name), name, pa.array(values, type=pa.int64()))
        pq.write_table(table, data_dir / f"episode_{j:06d}.parquet")
        if j != i:
            src.unlink()
            for d in video_dirs:
                if (d / f"episode_{i:06d}.mp4").is_file():
                    shutil.move(str(d / f"episode_{i:06d}.mp4"), d / f"episode_{j:06d}.mp4")
        kept.append({**ep, "episode_index": j})
        offset += n
    writer.episodes = kept
    writer.total_frames = offset
    writer._write_meta()
    extras = state_dir(root) / "episodes.jsonl"
    _write_jsonl(extras, [{**e, "episode_index": new_index[e["episode_index"]]}
                          for e in _jsonl(extras) if e["episode_index"] in new_index])
    (state_dir(root) / "validate.json").unlink(missing_ok=True)  # the last check was of the old dataset
    return len(old) - len(kept)
