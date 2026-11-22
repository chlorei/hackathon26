"""<data_dir>/datasets/<name>/: every dataset dreamkit keeps, in one place and one format.

A dataset is a LeRobot v2.1 folder (meta/, data/, videos/: what Dream Labs' validate_dataset and upload_dataset
read) plus dataset.json, dreamkit's own record of it:

    {"name", "source": harvester | gym | teleop | export | migrated, "created",
     "episodes": [{"index", "run", "episode", "task", "frames", "added"}],
     "validate": {"ok", "checked", "episodes", "frames", "fps", "error"},
     "upload":   {"dataset_id", "dataset_name", "status", "updated"},
     "finetune": {"job_id", "status", "model", "updated"}}

The harvester appends each successful attempt as it ends; DreamGym records into its own datasets here too. A
change to the episodes makes the last check stale (validate is cleared), so the status always tells the truth.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

## The LeRobot parts of a dataset folder: what is checked and uploaded (dataset.json and .pending/ are not).
LEROBOT_PARTS = ("meta", "data", "videos")
CHUNK = 1000  # episodes per chunk folder (LeRobot's default chunks_size)
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")


def slug(text: str, limit: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:limit].strip("-") or "task"


def harvest_name(task: str, template: str = "harvest-{task}") -> str:
    """The harvester's dataset for a task: `put apple into the bowl` -> harvest-put-apple-into-the-bowl."""
    return slug(template.replace("{task}", slug(task)), 80)


# -- dataset.json ---------------------------------------------------------------------------------------------------
def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def record(root: Path) -> dict[str, Any]:
    """dataset.json, or what a LeRobot folder without one implies."""
    rec = _read(root / "dataset.json")
    if rec:
        return rec
    info = _read(root / "meta" / "info.json")
    return {"name": root.name, "source": "unknown", "created": None,
            "episodes": [{"index": i} for i in range(int(info.get("total_episodes") or 0))]}


def _write(root: Path, rec: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / ".dataset.json.tmp"
    tmp.write_text(json.dumps(rec, indent=2, default=str))
    os.replace(tmp, root / "dataset.json")


@contextlib.contextmanager
def _locked(root: Path) -> Iterator[None]:
    """One writer at a time per dataset (two runs may harvest into the same one)."""
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def update(root: Path, **fields: Any) -> dict[str, Any]:
    """Sets top-level fields of dataset.json (validate, upload, finetune, source ...)."""
    with _locked(root):
        rec = record(root)
        rec.setdefault("name", root.name)
        for k, v in fields.items():
            rec[k] = {**rec.get(k, {}), **v} if isinstance(v, dict) and isinstance(rec.get(k), dict) else v
        _write(root, rec)
        return rec


def note_episode(root: Path, *, source: str, **link: Any) -> dict[str, Any]:
    """Records in dataset.json an episode a LeRobot writer already appended (DreamGym's writer, Teleop)."""
    with _locked(root):
        rec = record(root)
        rec.setdefault("name", root.name)
        rec.setdefault("source", source)
        if rec.get("source") == "unknown":
            rec["source"] = source
        rec["created"] = rec.get("created") or time.time()
        eps = rec.setdefault("episodes", [])
        index = int(_read(root / "meta" / "info.json").get("total_episodes") or len(eps) + 1) - 1
        eps.append({"index": index, **{k: v for k, v in link.items() if v is not None}, "added": time.time()})
        rec.pop("validate", None)  # the content changed: the last check no longer applies
        _write(root, rec)
        return rec


# -- appending a one-episode LeRobot dataset -------------------------------------------------------------------------
def append_episode(root: Path, src: Path, *, source: str, **link: Any) -> int:
    """Adds a one-episode LeRobot v2.1 dataset (the harvester writes one per attempt) as the next episode of the
    dataset at `root`; returns its episode index. Parquet indexes, task index and meta are rewritten to fit."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    with _locked(root):
        rec = record(root)  # before meta/ changes: a dataset without dataset.json is described by its meta
        meta = root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        info = _read(meta / "info.json")
        src_info = _read(src / "meta" / "info.json")
        if not src_info:
            raise ValueError(f"{src} is not a LeRobot dataset (no meta/info.json)")
        n = int(info.get("total_episodes") or 0)
        offset = int(info.get("total_frames") or 0)
        tasks = [json.loads(line) for line in _lines(meta / "tasks.jsonl")]
        src_tasks = [json.loads(line) for line in _lines(src / "meta" / "tasks.jsonl")]
        task = (src_tasks[0]["task"] if src_tasks else link.get("task")) or ""
        task_index = next((t["task_index"] for t in tasks if t["task"] == task), None)
        if task_index is None:
            task_index = len(tasks)
            tasks.append({"task_index": task_index, "task": task})

        table = pq.read_table(next((src / "data").rglob("episode_*.parquet")))
        length = table.num_rows
        cols = {name: table.column(name) for name in table.column_names}
        cols["episode_index"] = pa.array([n] * length, type=pa.int64())
        cols["task_index"] = pa.array([task_index] * length, type=pa.int64())
        cols["index"] = pa.array(list(range(offset, offset + length)), type=pa.int64())
        chunk = f"chunk-{n // CHUNK:03d}"
        out = root / "data" / chunk / f"episode_{n:06d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(cols), out)
        videos = 0
        for video in (src / "videos").rglob("episode_*.mp4"):
            key = video.parent.name  # observation.images.<role>
            dst = root / "videos" / chunk / key / f"episode_{n:06d}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, dst)
            videos += 1

        episodes = list(_lines(meta / "episodes.jsonl"))
        episodes.append(json.dumps({"episode_index": n, "tasks": [task], "length": length}))
        stats = list(_lines(meta / "episodes_stats.jsonl"))
        for line in _lines(src / "meta" / "episodes_stats.jsonl"):
            row = json.loads(line)
            row["episode_index"] = n
            stats.append(json.dumps(row))
        merged = {**src_info, **{k: v for k, v in info.items() if k == "features"}} if info else dict(src_info)
        merged.update(total_episodes=n + 1, total_frames=offset + length, total_tasks=len(tasks),
                      total_videos=int(info.get("total_videos") or 0) + videos, total_chunks=n // CHUNK + 1,
                      chunks_size=CHUNK, splits={"train": f"0:{n + 1}"})
        (meta / "info.json").write_text(json.dumps(merged, indent=2))
        (meta / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
        (meta / "episodes.jsonl").write_text("".join(e + "\n" for e in episodes))
        if stats:
            (meta / "episodes_stats.jsonl").write_text("".join(s + "\n" for s in stats))

        rec.setdefault("name", root.name)
        if rec.get("source") in (None, "unknown"):
            rec["source"] = source
        rec["created"] = rec.get("created") or time.time()
        rec.setdefault("episodes", []).append(
            {"index": n, **{k: v for k, v in link.items() if v is not None}, "task": task, "frames": length,
             "added": time.time()})
        rec.pop("validate", None)
        _write(root, rec)
        return n


def _lines(path: Path) -> Iterator[str]:
    try:
        for line in path.read_text().splitlines():
            if line.strip():
                yield line
    except OSError:
        return


# -- reading -------------------------------------------------------------------------------------------------------
def size_bytes(path: Path) -> int:
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


def status(rec: dict[str, Any]) -> str:
    """Where the dataset is on its way: empty -> collected -> checked -> uploaded -> training -> trained (or failed)."""
    ft, up, val = rec.get("finetune") or {}, rec.get("upload") or {}, rec.get("validate") or {}
    if ft.get("model"):
        return "trained"
    if str(ft.get("status", "")).lower() in ("failed", "cancelled") or str(up.get("status", "")).lower() == "failed":
        return "failed"
    if ft.get("job_id"):
        return "training"
    if up.get("dataset_id") or up.get("dataset_name"):
        return "uploaded"
    if val.get("ok"):
        return "checked"
    if val and not val.get("ok"):
        return "check failed"
    return "collected" if rec.get("episodes") else "empty"


def describe(root: Path, with_size: bool = True) -> dict[str, Any]:
    rec = record(root)
    info = _read(root / "meta" / "info.json")
    episodes = int(info.get("total_episodes") or len(rec.get("episodes") or []))
    return {
        "name": root.name, "source": rec.get("source", "unknown"), "episodes": episodes,
        "frames": int(info.get("total_frames") or 0), "fps": info.get("fps"),
        "tasks": sorted({e.get("task") for e in rec.get("episodes") or [] if e.get("task")}),
        "size_bytes": size_bytes(root) if with_size else None, "status": status(rec),
        "validate": rec.get("validate"), "upload": rec.get("upload"), "finetune": rec.get("finetune"),
        "model": (rec.get("finetune") or {}).get("model"), "created": rec.get("created"),
        "pending": len(list((root / ".pending").glob("*"))) if (root / ".pending").is_dir() else 0,
    }


def list_all(datasets_dir: Path, with_size: bool = True) -> list[dict[str, Any]]:
    if not datasets_dir.is_dir():
        return []
    out = [describe(d, with_size) for d in sorted(datasets_dir.iterdir())
           if d.is_dir() and not d.name.startswith(".")
           and ((d / "dataset.json").is_file() or (d / "meta" / "info.json").is_file() or (d / ".pending").is_dir())]
    return sorted(out, key=lambda d: d.get("created") or 0, reverse=True)


def find_episode(datasets_dir: Path, run: str, episode: int) -> dict[str, Any] | None:
    """The dataset an attempt went into, if any: {"dataset", "index"}."""
    if not datasets_dir.is_dir():
        return None
    for d in datasets_dir.iterdir():
        for e in record(d).get("episodes") or []:
            if e.get("run") == run and e.get("episode") == episode:
                return {"dataset": d.name, "index": e.get("index")}
    return None


# -- check -------------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def lerobot_view(root: Path) -> Iterator[Path]:
    """The dataset as Dream Labs' tools should see it: only meta/, data/, videos/ (links, nothing copied)."""
    tmp = Path(tempfile.mkdtemp(prefix="dreamkit-ds-"))
    view = tmp / root.name
    view.mkdir()
    try:
        for part in LEROBOT_PARTS:
            if (root / part).exists():
                os.symlink((root / part).resolve(), view / part)
        yield view
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def validate(path: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """DreamClient().validate_dataset: the checks upload_dataset runs (local only; no key, no network)."""
    from .plugins.harvester.plugin import YAM_CONFIG

    config = config or YAM_CONFIG
    try:
        import dream

        try:
            report = dream.DreamClient().validate_dataset(str(path), config)
        except Exception as error:  # a client that cannot be built without credentials: the same module-level checks
            if "validate_dataset" in str(error):
                raise
            from dream import datasets as dream_datasets

            report = dream_datasets.validate_dataset(str(path), config)
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    data = {k: getattr(report, k) for k in dir(report) if not k.startswith("_") and not callable(getattr(report, k))}
    data = json.loads(json.dumps(data, default=str))
    data.setdefault("ok", True)
    return data


def check(root: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """validate_dataset on the dataset, recorded in dataset.json (validate)."""
    if not (root / "meta" / "info.json").is_file():
        report: dict[str, Any] = {"ok": False, "error": "no episodes yet"}
    else:
        with lerobot_view(root) as view:
            report = validate(view, config)
    report["checked"] = time.time()
    update(root, validate=report)
    return report


# -- fine-tune: the one path (the Data tab and `dreamkit datasets train` both call these) --------------------------
## The embodiment every dreamkit dataset is uploaded under: the YAM body as the harvester and DreamGym write it.
EMBODIMENT = "dreamkit-yam"
TERMINAL = ("processed", "failed")


def train_config() -> dict[str, Any]:
    from .plugins.harvester.plugin import YAM_CONFIG

    return {**YAM_CONFIG, "head_camera": "overhead"}


def _client() -> Any:
    import dream

    return dream.DreamClient()


def _plain(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")} if hasattr(obj, "__dict__") else {"raw": str(obj)}


def train(root: Path, *, suffix: str = "dreamkit", steps: int | None = None, name: str | None = None,
          wait_s: float | None = None, poll_s: float = 20.0, log: Any = print) -> dict[str, Any]:
    """Checks the dataset, registers the embodiment once, uploads it, waits for Dream Labs to process it and starts the
    fine-tune (fine-tuning credits). Everything is recorded in dataset.json (validate, upload, finetune) as it happens.

    `wait_s`: how long to wait for processing (None: until it is done); if it runs out, the upload stays recorded as
    processing and `refresh()` starts the training later. Returns the finetune record."""
    root = Path(root)
    report = check(root, train_config())
    if not report.get("ok"):
        raise RuntimeError(f"dataset check failed: {report.get('error')}")
    log(f"validate_dataset ok: {report.get('episodes')} episodes, {report.get('frames')} frames")
    client = _client()
    mine = {e.name.split("/", 1)[-1]: e for e in client.get_embodiments()}
    embodiment = mine.get(EMBODIMENT) or client.add_embodiment(EMBODIMENT, train_config())
    rec = record(root)
    previous = list((rec.get("finetune") or {}).get("previous") or [])
    if (rec.get("finetune") or {}).get("job_id"):  # trained before: keep the old model ids
        old = rec["finetune"]
        previous.append({k: old.get(k) for k in ("model", "job_id", "updated")})
    upload_name = (name or root.name).replace("_", "-").lower()[:60]
    if (rec.get("upload") or {}).get("dataset_name"):  # uploaded before (maybe with fewer episodes): a new name
        upload_name = f"{upload_name}-{time.strftime('%m%d-%H%M')}"
    update(root, upload={"dataset_name": upload_name, "dataset_id": None, "status": "uploading", "updated": time.time()},
           finetune={"stage": "uploading", "job_id": None, "model": None, "previous": previous, "updated": time.time()})
    with lerobot_view(root) as view:
        ds = client.upload_dataset(str(view), upload_name, embodiment.name)
    update(root, upload={"dataset_id": ds.id, "dataset_name": ds.name, "status": "processing", "updated": time.time()},
           finetune={"stage": "processing", "updated": time.time()})
    log(f"uploaded {ds.name}; Dream Labs is processing it")
    deadline = None if wait_s is None else time.monotonic() + wait_s
    while (st := client.get_dataset_status(ds.id)).status not in TERMINAL:
        if deadline is not None and time.monotonic() > deadline:
            log("still processing; `dreamkit datasets status` starts the training once it is processed")
            return record(root).get("finetune") or {}
        time.sleep(poll_s)
    if st.status == "failed":
        update(root, upload={"status": "failed", "error": str(st.error)},
               finetune={"stage": "failed", "status": "failed", "error": str(st.error), "updated": time.time()})
        raise RuntimeError(f"dataset processing failed: {st.error}")
    return _start_job(root, client, ds.name, suffix, steps, log)


def _start_job(root: Path, client: Any, dataset_name: str, suffix: str, steps: int | None, log: Any) -> dict[str, Any]:
    job = _plain(client.finetune(dataset_name, suffix, steps=steps))
    rec = update(root, upload={"status": "processed"},
                 finetune={"stage": "training", "status": str(job.get("status") or "queued"), "job": job,
                           "job_id": job.get("id") or job.get("job_id"), "suffix": suffix, "steps": steps,
                           "started": time.time(), "updated": time.time()})
    log(f"fine-tune started: job {rec['finetune']['job_id']} (a few hours)")
    return rec["finetune"]


def refresh(root: Path, *, suffix: str | None = None, log: Any = print) -> dict[str, Any]:
    """Asks Dream Labs where the dataset's upload or job is: a processed upload starts the training, a finished job
    records its model id. Returns the finetune record."""
    root = Path(root)
    rec = record(root)
    up, ft = rec.get("upload") or {}, rec.get("finetune") or {}
    client = _client()
    if ft.get("stage") == "processing" and up.get("dataset_id"):
        st = client.get_dataset_status(up["dataset_id"])
        if st.status == "processed":
            return _start_job(root, client, up["dataset_name"], suffix or ft.get("suffix") or "dreamkit",
                              ft.get("steps"), log)
        if st.status == "failed":
            rec = update(root, upload={"status": "failed", "error": str(st.error)},
                         finetune={"stage": "failed", "status": "failed", "error": str(st.error), "updated": time.time()})
        return rec.get("finetune") or {}
    if not ft.get("job_id"):
        return ft
    job = _plain(client.get_finetuning_status(ft["job_id"]))
    model = job.get("fine_tuned_model")
    state = str(job.get("status", "")).lower()
    stage = "done" if model else ("failed" if state in ("failed", "cancelled") else "training")
    rec = update(root, finetune={"stage": stage, "status": job.get("status"), "model": model, "job": job,
                                 "updated": time.time()})
    if model:
        log(f"fine-tune ready: {model}")
    return rec["finetune"]
