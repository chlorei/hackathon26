"""Dataset folder -> validated -> uploaded -> fine-tuned Dream-1, with the dream library's own calls.

State is kept beside the dataset in a hidden folder (`<dataset>/.dreamgym/`), which the dataset checks and the
upload skip: validate.json (the last local check), finetune.json (the job, refreshed by status()).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .lerobot_writer import EMBODIMENT_CONFIG

EMBODIMENT = "dreamgym-yam"


def state_dir(dataset: Path) -> Path:
    d = Path(dataset) / ".dreamgym"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(dataset: Path, name: str, data: dict) -> None:
    (state_dir(dataset) / name).write_text(json.dumps(data, indent=2, default=str))


def _client():
    import dream

    return dream.DreamClient()


def validate(dataset: Path) -> dict:
    """Local check against the sim's embodiment; nothing is sent."""
    report = _client().validate_dataset(str(dataset), EMBODIMENT_CONFIG)
    out = {k: getattr(report, k, None) for k in ("ok", "error", "episodes", "frames", "fps", "state_dim", "action_dim")}
    out["checked_at"] = time.time()
    _save(dataset, "validate.json", out)
    return out


def start(dataset: Path, name: str | None = None, suffix: str = "dreamgym", steps: int | None = None, log=print) -> dict:
    """Validates, registers the embodiment once, uploads, waits for processing, queues the job."""
    dataset = Path(dataset)
    v = validate(dataset)
    log(f"validate: ok={v['ok']} episodes={v['episodes']} frames={v['frames']} fps={v['fps']}")
    if not v["ok"]:
        raise RuntimeError(f"dataset check failed: {v['error']}")
    client = _client()
    mine = {e.name.split("/", 1)[-1]: e for e in client.get_embodiments()}
    embodiment = mine.get(EMBODIMENT) or client.add_embodiment(EMBODIMENT, EMBODIMENT_CONFIG)
    log(f"embodiment: {embodiment.name}")
    path = state_dir(dataset) / "finetune.json"
    before = json.loads(path.read_text()) if path.is_file() else {}
    # earlier fine-tunes of this folder: kept, so a retrain does not lose the model ids
    previous = before.get("previous", []) + ([{k: before.get(k) for k in ("model", "job_id", "dataset_name", "started")}]
                                            if before.get("job_id") else [])
    name = name or dataset.name.replace("_", "-").lower()
    if before.get("dataset_name"):  # uploaded before (maybe with fewer episodes): a new name for the new upload
        name = f"{name}-{time.strftime('%m%d-%H%M')}"
    _save(dataset, "finetune.json", {"stage": "uploading", "dataset_name": name, "previous": previous,
                                     "updated": time.time()})
    ds = client.upload_dataset(str(dataset), name, embodiment.name)
    log(f"uploaded {ds.id} ({ds.status}); processing ...")
    _save(dataset, "finetune.json", {"stage": "processing", "dataset_id": ds.id, "dataset_name": ds.name,
                                     "previous": previous, "updated": time.time()})
    while (st := client.get_dataset_status(ds.id)).status not in ("processed", "failed"):
        time.sleep(20)
    if st.status == "failed":
        _save(dataset, "finetune.json", {"stage": "failed", "error": st.error, "dataset_id": ds.id,
                                         "dataset_name": ds.name, "previous": previous, "updated": time.time()})
        raise RuntimeError(f"dataset processing failed: {st.error}")
    job = client.finetune(ds.name, suffix, steps=steps)
    record = {"stage": "training", "dataset_id": ds.id, "dataset_name": ds.name, "job": job,
              "job_id": job.get("id") or job.get("job_id"), "previous": previous, "started": time.time(),
              "updated": time.time()}
    _save(dataset, "finetune.json", record)
    log(json.dumps(job, indent=2, default=str))
    return record


def status(dataset: Path) -> dict:
    """Refreshes the job from the service; once it has published, `model` is the id to run."""
    path = state_dir(dataset) / "finetune.json"
    record = json.loads(path.read_text()) if path.is_file() else {}
    if not record.get("job_id"):
        return record
    job = _client().get_finetuning_status(record["job_id"])
    job = job if isinstance(job, dict) else getattr(job, "__dict__", {"raw": str(job)})
    record["job"] = job
    record["model"] = job.get("fine_tuned_model")
    state = str(job.get("status", "")).lower()
    record["stage"] = "done" if record["model"] else ("failed" if state in ("failed", "cancelled") else "training")
    record["updated"] = time.time()
    _save(dataset, "finetune.json", record)
    return record
