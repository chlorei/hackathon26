"""fine-tune — Learn, closed: the harvester's dataset -> checked -> a Dream-1 fine-tune (PLUGIN_STANDARD.md).

The harvester appends every successful attempt of a task to <data_dir>/datasets/harvest-<task>/, across runs. At the
end of every run (on the ctx.defer worker, never in the loop) this plugin:
1. checks that dataset with Dream Labs' own validate_dataset (local, no key, no credits), recorded in its dataset.json;
2. only with `auto_train` on and at least `min_episodes`: registers the YAM embodiment once, uploads the dataset,
   waits up to `processing_wait_s` for Dream Labs to process it and starts the fine-tune (fine-tuning credits).
The upload, the job and the model id are written to the dataset's dataset.json (upload, finetune), where the Data
tab shows them. At the start of every run it asks Dream Labs about an unfinished job: a processed upload starts
training, a finished job announces its model id. A job takes hours, so the answer usually comes a few runs later.
Its own page reads run_dir/fine-tune/status.json and job.json.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from dreamkit import datasets
from dreamkit.plugin import Plugin, PluginContext
from dreamkit.plugins.harvester.plugin import YAM_CONFIG
from dreamkit.standard import Manifest

EMBODIMENT = "dreamkit-yam"
## The YAM body as the harvester writes it, plus the camera the model treats as the head.
CONFIG = {**YAM_CONFIG, "head_camera": "overhead"}
ACTIVE = ("uploading", "processing", "training")


class Params(BaseModel):
    dataset: str = Field("", description="The dataset to train on, in <data_dir>/datasets/; empty: the harvester's "
                         "dataset of this run's task (harvest-<task>)")
    min_episodes: int = Field(20, ge=1, le=10000, description="Successful attempts needed before training starts",
                              json_schema_extra={"unit": ""})
    auto_train: bool = Field(False, description="Upload and start a Dream-1 fine-tune when the dataset is ready "
                             "(uses fine-tuning credits)")
    train_again: bool = Field(False, description="Start a new job even when an earlier one has finished")
    suffix: str = Field("dreamkit", description="The fine-tune is named dream-1-ft:<id>-<suffix>")
    train_steps: int = Field(0, ge=0, le=10000, description="Training steps, a multiple of 100; 0 for the service "
                             "default", json_schema_extra={"unit": ""})
    processing_wait_s: float = Field(60, ge=0, le=110, description="How long to wait for Dream Labs to process an "
                                     "upload before leaving training to the next run", json_schema_extra={"unit": "s"})
    check_status: bool = Field(True, description="At run start, ask Dream Labs about the last job")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "task"


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _plain(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")} if hasattr(obj, "__dict__") else {"raw": str(obj)}


def _say(ctx: PluginContext, level: str, text: str, **data: Any) -> None:
    """ctx.emit with the message in the form §7 asks (lowercase start, no trailing period), whatever an error said."""
    text = " ".join(text.split()).rstrip(". ")
    ctx.emit(level, text[:1].lower() + text[1:], **data)


class FineTune(Plugin):
    """Builds a fine-tuning dataset from successful attempts and trains Dream-1 on it."""

    name = "fine-tune"
    Params = Params
    manifest = Manifest(group="output", icon="GraduationCap", capability="observe",
                        hooks=["on_run_start", "on_run_end"], step_rate_hz=1, requires=["harvester"],
                        ui="ui/index.html")

    # -- run start: an unfinished job of any dataset, asked about ---------------------------------------------------
    def on_run_start(self, ctx: PluginContext) -> None:
        self.root: Path | None = None
        self.job = self._previous_job(ctx)
        if not self.job:
            return
        self._save_job(ctx)
        if self.params.check_status and self.job.get("stage") in ("processing", "training"):
            ctx.defer(self._refresh, ctx)

    def _previous_job(self, ctx: PluginContext) -> dict[str, Any]:
        """The newest dataset with an upload or a job on it (dataset.json: upload, finetune)."""
        if ctx.datasets_dir is None or not Path(ctx.datasets_dir).is_dir():
            return {}
        best: tuple[float, Path, dict[str, Any]] | None = None
        for root in Path(ctx.datasets_dir).iterdir():
            rec = datasets.record(root) if root.is_dir() else {}
            job = rec.get("finetune") or {}
            if not job.get("stage"):
                continue
            when = float(job.get("updated") or 0)
            if best is None or when > best[0]:
                best = (when, root, job)
        if best is None:
            return {}
        self.root = best[1]
        return dict(best[2])

    def _save_job(self, ctx: PluginContext) -> None:
        self.job["updated"] = time.time()
        ctx.files.path("job.json").write_text(json.dumps(self.job, indent=2, default=str))  # its page reads this
        if self.root is not None:
            upload = {k: self.job.get(k) for k in ("dataset_id", "dataset_name") if self.job.get(k)}
            stage = self.job.get("stage")
            upload["status"] = stage if stage in ("uploading", "processing", "processed", "failed") else "processed"
            datasets.update(self.root, upload=upload, finetune={
                "stage": stage, "job_id": self.job.get("job_id"), "model": self.job.get("model"),
                "status": "failed" if stage == "failed" else self.job.get("stage"), "error": self.job.get("error"),
                "updated": self.job["updated"]})

    def _refresh(self, ctx: PluginContext) -> None:
        try:
            import dream

            client = dream.DreamClient()
            if self.job["stage"] == "processing":
                st = client.get_dataset_status(self.job["dataset_id"])
                if st.status == "failed":
                    self.job.update(stage="failed", error=str(st.error))
                    _say(ctx, "error", f"dataset {self.job['dataset_name']} failed processing: {st.error}")
                elif st.status == "processed":
                    self._train(ctx, client)
            elif self.job["stage"] == "training":
                job = _plain(client.get_finetuning_status(self.job["job_id"]))
                self.job["job"] = job
                model = job.get("fine_tuned_model")
                if model:
                    self.job.update(stage="done", model=model)
                    _say(ctx, "success", f"fine-tune ready: {model}", model=model)
                elif str(job.get("status", "")).lower() in ("failed", "cancelled"):
                    self.job.update(stage="failed", error=str(job.get("error") or job.get("status")))
                    _say(ctx, "error", f"fine-tune {self.job['job_id']} {job.get('status')}")
                else:
                    _say(ctx, "info", f"fine-tune {self.job['job_id']} still {job.get('status', 'training')}")
        except Exception as error:  # no key, no network: say so, the loop goes on
            _say(ctx, "warn", f"could not ask dream labs about the fine-tune: {type(error).__name__}: {error}")
        self._save_job(ctx)

    # -- run end: check, maybe train -------------------------------------------------------------------------------
    def on_run_end(self, ctx: PluginContext) -> None:
        ctx.defer(self._collect, ctx, str(ctx.task or ""))

    def _dataset(self, ctx: PluginContext, task: str) -> Path | None:
        if ctx.datasets_dir is None:
            return None
        return Path(ctx.datasets_dir) / (self.params.dataset or datasets.harvest_name(task or "task"))

    def _collect(self, ctx: PluginContext, task: str) -> None:
        root = self._dataset(ctx, task)
        info = datasets.describe(root) if root is not None and root.is_dir() else {"episodes": 0, "frames": 0, "fps": 10}
        episodes, frames = int(info["episodes"]), int(info.get("frames") or 0)
        seconds = frames / max(1, int(info.get("fps") or 10))
        report: dict[str, Any] = {"ok": False, "error": "no successful attempts yet"}
        if episodes and root is not None:
            report = datasets.check(root, CONFIG)
        status = {"run": ctx.run_id, "task": task, "dataset": root.name if root else None, "episodes": episodes,
                  "frames": frames, "duration_s": round(seconds, 1), "min_episodes": self.params.min_episodes,
                  "auto_train": self.params.auto_train, "validate": report, "updated": time.time()}
        ctx.files.path("status.json").write_text(json.dumps(status, indent=2, default=str))
        ready = bool(report.get("ok")) and episodes >= self.params.min_episodes
        if not episodes:
            _say(ctx, "info", "no successful attempts to train on yet")
        elif not report.get("ok"):
            _say(ctx, "error", f"dataset of {episodes} attempts fails validate_dataset: {report.get('error')}",
                     episodes=episodes)
        else:
            more = "" if ready else f", {self.params.min_episodes - episodes} more to train"
            _say(ctx, "success" if ready else "info",
                     f"dataset {root.name}: {episodes} attempts, {seconds / 60:.1f} min, validate_dataset ok{more}",
                     episodes=episodes, duration_s=round(seconds, 1))
        busy = self.job.get("stage") in ACTIVE
        done = self.job.get("stage") == "done" and not self.params.train_again
        if ready and self.params.auto_train and not busy and not done:
            self.root = root
            self._upload(ctx, root, task, episodes)
        elif ready and not self.params.auto_train and not busy:
            _say(ctx, "info", "ready to train: switch on auto_train in fine-tune settings (uses fine-tuning credits)")

    def _upload(self, ctx: PluginContext, dataset: Path, task: str, episodes: int) -> None:
        try:
            import dream

            client = dream.DreamClient()
            mine = {e.name.split("/", 1)[-1]: e for e in client.get_embodiments()}
            embodiment = mine.get(EMBODIMENT) or client.add_embodiment(EMBODIMENT, CONFIG)
            name = f"dreamkit-{_slug(task)}-{_slug(ctx.run_id)}"
            self.job = {"stage": "uploading", "dataset_name": name, "episodes": episodes, "run": ctx.run_id}
            self._save_job(ctx)
            with datasets.lerobot_view(dataset) as view:  # meta/, data/, videos/: not dataset.json
                ds = client.upload_dataset(str(view), name, embodiment.name)
            self.job.update(stage="processing", dataset_id=ds.id, dataset_name=ds.name)
            self._save_job(ctx)
            _say(ctx, "info", f"uploaded {episodes} attempts as {ds.name}; dream labs is processing it")
            deadline = time.monotonic() + self.params.processing_wait_s
            while time.monotonic() < deadline:
                st = client.get_dataset_status(ds.id)
                if st.status == "processed":
                    self._train(ctx, client)
                    break
                if st.status == "failed":
                    self.job.update(stage="failed", error=str(st.error))
                    _say(ctx, "error", f"dataset {ds.name} failed processing: {st.error}")
                    break
                time.sleep(5)
            else:
                _say(ctx, "info", "still processing; training starts at the next run")
        except Exception as error:
            self.job.update(stage="failed", error=f"{type(error).__name__}: {error}")
            _say(ctx, "error", f"upload failed: {type(error).__name__}: {error}")
        self._save_job(ctx)

    def _train(self, ctx: PluginContext, client: Any) -> None:
        if not self.params.auto_train:
            self.job["stage"] = "processed"
            _say(ctx, "info", f"dataset {self.job['dataset_name']} processed; auto_train is off, not training")
            return
        job = _plain(client.finetune(self.job["dataset_name"], self.params.suffix,
                                     steps=self.params.train_steps or None))
        self.job.update(stage="training", job=job, job_id=job.get("id") or job.get("job_id"), started=time.time())
        _say(ctx, "success", f"fine-tune started: job {self.job['job_id']}, a few hours")
