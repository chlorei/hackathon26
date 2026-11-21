"""fine-tune — Learn, closed: successful attempts -> one checked dataset -> a Dream-1 fine-tune (PLUGIN_STANDARD.md).

At the end of every run (on the ctx.defer worker, never in the loop):
1. gathers the harvester's successful attempts of this run, and of earlier runs of the same task, into
   run_dir/fine-tune/dataset/ (one LeRobot v2.1 dataset per subfolder, as `dreamkit harvest export` builds it);
2. checks it with Dream Labs' own validate_dataset (local, no key, no credits) and writes status.json;
3. only with `auto_train` on and at least `min_episodes`: registers the YAM embodiment once, uploads the dataset,
   waits up to `processing_wait_s` for Dream Labs to process it and starts the fine-tune (fine-tuning credits).

At the start of every run it asks Dream Labs about the last job (found in the newest earlier run's job.json): a
processed upload starts training, a finished job announces its model id. A job takes hours, so the answer usually
comes a few runs later. Ported from DreamGym's Data & fine-tune.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from dreamkit.plugin import Plugin, PluginContext
from dreamkit.plugins.harvester.plugin import YAM_CONFIG
from dreamkit.standard import Manifest

EMBODIMENT = "dreamkit-yam"
## The YAM body as the harvester writes it, plus the camera the model treats as the head.
CONFIG = {**YAM_CONFIG, "head_camera": "overhead"}
ACTIVE = ("uploading", "processing", "training")


class Params(BaseModel):
    include_previous_runs: bool = Field(True, description="Also gather successful attempts of earlier runs")
    same_task_only: bool = Field(True, description="Only attempts of this run's task")
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

    # -- run start: carry the last job over and ask about it ----------------------------------------------------------
    def on_run_start(self, ctx: PluginContext) -> None:
        self.job = self._previous_job(ctx)
        if not self.job:
            return
        self._save_job(ctx)
        if self.params.check_status and self.job.get("stage") in ("processing", "training"):
            ctx.defer(self._refresh, ctx)

    def _previous_job(self, ctx: PluginContext) -> dict[str, Any]:
        root = Path(ctx.run_dir).parent
        for run in sorted((p for p in root.iterdir() if p.is_dir() and p.name != ctx.run_id), reverse=True):
            job = _read(run / self.name / "job.json")
            if job:
                return job
        return {}

    def _save_job(self, ctx: PluginContext) -> None:
        self.job["updated"] = time.time()
        ctx.files.path("job.json").write_text(json.dumps(self.job, indent=2, default=str))

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

    # -- run end: build, check, maybe train --------------------------------------------------------------------------
    def on_run_end(self, ctx: PluginContext) -> None:
        ctx.defer(self._collect, ctx, str(ctx.task or ""))

    def _sources(self, ctx: PluginContext, task: str) -> list[Path]:
        root = Path(ctx.run_dir).parent
        runs = [Path(ctx.run_dir)]
        if self.params.include_previous_runs:
            runs += sorted(p for p in root.iterdir() if p.is_dir() and p.name != ctx.run_id)
        found = []
        for run in runs:
            for ep in sorted((run / "harvester" / "success").glob("ep*/meta/info.json")):
                folder = ep.parent.parent
                if self.params.same_task_only and task and _read(folder / "dreamkit.json").get("task") != task:
                    continue
                found.append(folder)
        return found

    def _collect(self, ctx: PluginContext, task: str) -> None:
        dataset = ctx.files.path("dataset")
        dataset.mkdir(parents=True, exist_ok=True)
        sources = self._sources(ctx, task)
        for ep in sources:
            target = dataset / f"{ep.parent.parent.parent.name}-{ep.name}"
            if not target.exists():
                shutil.copytree(ep, target, ignore=shutil.ignore_patterns("dreamkit.json"))
        frames, seconds = 0, 0.0
        for info in dataset.glob("*/meta/info.json"):
            meta = _read(info)
            frames += int(meta.get("total_frames") or 0)
            seconds += int(meta.get("total_frames") or 0) / max(1, int(meta.get("fps") or 10))
        episodes = len(list(dataset.glob("*/meta/info.json")))
        report: dict[str, Any] = {"ok": False, "error": "no successful attempts yet"}
        if episodes:
            from dreamkit import harvest

            report = harvest.validate(dataset, CONFIG)
        status = {"run": ctx.run_id, "task": task, "episodes": episodes, "frames": frames,
                  "duration_s": round(seconds, 1), "min_episodes": self.params.min_episodes,
                  "auto_train": self.params.auto_train, "validate": report,
                  "sources": sorted({p.parent.parent.parent.name for p in sources}), "updated": time.time()}
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
                     f"dataset {episodes} attempts, {seconds / 60:.1f} min, validate_dataset ok{more}",
                     episodes=episodes, duration_s=round(seconds, 1))
        busy = self.job.get("stage") in ACTIVE
        done = self.job.get("stage") == "done" and not self.params.train_again
        if ready and self.params.auto_train and not busy and not done:
            self._upload(ctx, dataset, task, episodes)
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
            ds = client.upload_dataset(str(dataset), name, embodiment.name)
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
