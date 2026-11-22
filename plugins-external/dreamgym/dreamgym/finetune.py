"""Dataset folder -> validated -> uploaded -> fine-tuned Dream-1: the Data tab's buttons, on dreamkit's one path.

The work is dreamkit.datasets.check / train / refresh (the same functions `dreamkit datasets check|train|status`
runs): one way to upload and train, recorded in the dataset's dataset.json. This module only keeps the Data tab's
view of it in `<dataset>/.dreamgym/` (validate.json, finetune.json), which the page reads.
"""

from __future__ import annotations

import json
from pathlib import Path

from dreamkit import datasets

EMBODIMENT = datasets.EMBODIMENT


def state_dir(dataset: Path) -> Path:
    d = Path(dataset) / ".dreamgym"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _mirror(dataset: Path) -> dict:
    """The Data tab's copy of dataset.json's validate / upload / finetune records."""
    rec = datasets.record(Path(dataset))
    v = rec.get("validate") or {}
    if v:
        (state_dir(dataset) / "validate.json").write_text(
            json.dumps({**v, "checked_at": v.get("checked")}, indent=2, default=str))
    ft = {**(rec.get("upload") or {}), **(rec.get("finetune") or {})}
    if rec.get("finetune"):
        (state_dir(dataset) / "finetune.json").write_text(json.dumps(ft, indent=2, default=str))
    return ft


def validate(dataset: Path) -> dict:
    """Local check against the YAM embodiment; nothing is sent."""
    report = datasets.check(Path(dataset), datasets.train_config())
    _mirror(dataset)
    return {k: report.get(k) for k in ("ok", "error", "episodes", "frames", "fps", "state_dim", "action_dim")}


def start(dataset: Path, name: str | None = None, suffix: str = "dreamgym", steps: int | None = None, log=print) -> dict:
    """Checks, registers the embodiment once, uploads, waits for processing, queues the job (dreamkit.datasets.train)."""
    try:
        return datasets.train(Path(dataset), suffix=suffix, steps=steps, name=name, log=log)
    finally:
        _mirror(dataset)


def status(dataset: Path) -> dict:
    """Refreshes the job from the service; once it has published, `model` is the id to run."""
    try:
        datasets.refresh(Path(dataset))
    finally:
        out = _mirror(dataset)
    return out
