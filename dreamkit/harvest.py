"""`dreamkit harvest`: gather the successful episodes of runs into one folder and check it with Dream Labs' own
dataset checks (DreamClient.validate_dataset, local only). Nothing is uploaded here."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .plugins.harvester import YAM_CONFIG


def episodes(run_dir: Path, split: str = "success") -> list[Path]:
    return sorted(p.parent.parent for p in (run_dir / "dataset" / split).glob("ep*/meta/info.json"))


def validate(path: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Runs DreamClient().validate_dataset (the checks upload_dataset runs; no network, no key needed)."""
    config = config or YAM_CONFIG
    try:
        import dream

        try:
            report = dream.DreamClient().validate_dataset(str(path), config)
        except Exception as error:  # a client that cannot be built without credentials: same checks, module-level
            if "validate_dataset" in str(error):
                raise
            from dream import datasets

            report = datasets.validate_dataset(str(path), config)
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    data = {k: getattr(report, k) for k in dir(report) if not k.startswith("_") and not callable(getattr(report, k))}
    return json.loads(json.dumps(data, default=str))


def export(run_dirs: list[Path], out: Path) -> list[Path]:
    """Copies each run's successful episodes into `out/<run>-<ep>/` (one LeRobot dataset per subfolder)."""
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for run in run_dirs:
        for ep in episodes(run):
            target = out / f"{run.name}-{ep.name}"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(ep, target, ignore=shutil.ignore_patterns("dreamkit.json"))
            copied.append(target)
    return copied
