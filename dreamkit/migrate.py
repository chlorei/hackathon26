"""`dreamkit migrate`: the old places dreamkit saved to, moved into one data folder.

    before                                   after (<data_dir>, default ./dreamkit-data)
    runs/<id>/                               runs/<id>/
      recorder/ep003_top.mp4                   ep003/top.mp4
      history/ep003_recap.mp4                  ep003/recap.mp4
      failure-monitor/ep003_trace.json         ep003/trace.json
      ep3/top.mp4 (oldest runs)                ep003/top.mp4                      (+ ep003/result.json)
      harvester/success/ep003/ (one dataset    datasets/harvest-<task>/ (one dataset, the episode appended;
        per attempt), dataset/success/...        the old folder kept in runs/<id>/.migrated/)
    gym/data/<name>/                         datasets/<name>/  (+ dataset.json)
    gym/runs/<id>/                           runs/gym-<id>/
    harvest/ (dreamkit harvest export)       datasets/export/  (the original in .migrated/harvest/)

Files are moved, never deleted; anything that does not fit is kept under .migrated/. Every move is printed.
run.json is rewritten so the panel finds the files where they are now. Running it twice does nothing new.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable

import yaml

from . import datasets, storage
from . import config as config_mod

OLD_FILE = [  # (pattern of a path inside a run, its new place); NNN is the zero-padded attempt
    (re.compile(r"^recorder/ep(\d+)_(.+\.mp4)$"), "ep{n}/{rest}"),
    (re.compile(r"^history/ep(\d+)_recap\.mp4$"), "ep{n}/recap.mp4"),
    (re.compile(r"^failure-monitor/ep(\d+)_trace\.json$"), "ep{n}/trace.json"),
    (re.compile(r"^ep(\d+)/(.+)$"), "ep{n}/{rest}"),
]


def _new_path(rel: str) -> str | None:
    for pattern, template in OLD_FILE:
        m = pattern.match(rel)
        if m:
            rest = m.group(2) if m.re.groups > 1 else ""
            new = template.format(n=f"{int(m.group(1)):03d}", rest=rest)
            return None if new == rel else new
    return None


def _rewrite(value: Any, moved: dict[str, str]) -> Any:
    if isinstance(value, str):
        return moved.get(value, value)
    if isinstance(value, list):
        return [_rewrite(v, moved) for v in value]
    if isinstance(value, dict):
        return {k: _rewrite(v, moved) for k, v in value.items()}
    return value


class Migration:
    def __init__(self, config: str | Path | None, say: Callable[[str], None] = print, dry_run: bool = False):
        path = Path(config) if config and Path(config).is_file() else None
        self.config_path = path
        self.cfg, self.base = config_mod.load(path)
        self.say, self.dry = say, dry_run
        self.old_runs = self.cfg.runs_path(self.base)  # where runs are now (runs_dir, or data_dir/runs)
        self.data = (self.base / Path(self.cfg.data_dir).expanduser()).resolve() \
            if not Path(self.cfg.data_dir).is_absolute() else Path(self.cfg.data_dir)
        self.runs = self.data / "runs"
        self.datasets = self.data / "datasets"
        self.moves = 0

    # -- helpers ------------------------------------------------------------------------------------------------------
    def _mv(self, src: Path, dst: Path) -> bool:
        if not src.exists():
            return False
        if dst.exists():
            self.say(f"  kept {self._rel(src)}: {self._rel(dst)} already exists")
            return False
        self.say(f"  {self._rel(src)} -> {self._rel(dst)}")
        self.moves += 1
        if not self.dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        return True

    def _rel(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(self.base.resolve()))
        except ValueError:
            return str(p)

    # -- steps -----------------------------------------------------------------------------------------------------------
    def run(self) -> int:
        self.say(f"dreamkit data folder: {self._rel(self.data)} (runs/, datasets/)")
        self.move_runs()
        for run in sorted(self.runs.iterdir()) if self.runs.is_dir() else []:
            if run.is_dir() and (run / "run.json").is_file():
                self.restructure(run)
        self.move_gym()
        self.move_export()
        self.update_config()
        self.say(f"{self.moves} move{'s' if self.moves != 1 else ''}" + (" planned (dry run, nothing changed)" if self.dry else
                                                                          "; nothing was deleted"))
        return 0

    def move_runs(self) -> None:
        legacy = [self.old_runs] + ([self.base / "runs"] if (self.base / "runs").resolve() != self.old_runs.resolve() else [])
        for old in legacy:
            if old.is_dir() and old.resolve() != self.runs.resolve():
                self.say(f"runs: {self._rel(old)}/ -> {self._rel(self.runs)}/")
                for run in sorted(old.iterdir()):
                    if run.is_dir():
                        self._mv(run, self.runs / run.name)
                if not self.dry and old.is_dir() and not any(old.iterdir()):
                    old.rmdir()

    def restructure(self, run: Path) -> None:
        """One run: attempt files into ep003/, its harvested episodes into datasets/, run.json rewritten."""
        info = storage.read_run(run)
        if info is None:
            return
        moved: dict[str, str] = {}
        files = [p for p in run.rglob("*") if p.is_file() and ".migrated" not in p.parts]
        for f in files:
            rel = str(f.relative_to(run))
            new = _new_path(rel)
            if new and not (run / new).exists():
                if not moved:
                    self.say(f"run {run.name}:")
                self._mv(f, run / new)
                moved[rel] = new
        for d in sorted(run.glob("ep*")):  # the oldest runs' ep3/ folders, now empty
            if d.is_dir() and re.fullmatch(r"ep\d+", d.name) and not re.fullmatch(r"ep\d{3}", d.name) and not self.dry:
                try:
                    d.rmdir()
                except OSError:
                    pass
        harvested = self.harvest(run, info)
        if moved or harvested:
            info = _rewrite(info, moved)
            for ep in info.get("episodes", []):
                n = int(ep.get("episode") or 0)
                if n in harvested:
                    ep["harvester"] = harvested[n]
            if not self.dry:
                storage.write_json(run / "run.json", info)
        if not self.dry:
            for ep in info.get("episodes", []):
                n = int(ep.get("episode") or 0)
                if n and not (run / f"ep{n:03d}" / "result.json").exists():
                    (run / f"ep{n:03d}").mkdir(exist_ok=True)
                    storage.write_json(run / f"ep{n:03d}" / "result.json", ep)

    def harvest(self, run: Path, info: dict[str, Any]) -> dict[int, dict[str, Any]]:
        """The harvester's one-episode datasets of a run (harvester/success/, or the older dataset/success/)
        appended to datasets/harvest-<task>/; the old folders kept in runs/<id>/.migrated/."""
        out: dict[int, dict[str, Any]] = {}
        for old in ("harvester", "dataset"):
            base = run / old
            if not base.is_dir():
                continue
            for ep in sorted((base / "success").glob("ep*")):
                if not (ep / "meta" / "info.json").is_file():
                    continue
                n = int(re.sub(r"\D", "", ep.name) or 0)
                meta = json.loads((ep / "dreamkit.json").read_text()) if (ep / "dreamkit.json").is_file() else {}
                task = meta.get("task") or next((e.get("task") for e in info.get("episodes", [])
                                                 if e.get("episode") == n), None) or info.get("task") or "task"
                name = datasets.harvest_name(task)
                if datasets.find_episode(self.datasets, run.name, n):
                    continue
                self.say(f"  {self._rel(ep)} -> datasets/{name}/ (episode appended)")
                self.moves += 1
                if not self.dry:
                    index = datasets.append_episode(self.datasets / name, ep, source="harvester", run=run.name,
                                                    episode=n, task=task, model=info.get("model"))
                    out[n] = {"dataset": name, "index": index}
            self._mv(base, run / ".migrated" / old)
        return out

    def move_gym(self) -> None:
        gym = self.base / "gym"
        for ds in sorted((gym / "data").iterdir()) if (gym / "data").is_dir() else []:
            if ds.is_dir() and self._mv(ds, self.datasets / ds.name) and not self.dry:
                target = self.datasets / ds.name
                if not (target / "dataset.json").is_file():
                    info = json.loads((target / "meta" / "info.json").read_text()) if (target / "meta" / "info.json").is_file() else {}
                    datasets.update(target, name=ds.name, source="gym", created=target.stat().st_mtime,
                                    episodes=[{"index": i} for i in range(int(info.get("total_episodes") or 0))])
        for run in sorted((gym / "runs").iterdir()) if (gym / "runs").is_dir() else []:
            if run.is_dir():
                self._mv(run, self.runs / (run.name if run.name.startswith("gym-") else f"gym-{run.name}"))
        for sub in ("data", "runs"):
            if not self.dry and (gym / sub).is_dir() and not any((gym / sub).iterdir()):
                (gym / sub).rmdir()
        if not self.dry and gym.is_dir() and not any(gym.iterdir()):
            gym.rmdir()

    def move_export(self) -> None:
        """`dreamkit harvest export`'s folder (one dataset per subfolder) -> datasets/export/."""
        old = self.base / "harvest"
        if not old.is_dir():
            return
        eps = sorted(p for p in old.iterdir() if (p / "meta" / "info.json").is_file())
        if eps:
            self.say(f"harvest/: {len(eps)} episodes -> datasets/export/")
            for ep in eps:
                run, _, n = ep.name.rpartition("-ep")
                if not self.dry and not datasets.find_episode(self.datasets, run, int(n or 0)):
                    datasets.append_episode(self.datasets / "export", ep, source="export", run=run,
                                            episode=int(n or 0))
        self._mv(old, self.data / ".migrated" / "harvest")

    def update_config(self) -> None:
        if self.config_path is None:
            return
        raw = yaml.safe_load(self.config_path.read_text()) or {}
        if "runs_dir" not in raw and "data_dir" in raw:
            return
        raw.pop("runs_dir", None)
        raw.setdefault("data_dir", self.cfg.data_dir)
        self.say(f"{self.config_path.name}: runs_dir removed, data_dir: {raw['data_dir']}")
        if not self.dry:
            text = self.config_path.read_text()
            lines = [ln for ln in text.splitlines() if not ln.startswith("runs_dir:")]
            if not any(ln.startswith("data_dir:") for ln in lines):
                at = next((i + 1 for i, ln in enumerate(lines) if ln.startswith("sim_path:")), 0)
                lines.insert(at, f"data_dir: {raw['data_dir']}")
            self.config_path.write_text("\n".join(lines) + "\n")


def migrate(config: str | Path | None, say: Callable[[str], None] = print, dry_run: bool = False) -> int:
    return Migration(config, say, dry_run).run()
