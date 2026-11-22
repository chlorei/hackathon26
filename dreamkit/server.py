"""The panel's backend. It never runs the robot and never sends prompts: it manages plugins (dreamkit.yaml) and
monitors runs, which the user's own loop writes to runs/<id>/ through dreamkit.wrap()."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config as config_mod
from . import cost, registry, storage
from .manager import describe, discover

UI_DIST = Path(__file__).resolve().parent / "ui_dist"
THEME = Path(__file__).resolve().parent / "theme" / "dreamkit-theme.css"  # shared with plugin pages (§7a)
THEME_LINK = '<link rel="stylesheet" href="/theme/dreamkit-theme.css">'
DEFAULT_PAGE = THEME.parent / "plugin-page.html"  # every plugin without a page of its own gets this one (§7a)
LIVE_FRESH_S = 10.0  # a run whose heartbeat is older than this is not live
POLL_S = 0.12


class PluginUpdate(BaseModel):
    enabled: bool | None = None
    params: dict[str, Any] | None = None


class InstallRequest(BaseModel):
    name: str


class CreateRequest(BaseModel):
    messages: list[dict[str, str]]  # the chat so far: [{"role": "user" | "assistant", "text": ...}]
    files: dict[str, str] | None = None  # the current package, when the chat refines it


class DemoRequest(BaseModel):
    policy: str = "mock"  # mock: free, scripted failures; dream: Dream-1 on the simulator (uses credits)


SIM_LOOP = Path(__file__).resolve().parent.parent / "examples" / "sim_loop.py"  # a source checkout has it
DEMO_ARGS = {
    # the free demo: scripted arm in the simulator, judged by the simulator's world (so recovery can be seen to work),
    # and no plugin that calls an outside model (no network, no credits, nothing to time out on camera)
    "mock": ["--mock", "--episodes", "3", "--behaviors", "dropped,success,dropped", "--with-world",
             "--without", "vision-judge", "--without", "monitor-stats"],  # its name before and after the rename
    "dream": ["--episodes", "2", "--seconds", "30"],
}


class CatalogRequest(BaseModel):
    enable: bool = False  # also switch it on in dreamkit.yaml


class Label(BaseModel):
    success: bool


def _heartbeat(folder: Path) -> dict[str, Any] | None:
    try:
        return json.loads((folder / "live.json").read_text())
    except (OSError, ValueError):
        return None


def _is_live(folder: Path, info: dict[str, Any] | None) -> bool:
    beat = _heartbeat(folder)
    return bool(info and info.get("status") == "running" and beat and time.time() - beat.get("updated", 0) < LIVE_FRESH_S)


def _annotate(run: dict[str, Any]) -> None:
    """Fields the panel reads on every run, filled for runs written before they existed.

    - source: "gym" for DreamGym's runs, "loop" for everything else.
    - kind: "dream" when Dream-1 (or a fine-tune of it) drove the robot, "scripted" for the mock, DreamGym's scripted
      expert, teleop and test policies: only "dream" runs say anything about the model.
    - episodes[].task_readable: whether dreamkit can read an object and a target out of the attempt's task; without
      them nothing judged success, so such attempts are counted apart (invalid), not as failures."""
    from .task import parse

    run.setdefault("source", "loop")
    model = str(run.get("model") or "")
    scripted = bool(run.get("mock")) or not model or any(w in model.lower() for w in ("scripted", "fake", "mock", "teleop"))
    run["kind"] = "scripted" if scripted or "dream" not in model.lower() else "dream"
    # every field the panel reads is there, whatever wrote the run (older dreamkit, a crash, another tool)
    run.setdefault("task", "")
    run.setdefault("model", "")
    run.setdefault("plugins", [])
    episodes = run.setdefault("episodes", [])
    run.setdefault("summary", {"attempts": len(episodes), "successes": sum(1 for e in episodes if e.get("success")),
                               "scenes": 0, "scene_successes": 0, "reasons": {}})
    if not isinstance(run.get("seconds"), (int, float)):  # the attempt length the loop allowed; older runs lack it
        run["seconds"] = max([float(e.get("duration_s") or 0) for e in episodes] + [0.0])
    for e in episodes:
        e.setdefault("duration_s", 0.0)
        if e.get("cost_usd") is None:  # older runs: the core's formula on their recorded calls (dreamkit.cost)
            from . import cost as cost_mod

            e["cost_usd"] = round(cost_mod.usd(int(e.get("requests") or 0),
                                               cost_mod.price_per_hour(run.get("model"), bool(run.get("mock")))), 4)
        e.setdefault("videos", [])
        e.setdefault("requests", 0)
        e.setdefault("scene", 0)
        e.setdefault("attempt", 1)
        if "task_readable" not in e:
            obj, target = parse(str(e.get("task") or run.get("task") or ""))
            e["task_readable"] = bool(obj and target)
    from . import stats

    run["stats"] = stats.summary(episodes)  # the numbers every screen shows, from one place


class AppsHost:
    """Serves every app plugin under /apps/<name>/: its router at api/, its page at /. Apps are looked up again on
    refresh() (after an install, on /api/apps), so a new app works without restarting the panel (§13)."""

    def __init__(self, context):
        self.context = context
        self.apps: dict[str, Any] = {}
        self.problems: list[str] = []
        self.subs: dict[str, Any] = {}
        self.refresh()

    def refresh(self) -> None:
        import importlib

        from . import apps as apps_mod

        importlib.invalidate_caches()
        found, problems = apps_mod.discover()
        for name in list(self.subs):
            if name not in found or found[name] is not self.apps.get(name):
                del self.subs[name]
        self.apps, self.problems = found, problems

    def sub(self, name: str) -> Any:
        if name not in self.subs:
            a = self.apps[name]
            sub = FastAPI(title=f"dreamkit app {name}")
            if a.router is not None:
                sub.include_router(a.router(self.context()), prefix="/api")
            if a.page is not None:
                sub.mount("/", StaticFiles(directory=a.page, html=True), name="page")
            self.subs[name] = sub
        return self.subs[name]

    async def __call__(self, scope, receive, send) -> None:
        from starlette.responses import JSONResponse, RedirectResponse
        from starlette.routing import get_route_path

        rest = get_route_path(scope)  # the path under /apps
        name = rest.strip("/").split("/", 1)[0]
        if name and name not in self.apps and scope["type"] == "http":
            self.refresh()  # installed since the panel started
        if not name or name not in self.apps:
            response = JSONResponse({"detail": f"no app plugin named {name!r}"}, status_code=404)
            return await response(scope, receive, send)
        if rest.rstrip("/") == f"/{name}" and not rest.endswith("/") and scope["type"] == "http":
            return await RedirectResponse(scope.get("root_path", "") + f"/{name}/")(scope, receive, send)
        try:
            sub = self.sub(name)
        except Exception as error:  # noqa: BLE001 - one broken app never takes the panel down
            response = JSONResponse({"detail": f"app {name} failed to start: {type(error).__name__}: {error}"},
                                    status_code=500)
            return await response(scope, receive, send)
        child = dict(scope, root_path=scope.get("root_path", "") + f"/{name}")
        await sub(child, receive, send)


def create_app(config_path: str | Path) -> FastAPI:
    config_path = Path(config_path).expanduser().resolve()
    app = FastAPI(title="dreamkit")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])

    def load_cfg() -> tuple[config_mod.Config, Path]:
        if config_path.is_file():
            return config_mod.load(config_path)
        return config_mod.Config(), config_path.parent

    def runs_root() -> Path:
        cfg, base = load_cfg()
        return cfg.runs_path(base)

    def datasets_root() -> Path:
        cfg, base = load_cfg()
        return cfg.datasets_path(base)

    def dataset_folder(name: str) -> Path:
        from . import datasets

        folder = (datasets_root() / name).resolve()
        if not datasets.NAME_RE.match(name) or folder.parent != datasets_root().resolve() or not folder.is_dir():
            raise HTTPException(404, "no such dataset")
        return folder

    def run_folder(run_id: str) -> Path:
        folder = (runs_root() / run_id).resolve()
        if folder.parent != runs_root().resolve() or not folder.is_dir():
            raise HTTPException(404, "no such run")
        return folder

    def save_raw(update) -> None:
        raw = (yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}) or {}
        update(raw)
        config_mod.Config.model_validate(raw)  # never write a config wrap() would refuse
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))

    # -- config & plugins --------------------------------------------------------------------------------------
    @app.get("/api/config")
    def get_config():
        cfg, base = load_cfg()
        return {"config_file": config_path.name, "sim_found": (cfg.sim_dir(base) / "yam_env.py").is_file(),
                "light": cfg.light, "data_dir": str(cfg.data_path(base)), "runs_dir": str(cfg.runs_path(base)),
                "datasets_dir": str(cfg.datasets_path(base))}

    def latest_stats() -> tuple[dict[str, Any], dict[str, str], str | None]:
        """Each plugin's timings/errors and refusals from the most recent run (what the monitor knows)."""
        for info in storage.list_runs(runs_root()):
            if info.get("plugin_stats") is not None:
                return info.get("plugin_stats") or {}, info.get("refused") or {}, info.get("id")
        return {}, {}, None

    def plugin_list():
        cfg, base = load_cfg()
        found, problems = discover([base / "plugins"])
        info = {p["name"]: p for p in describe(found)}
        stats, refused, run_id = latest_stats()
        enabled_control = [s.name for s in cfg.plugins if s.enabled and info.get(s.name, {}).get("capability") == "control"]
        all_control = [n for n, e in info.items() if e["capability"] == "control"]
        out = []
        order = {s.name: i for i, s in enumerate(cfg.plugins)}
        names = [s.name for s in cfg.plugins] + [n for n in sorted(info) if n not in order]
        specs = {s.name: s for s in cfg.plugins}
        for name in names:
            entry = info.get(name)
            spec = specs.get(name)
            enabled = bool(spec and spec.enabled)
            row: dict[str, Any] = {
                "name": name, "installed": entry is not None, "enabled": enabled, "in_config": spec is not None,
                "order": order.get(name, 999),
                "description": entry["description"] if entry else "not installed",
                "source": entry["source"] if entry else None,
                "group": entry["group"] if entry else "diagnosis", "icon": entry["icon"] if entry else "Plug",
                "capability": entry["capability"] if entry else "observe",
                "needs": entry["needs"] if entry else ["robot"],
                "manifest": entry["manifest"] if entry else {},
                "schema": entry["schema"] if entry else {},
                "defaults": entry["defaults"] if entry else {},
                "params": {**(entry["defaults"] if entry else {}), **(spec.params if spec else {})},
                "stats": stats.get(name), "refused": refused.get(name) or (stats.get(name) or {}).get("inactive"),
                "stats_run": run_id,
            }
            if row["capability"] == "control":
                others = [n for n in all_control if n != name]
                row["conflicts"] = sorted(set(others + row["manifest"].get("conflicts", [])))
                if enabled and len(enabled_control) > 1 and enabled_control[0] != name:
                    row["refused"] = f"conflicts with {enabled_control[0]} (only one control plugin may be on)"
            else:
                row["conflicts"] = row["manifest"].get("conflicts", [])
            out.append(row)
        return {"plugins": out, "problems": problems}

    @app.get("/api/plugins")
    def get_plugins():
        return plugin_list()

    @app.post("/api/plugins/{name}")
    def update_plugin(name: str, body: PluginUpdate):
        _, base = load_cfg()
        found, _ = discover([base / "plugins"])
        defaults = found[name]["cls"].Params().model_dump(mode="json") if name in found else {}

        def update(raw):
            specs = config_mod.Config.model_validate({"plugins": raw.get("plugins") or []}).plugins
            spec = next((s for s in specs if s.name == name), None)
            if spec is None:
                spec = config_mod.PluginSpec(name=name, enabled=False)
                at = next((i for i, s in enumerate(specs) if s.name == "report"), len(specs))  # report stays last
                specs.insert(at, spec)
            if body.enabled is not None:
                spec.enabled = body.enabled
            if body.params is not None:
                spec.params = {k: v for k, v in body.params.items() if defaults.get(k, object()) != v}
            raw["plugins"] = config_mod.dump_plugins(specs)

        try:
            save_raw(update)
        except Exception as error:
            raise HTTPException(400, f"invalid plugin settings: {error}") from error
        return plugin_list()

    @app.delete("/api/plugins/{name}")
    async def remove_plugin(name: str):
        _, base = load_cfg()
        found, _ = discover([base / "plugins"])
        result = await asyncio.to_thread(registry.uninstall, name, found)
        if result.returncode != 0:
            raise HTTPException(400, (result.stdout + result.stderr)[-2000:])

        def update(raw):
            specs = config_mod.Config.model_validate({"plugins": raw.get("plugins") or []}).plugins
            raw["plugins"] = config_mod.dump_plugins([s for s in specs if s.name != name])

        save_raw(update)
        return {"ok": True}

    @app.get("/api/registry")
    def get_registry():
        cfg, base = load_cfg()
        entries, source = registry.load(cfg.registry_url)
        found, _ = discover([base / "plugins"])
        host.refresh()
        for e in entries:
            e["installed"] = e["name"] in found or e["name"] in host.apps
        return {"source": "GitHub" if source.startswith("http") else "local registry.yaml", "plugins": entries}

    @app.post("/api/install")
    async def install(body: InstallRequest):
        from . import pdk

        ok, log = await asyncio.to_thread(pdk.install, body.name)
        import importlib

        importlib.invalidate_caches()
        host.refresh()
        return {"ok": ok, "log": log, "app": next((a.title for a in host.apps.values()
                                                  if body.name in (a.name, *a.aliases)), None)}

    # -- Create: a plugin described in words, written by a model, checked like dreamkit install ---------------------
    def create_job(job_id: str):
        from . import create

        job = create.JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "no such plugin draft")
        return job

    @app.post("/api/create")
    def create_start(body: CreateRequest):
        from . import create

        if not any(m.get("role") == "user" and str(m.get("text", "")).strip() for m in body.messages):
            raise HTTPException(400, "describe the plugin first")
        _, base = load_cfg()
        found, _ = discover([base / "plugins"])
        taken = {n for n, e in found.items() if e["source"] != "generated"} | set(host.apps)
        return create.start(body.messages, body.files, taken).public()

    @app.get("/api/llm/usage")
    def llm_usage():
        from . import llm

        return {"today": llm.usage_today()}

    @app.get("/api/create/{job_id}")
    def create_get(job_id: str):
        return create_job(job_id).public()

    @app.post("/api/create/{job_id}/catalog")
    def create_catalog(job_id: str, body: CatalogRequest):
        from . import create

        job = create_job(job_id)
        _, base = load_cfg()
        try:
            create.add_to_catalog(job, base)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        return update_plugin(job.name, PluginUpdate(enabled=body.enable)) if body.enable else plugin_list()

    @app.get("/api/create/{job_id}/zip")
    def create_zip(job_id: str):
        from . import create

        job = create_job(job_id)
        try:
            data = create.zip_bytes(job)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{job.name}.zip"'})

    # -- runs (the monitor) ------------------------------------------------------------------------------------
    # -- the demo button on Live: the panel starts the reference loop (examples/sim_loop.py) as its own process. The
    # loop stays what it always is, independent of the panel: this only presses start and stop, one run at a time.
    demo: dict[str, Any] = {"proc": None, "policy": None, "started": None, "stopped": False}
    demo_log = Path("~/.dreamkit/demo.log").expanduser()

    def demo_failed() -> str | None:
        """The last lines of a demo that ended with an error (not one the user stopped): the panel shows them."""
        proc = demo["proc"]
        if proc is None or proc.poll() in (None, 0) or demo["stopped"]:
            return None
        try:
            lines = [ln for ln in demo_log.read_text(errors="replace").splitlines() if ln.strip()]
        except OSError:
            lines = []
        return "\n".join(lines[-8:]) or f"the demo exited with code {proc.poll()}"

    def dream_key(base: Path) -> bool:
        from . import llm

        llm.load_env(base)  # DREAM_API_KEY from the project's .env, for the button and for the loop it starts
        if os.getenv("DREAM_API_KEY"):
            return True
        creds = Path(os.getenv("DREAM_CREDENTIALS") or "~/.dreamlabs/credentials").expanduser()
        try:
            return any(line.strip().startswith("DREAM_API_KEY=") and line.split("=", 1)[1].strip().strip("\"'")
                       for line in creds.read_text().splitlines())
        except OSError:
            return False

    def demo_running() -> bool:
        proc = demo["proc"]
        return proc is not None and proc.poll() is None

    @app.get("/api/demo")
    def demo_state():
        cfg, base = load_cfg()
        return {"running": demo_running(), "policy": demo["policy"] if demo_running() else None,
                "started": demo["started"] if demo_running() else None, "available": SIM_LOOP.is_file(),
                "dream_key": dream_key(base), "plugins": [s.name for s in cfg.plugins if s.enabled],
                "failed": demo_failed()}

    @app.post("/api/demo")
    def demo_start(body: DemoRequest):
        if body.policy not in DEMO_ARGS:
            raise HTTPException(400, "policy must be mock or dream")
        if not SIM_LOOP.is_file():
            raise HTTPException(404, "examples/sim_loop.py is not here (a source checkout of dreamkit has it)")
        if demo_running() or live_run()["id"]:
            raise HTTPException(409, "a run is already going: one at a time")
        _, base = load_cfg()
        if body.policy == "dream" and not dream_key(base):
            raise HTTPException(400, "no Dream API key: add DREAM_API_KEY to .env (or ~/.dreamlabs/credentials)")
        log = demo_log
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(SIM_LOOP), *DEMO_ARGS[body.policy], "--config", str(config_path)]
        demo.update(proc=subprocess.Popen(cmd, cwd=str(base), stdout=log.open("w"), stderr=subprocess.STDOUT,
                                          env=dict(os.environ), start_new_session=True),
                    policy=body.policy, started=time.time(), stopped=False)
        return demo_state()

    @app.delete("/api/demo")
    def demo_stop():
        proc = demo["proc"]
        if proc is not None and proc.poll() is None:
            demo["stopped"] = True  # its exit code is not a failure to show
            proc.send_signal(signal.SIGINT)  # the loop's Ctrl-C: it closes the run as stopped
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.terminate()
        return demo_state()

    @app.get("/api/live")
    def live_run():
        """The run the user's loop is writing right now, if any."""
        for info in storage.list_runs(runs_root()):
            if _is_live(runs_root() / info["id"], info):
                return {"id": info["id"], "task": info.get("task"), "model": info.get("model"), "mock": info.get("mock")}
        return {"id": None}

    @app.get("/api/runs")
    def list_runs():
        runs = storage.list_runs(runs_root())
        cfg, _ = load_cfg()
        for r in runs:
            if r.get("cost_usd") is None:  # a run from before dreamkit.cost: the same formula, from its calls
                calls = sum(int(e.get("requests") or 0) for e in r.get("episodes") or [])
                r["cost_usd"] = round(cost.usd(calls, cost.price_per_hour(r.get("model"), bool(r.get("mock")), cfg.prices)), 4)
            if r.get("status") == "running" and not _is_live(runs_root() / r["id"], r):
                r["status"] = "interrupted"  # its loop is gone
            _annotate(r)
        return {"runs": runs}

    @app.get("/api/stats")
    def get_stats(kind: str = "all"):
        """dreamkit.stats over the attempts of the runs of one kind (dream | scripted | all): History's top card.
        `real` is the same over the runs Dream-1 drove, where money was spent, whichever kind is shown."""
        from . import stats

        runs = storage.list_runs(runs_root())
        for r in runs:
            _annotate(r)
        chosen = [r for r in runs if kind == "all" or r.get("kind") == kind]
        real = [r for r in runs if r.get("kind") == "dream"]
        eps = lambda rs: [e for r in rs for e in r.get("episodes") or []]  # noqa: E731
        return {"kind": kind, "runs": len(chosen), "empty_runs": sum(1 for r in chosen if not r.get("episodes")),
                **stats.summary(eps(chosen)), "real": stats.summary(eps(real))}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        folder = run_folder(run_id)
        info = storage.read_run(folder)
        if info is None:
            raise HTTPException(404, "no such run")
        info["live"] = _is_live(folder, info)
        if info.get("status") == "running" and not info["live"]:
            info["status"] = "interrupted"
        info["recap"] = None
        for recap in (folder / "history" / "summary.json", folder / "recap" / "summary.json", folder / "recap.json"):
            try:
                info["recap"] = json.loads(recap.read_text())
                break
            except (OSError, ValueError):
                pass
        report = next((r for r in (folder / "history" / "report.html", folder / "report" / "report.html",
                                   folder / "report.html") if r.is_file()), None)
        info["report"] = str(report.relative_to(folder)) if report else None
        _annotate(info)
        return info

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str):
        folder = run_folder(run_id)
        if _is_live(folder, storage.read_run(folder)):
            raise HTTPException(409, "this run is still being written; stop the loop first")
        shutil.rmtree(folder)
        return {"ok": True}

    @app.post("/api/runs/{run_id}/episodes/{episode}/label")
    def label_episode(run_id: str, episode: int, body: Label):
        """A person's verdict for an attempt the robot could not judge (no world): ✓ or ✗ in History."""
        from . import history

        folder = run_folder(run_id)
        info = storage.read_run(folder)
        ep = next((e for e in (info or {}).get("episodes", []) if e.get("episode") == episode), None)
        if ep is None:
            raise HTTPException(404, "no such attempt")
        ep["success"], ep["labeled_by"], ep["judged_by"] = body.success, "person", "person"  # a person decides last
        if body.success:
            ep["reason"] = None
        elif not ep.get("reason"):
            ep["reason"] = "marked_failed"
            ep["message"] = "marked failed by a person"
        # the harvester kept this attempt waiting in its dataset's .pending/: ✓ appends it, ✗ drops it
        from . import datasets

        if ep.get("harvester"):
            ep["harvester"] = datasets.resolve_pending(datasets_root(), ep["harvester"], body.success)
        from .kit import _summary

        info["summary"] = _summary(info["episodes"])
        storage.write_json(folder / "run.json", info)
        history.write_run(folder, info, info["episodes"])
        return {"ok": True, "episode": ep}

    @app.post("/api/runs/{run_id}/reveal")
    def reveal(run_id: str, episode: int | None = None):
        """Opens the run's folder in Finder / the file manager (only works where the panel runs)."""
        import platform
        import subprocess

        folder = run_folder(run_id)
        for name in (f"ep{episode:03d}", f"ep{episode}") if episode is not None else ():
            if (folder / name).is_dir():
                folder = folder / name
                break
        opener = {"Darwin": ["open"], "Windows": ["explorer"]}.get(platform.system(), ["xdg-open"])
        try:
            subprocess.Popen([*opener, str(folder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as error:
            raise HTTPException(500, f"cannot open a file manager here: {error}") from error
        return {"ok": True}

    @app.get("/api/runs/{run_id}/episodes/{episode}/files")
    def episode_files(run_id: str, episode: int):
        """The attempt's folder (runs/<run>/ep003/): its files, and the dataset it went into, if any."""
        from . import datasets

        folder = run_folder(run_id)
        ep_dir = folder / f"ep{episode:03d}"
        files = []
        if ep_dir.is_dir():
            for f in sorted(ep_dir.rglob("*")):
                if f.is_file():
                    files.append({"name": str(f.relative_to(ep_dir)), "path": str(f.relative_to(folder)),
                                  "size_bytes": f.stat().st_size})
        info = storage.read_run(folder) or {}
        ep = next((e for e in info.get("episodes", []) if e.get("episode") == episode), {})
        h = ep.get("harvester") or {}
        dataset = ({"dataset": h["dataset"], "index": h.get("index"), "pending": bool(h.get("pending"))}
                   if h.get("dataset") else datasets.find_episode(datasets_root(), run_id, episode))
        return {"folder": ep_dir.name, "files": files, "dataset": dataset}

    @app.get("/api/datasets")
    def list_datasets():
        from . import datasets

        return {"datasets": datasets.list_all(datasets_root()), "dir": str(datasets_root())}

    @app.post("/api/datasets/{name}/check")
    async def check_dataset(name: str):
        """Dream Labs' own dataset checks (validate_dataset). Local only: nothing is uploaded."""
        from . import datasets

        return await asyncio.to_thread(datasets.check, dataset_folder(name))

    @app.post("/api/datasets/{name}/reveal")
    def reveal_dataset(name: str):
        import platform
        import subprocess

        opener = {"Darwin": ["open"], "Windows": ["explorer"]}.get(platform.system(), ["xdg-open"])
        subprocess.Popen([*opener, str(dataset_folder(name))], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"ok": True}

    @app.get("/api/storage")
    def get_storage():
        """Where everything is and how big it is; mock runs, which can go."""
        from . import datasets

        runs = storage.list_runs(runs_root())
        mock = [r for r in runs if r.get("mock")]
        return {"data_dir": str(runs_root().parent), "runs_dir": str(runs_root()), "datasets_dir": str(datasets_root()),
                "runs": len(runs), "runs_bytes": datasets.size_bytes(runs_root()) if runs_root().is_dir() else 0,
                "datasets": len(datasets.list_all(datasets_root(), with_size=False)),
                "datasets_bytes": datasets.size_bytes(datasets_root()) if datasets_root().is_dir() else 0,
                "mock_runs": len(mock),
                "mock_bytes": sum(datasets.size_bytes(runs_root() / r["id"]) for r in mock if r.get("id"))}

    @app.post("/api/storage/delete-mock-runs")
    def delete_mock_runs(body: dict[str, Any]):
        """Deletes the runs of the mock policy (scripted failures, no model). Needs {"confirm": true}."""
        if body.get("confirm") is not True:
            raise HTTPException(400, 'confirm with {"confirm": true}')
        gone = []
        for r in storage.list_runs(runs_root()):
            folder = runs_root() / str(r.get("id", ""))
            if r.get("mock") and folder.is_dir() and not _is_live(folder, r):
                shutil.rmtree(folder, ignore_errors=True)
                gone.append(r["id"])
        return {"deleted": len(gone)}

    @app.get("/api/runs/{run_id}/trace/{episode}")
    def get_trace(run_id: str, episode: int):
        folder = run_folder(run_id)
        for path in (folder / f"ep{episode:03d}" / "trace.json",  # the attempt's folder (§8)
                     folder / "failure-monitor" / f"ep{episode:03d}_trace.json", folder / f"ep{episode}" / "trace.json"):
            if path.is_file():
                break
        else:
            raise HTTPException(404, "no trace")
        return json.loads(path.read_text())

    @app.websocket("/ws/live/{run_id}")
    async def ws_live(socket: WebSocket, run_id: str, frames: int = 0):
        """Follows a run's files: new lines of events.jsonl, the live camera frames, the heartbeat's telemetry."""
        await socket.accept()
        try:
            folder = run_folder(run_id)
        except HTTPException:
            await socket.close()
            return
        offset = 0
        frame_mtimes: dict[str, float] = {}
        beat_mtime = 0.0
        idle_since = time.time()
        status_checked = 0.0
        watching = folder / "live" / ".watching"
        last_touch = 0.0
        try:
            while True:
                if frames and time.time() - last_touch > 1.0:  # the loop renders cameras only while they are shown (§5)
                    watching.parent.mkdir(exist_ok=True)
                    watching.touch()
                    last_touch = time.time()
                sent = False
                events = folder / "events.jsonl"
                if events.is_file():
                    with events.open("rb") as f:
                        f.seek(offset)
                        chunk = f.read()
                    complete = chunk[: chunk.rfind(b"\n") + 1]
                    offset += len(complete)
                    for line in complete.decode(errors="replace").splitlines():
                        if line.strip():
                            await socket.send_text(line)
                            sent = True
                for jpg in sorted((folder / "live").glob("*.jpg")):
                    mtime = jpg.stat().st_mtime
                    if frame_mtimes.get(jpg.stem) != mtime:
                        frame_mtimes[jpg.stem] = mtime
                        data = jpg.read_bytes()
                        await socket.send_text(json.dumps({"type": "frame", "cam": jpg.stem,
                                                           "jpeg": base64.b64encode(data).decode("ascii")}))
                        sent = True
                beat_file = folder / "live.json"
                if beat_file.is_file() and beat_file.stat().st_mtime != beat_mtime:
                    beat_mtime = beat_file.stat().st_mtime
                    beat = _heartbeat(folder)
                    if beat:
                        await socket.send_text(json.dumps({"type": "heartbeat", **beat}))
                        sent = True
                if sent:
                    idle_since = time.time()
                if time.time() - status_checked > 1.0:  # run.json: once a second is enough to notice the end
                    status_checked = time.time()
                    info = storage.read_run(folder)
                    if info and info.get("status") != "running" and time.time() - idle_since > 1.0:
                        break  # the run ended and everything has been sent
                if time.time() - idle_since > LIVE_FRESH_S * 3:
                    break  # the loop went quiet for good
                await asyncio.sleep(POLL_S)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            try:
                await socket.close()
            except RuntimeError:
                pass

    @app.get("/media/{run_id}/{path:path}")
    def media(run_id: str, path: str, request: Request):
        root = run_folder(run_id)
        target = (root / path).resolve()
        # §7a: a plugin page runs sandboxed (origin "null"); it may read its own plugin's files and the run's record
        # (run.json, events.jsonl), nothing else. A missing file of its own is a plain 404 it can read, too.
        cors = {}
        if request.headers.get("origin") == "null":
            parts = Path(path).parts
            if parts in (("run.json",), ("events.jsonl",)) or (len(parts) > 1 and (
                    parts[0] in plugin_pages() or re.fullmatch(r"ep\d{3}", parts[0]))):
                cors = {"Access-Control-Allow-Origin": "null"}
        if root not in target.parents or not target.is_file():
            raise HTTPException(404, "not found", headers=cors or None)
        return FileResponse(target, headers=cors or None)

    def plugin_pages() -> dict[str, Path]:
        """§7a: plugin name -> its page: its own (manifest ui), or the standard one every plugin gets."""
        _, base = load_cfg()
        found, _ = discover([base / "plugins"])
        out = {}
        for name, entry in found.items():
            ui = getattr(entry["cls"].manifest, "ui", None)
            out[name] = (Path(inspect.getfile(entry["cls"])).resolve().parent / ui).resolve() if ui else DEFAULT_PAGE
        return out

    @app.get("/theme/{file}")
    def theme(file: str):
        types = {"dreamkit-theme.css": "text/css", "dreamkit-page.js": "text/javascript", "icons.json": "application/json"}
        if file not in types:
            raise HTTPException(404, "not found")
        # no-cache: a browser that kept an older theme draws new pages unstyled (black charts, no margins)
        return FileResponse(THEME.parent / file, media_type=types[file],
                            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"})

    @app.get("/plugin-ui/{name}/{path:path}")
    def plugin_ui(name: str, path: str = ""):
        """A plugin's own page with the theme linked in; files next to it are served from the same folder."""
        page = plugin_pages().get(name)
        if page is None or not page.is_file():
            raise HTTPException(404, "this plugin has no page")
        folder = page.parent
        target = page if not path else (folder / path).resolve()
        if target != page and (folder not in target.parents or not target.is_file()):
            raise HTTPException(404, "not found")
        if target.suffix.lower() not in (".html", ".htm"):
            return FileResponse(target)
        html = target.read_text(errors="replace")
        # the plugin as the Plugins page knows it (description, icon, settings), for the page's header and Settings
        try:
            row = next((p for p in plugin_list()["plugins"] if p["name"] == name), {"name": name})
        except Exception:  # noqa: BLE001 - the page still shows, with only its name
            row = {"name": name}
        if name == "vision-judge":  # its page tells how many model calls this machine made today
            from . import llm

            row = {**row, "llm_usage": llm.usage_today()}
        info = json.dumps(row, default=str).replace("</", "<\\/")
        head = f'<base href="/plugin-ui/{name}/">{THEME_LINK}<script>window.DK_PLUGIN = {info};</script>'
        lower = html.lower()
        at = lower.find("<head>")
        html = html[: at + 6] + head + html[at + 6 :] if at >= 0 else head + html
        return Response(html, media_type="text/html", headers={"Cache-Control": "no-cache"})

    # -- app plugins (§13): their API and page under /apps/<name>/, found again after an install -----------------
    from . import apps as apps_mod

    def app_context() -> "apps_mod.AppContext":
        cfg, base = load_cfg()
        return apps_mod.AppContext(config_path=config_path, base=base, runs_dir=runs_root(), datasets_dir=datasets_root(),
                                   world=apps_mod.world_available(cfg, base), config=cfg)

    host = AppsHost(app_context)
    app.state.apps = host
    app.mount("/apps", host, name="apps")

    @app.get("/api/apps")
    def list_apps():
        host.refresh()  # an app installed a moment ago gets its tab without restarting the panel
        cfg, base = load_cfg()
        world = apps_mod.world_available(cfg, base)
        out = []
        for a in host.apps.values():
            d = a.describe()
            d["inactive"] = ("needs Dream Labs' simulator (set sim_path in dreamkit.yaml, or pip install dreamlabs)"
                             if "world" in a.manifest.needs and not world else None)
            out.append(d)
        return {"apps": out, "problems": host.problems, "world": world}

    @app.delete("/api/apps/{name}")
    async def remove_app(name: str):
        """Uninstalls an app plugin installed with pip: its tab goes (the panel finds apps again, §13)."""
        host.refresh()
        a = host.apps.get(name)
        if a is None:
            raise HTTPException(404, f"no app plugin named {name!r}")
        origin = getattr(a, "origin", "") or ""
        if not origin.startswith("pip "):
            raise HTTPException(400, f"{a.title} is loaded from a folder ({origin}), not installed with pip: "
                                     "take it out of DREAMKIT_PLUGIN_PATH instead")
        result = await asyncio.to_thread(registry.uninstall_dist, origin.split(" ", 1)[1])
        if result.returncode != 0:
            raise HTTPException(400, (result.stdout + result.stderr)[-2000:])
        host.refresh()
        if name in host.apps:
            raise HTTPException(400, f"{a.title} is still found after pip uninstall (another copy is installed)")
        return {"ok": True, "title": a.title}

    if (UI_DIST / "index.html").is_file():
        @app.middleware("http")
        async def fresh_panel(request: Request, call_next):
            # the page that names the current build is never cached (its assets are, by their hashed names): a browser
            # that kept an old index.html would keep showing the old panel after an update
            response = await call_next(request)
            if request.url.path in ("/", "/index.html"):
                response.headers["Cache-Control"] = "no-cache"
            return response

        app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")
    else:
        @app.get("/")
        def no_ui():
            return HTMLResponse("<h3>dreamkit API is up.</h3><p>The UI is not built: <code>cd ui && npm install && "
                                "npm run build</code>. API: <a href='/docs'>/docs</a></p>")

    return app


def serve(config_path: str, host: str = "127.0.0.1", port: int = 8800) -> None:
    import uvicorn

    if host not in ("127.0.0.1", "localhost"):
        print(f"dreamkit: listening on {host} — anyone on this network can see your runs", file=sys.stderr)
    print(f"dreamkit panel: http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}")
    uvicorn.run(create_app(config_path), host=host, port=port, log_level="warning")
