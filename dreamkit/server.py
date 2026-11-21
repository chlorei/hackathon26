"""The panel's backend: REST + WebSocket over Runner processes (MuJoCo never runs in this process)."""

from __future__ import annotations

import asyncio
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config as config_mod
from . import registry, storage
from .manager import describe, discover

UI_DIST = Path(__file__).resolve().parent / "ui_dist"


class LiveRun:
    """One Runner subprocess and everything it has printed so far."""

    def __init__(self, run_id: str):
        self.id = run_id
        self.proc: asyncio.subprocess.Process | None = None
        self.backlog: list[dict[str, Any]] = []
        self.frames: dict[str, dict[str, Any]] = {}
        self.subscribers: set[asyncio.Queue] = set()
        self.stderr: list[str] = []
        self.done = asyncio.Event()
        self.returncode: int | None = None

    def publish(self, record: dict[str, Any]) -> None:
        if record.get("type") == "frame":
            self.frames[record["cam"]] = record
        else:
            self.backlog.append(record)
        for q in list(self.subscribers):
            try:
                q.put_nowait(record)
            except asyncio.QueueFull:
                if record.get("type") != "frame":  # never lose an event; frames may be dropped
                    try:
                        q.get_nowait()
                        q.put_nowait(record)
                    except (asyncio.QueueEmpty, asyncio.QueueFull):
                        pass


class RunRequest(BaseModel):
    task: str | None = None
    model: str | None = None
    episodes: int | None = None
    seconds: float | None = None
    mock: bool = True
    overrides: dict[str, Any] = {}


class PluginUpdate(BaseModel):
    enabled: bool | None = None
    params: dict[str, Any] | None = None


class InstallRequest(BaseModel):
    name: str


def create_app(config_path: str | Path) -> FastAPI:
    config_path = Path(config_path).expanduser().resolve()
    app = FastAPI(title="dreamkit")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])
    live: dict[str, LiveRun] = {}

    def load_cfg() -> tuple[config_mod.Config, Path]:
        if config_path.is_file():
            return config_mod.load(config_path)
        return config_mod.Config(), config_path.parent

    def runs_root() -> Path:
        cfg, base = load_cfg()
        root = Path(cfg.runs_dir).expanduser()
        return root if root.is_absolute() else base / root

    def save_raw(update) -> None:
        raw = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
        raw = raw or {}
        update(raw)
        config_mod.Config.model_validate(raw)  # never write a config the runner would refuse
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))

    def active() -> LiveRun | None:
        return next((r for r in live.values() if not r.done.is_set()), None)

    # -- config & plugins --------------------------------------------------------------------------------------
    @app.get("/api/config")
    def get_config():
        cfg, base = load_cfg()
        data = cfg.model_dump()
        data["config_path"] = str(config_path)
        data["sim_found"] = (cfg.sim_dir(base) / "yam_env.py").is_file()
        return data

    @app.get("/api/plugins")
    def get_plugins():
        cfg, base = load_cfg()
        found, problems = discover([base / "plugins"])
        info = {p["name"]: p for p in describe(found)}
        out = []
        for spec in cfg.plugins:
            entry = info.pop(spec.name, None)
            out.append({
                "name": spec.name,
                "installed": entry is not None,
                "enabled": spec.enabled,
                "in_config": True,
                "description": entry["description"] if entry else "not installed",
                "source": entry["source"] if entry else None,
                "params": {**(entry["defaults"] if entry else {}), **spec.params},
                "defaults": entry["defaults"] if entry else {},
            })
        for name, entry in info.items():
            out.append({"name": name, "installed": True, "enabled": False, "in_config": False,
                        "description": entry["description"], "source": entry["source"],
                        "params": dict(entry["defaults"]), "defaults": entry["defaults"]})
        return {"plugins": out, "problems": problems}

    @app.post("/api/plugins/{name}")
    def update_plugin(name: str, body: PluginUpdate):
        cfg, base = load_cfg()
        found, _ = discover([base / "plugins"])
        defaults = found[name]["cls"].defaults if name in found else {}

        def update(raw):
            specs = config_mod.Config.model_validate({"plugins": raw.get("plugins") or []}).plugins
            spec = next((s for s in specs if s.name == name), None)
            if spec is None:
                spec = config_mod.PluginSpec(name=name, enabled=False)
                # keep `report` last: it summarises what the others wrote
                at = next((i for i, s in enumerate(specs) if s.name == "report"), len(specs))
                specs.insert(at, spec)
            if body.enabled is not None:
                spec.enabled = body.enabled
            if body.params is not None:
                # store only what differs from the plugin's defaults
                spec.params = {k: v for k, v in body.params.items() if defaults.get(k, object()) != v}
            raw["plugins"] = config_mod.dump_plugins(specs)

        try:
            save_raw(update)
        except Exception as error:
            raise HTTPException(400, f"invalid plugin settings: {error}") from error
        return get_plugins()

    @app.get("/api/registry")
    def get_registry():
        cfg, base = load_cfg()
        entries, source = registry.load(cfg.registry_url)
        found, _ = discover([base / "plugins"])
        for e in entries:
            e["installed"] = e["name"] in found
        return {"source": source, "plugins": entries}

    @app.post("/api/install")
    async def install(body: InstallRequest):
        cfg, _ = load_cfg()
        entries, _ = registry.load(cfg.registry_url)
        result = await asyncio.to_thread(registry.install, body.name, entries)
        import importlib

        importlib.invalidate_caches()
        if result.returncode != 0:
            raise HTTPException(500, (result.stdout + result.stderr)[-3000:])
        return {"ok": True, "output": (result.stdout + result.stderr)[-3000:]}

    # -- runs --------------------------------------------------------------------------------------------------
    async def pump(run: LiveRun) -> None:
        proc = run.proc
        assert proc is not None and proc.stdout is not None and proc.stderr is not None

        async def read_err():
            async for line in proc.stderr:
                text = line.decode(errors="replace").rstrip()
                if text and "arena memory" not in text.lower():
                    run.stderr = (run.stderr + [text])[-50:]

        err_task = asyncio.create_task(read_err())
        async for line in proc.stdout:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            run.publish(record)
        run.returncode = await proc.wait()
        await err_task
        if run.returncode not in (0, None) and not any(r.get("type") == "run_end" for r in run.backlog):
            run.publish({"type": "event", "ts": time.time(), "plugin": "dreamkit", "level": "error",
                         "message": "runner exited: " + (" | ".join(run.stderr[-5:]) or f"code {run.returncode}"),
                         "episode": None, "t": None, "data": {}})
            run.publish({"type": "run_end", "run_id": run.id, "status": "error"})
        run.done.set()
        for q in list(run.subscribers):
            q.put_nowait(None)

    @app.post("/api/runs")
    async def start_run(body: RunRequest):
        if active() is not None:
            raise HTTPException(409, f"run {active().id} is still running; stop it first")
        cfg, base = load_cfg()
        data = cfg.model_dump()
        data.update({k: v for k, v in body.overrides.items() if k in data})
        for key in ("task", "model", "episodes", "seconds"):
            if getattr(body, key) is not None:
                data[key] = getattr(body, key)
        data["sim_path"] = str(cfg.sim_dir(base))
        data["runs_dir"] = str(runs_root())
        data.setdefault("realtime", True)
        try:
            config_mod.Config.model_validate(data)
        except Exception as error:
            raise HTTPException(400, str(error)) from error
        run_id = storage.new_run_id()
        folder = storage.run_dir(runs_root(), run_id)
        request = folder / "request.yaml"
        request.write_text(yaml.safe_dump(data, sort_keys=False))
        args = [sys.executable, "-m", "dreamkit.runner", "--config", str(request), "--json-events", "--run-id", run_id]
        if body.mock:
            args.append("--mock")
        run = LiveRun(run_id)
        run.proc = await asyncio.create_subprocess_exec(
            *args, cwd=str(base), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=2**23,
            start_new_session=True,
        )
        live[run_id] = run
        asyncio.create_task(pump(run))
        return {"id": run_id}

    @app.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: str):
        run = live.get(run_id)
        if run is None or run.done.is_set() or run.proc is None:
            raise HTTPException(404, "no such active run")
        run.proc.send_signal(signal.SIGINT)
        return {"ok": True}

    @app.get("/api/runs")
    def list_runs():
        runs = storage.list_runs(runs_root())
        for r in runs:
            lr = live.get(r["id"])
            if r.get("status") in ("running", "starting") and (lr is None or lr.done.is_set()):
                r["status"] = "interrupted"  # its runner is gone
        return {"runs": runs, "active": active().id if active() else None}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        folder = runs_root() / run_id
        info = storage.read_run(folder)
        if info is None:
            raise HTTPException(404, "no such run")
        info["events"] = [e for e in storage.read_events(folder) if e.get("type") != "frame"]
        info["report"] = (folder / "report.html").is_file()
        info["live"] = run_id in live and not live[run_id].done.is_set()
        return info

    @app.get("/api/runs/{run_id}/trace/{episode}")
    def get_trace(run_id: str, episode: int):
        path = runs_root() / run_id / f"ep{episode}" / "trace.json"
        if not path.is_file():
            raise HTTPException(404, "no trace")
        return json.loads(path.read_text())

    @app.websocket("/ws/runs/{run_id}")
    async def ws_run(socket: WebSocket, run_id: str):
        await socket.accept()
        run = live.get(run_id)
        if run is None:
            folder = runs_root() / run_id
            for e in storage.read_events(folder):
                await socket.send_text(json.dumps(e))
            await socket.close()
            return
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        backlog = list(run.backlog)
        frames = list(run.frames.values())
        run.subscribers.add(queue)
        try:
            for record in backlog + frames:
                await socket.send_text(json.dumps(record))
            if run.done.is_set():
                return
            while True:
                record = await queue.get()
                if record is None:
                    break
                await socket.send_text(json.dumps(record))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            run.subscribers.discard(queue)
            try:
                await socket.close()
            except RuntimeError:
                pass

    @app.get("/media/{run_id}/{path:path}")
    def media(run_id: str, path: str):
        root = (runs_root() / run_id).resolve()
        target = (root / path).resolve()
        if root not in target.parents or not target.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(target)

    @app.on_event("shutdown")
    def stop_children():
        for run in live.values():
            if not run.done.is_set() and run.proc is not None and run.proc.returncode is None:
                run.proc.send_signal(signal.SIGINT)

    if (UI_DIST / "index.html").is_file():
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
        print(f"dreamkit: listening on {host} — anyone on this network can start the robot", file=sys.stderr)
    print(f"dreamkit panel: http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}")
    uvicorn.run(create_app(config_path), host=host, port=port, log_level="warning")
