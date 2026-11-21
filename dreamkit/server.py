"""The panel's backend. It never runs the robot and never sends prompts: it manages plugins (dreamkit.yaml) and
monitors runs, which the user's own loop writes to runs/<id>/ through dreamkit.wrap()."""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
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
LIVE_FRESH_S = 10.0  # a run whose heartbeat is older than this is not live
POLL_S = 0.12


class PluginUpdate(BaseModel):
    enabled: bool | None = None
    params: dict[str, Any] | None = None


class InstallRequest(BaseModel):
    name: str


def _heartbeat(folder: Path) -> dict[str, Any] | None:
    try:
        return json.loads((folder / "live.json").read_text())
    except (OSError, ValueError):
        return None


def _is_live(folder: Path, info: dict[str, Any] | None) -> bool:
    beat = _heartbeat(folder)
    return bool(info and info.get("status") == "running" and beat and time.time() - beat.get("updated", 0) < LIVE_FRESH_S)


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
        root = Path(cfg.runs_dir).expanduser()
        return root if root.is_absolute() else base / root

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
                "light": cfg.light, "runs_dir": cfg.runs_dir}

    def plugin_list():
        cfg, base = load_cfg()
        found, problems = discover([base / "plugins"])
        info = {p["name"]: p for p in describe(found)}
        out = []
        for i, spec in enumerate(cfg.plugins):
            entry = info.pop(spec.name, None)
            out.append({
                "name": spec.name, "installed": entry is not None, "enabled": spec.enabled, "in_config": True,
                "order": i, "group": (entry or {}).get("group", "other"),
                "description": entry["description"] if entry else "not installed",
                "source": entry["source"] if entry else None,
                "params": {**(entry["defaults"] if entry else {}), **spec.params},
                "defaults": entry["defaults"] if entry else {},
            })
        for name, entry in info.items():
            out.append({"name": name, "installed": True, "enabled": False, "in_config": False, "order": 999,
                        "group": entry.get("group", "other"), "description": entry["description"],
                        "source": entry["source"], "params": dict(entry["defaults"]), "defaults": entry["defaults"]})
        return {"plugins": out, "problems": problems}

    @app.get("/api/plugins")
    def get_plugins():
        return plugin_list()

    @app.post("/api/plugins/{name}")
    def update_plugin(name: str, body: PluginUpdate):
        _, base = load_cfg()
        found, _ = discover([base / "plugins"])
        defaults = found[name]["cls"].defaults if name in found else {}

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
        for e in entries:
            e["installed"] = e["name"] in found
        return {"source": "GitHub" if source.startswith("http") else "local registry.yaml", "plugins": entries}

    @app.post("/api/install")
    async def install(body: InstallRequest):
        cfg, _ = load_cfg()
        entries, _ = registry.load(cfg.registry_url)
        result = await asyncio.to_thread(registry.install, body.name, entries)
        import importlib

        importlib.invalidate_caches()
        if result.returncode != 0:
            raise HTTPException(500, (result.stdout + result.stderr)[-3000:])
        return {"ok": True}

    # -- runs (the monitor) ------------------------------------------------------------------------------------
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
        for r in runs:
            if r.get("status") == "running" and not _is_live(runs_root() / r["id"], r):
                r["status"] = "interrupted"  # its loop is gone
        return {"runs": runs}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        folder = run_folder(run_id)
        info = storage.read_run(folder)
        if info is None:
            raise HTTPException(404, "no such run")
        info["live"] = _is_live(folder, info)
        if info.get("status") == "running" and not info["live"]:
            info["status"] = "interrupted"
        info["report"] = (folder / "report.html").is_file()
        try:
            info["recap"] = json.loads((folder / "recap.json").read_text())
        except (OSError, ValueError):
            info["recap"] = None
        return info

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str):
        folder = run_folder(run_id)
        if _is_live(folder, storage.read_run(folder)):
            raise HTTPException(409, "this run is still being written; stop the loop first")
        shutil.rmtree(folder)
        return {"ok": True}

    @app.post("/api/runs/{run_id}/reveal")
    def reveal(run_id: str, episode: int | None = None):
        """Opens the run's folder in Finder / the file manager (only works where the panel runs)."""
        import platform
        import subprocess

        folder = run_folder(run_id)
        if episode is not None and (folder / f"ep{episode}").is_dir():
            folder = folder / f"ep{episode}"
        opener = {"Darwin": ["open"], "Windows": ["explorer"]}.get(platform.system(), ["xdg-open"])
        try:
            subprocess.Popen([*opener, str(folder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as error:
            raise HTTPException(500, f"cannot open a file manager here: {error}") from error
        return {"ok": True}

    @app.get("/api/runs/{run_id}/trace/{episode}")
    def get_trace(run_id: str, episode: int):
        path = run_folder(run_id) / f"ep{episode}" / "trace.json"
        if not path.is_file():
            raise HTTPException(404, "no trace")
        return json.loads(path.read_text())

    @app.websocket("/ws/live/{run_id}")
    async def ws_live(socket: WebSocket, run_id: str):
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
        try:
            while True:
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
    def media(run_id: str, path: str):
        root = run_folder(run_id)
        target = (root / path).resolve()
        if root not in target.parents or not target.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(target)

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
        print(f"dreamkit: listening on {host} — anyone on this network can see your runs", file=sys.stderr)
    print(f"dreamkit panel: http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}")
    uvicorn.run(create_app(config_path), host=host, port=port, log_level="warning")
