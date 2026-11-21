"""The panel's backend. It never runs the robot and never sends prompts: it manages plugins (dreamkit.yaml) and
monitors runs, which the user's own loop writes to runs/<id>/ through dreamkit.wrap()."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import shutil
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
from . import registry, storage
from .manager import describe, discover

UI_DIST = Path(__file__).resolve().parent / "ui_dist"
THEME = Path(__file__).resolve().parent / "theme" / "dreamkit-theme.css"  # shared with plugin pages (§7a)
THEME_LINK = '<link rel="stylesheet" href="/theme/dreamkit-theme.css">'
LIVE_FRESH_S = 10.0  # a run whose heartbeat is older than this is not live
POLL_S = 0.12


class PluginUpdate(BaseModel):
    enabled: bool | None = None
    params: dict[str, Any] | None = None


class InstallRequest(BaseModel):
    name: str


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
        ep["success"], ep["labeled_by"] = body.success, "person"
        if body.success:
            ep["reason"] = None
        elif not ep.get("reason"):
            ep["reason"] = "marked_failed"
            ep["message"] = "marked failed by a person"
        # the harvester's dataset of this attempt moves out of unlabeled/
        src = folder / "harvester" / "unlabeled" / f"ep{episode:03d}"
        if src.is_dir():
            split = "success" if body.success else "failed"
            dst = folder / "harvester" / split / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), dst)
            meta = dst / "dreamkit.json"
            if meta.is_file():
                data = json.loads(meta.read_text())
                data.update(success=body.success, labeled_by="person")
                meta.write_text(json.dumps(data, indent=2))
            ep["harvester"] = {**(ep.get("harvester") or {}), "split": split, "path": f"harvester/{split}/{src.name}"}
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
        if episode is not None and (folder / f"ep{episode}").is_dir():
            folder = folder / f"ep{episode}"
        opener = {"Darwin": ["open"], "Windows": ["explorer"]}.get(platform.system(), ["xdg-open"])
        try:
            subprocess.Popen([*opener, str(folder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as error:
            raise HTTPException(500, f"cannot open a file manager here: {error}") from error
        return {"ok": True}

    @app.get("/api/runs/{run_id}/harvest")
    def get_harvest(run_id: str):
        from . import harvest

        folder = run_folder(run_id)
        ready, failed = harvest.episodes(folder), harvest.episodes(folder, "failed")
        try:
            check = json.loads((folder / "harvester" / "validation.json").read_text())
        except (OSError, ValueError):
            check = None
        return {"success": len(ready), "failed": len(failed), "validation": check}

    @app.post("/api/runs/{run_id}/harvest/validate")
    async def validate_harvest(run_id: str):
        """Dream Labs' own dataset checks on this run's successful episodes. Local only: nothing is uploaded."""
        from . import harvest

        folder = run_folder(run_id)
        if not harvest.episodes(folder):
            raise HTTPException(404, "no successful episodes in this run (is the harvester plugin on?)")
        report = await asyncio.to_thread(harvest.validate, harvest.episodes(folder)[0].parent)
        report["checked"] = time.time()
        (folder / "harvester").mkdir(exist_ok=True)
        storage.write_json(folder / "harvester" / "validation.json", report)
        return report

    @app.get("/api/runs/{run_id}/trace/{episode}")
    def get_trace(run_id: str, episode: int):
        folder = run_folder(run_id)
        path = folder / "failure-monitor" / f"ep{episode:03d}_trace.json"
        if not path.is_file():
            path = folder / f"ep{episode}" / "trace.json"  # older runs
        if not path.is_file():
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
        if root not in target.parents or not target.is_file():
            raise HTTPException(404, "not found")
        response = FileResponse(target)
        # §7a: a plugin page runs sandboxed (origin "null"); it may read its own plugin's files, nothing else.
        if request.headers.get("origin") == "null" and len(Path(path).parts) > 1:
            if Path(path).parts[0] in plugin_pages():
                response.headers["Access-Control-Allow-Origin"] = "null"
        return response

    def plugin_pages() -> dict[str, Path]:
        """§7a: plugin name -> its page (manifest ui), for the plugins that ship one."""
        _, base = load_cfg()
        found, _ = discover([base / "plugins"])
        out = {}
        for name, entry in found.items():
            ui = getattr(entry["cls"].manifest, "ui", None)
            if ui:
                out[name] = (Path(inspect.getfile(entry["cls"])).resolve().parent / ui).resolve()
        return out

    @app.get("/theme/dreamkit-theme.css")
    def theme():
        return FileResponse(THEME, media_type="text/css")

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
        head = f'<base href="/plugin-ui/{name}/">{THEME_LINK}'
        lower = html.lower()
        at = lower.find("<head>")
        html = html[: at + 6] + head + html[at + 6 :] if at >= 0 else head + html
        return Response(html, media_type="text/html")

    # -- app plugins (§13): their API and page under /apps/<name>/, found again after an install -----------------
    from . import apps as apps_mod

    def app_context() -> "apps_mod.AppContext":
        cfg, base = load_cfg()
        return apps_mod.AppContext(config_path=config_path, base=base, runs_dir=runs_root(),
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
