"""DreamGym dashboard: a local web app over the runs/ folder. Standard library only.

  dreamgym serve                 # http://127.0.0.1:8765, opens the browser

It lists every Arena and Compare run, computes live statistics from episodes.jsonl while a run is going, shows
failure GIFs, and starts new runs as `dreamgym` subprocesses (so a crash in the sim never takes the page down).
Bound to 127.0.0.1 only: nobody else on the hackathon Wi-Fi can start runs on your credits.
"""

from __future__ import annotations

import json
import mimetypes
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..arena import load_results, slug
from ..stats import paired, summarize
from ..tasks import TASKS, prompt_variants

INDEX = Path(__file__).with_name("index.html")
PROCS: dict[str, subprocess.Popen] = {}
LOCK = threading.Lock()
TELEOP = {"session": None}
SAFE = __import__("re").compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _running(run_id: str) -> bool:
    with LOCK:
        proc = PROCS.get(run_id)
        return proc is not None and proc.poll() is None


def _sub_runs(folder: Path) -> list[tuple[str, Path]]:
    """(label, folder) of each variant of a compare run, in launch order."""
    meta = _read_json(folder / "meta.json") or {}
    report = _read_json(folder / "compare.json")
    if report:
        return list(zip(report["variants"], (folder / r for r in report["runs"])))
    if meta.get("variants"):
        return [(label, folder / slug(label)) for label in meta["variants"]]
    return [(p.name, p) for p in sorted(folder.iterdir()) if (p / "episodes.jsonl").is_file()]


def _summary(rows: list[dict], meta: dict, label: str | None = None) -> dict:
    s = summarize(rows)
    s.update(policy=label or meta.get("policy", "?"), task=meta.get("task", "?"), prompt=meta.get("prompt") or "")
    return s


def describe(root: Path, folder: Path, full: bool = False) -> dict | None:
    meta = _read_json(folder / "meta.json") or {}
    is_compare = meta.get("mode") in ("compare", "tune") or (folder / "compare.json").is_file()
    is_arena = (folder / "episodes.jsonl").is_file() or (folder / "summary.json").is_file() or meta.get("mode") == "arena"
    if not (is_compare or is_arena):
        return None
    if not meta:  # a run started from the CLI: take what summary.json knows
        s = _read_json(folder / "summary.json") or {}
        meta = {"task": s.get("task"), "policy": s.get("policy"), "prompt": s.get("prompt"),
                "n": s.get("episodes"), "mode": "compare" if is_compare else "arena"}
        if is_compare:
            c = _read_json(folder / "compare.json") or {}
            meta.update(task=c.get("task"), variants=c.get("variants"))
    run_id = folder.name
    out = {
        "id": run_id,
        "mode": meta.get("mode") or ("compare" if is_compare else "arena"),
        "task": meta.get("task"),
        "created": meta.get("started") or folder.stat().st_mtime,
        "running": _running(run_id),
        "status": meta.get("status", "done"),
        "n": meta.get("n"),
        "settings": {k: meta.get(k) for k in ("jitter", "lighting", "max_seconds", "seed")},
    }
    if out["running"]:
        out["status"] = "running"
    elif out["status"] == "running":  # the server restarted, or the process ended without us noticing
        out["status"] = "stopped"
    variants = _sub_runs(folder) if is_compare else [(meta.get("policy") or "?", folder)]
    rows_by_variant = [(label, load_results(path), path) for label, path in variants]
    out["summaries"] = [_summary(rows, meta, label) for label, rows, _ in rows_by_variant]
    out["done"] = sum(len(rows) for _, rows, _ in rows_by_variant)
    out["total"] = (meta.get("n") or 0) * max(1, len(variants)) or out["done"]
    out["cost_usd"] = round(sum(s["cost_usd"] for s in out["summaries"]), 4)
    if is_compare and len(rows_by_variant) > 1:
        base = rows_by_variant[0]
        out["paired"] = [{"a": base[0], "b": label, **paired(base[1], rows)} for label, rows, _ in rows_by_variant[1:]]
    if full:
        out["variants"] = [
            {"label": label, "dir": str(path.relative_to(root)), "episodes": rows} for label, rows, path in rows_by_variant
        ]
        log = folder / "log.txt"
        out["log"] = log.read_text(errors="replace")[-6000:] if log.is_file() else ""
    return out


def list_runs(root: Path) -> list[dict]:
    runs = []
    if root.is_dir():
        for folder in root.iterdir():
            if folder.is_dir():
                d = describe(root, folder)
                if d:
                    runs.append(d)
    runs.sort(key=lambda r: r["created"] if isinstance(r["created"], (int, float)) else 0, reverse=True)
    return runs


def launch(root: Path, body: dict) -> dict:
    mode = body.get("mode", "arena")
    task = body["task"]
    if task not in TASKS:
        raise ValueError(f"unknown task {task}")
    policies = [p.strip() for p in body.get("policies", []) if p.strip()] or ["dream-1"]
    prompts = [p.strip() for p in body.get("prompts", []) if p.strip()]
    if mode == "tune":  # Prompt Tuner: one policy, the task's prompt first (the baseline), then the variants
        policies = policies[:1]
        prompts = prompts or prompt_variants(TASKS[task])
        if TASKS[task].prompt not in prompts:
            prompts = [TASKS[task].prompt] + prompts
        if len(prompts) < 2:
            raise ValueError("the tuner needs at least two prompts")
    n = int(body.get("n", 10))
    if not 1 <= n <= 500:
        raise ValueError("n must be 1..500")
    run_id = f"{time.strftime('%m%d-%H%M%S')}_{mode}_{task}"
    folder = root / run_id
    cmd = [sys.executable, "-u", "-m", "dreamgym.cli", "arena" if mode == "arena" else "compare", "--task", task, "--n", str(n), "--out", str(folder)]
    for key, flag in (("seed", "--seed"), ("jitter", "--jitter"), ("lighting", "--lighting"), ("max_seconds", "--max-seconds")):
        if body.get(key) not in (None, ""):
            cmd += [flag, str(float(body[key]) if key != "seed" else int(body[key]))]
    cmd += ["--gifs", body.get("gifs", "failures")]
    if mode == "arena":
        cmd += ["--policy", policies[0]] + (["--prompt", prompts[0]] if prompts else [])
        variants = None
    else:
        for p in policies:
            cmd += ["--policy", p]
        for q in prompts:
            cmd += ["--prompt", q]
        variants = [p + (f" | {q}" if q else "") for p in policies for q in (prompts or [None])]
        if len(variants) < 2:
            raise ValueError("compare needs two policies or two prompts")
    folder.mkdir(parents=True)
    meta = {"mode": mode, "task": task, "policy": policies[0] if mode == "arena" else None,
            "prompt": prompts[0] if prompts else TASKS[task].prompt, "variants": variants, "n": n,
            "seed": body.get("seed"), "jitter": body.get("jitter"), "lighting": body.get("lighting"),
            "max_seconds": body.get("max_seconds"), "started": time.time(), "status": "running", "cmd": cmd}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    log = open(folder / "log.txt", "w")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=root.parent, env=env, start_new_session=True)
    with LOCK:
        PROCS[run_id] = proc
    threading.Thread(target=_watch, args=(folder, proc, log), daemon=True).start()
    return {"id": run_id}


def _watch(folder: Path, proc: subprocess.Popen, log) -> None:
    code = proc.wait()
    log.close()
    meta = _read_json(folder / "meta.json") or {}
    if meta.get("status") == "running":
        meta["status"] = "done" if code == 0 else ("stopped" if code < 0 else "failed")
        meta["exit_code"] = code
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))


def stop(root: Path, run_id: str) -> None:
    with LOCK:
        proc = PROCS.get(run_id)
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)  # Ctrl-C: the arena closes the policy and keeps what it wrote
        meta_path = root / run_id / "meta.json"
        meta = _read_json(meta_path) or {}
        meta["status"] = "stopped"
        meta_path.write_text(json.dumps(meta, indent=2))


# --- datasets: record / harvest / validate / fine-tune --------------------------------------------------------------

def data_root(root: Path) -> Path:
    return root.parent / "data"


def _spawn_data_job(data: Path, name: str, kind: str, cmd: list[str]) -> dict:
    key = f"data:{name}:{kind}"
    if _running(key):
        raise ValueError(f"{kind} is already running for {name}")
    state = data / name / ".dreamgym"
    state.mkdir(parents=True, exist_ok=True)
    job_path = state / f"{kind}.job.json"
    job_path.write_text(json.dumps({"kind": kind, "status": "running", "started": time.time(), "cmd": cmd}, indent=2))
    log = open(state / f"{kind}.log", "w")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=data.parent, env=env, start_new_session=True)
    with LOCK:
        PROCS[key] = proc
    threading.Thread(target=_watch_job, args=(job_path, proc, log), daemon=True).start()
    return {"id": key}


def _watch_job(job_path: Path, proc: subprocess.Popen, log) -> None:
    code = proc.wait()
    log.close()
    job = _read_json(job_path) or {}
    if job.get("status") == "running":
        job["status"] = "done" if code == 0 else ("stopped" if code < 0 else "failed")
    job.update(exit_code=code, ended=time.time())
    job_path.write_text(json.dumps(job, indent=2))


def list_datasets(data: Path) -> list[dict]:
    out = []
    if not data.is_dir():
        return out
    for folder in data.iterdir():
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        info = _read_json(folder / "meta" / "info.json") or {}
        state = folder / ".dreamgym"
        if not info and not state.is_dir():
            continue
        tasks = []
        tasks_file = folder / "meta" / "tasks.jsonl"
        if tasks_file.is_file():
            tasks = [json.loads(l)["task"] for l in tasks_file.read_text().splitlines() if l.strip()]
        jobs = {}
        for kind in ("record", "validate", "finetune", "status"):
            job = _read_json(state / f"{kind}.job.json")
            if not job:
                continue
            running = _running(f"data:{folder.name}:{kind}")
            if job.get("status") == "running" and not running:
                job["status"] = "stopped"
            log = state / f"{kind}.log"
            job["log"] = log.read_text(errors="replace")[-2500:] if log.is_file() else ""
            job["running"] = running
            jobs[kind] = job
        fps = info.get("fps") or 0
        frames = info.get("total_frames") or 0
        out.append({
            "name": folder.name, "path": str(folder), "episodes": info.get("total_episodes", 0), "frames": frames,
            "fps": fps, "minutes": round(frames / fps / 60, 1) if fps else 0, "tasks": tasks, "jobs": jobs,
            "validate": _read_json(state / "validate.json"), "finetune": _read_json(state / "finetune.json"),
            "modified": max([p.stat().st_mtime for p in (folder, state) if p.exists()]),
        })
    out.sort(key=lambda d: d["modified"], reverse=True)
    return out


def launch_record(root: Path, body: dict) -> dict:
    data = data_root(root)
    task = body["task"]
    if task not in TASKS:
        raise ValueError(f"unknown task {task}")
    kind = body.get("kind", "scripted")
    policy = (body.get("policy") or "").strip() or ("scripted:0.004" if kind == "scripted" else "dream-1")
    name = (body.get("name") or "").strip() or f"{task}_{'demos' if kind == 'scripted' else 'harvest'}"
    if not SAFE.match(name):
        raise ValueError("dataset name: lowercase letters, digits, - and _")
    n = int(body.get("n", 20))
    if not 1 <= n <= 500:
        raise ValueError("n must be 1..500")
    attempts = int(body.get("max_attempts") or n * (3 if kind == "scripted" else 5))
    cmd = [sys.executable, "-u", "-m", "dreamgym.cli", "record", "--task", task, "--n", str(n), "--out", str(data / name),
           "--policy", policy, "--max-attempts", str(attempts), "--seed", str(int(body.get("seed") or 0))]
    for key, flag in (("jitter", "--jitter"), ("lighting", "--lighting"), ("max_seconds", "--max-seconds")):
        if body.get(key) not in (None, ""):
            cmd += [flag, str(float(body[key]))]
    (data / name).mkdir(parents=True, exist_ok=True)
    return _spawn_data_job(data, name, "record", cmd)


def dataset_action(root: Path, name: str, action: str, body: dict) -> dict:
    data = data_root(root)
    if not SAFE.match(name) or not (data / name).is_dir():
        raise ValueError("no such dataset")
    path = str(data / name)
    base = [sys.executable, "-u", "-m", "dreamgym.cli", "finetune", path]
    if action == "validate":
        return _spawn_data_job(data, name, "validate", base + ["--validate-only"])
    if action == "finetune":
        suffix = (body.get("suffix") or name).replace("_", "-").lower()[:40]
        cmd = base + ["--suffix", suffix]
        if body.get("steps"):
            cmd += ["--steps", str(int(body["steps"]))]
        return _spawn_data_job(data, name, "finetune", cmd)
    if action == "status":
        return _spawn_data_job(data, name, "status", base + ["--status"])
    if action == "stop":
        with LOCK:
            proc = PROCS.get(f"data:{name}:record")
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
        return {"ok": True}
    raise ValueError(f"unknown action {action}")


# --- teleop ----------------------------------------------------------------------------------------------------------

def teleop_start(root: Path, body: dict) -> dict:
    from .teleop import TeleopSession

    task = body.get("task")
    if task not in TASKS:
        raise ValueError(f"unknown task {task}")
    name = (body.get("dataset") or "").strip() or f"{task}_teleop"
    if not SAFE.match(name):
        raise ValueError("dataset name: lowercase letters, digits, - and _")
    if TELEOP["session"] is not None:
        TELEOP["session"].stop()
        TELEOP["session"].thread.join(timeout=5)
    TELEOP["session"] = TeleopSession(task, data_root(root) / name, seed=int(body.get("seed") or 0))
    return {"ok": True}


def make_handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet console
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code: int = 200) -> None:
            self._send(code, json.dumps(data, default=str).encode(), "application/json")

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
                if path == "/api/runs":
                    return self._json(list_runs(root))
                if path.startswith("/api/run/"):
                    folder = (root / path.removeprefix("/api/run/")).resolve()
                    if folder.parent != root.resolve() or not folder.is_dir():
                        return self._json({"error": "not found"}, 404)
                    return self._json(describe(root, folder, full=True))
                if path == "/api/datasets":
                    return self._json(list_datasets(data_root(root)))
                if path == "/api/prompts":
                    task = TASKS.get(urlparse(self.path).query.removeprefix("task="))
                    return self._json(prompt_variants(task) if task else [])
                if path == "/api/teleop/state":
                    session = TELEOP["session"]
                    return self._json(session.state if session else {"status": "off"})
                if path == "/api/teleop/frame":
                    session = TELEOP["session"]
                    if session is None or not session.jpeg:
                        return self._send(204, b"", "image/jpeg")
                    return self._send(200, session.jpeg, "image/jpeg")
                if path == "/api/tasks":
                    return self._json([{"name": t.name, "prompt": t.prompt, "jitter": t.jitter_m,
                                        "lighting": t.lighting} for t in TASKS.values()])
                if path.startswith("/files/"):
                    file = (root / path.removeprefix("/files/")).resolve()
                    if root.resolve() not in file.parents or not file.is_file():
                        return self._json({"error": "not found"}, 404)
                    ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                    return self._send(200, file.read_bytes(), ctype)
                self._json({"error": "not found"}, 404)
            except Exception as exc:  # keep the dashboard up whatever one folder holds
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if path == "/api/launch":
                    return self._json(launch(root, body))
                if path == "/api/record":
                    return self._json(launch_record(root, body))
                if path.startswith("/api/dataset/"):
                    name, _, action = path.removeprefix("/api/dataset/").partition("/")
                    return self._json(dataset_action(root, name, action, body))
                if path == "/api/teleop/start":
                    return self._json(teleop_start(root, body))
                if path.startswith("/api/teleop/"):
                    session = TELEOP["session"]
                    if session is None:
                        raise ValueError("no teleop session: start one first")
                    what = path.removeprefix("/api/teleop/")
                    if what == "keys":
                        session.set_keys(body.get("keys", []))
                    elif what == "cmd":
                        session.command(body.get("cmd", ""))
                    elif what == "stop":
                        session.stop()
                    return self._json({"ok": True})
                if path.startswith("/api/stop/"):
                    stop(root, path.removeprefix("/api/stop/"))
                    return self._json({"ok": True})
                self._json({"error": "not found"}, 404)
            except (ValueError, KeyError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def serve(runs: Path = Path("runs"), port: int = 8765, open_browser: bool = True) -> None:
    root = Path(runs).resolve()
    root.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(root))
    url = f"http://127.0.0.1:{port}"
    print(f"DreamGym dashboard: {url}   (runs: {root})   Ctrl-C to stop")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if TELEOP["session"] is not None:
            TELEOP["session"].stop()
        server.server_close()
