"""DreamGym's API inside dreamkit's panel (an app plugin, PLUGIN_STANDARD.md §13): /apps/dreamgym/api/...

It lists every Arena and Compare run, computes live statistics from episodes.jsonl while a run is going, shows
failure GIFs, and starts new runs as `dreamgym` subprocesses (so a crash in the sim never takes the panel down).
The runs go through dreamkit.wrap: DREAMKIT_CONFIG points them at the panel's dreamkit.yaml, so the loop plugins run
in them and they appear in History. The panel binds 127.0.0.1: nobody else on the Wi-Fi can start runs on your credits.
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
from http import HTTPStatus
from pathlib import Path

from ..arena import load_results, slug
from ..stats import paired, summarize
from ..tasks import TASKS, prompt_variants

PAGE = Path(__file__).with_name("page")
PROCS: dict[str, subprocess.Popen] = {}
LOCK = threading.Lock()
TELEOP = {"session": None}
SAFE = __import__("re").compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
## The dreamkit.yaml the panel serves; handed to every run DreamGym starts (DREAMKIT_CONFIG).
DREAMKIT = {"config": None}


def _env() -> dict:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if DREAMKIT["config"]:
        env["DREAMKIT_CONFIG"] = str(DREAMKIT["config"])
    return env


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
    # the dreamkit runs these episodes belong to (History), one per variant
    out["dreamkit_runs"] = [{"label": label, "id": next((r["run_id"] for r in reversed(rows) if r.get("run_id")), None)}
                            for label, rows, _ in rows_by_variant]
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
    from ..compare import Variant, with_recovery_ab

    recovery_ab = bool(body.get("recovery_ab"))
    if mode == "arena":
        cmd += ["--policy", policies[0]] + (["--prompt", prompts[0]] if prompts else [])
        variants = None
        if recovery_ab:  # the same seeds with dreamkit's recovery off, then on (a compare underneath)
            cmd.append("--recovery-ab")
            variants = [v.label for v in with_recovery_ab([Variant(policies[0], prompts[0] if prompts else None)])]
    else:
        for p in policies:
            cmd += ["--policy", p]
        for q in prompts:
            cmd += ["--prompt", q]
        vs = [Variant(p, q) for p in policies for q in (prompts or [None])]
        if recovery_ab:
            cmd.append("--recovery-ab")
            vs = with_recovery_ab(vs)
        variants = [v.label for v in vs]
        if len(variants) < 2:
            raise ValueError("compare needs two policies, two prompts, or recovery on vs off")
    folder.mkdir(parents=True)
    meta = {"mode": "compare" if variants else mode, "recovery_ab": recovery_ab, "task": task, "policy": policies[0] if mode == "arena" else None,
            "prompt": prompts[0] if prompts else TASKS[task].prompt, "variants": variants, "n": n,
            "seed": body.get("seed"), "jitter": body.get("jitter"), "lighting": body.get("lighting"),
            "max_seconds": body.get("max_seconds"), "started": time.time(), "status": "running", "cmd": cmd}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    log = open(folder / "log.txt", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=root.parent, env=_env(), start_new_session=True)
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
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=data.parent, env=_env(), start_new_session=True)
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


def make_router(root: Path, config_path: Path | None = None):
    """DreamGym's API as a FastAPI router; the panel mounts it at /apps/dreamgym/api."""
    from fastapi import APIRouter, Body, HTTPException
    from fastapi.responses import FileResponse, Response

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    DREAMKIT["config"] = config_path
    r = APIRouter()

    def guard(fn, *args):
        try:
            return fn(*args)
        except (ValueError, KeyError) as exc:
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(exc)) from exc

    @r.get("/runs")
    def runs():
        return list_runs(root)

    @r.get("/run/{run_id}")
    def run(run_id: str):
        folder = (root / run_id).resolve()
        if folder.parent != root or not folder.is_dir():
            raise HTTPException(404, "not found")
        return describe(root, folder, full=True)

    @r.get("/datasets")
    def datasets():
        return list_datasets(data_root(root))

    @r.get("/prompts")
    def prompts(task: str = ""):
        t = TASKS.get(task)
        return prompt_variants(t) if t else []

    @r.get("/tasks")
    def tasks():
        return [{"name": t.name, "prompt": t.prompt, "jitter": t.jitter_m, "lighting": t.lighting} for t in TASKS.values()]

    @r.get("/files/{path:path}")
    def files(path: str):
        file = (root / path).resolve()
        if root not in file.parents or not file.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(file, media_type=mimetypes.guess_type(file.name)[0] or "application/octet-stream")

    @r.get("/teleop/state")
    def teleop_state():
        session = TELEOP["session"]
        return session.state if session else {"status": "off"}

    @r.get("/teleop/frame")
    def teleop_frame():
        session = TELEOP["session"]
        if session is None or not session.jpeg:
            return Response(status_code=204)
        return Response(session.jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @r.post("/launch")
    def post_launch(body: dict = Body(default_factory=dict)):
        return guard(launch, root, body)

    @r.post("/record")
    def post_record(body: dict = Body(default_factory=dict)):
        return guard(launch_record, root, body)

    @r.post("/dataset/{name}/{action}")
    def post_dataset(name: str, action: str, body: dict = Body(default_factory=dict)):
        return guard(dataset_action, root, name, action, body)

    @r.post("/teleop/start")
    def post_teleop_start(body: dict = Body(default_factory=dict)):
        return guard(teleop_start, root, body)

    @r.post("/teleop/{what}")
    def post_teleop(what: str, body: dict = Body(default_factory=dict)):
        session = TELEOP["session"]
        if session is None:
            raise HTTPException(HTTPStatus.BAD_REQUEST, "no teleop session: start one first")
        if what == "keys":
            session.set_keys(body.get("keys", []))
        elif what == "cmd":
            session.command(body.get("cmd", ""))
        elif what == "stop":
            session.stop()
        return {"ok": True}

    @r.post("/stop/{run_id}")
    def post_stop(run_id: str):
        stop(root, run_id)
        return {"ok": True}

    return r
