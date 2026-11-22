"""monitor-stats: a vision model judges every attempt from its camera frames, and keeps the statistics across runs.

success-check decides from the simulator's object positions; this plugin decides from what the camera saw, so it also
works on a real robot, says why in words, and tells how often it agrees with success-check. Per attempt it keeps a
thinned buffer of one camera's frames (cheap, on_step), and on_episode_end sends evenly picked frames + the task to
Gemini (or Claude) on the ctx.defer worker. The frames it saw and its verdicts go to run_dir/monitor-stats/
(ep003_f0.jpg…, attempts.json) for the plugin's page (ui/index.html). At the end of a run it aggregates every judged attempt in runs/: success
rate with a Wilson interval, by task, top failure reasons, agreement with success-check, and which task needs more
demonstrations.
"""

from __future__ import annotations

import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from dreamkit import reasons
from dreamkit import stats
from dreamkit.plugin import Plugin, PluginContext
from dreamkit.standard import Manifest

from . import judge as judge_mod

KEY = "monitor-stats"


class Params(BaseModel):
    camera: str = Field("top", description="Camera the judge looks at (the first one if it is missing)",
                        json_schema_extra={"choices": ["top", "left", "right"]})
    frames: int = Field(10, ge=2, le=16, description="Frames per try sent to the judge (each recovery starts a new try)",
                        json_schema_extra={"unit": ""})
    image_width: int = Field(512, ge=64, le=1280, description="Frames are scaled down to at most this many pixels "
                             "wide", json_schema_extra={"unit": ""})
    buffer: int = Field(64, ge=16, le=512, description="Frames kept per attempt; above this every other one is "
                        "dropped", json_schema_extra={"unit": ""})
    provider: str = Field("gemini", description="Vision model that judges (GEMINI_API_KEY or ANTHROPIC_API_KEY)",
                          json_schema_extra={"choices": ["gemini", "claude"]})
    timeout_s: float = Field(10.0, gt=0, le=300, description="Longest wait for one model call",
                             json_schema_extra={"unit": "s"})
    max_wait_s: float = Field(20.0, gt=5, le=115, description="Longest wait for a verdict, fallbacks included; "
                              "dreamkit waits at most 120 s before saving the attempt",
                              json_schema_extra={"unit": "s"})
    confidence: float = Field(0.95, ge=0.5, le=0.999, description="Confidence level of the success-rate interval",
                              json_schema_extra={"unit": ""})


# --- statistics (pure, tested) ------------------------------------------------------------------------------------

wilson = stats.wilson  # the one statistics module (History, Data and reliability use it too)


def _rate(verdicts: list[dict[str, Any]], z: float) -> dict[str, Any]:
    r = stats.rate(sum(1 for v in verdicts if v["success"]), len(verdicts), z)
    n = len(verdicts)
    return {"attempts": n, "successes": r["successes"], "rate": r["success_rate"], "interval": r["ci95"],
            "mean_score": round(sum(v["score"] for v in verdicts) / n, 1) if n else None}


def aggregate(episodes: list[dict[str, Any]], z: float = 1.96) -> dict[str, Any]:
    """Every attempt the judge saw (run.json episodes) -> overall, by task, reasons, agreement, recommendation."""
    judged = [e for e in episodes if isinstance(e.get(KEY), dict) and "success" in e[KEY]]
    verdicts = [e[KEY] for e in judged]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in judged:
        by_task[str(e.get("task") or "")].append(e[KEY])
    tasks = sorted(({"task": t, **_rate(v, z)} for t, v in by_task.items()), key=lambda t: (t["rate"], -t["attempts"]))

    failed = [v for v in verdicts if not v["success"]]
    codes = Counter(v.get("reason_code") or "unknown" for v in failed)
    examples: dict[str, list[str]] = defaultdict(list)
    for v in failed:
        code = v.get("reason_code") or "unknown"
        if v.get("reason") and v["reason"] not in examples[code] and len(examples[code]) < 3:
            examples[code].append(v["reason"])
    top = [{"code": c, "label": reasons.label(c) if c != "unknown" else "other", "count": n,
            "share": round(n / len(failed), 4), "examples": examples[c]} for c, n in codes.most_common()]

    both = [e for e in judged if e.get("success") is not None]  # success-check (or failure-monitor) also decided
    agree = sum(1 for e in both if bool(e["success"]) == e[KEY]["success"])

    recommendation = None
    if tasks and tasks[0]["rate"] < 1:
        t = tasks[0]
        worst = Counter(v.get("reason_code") or "unknown" for v in by_task[t["task"]] if not v["success"])
        code = worst.most_common(1)[0][0]
        recommendation = {"task": t["task"], "rate": t["rate"], "reason": code,
                          "text": f"record more demonstrations of '{t['task'].rstrip('.')}': {t['successes']} of "
                                  f"{t['attempts']} judged successful, mostly "
                                  f"{reasons.label(code) if code != 'unknown' else 'other failures'}"}
    return {"overall": _rate(verdicts, z), "by_task": tasks, "reasons": top,
            "agreement": {"attempts": len(both), "agree": agree, "rate": round(agree / len(both), 4) if both else None},
            "recommendation": recommendation}


def _jpeg(rgb: np.ndarray, width: int) -> bytes:
    img = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8))
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _past_episodes(runs_dir: Path, this_run: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run in sorted(runs_dir.glob("*/run.json")):
        if run.parent.resolve() == this_run.resolve():
            continue
        try:
            out.extend(e for e in json.loads(run.read_text()).get("episodes", []) if isinstance(e, dict))
        except (OSError, ValueError):
            continue  # a run being written or a broken file: skip it, the stats still come out
    return out


# --- the plugin ------------------------------------------------------------------------------------------------------

class MonitorStats(Plugin):
    """Judges each attempt from camera frames with a vision model and keeps stats across runs."""

    name = KEY
    Params = Params
    manifest = Manifest(group="diagnosis", icon="ScanEye", capability="observe", needs_frames=True,
                        hooks=["on_run_start", "on_episode_start", "on_step", "on_episode_end", "on_run_end"],
                        step_rate_hz=2, ui="ui/index.html")

    def on_run_start(self, ctx: PluginContext) -> None:
        self.judged: list[dict[str, Any]] = []
        self.attempts: list[dict[str, Any]] = []  # what the page shows: attempts.json
        self.z = NormalDist().inv_cdf(0.5 + self.params.confidence / 2)  # 0.95 -> 1.96

    def on_episode_start(self, ctx: PluginContext) -> None:
        self.buf: list[tuple[float, np.ndarray]] = []
        self.stride, self.seen = 1, 0
        self.segment = 0  # recovery bumps ctx.store["segment"] when it sends the robot to try again
        self.tries: list[dict[str, Any]] = []  # verdicts of the tries before the last one

    def on_step(self, ctx: PluginContext) -> None:
        segment = ctx.store.get("segment", 0)
        if segment != self.segment:  # the try before this one ended in a recovery: judge it on its own
            self._send(ctx, final=False)
            self.segment = segment
            self.buf, self.stride, self.seen = [], 1, 0
        if not ctx.frames_new or not ctx.frames:
            return
        self.seen += 1
        if (self.seen - 1) % self.stride:
            return
        frames = ctx.frames
        rgb = frames.get(self.params.camera)
        if rgb is None:
            rgb = next(iter(frames.values()))
        step = max(1, rgb.shape[1] // self.params.image_width)
        self.buf.append((float(ctx.t), np.array(rgb[::step, ::step])))  # a small copy: the core reuses its arrays
        if len(self.buf) > self.params.buffer:
            self.buf, self.stride = self.buf[::2], self.stride * 2

    def on_episode_end(self, ctx: PluginContext) -> None:
        self._send(ctx, final=True)

    def _send(self, ctx: PluginContext, final: bool) -> None:
        """Evenly picked frames of the current try to the judge, on the defer worker."""
        if not self.buf:
            return
        n = min(self.params.frames, len(self.buf))
        picked = [self.buf[round(i * (len(self.buf) - 1) / max(1, n - 1))] for i in range(n)]
        self.buf = []
        try_no = self.segment + 1
        ctx.emit("info", "judging attempt" if final else f"judging try {try_no}", frames=n, try_no=try_no)
        ctx.defer(self._judge, ctx, picked, str(ctx.task or ""), ctx.episode, try_no, final)

    def _judge(self, ctx: PluginContext, picked: list[tuple[float, np.ndarray]], task: str, episode: int,
               try_no: int = 1, final: bool = True) -> None:
        p = self.params
        times = [round(t, 2) for t, _ in picked]
        images = [_jpeg(rgb, p.image_width) for _, rgb in picked]
        files = self._save_frames(ctx, images, episode, try_no)
        # success-check speaks for the whole attempt, i.e. its last try; an earlier try ended in a recovery
        row: dict[str, Any] = {"episode": episode, "try": try_no, "final": final, "task": task,
                               "check": ctx.result.get("success") if final else None}
        tag = "" if final and try_no == 1 else f"try {try_no}, "
        note = "" if final else ("After the last frame the robot was stopped and sent back to try again; "
                                 "judge what the frames show up to that point.")
        try:
            v = judge_mod.judge(images, task, p.provider, p.timeout_s, p.max_wait_s, note=note)
        except Exception as error:  # the robot goes on; the attempt is just not judged
            msg = " ".join(str(error).split())[:160].rstrip(".")
            self._save_attempt(ctx, {**row, "error": msg, "frames": [{"t_s": t, "file": f} for t, f in
                                                                     zip(times, files, strict=True)]})
            msg = msg if msg[:2].isupper() else msg[:1].lower() + msg[1:]
            if final:
                ctx.result[KEY] = {"error": msg, **({"tries": self.tries} if self.tries else {})}
                ctx.ui.badge("not judged", "warn")
            else:
                self.tries.append({"try": try_no, "error": msg})
            ctx.emit("warn", f"not judged{f' (try {try_no})' if tag else ''}: {msg}")
            return
        for f, t in zip(v["frames"], times, strict=True):
            f["t_s"] = t
            if f["status"] == "error":
                ctx.ui.mark(t, f["comment"] or "camera: error", "error")
        if v["reason_code"]:
            row["reason_label"] = reasons.label(v["reason_code"])
        self._save_attempt(ctx, {**row, **v, "frames": [{**f, "file": name} for f, name in
                                                         zip(v["frames"], files, strict=True)]})
        bad = sum(f["status"] == "error" for f in v["frames"])
        data: dict[str, Any] = {"score": v["score"], "frames_off": bad, "model_s": v["model_s"], "try_no": try_no}
        if v["success"]:
            ctx.emit("success", f"camera verdict: success ({tag}score {v['score']}, {bad} of {len(times)} frames off)",
                     **data)
        else:
            first = next((f["t_s"] for f in v["frames"] if f["status"] == "error"), None)
            if v["reason_code"]:
                data["reason"] = v["reason_code"]
            if first is not None:
                data["at_s"] = first  # the panel offers "watch from" on it
            # a warning, not a verdict: the core decides success (world, then vision, then a person), §7a-b
            ctx.emit("warn", f"camera verdict: {v['reason']} ({tag}score {v['score']})", **data)
        if not final:
            self.tries.append({"try": try_no, **v})
            return
        ctx.result[KEY] = {**v, "judge": "vision", **({"tries": self.tries} if self.tries else {})}
        self.judged.append(v)
        ctx.ui.badge(f"camera: {'success' if v['success'] else 'failed'} · {v['score']}",
                     "success" if v["success"] else "error")
        ctx.ui.metric("judged success", 100 * _rate(self.judged, self.z)["rate"], unit="%")
        ctx.ui.metric("judge score", float(v["score"]), unit="")

    def _save_frames(self, ctx: PluginContext, images: list[bytes], episode: int, try_no: int) -> list[str | None]:
        """The judged frames as JPEGs for the page: ep003_t1_f0.jpg… (None without a run folder)."""
        if ctx.run_dir is None:
            return [None] * len(images)
        out: list[str | None] = []
        for i, img in enumerate(images):
            name = f"ep{episode:03d}_t{try_no}_f{i}.jpg"
            ctx.files.path(name).write_bytes(img)
            out.append(name)
        return out

    def _save_attempt(self, ctx: PluginContext, row: dict[str, Any]) -> None:
        """attempts.json (one row per judged try), rewritten each time (the defer worker runs one job at a time)."""
        self.attempts.append(row)
        if ctx.run_dir is not None:
            ctx.files.path("attempts.json").write_text(json.dumps(self.attempts, indent=1, ensure_ascii=False))

    def on_run_end(self, ctx: PluginContext) -> None:
        ctx.defer(self._summary, ctx, ctx.episodes)

    def _summary(self, ctx: PluginContext, episodes: list[dict[str, Any]]) -> None:
        run_dir = Path(ctx.run_dir) if ctx.run_dir is not None else None
        past = _past_episodes(run_dir.parent, run_dir) if run_dir is not None else []
        s = aggregate(past + episodes, self.z)
        o = s["overall"]
        if not o["attempts"]:
            return
        lo, hi = o["interval"]
        conf = round(100 * self.params.confidence)
        ctx.emit("info", f"judge over {o['attempts']} attempts: {100 * o['rate']:.0f}% success "
                 f"({conf}% interval {100 * lo:.0f}–{100 * hi:.0f}%), mean score {o['mean_score']:.0f}",
                 attempts=o["attempts"], successes=o["successes"], rate_pct=round(100 * o["rate"], 1))
        a = s["agreement"]
        if a["attempts"]:
            ctx.emit("info", f"judge agrees with success-check on {a['agree']} of {a['attempts']} attempts",
                     agree=a["agree"], attempts=a["attempts"])
        if s["recommendation"]:
            ctx.emit("warn", s["recommendation"]["text"])
        out = {"run": ctx.run_id, "confidence": self.params.confidence, "provider": self.params.provider, **s}
        if run_dir is not None:
            ctx.files.path("summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
