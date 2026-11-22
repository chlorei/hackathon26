"""History: what the panel and the run folder keep about every run, made by the core (not plugins).

- per attempt: a one-sentence story and a captioned highlights clip (runs/<run>/ep003/recap.mp4), cut from the
  recorder's video when there is one;
- per run: a plain-language summary (history/summary.json) and a self-contained report (history/report.html).
"""

from __future__ import annotations

import html
import json
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BEFORE_S, AFTER_S, MAX_S, CAMERA = 1.5, 1.2, 14.0, "top"

## Milestones worth a highlight, and how they read in a sentence.
KEY = {"grasp", "lifted", "separated", "fell", "collision", "stalled"}
REASON_TEXT = {
    "dropped": "dropped the {obj} (it slipped out of the closed gripper)",
    "released_off_target": "let go of the {obj} outside the {target}",
    "grasped_air": "closed the gripper next to the {obj}, not on it",
    "not_reached": "never got to the {obj}",
    "collision": "hit something",
    "stalled": "stopped moving",
    "fell_off_table": "knocked the {obj} off the table",
    "slow_model": "waited too long on the model",
    "timeout": "ran out of time",
    "camera_failure": "lost a camera and was stopped",
    "stopped": "was stopped",
    "model_error": "got an error from the model",
}


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _caption(rgb: np.ndarray, text: str, sub: str, tone: str) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    draw.rectangle([0, h - 54, w, h], fill=(12, 16, 22, 200))
    color = {"bad": (239, 98, 98), "good": (74, 222, 128), "info": (240, 180, 60)}[tone]
    draw.rectangle([0, h - 54, 5, h], fill=color + (255,))
    draw.text((16, h - 48), text, fill=(240, 242, 245), font=_font(20))
    draw.text((16, h - 22), sub, fill=(160, 168, 180), font=_font(14))
    return np.asarray(img)





def story(core) -> tuple[list[dict[str, Any]], str]:
    """The attempt's key moments and its one-sentence headline."""
    r = core.result
    obj, target = core.target["object"] or "object", core.target["target"] or "target"
    moments = [{"t": m["t"], "text": m["message"], "tone": "bad" if m["kind"] in ("separated", "fell", "collision",
                                                                                  "stalled") else "info"}
               for m in (r.get("timeline") or []) if m.get("kind") in KEY]
    for v in core.store.get("verdicts", []):
        if v.get("by") != "failure-monitor":
            moments.append({"t": v["at_s"], "text": v["message"], "tone": "bad"})
    if r.get("success"):
        moments.append({"t": core.t, "text": "task done", "tone": "good"})
        rec = (r.get("recovery") or {}).get("count")
        headline = f"Attempt {core.episode}: done in {core.t:.1f} s" + (
            f" after {rec} recover{'y' if rec == 1 else 'ies'}." if rec else ".")
    elif r.get("success") is None and not r.get("reason"):
        headline = f"Attempt {core.episode}: ran {core.t:.1f} s; the outcome is not known yet (mark it in History)."
    else:
        reason = r.get("reason") or "timeout"
        what = REASON_TEXT.get(reason, reason.replace("_", " ")).format(obj=obj, target=target)
        at = r.get("at_s")
        if at is not None:
            moments.append({"t": at, "text": r.get("message") or what, "tone": "bad", "verdict": True})
        steps = [m["text"] for m in moments if not m.get("verdict")][:3]
        lead = (", ".join(steps) + ", then ") if steps else ""
        headline = f"Attempt {core.episode}: {lead}{what}" + (f" at {at:.1f} s." if at is not None else ".")
    moments.sort(key=lambda m: m["t"])
    return moments, headline


def clip(src: Path, out: Path, moments: list[dict[str, Any]], episode: int) -> None:
    """The key moments of the recorder's video, captioned (runs on the background worker)."""
    import av  # noqa: PLC0415

    if not src.is_file() or not moments:
        return
    windows: list[list[Any]] = []
    for m in moments:
        lo, hi = max(0.0, m["t"] - BEFORE_S), m["t"] + AFTER_S
        if windows and lo <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], hi)
            windows[-1][2].append(m)
        else:
            windows.append([lo, hi, [m]])
    kept, total = [], 0.0
    for w in sorted(windows, key=lambda w: (not any(m.get("verdict") for m in w[2]), w[0])):
        if total + (w[1] - w[0]) <= MAX_S or not kept:
            kept.append(w)
            total += w[1] - w[0]
    kept.sort(key=lambda w: w[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(src)) as source:
        stream_in = source.streams.video[0]
        dst = av.open(str(out), "w", options={"movflags": "+faststart"})
        stream = dst.add_stream("libx264", rate=10)
        stream.width, stream.height = stream_in.width, stream_in.height
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "28"}
        pts = 0
        for frame in source.decode(stream_in):
            t = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
            window = next((w for w in kept if w[0] <= t <= w[1]), None)
            if window is None:
                continue
            near = min(window[2], key=lambda m: abs(m["t"] - t))
            rgb = _caption(frame.to_ndarray(format="rgb24"), near["text"], f"{t:5.1f} s  attempt {episode}", near["tone"])
            new = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            new.pts, new.time_base = pts, Fraction(1, 10)
            pts += 1
            for packet in stream.encode(new):
                dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)
        dst.close()


def episode_recap(core, run_dir: Path, defer) -> dict[str, Any]:
    moments, headline = story(core)
    recap: dict[str, Any] = {"headline": headline, "moments": moments, "video": None}
    ep_dir = run_dir / f"ep{core.episode:03d}"
    src = ep_dir / f"{CAMERA}.mp4"
    recorded = (core.result.get("recorder") or {}).get("videos") or []  # the recorder names its files as it ends
    if any(Path(v).name == src.name for v in recorded) or src.exists():
        out = ep_dir / "recap.mp4"
        defer("history", clip, (src, out, moments, core.episode), {})
        recap["video"] = str(out.relative_to(run_dir))
    return recap


def latency(episodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    stats = [(e.get("failure-monitor") or {}).get("latency") or {} for e in episodes]
    real = [s for s in stats if s.get("mean_ms") is not None]
    if real:
        calls = sum(s["calls"] for s in real)
        mean = sum(s["mean_ms"] * s["calls"] for s in real) / max(calls, 1)
        return {"mean_ms": mean, "p95_ms": max(s["p95_ms"] for s in real), "calls": calls}
    if any(s.get("mock") for s in stats):
        return {"mock": True}
    return None


def summary(episodes: list[dict[str, Any]], mock: bool) -> dict[str, Any]:
    ok = [e for e in episodes if e.get("success")]
    scenes = sorted({e["scene"] for e in episodes})
    solved = {e["scene"] for e in ok}
    count: dict[str, int] = {}
    for e in episodes:
        if e.get("success") is False:
            count[e.get("reason") or "unknown"] = count.get(e.get("reason") or "unknown", 0) + 1
    unknown = sum(1 for e in episodes if e.get("success") is None)
    lines = [f"{len(ok)} of {len(episodes)} attempts worked; {len(solved)} of {len(scenes)} scenes solved."]
    if unknown:
        lines.append(f"{unknown} attempt{'s' if unknown > 1 else ''} not judged yet (mark them in History).")
    if count:
        top, n = max(count.items(), key=lambda kv: kv[1])
        lines.append(f"Most common failure: {top.replace('_', ' ')} ({n}x).")
    rescued = [e for e in ok if (e.get("recovery") or {}).get("count")]
    if rescued:
        lines.append(f"Recovery turned {len(rescued)} failing attempt{'s' if len(rescued) > 1 else ''} into a success.")
    retried = [s for s in solved if any(e["scene"] == s and e.get("success") is False for e in episodes)]
    if retried:
        lines.append(f"Retries saved {len(retried)} scene{'s' if len(retried) > 1 else ''}.")
    times = [e["duration_s"] for e in ok]
    if times:
        lo, hi = min(times), max(times)
        span = f"{lo:.1f} s" if round(lo, 1) == round(hi, 1) else f"{lo:.1f}-{hi:.1f} s"
        lines.append(f"Successful attempt{'s' if len(times) > 1 else ''} took {span}.")
    lat = latency(episodes)
    if mock:
        lines.append("Mock run: the failures were scripted on purpose to test the plugins; no model calls.")
    elif lat and lat.get("mean_ms") is not None:
        lines.append(f"Model round trip: {lat['mean_ms']:.0f} ms on average, {lat['p95_ms']:.0f} ms p95.")
    return {"text": " ".join(lines), "attempts": [(e.get("recap") or {}).get("headline") for e in episodes],
            "mock": mock}


COLORS = {"success": "#22c55e", "fail": "#ef4444", "unknown": "#a1a1aa"}


def render(ctx_info: dict, episodes: list[dict], latency: dict | None) -> str:
    attempts = len(episodes)
    ok = sum(1 for e in episodes if e.get("success"))
    scenes = sorted({e.get("scene") for e in episodes})
    scene_ok = len({e.get("scene") for e in episodes if e.get("success")})
    reasons: dict[str, int] = {}
    for e in episodes:
        if e.get("success") is False:
            reasons[e.get("reason") or "unknown"] = reasons.get(e.get("reason") or "unknown", 0) + 1
    rate = 100.0 * ok / attempts if attempts else 0.0
    biggest = max(reasons.values()) if reasons else 1
    bars = "".join(
        f'<div class="bar"><span class="lbl">{html.escape(r)}</span>'
        f'<span class="fill" style="width:{100 * n / biggest:.0f}%"></span><span class="n">{n}</span></div>'
        for r, n in sorted(reasons.items(), key=lambda kv: -kv[1])
    ) or '<p class="muted">no failures</p>'
    rows = []
    for e in episodes:
        status = "success" if e.get("success") else ("fail" if e.get("success") is False else "unknown")
        mark = {"success": "✓", "fail": "✗", "unknown": "?"}[status]
        at = e.get("at_s", e.get("at"))
        video = (e.get("videos") or [None])[0]
        top = next((v for v in e.get("videos") or [] if v.endswith("top.mp4")), video)
        top = f"../{top}" if top else top  # the report lives in run_dir/history/
        link = ""
        if top:
            seek = f"#t={max(0.0, float(at) - 1):.1f}" if isinstance(at, (int, float)) and status == "fail" else ""
            link = f'<a href="{html.escape(top)}{seek}">video{" @ " + format(at, ".1f") + "s" if seek else ""}</a>'
        lat = ((e.get("failure-monitor") or {}).get("latency") or e.get("latency-monitor") or e.get("latency") or {}).get("mean_ms")
        rows.append(
            f'<tr><td>{e.get("episode")}</td><td>{(e.get("scene") or 0) + 1}.{e.get("attempt")}</td>'
            f'<td style="color:{COLORS[status]}">{mark}</td>'
            f'<td>{html.escape(str(e.get("message") or ("success" if status == "success" else e.get("reason") or "")))}</td>'
            f'<td>{"" if at is None else format(at, ".1f") + " s"}</td><td>{e.get("duration_s", 0):.1f} s</td>'
            f'<td>{"" if lat is None else format(lat, ".0f") + " ms"}</td><td>{link}</td></tr>'
        )
    lat_html = (
        f'<div class="card"><div class="k">latency mean / p95</div><div class="v">{latency["mean_ms"]:.0f} / '
        f'{latency["p95_ms"]:.0f} ms</div></div>' if latency and "mean_ms" in latency else
        '<div class="card"><div class="k">latency</div><div class="v">mock</div><div class="muted">no model calls</div></div>'
        if latency else ""
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>dreamkit report {html.escape(ctx_info['id'])}</title>
<style>
body{{background:#09090b;color:#e4e4e7;font:14px/1.5 ui-sans-serif,system-ui,sans-serif;margin:0;padding:32px;}}
h1{{font-size:20px;margin:0 0 4px}} .muted{{color:#71717a}} a{{color:#60a5fa}}
.cards{{display:flex;gap:12px;margin:20px 0}} .card{{background:#18181b;border:1px solid #27272a;border-radius:10px;padding:14px 18px;min-width:150px}}
.k{{color:#a1a1aa;font-size:12px;text-transform:uppercase;letter-spacing:.05em}} .v{{font-size:24px;font-weight:600}}
table{{border-collapse:collapse;width:100%;background:#18181b;border:1px solid #27272a;border-radius:10px;overflow:hidden}}
td,th{{padding:8px 12px;border-bottom:1px solid #27272a;text-align:left}} th{{color:#a1a1aa;font-weight:500}}
.bar{{display:flex;align-items:center;gap:8px;margin:4px 0}} .lbl{{width:130px;color:#d4d4d8}}
.fill{{height:14px;background:#ef4444;border-radius:4px}} .n{{color:#a1a1aa}}
section{{margin:24px 0}}
</style></head><body>
<h1>dreamkit report</h1>
<div class="muted">run {html.escape(ctx_info['id'])} · task “{html.escape(ctx_info['task'])}” · model {html.escape(ctx_info['model'])}{' · <b style="color:#b7791f">mock: scripted failures</b>' if ctx_info.get('mock') else ''}
 · {time.strftime('%Y-%m-%d %H:%M')}</div>
<div class="cards">
<div class="card"><div class="k">success rate</div><div class="v" style="color:{COLORS['success'] if rate >= 50 else COLORS['fail']}">{rate:.0f}%</div><div class="muted">{ok}/{attempts} attempts</div></div>
<div class="card"><div class="k">scenes solved</div><div class="v">{scene_ok}/{len(scenes)}</div><div class="muted">with retries</div></div>
{lat_html}
</div>
<section><h3>failure reasons</h3>{bars}</section>
<section><h3>attempts</h3><table><tr><th>#</th><th>scene.try</th><th></th><th>what happened</th><th>at</th><th>length</th><th>latency</th><th>video</th></tr>
{''.join(rows)}</table></section>
</body></html>"""




def write_run(run_dir: Path, info: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    out = run_dir / "history"
    out.mkdir(parents=True, exist_ok=True)
    s = summary(episodes, bool(info.get("mock")))
    (out / "summary.json").write_text(json.dumps(s))
    page = render({"id": info["id"], "task": info.get("task") or "", "model": info.get("model") or "mock",
                   "mock": info.get("mock")}, episodes, latency(episodes))
    (out / "report.html").write_text(page)
    return s
