"""recap: what happened, told short — a captioned highlights clip of each attempt and a plain-language summary of
the run, in run_dir/recap/ (PLUGIN_STANDARD.md). Reads the recorder's video; all encoding runs on ctx.defer."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from dreamkit.plugin import Plugin, PluginContext
from dreamkit.standard import Manifest

## Milestones worth a highlight, and how they read in a sentence.
KEY = {"grasp", "lifted", "separated", "fell", "collision", "stalled"}
REASON_TEXT = {
    "dropped": "dropped the {obj} (it slipped out of the closed gripper)",
    "released_off_target": "let go of the {obj} away from the {target}",
    "missed_target": "missed the {target}",
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



class Params(BaseModel):
    before_s: float = Field(1.5, ge=0, le=10, description="Seconds of video before each key moment",
                            json_schema_extra={"unit": "s"})
    after_s: float = Field(1.2, ge=0, le=10, description="Seconds of video after each key moment",
                           json_schema_extra={"unit": "s"})
    max_s: float = Field(14.0, gt=0, le=120, description="Longest clip", json_schema_extra={"unit": "s"})
    camera: str = Field("top", description="Which camera the clip shows", json_schema_extra={"choices": ["top", "left", "right"]})
    video: bool = Field(True, description="Make the clip (the text recap is always written)")


class Recap(Plugin):
    """Tells each attempt in one sentence and cuts a captioned clip of its key moments."""

    name = "recap"
    Params = Params
    manifest = Manifest(group="output", icon="Clapperboard", capability="observe",
                        hooks=["on_episode_end", "on_run_end"])

    def _story(self, ctx: PluginContext) -> tuple[list[dict[str, Any]], str]:
        r = ctx.result
        obj, target = ctx.target["object"] or "object", ctx.target["target"] or "target"
        moments = [{"t": m["t"], "text": m["message"], "tone": "bad" if m["kind"] in ("separated", "fell", "collision",
                                                                                      "stalled") else "info"}
                   for m in (r.get("timeline") or []) if m.get("kind") in KEY]
        for v in ctx.store.get("verdicts", []):
            if v.get("by") != "failure-monitor":
                moments.append({"t": v["at_s"], "text": v["message"], "tone": "bad"})
        if r.get("success"):
            moments.append({"t": ctx.t, "text": "task done", "tone": "good"})
            rec = (r.get("recovery") or {}).get("count")
            headline = f"Attempt {ctx.episode}: done in {ctx.t:.1f} s" + (f" after {rec} recover{'y' if rec == 1 else 'ies'}." if rec else ".")
        else:
            reason = r.get("reason") or "timeout"
            what = REASON_TEXT.get(reason, reason.replace("_", " ")).format(obj=obj, target=target)
            at = r.get("at_s")
            if at is not None:
                moments.append({"t": at, "text": r.get("message") or what, "tone": "bad", "verdict": True})
            steps = [m["text"] for m in moments if not m.get("verdict")][:3]
            lead = (", ".join(steps) + ", then ") if steps else ""
            headline = f"Attempt {ctx.episode}: {lead}{what}" + (f" at {at:.1f} s." if at is not None else ".")
        moments.sort(key=lambda m: m["t"])
        return moments, headline

    def _clip(self, src: Path, out: Path, moments: list[dict[str, Any]], episode: int) -> None:
        import av  # noqa: PLC0415

        if not src.is_file() or not moments:
            return
        p = self.params
        windows: list[list[Any]] = []
        for m in moments:
            lo, hi = max(0.0, m["t"] - p.before_s), m["t"] + p.after_s
            if windows and lo <= windows[-1][1]:
                windows[-1][1] = max(windows[-1][1], hi)
                windows[-1][2].append(m)
            else:
                windows.append([lo, hi, [m]])
        kept, total = [], 0.0
        for w in sorted(windows, key=lambda w: (not any(m.get("verdict") for m in w[2]), w[0])):
            if total + (w[1] - w[0]) <= p.max_s or not kept:
                kept.append(w)
                total += w[1] - w[0]
        kept.sort(key=lambda w: w[0])
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

    def on_episode_end(self, ctx: PluginContext) -> None:
        moments, headline = self._story(ctx)
        recap: dict[str, Any] = {"headline": headline, "moments": moments, "video": None}
        videos = (ctx.result.get("recorder") or {}).get("videos") or []
        src = next((v for v in videos if v.endswith(f"_{self.params.camera}.mp4")), None)
        if self.params.video and src and ctx.run_dir is not None:
            out = ctx.files.path(ctx.files.episode_name("recap.mp4"))
            ctx.defer(self._clip, ctx.run_dir / src, out, moments, ctx.episode)  # after the recorder closed it
            recap["video"] = ctx.files.relative(out)
        ctx.result["recap"] = recap

    def on_run_end(self, ctx: PluginContext) -> None:
        eps = ctx.episodes
        if not eps:
            return
        ok = [e for e in eps if e.get("success")]
        scenes = sorted({e["scene"] for e in eps})
        solved = {e["scene"] for e in ok}
        count: dict[str, int] = {}
        for e in eps:
            if e.get("success") is False:
                count[e.get("reason") or "unknown"] = count.get(e.get("reason") or "unknown", 0) + 1
        lines = [f"{len(ok)} of {len(eps)} attempts worked; {len(solved)} of {len(scenes)} scenes solved."]
        if count:
            top, n = max(count.items(), key=lambda kv: kv[1])
            lines.append(f"Most common failure: {top.replace('_', ' ')} ({n}x).")
        rescued = [e for e in ok if (e.get("recovery") or {}).get("count")]
        if rescued:
            lines.append(f"Recovery turned {len(rescued)} failing attempt{'s' if len(rescued) > 1 else ''} into a success.")
        retried = [s for s in solved if any(e["scene"] == s and e.get("success") is False for e in eps)]
        if retried:
            lines.append(f"Retries saved {len(retried)} scene{'s' if len(retried) > 1 else ''}.")
        times = [e["duration_s"] for e in ok]
        if times:
            lo, hi = min(times), max(times)
            span = f"{lo:.1f} s" if round(lo, 1) == round(hi, 1) else f"{lo:.1f}-{hi:.1f} s"
            lines.append(f"Successful attempt{'s' if len(times) > 1 else ''} took {span}.")
        lat = ctx.store.get("latency-monitor")
        if lat and lat.get("mock"):
            lines.append("Mock: no model calls.")
        elif lat:
            lines.append(f"Model round trip: {lat['mean_ms']:.0f} ms on average, {lat['p95_ms']:.0f} ms p95.")
        summary = {"text": " ".join(lines), "attempts": [(e.get("recap") or {}).get("headline") for e in eps]}
        ctx.defer(ctx.files.path("summary.json").write_text, json.dumps(summary))
        ctx.store["recap"] = summary
        ctx.emit("success" if ok else "info", summary["text"][0].lower() + summary["text"][1:].rstrip("."))
