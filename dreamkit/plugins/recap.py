"""recap: what happened, told short — a highlights clip of each attempt and a plain-language summary of the run."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..plugin import Context, Plugin
from ..task import parse

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
    "stopped_by_plugin": "was stopped by a plugin",
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


class Recap(Plugin):
    """Per attempt: recap.mp4 (the key moments of the top camera, captioned) and recap.json (the story in sentences).
    Per run: one paragraph and the numbers. Reads the recorder's video, so put it after `recorder`."""

    name = "recap"
    group = "output"
    description = "Highlights clip of each attempt + a plain-language recap of the run"
    defaults = {"before_s": 1.5, "after_s": 1.2, "max_s": 14.0, "camera": "top", "video": True}

    def _story(self, ctx: Context) -> tuple[list[dict[str, Any]], str]:
        r = ctx.result
        timeline = r.get("timeline") or []
        obj, target = parse(ctx.task)
        obj, target = obj or "object", target or "target"
        moments = [{"t": m["t"], "text": m["message"], "tone": "bad" if m["kind"] in ("separated", "fell", "collision",
                                                                                      "stalled") else "info"}
                   for m in timeline if m.get("kind") in KEY]
        if r.get("success"):
            moments.append({"t": r.get("at", ctx.t), "text": "task done", "tone": "good"})
            headline = f"Attempt {ctx.episode}: done in {ctx.t:.1f} s."
        else:
            reason = r.get("reason") or "unknown"
            what = REASON_TEXT.get(reason, reason).format(obj=obj, target=target)
            at = r.get("at")
            if at is not None:
                moments.append({"t": at, "text": r.get("message") or what, "tone": "bad", "verdict": True})
            steps = [m["text"] for m in moments if not m.get("verdict")][:3]
            lead = (", ".join(steps) + ", then ") if steps else ""
            headline = f"Attempt {ctx.episode}: {lead}{what}" + (f" at {at:.1f} s." if at is not None else ".")
        moments.sort(key=lambda m: m["t"])
        return moments, headline

    def _clip(self, ctx: Context, moments: list[dict[str, Any]]) -> str | None:
        import av

        src = ctx.episode_dir / f"{self.params['camera']}.mp4"
        if not src.is_file() or not moments:
            return None
        before, after = float(self.params["before_s"]), float(self.params["after_s"])
        windows: list[list[Any]] = []
        for m in moments:
            lo, hi = max(0.0, m["t"] - before), m["t"] + after
            if windows and lo <= windows[-1][1]:
                windows[-1][1] = max(windows[-1][1], hi)
                windows[-1][2].append(m)
            else:
                windows.append([lo, hi, [m]])
        budget = float(self.params["max_s"])
        kept, total = [], 0.0
        for w in sorted(windows, key=lambda w: (not any(m.get("verdict") for m in w[2]), w[0])):
            if total + (w[1] - w[0]) <= budget or not kept:
                kept.append(w)
                total += w[1] - w[0]
        kept.sort(key=lambda w: w[0])
        out_path = ctx.episode_dir / "recap.mp4"
        with av.open(str(src)) as source:
            stream_in = source.streams.video[0]
            out = av.open(str(out_path), "w", options={"movflags": "+faststart"})
            stream = out.add_stream("libx264", rate=10)
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
                rgb = _caption(frame.to_ndarray(format="rgb24"), near["text"], f"{t:5.1f} s  ·  attempt {ctx.episode}",
                               near["tone"])
                new = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                new.pts = pts
                new.time_base = Fraction(1, 10)
                pts += 1
                for packet in stream.encode(new):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
            out.close()
        return f"ep{ctx.episode}/recap.mp4" if pts else None

    def on_episode_end(self, ctx: Context) -> None:
        if ctx.episode_dir is None:
            return
        moments, headline = self._story(ctx)
        video = None
        if self.params["video"]:
            try:
                video = self._clip(ctx, moments)
            except Exception as error:  # a clip is a nicety; the story still stands
                ctx.emit("warn", f"recap clip skipped: {error}")
        recap = {"headline": headline, "moments": moments, "video": video}
        (ctx.episode_dir / "recap.json").write_text(json.dumps(recap))
        ctx.result["recap"] = recap

    def on_run_end(self, ctx: Context) -> None:
        eps = ctx.episodes
        if not eps:
            return
        ok = [e for e in eps if e.get("success")]
        scenes = sorted({e["scene"] for e in eps})
        solved = {e["scene"] for e in ok}
        reasons: dict[str, int] = {}
        for e in eps:
            if e.get("success") is False:
                reasons[e.get("reason") or "unknown"] = reasons.get(e.get("reason") or "unknown", 0) + 1
        lines = [f"{len(ok)} of {len(eps)} attempts worked; {len(solved)} of {len(scenes)} scenes solved."]
        if reasons:
            top, n = max(reasons.items(), key=lambda kv: kv[1])
            lines.append(f"Most common failure: {top.replace('_', ' ')} ({n}×).")
        rescued = [e for e in ok if e.get("recoveries")]
        if rescued:
            lines.append(f"Recovery turned {len(rescued)} failing attempt{'s' if len(rescued) > 1 else ''} into a success.")
        retried = [s for s in solved if any(e["scene"] == s and e.get("success") is False for e in eps)]
        if retried:
            lines.append(f"Retries saved {len(retried)} scene{'s' if len(retried) > 1 else ''}.")
        times = [e["duration_s"] for e in ok]
        if times:
            lo, hi = min(times), max(times)
            span = f"{lo:.1f} s" if round(lo, 1) == round(hi, 1) else f"{lo:.1f}–{hi:.1f} s"
            lines.append(f"Successful attempt{'s' if len(times) > 1 else ''} took {span}.")
        lat = ctx.store.get("latency")
        if ctx.store.get("latency_mock") and not lat:
            lines.append("Mock: no model calls.")
        elif lat:
            lines.append(f"Model latency: {lat['mean_ms']:.0f} ms on average, {lat['p95_ms']:.0f} ms p95.")
        summary = {"text": " ".join(lines), "attempts": [e.get("recap", {}).get("headline") for e in eps]}
        if ctx.run_dir is not None:
            (ctx.run_dir / "recap.json").write_text(json.dumps(summary))
        ctx.store["recap"] = summary
        ctx.emit("success" if ok else "info", summary["text"])
