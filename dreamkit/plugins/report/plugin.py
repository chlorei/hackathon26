"""report: a self-contained HTML summary of the run, in run_dir/report/report.html (PLUGIN_STANDARD.md)."""

from __future__ import annotations

import html
import time

from pydantic import BaseModel

from dreamkit.plugin import Plugin, PluginContext
from dreamkit.standard import Manifest

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
        top = f"../{top}" if top else top  # the report lives in run_dir/report/
        link = ""
        if top:
            seek = f"#t={max(0.0, float(at) - 1):.1f}" if isinstance(at, (int, float)) and status == "fail" else ""
            link = f'<a href="{html.escape(top)}{seek}">video{" @ " + format(at, ".1f") + "s" if seek else ""}</a>'
        lat = (e.get("latency-monitor") or e.get("latency") or {}).get("mean_ms")
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
<div class="muted">run {html.escape(ctx_info['id'])} · task “{html.escape(ctx_info['task'])}” · model {html.escape(ctx_info['model'])}
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


class Params(BaseModel):
    """No settings."""


class Report(Plugin):
    """Writes a self-contained HTML report: success rate, failure reasons, latency, video links."""

    name = "report"
    Params = Params
    manifest = Manifest(group="output", icon="FileText", capability="observe", hooks=["on_run_end"])

    def on_run_end(self, ctx: PluginContext) -> None:
        info = {"id": ctx.run_id, "task": ctx.task, "model": ctx.model or "mock"}
        path = ctx.files.path("report.html")
        page = render(info, ctx.episodes, ctx.store.get("latency-monitor"))
        ctx.defer(path.write_text, page)
        ctx.emit("info", f"report written to {ctx.files.relative(path)}")
