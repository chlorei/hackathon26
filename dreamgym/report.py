"""A static HTML page for an Arena run or a Compare folder: rates with intervals, failure breakdown, failure GIFs.

Open `report.html` in a browser; it needs no server. The dashboard owner can replace this with anything that
reads summary.json / compare.json / episodes.jsonl.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from .arena import load_results

CSS = """
body{font:15px/1.45 system-ui,sans-serif;margin:32px auto;max-width:1100px;padding:0 16px;color:#1d1d1f}
h1{font-size:24px;margin:0 0 4px} h2{font-size:18px;margin:28px 0 8px} .muted{color:#6e6e73}
table{border-collapse:collapse;width:100%} th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #e5e5ea}
.bar{height:10px;background:#e5e5ea;border-radius:5px;position:relative;min-width:160px}
.fill{height:10px;background:#2f7d5b;border-radius:5px} .ci{position:absolute;top:-3px;height:16px;border:2px solid #1d1d1f;border-top:0;border-bottom:0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.card{border:1px solid #e5e5ea;border-radius:8px;padding:8px} .card img{width:100%;border-radius:4px}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;background:#fdecea;color:#a1271b;font-size:13px}
"""


def _bar(s: dict) -> str:
    lo, hi = s["ci95"]
    return (f'<div class="bar"><div class="fill" style="width:{s["success_rate"] * 100:.1f}%"></div>'
            f'<div class="ci" style="left:{lo * 100:.1f}%;width:{(hi - lo) * 100:.1f}%"></div></div>')


def _row(s: dict) -> str:
    lo, hi = s["ci95"]
    cps = f'${s["cost_per_success_usd"]:.3f}' if s["cost_per_success_usd"] is not None else "–"
    fails = ", ".join(f"{k} {v}" for k, v in s["failures"].items()) or "–"
    return (f"<tr><td>{html.escape(s['policy'])}<div class='muted'>{html.escape(s['prompt'])}</div></td>"
            f"<td><b>{s['successes']}/{s['episodes']}</b> = {s['success_rate']:.0%}"
            f"<div class='muted'>95% CI {lo:.0%}–{hi:.0%}</div></td><td>{_bar(s)}</td>"
            f"<td>{html.escape(fails)}</td><td>{s['mean_seconds_to_success'] or '–'} s</td><td>{cps}</td></tr>")


def _gallery(run_dir: Path, prefix: str = "") -> str:
    cards = []
    for r in load_results(run_dir):
        if r.get("gif"):
            cards.append(f"<div class='card'><img src='{html.escape(prefix + r['gif'])}' loading='lazy'>"
                         f"<div>seed {r['seed']} <span class='tag'>{html.escape(r['reason'])}</span></div></div>")
    return "<div class='grid'>" + "".join(cards) + "</div>" if cards else "<p class='muted'>No GIFs.</p>"


def build(folder: Path) -> Path:
    folder = Path(folder)
    head = "<tr><th>Policy / prompt</th><th>Success</th><th></th><th>Failures</th><th>Time to success</th><th>$ / success</th></tr>"
    if (folder / "compare.json").is_file():
        report = json.loads((folder / "compare.json").read_text())
        title = f"Compare · {report['task']}"
        body = "<table>" + head + "".join(_row(s) for s in report["summaries"]) + "</table>"
        for t in report["paired_vs_first"]:
            verdict = "significant (p < 0.05)" if t["p_value"] < 0.05 else "not significant: could be luck"
            body += (f"<p><b>{html.escape(t['b'])}</b> vs <b>{html.escape(t['a'])}</b>: "
                     f"{t['rate_b']:.0%} vs {t['rate_a']:.0%} on {t['seeds']} identical scenes, "
                     f"Δ {t['delta']:+.0%}, McNemar p = {t['p_value']} — {verdict}</p>")
        for name, label in zip(report["runs"], report["variants"]):
            body += f"<h2>Failures · {html.escape(label)}</h2>" + _gallery(folder / name, f"{name}/")
    else:
        s = json.loads((folder / "summary.json").read_text())
        title = f"Arena · {s['task']}"
        body = "<table>" + head + _row(s) + "</table><h2>Failures</h2>" + _gallery(folder)
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body><h1>{html.escape(title)}</h1>"
            f"<p class='muted'>DreamGym · {html.escape(str(folder))}</p>{body}</body></html>")
    out = folder / "report.html"
    out.write_text(page, encoding="utf-8")
    return out
