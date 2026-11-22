"""The numbers in docs/RESULTS.md, from real Dream-1 runs: success rate with its 95% interval, why attempts failed,
the model's round trip, cost, what recovery and the camera judge did.

    python tools/results.py                                 # from docs/results/dream1-attempts.json (in git)
    python tools/results.py --runs runs --export docs/results/dream1-attempts.json   # re-export from run folders

The run folders (videos, frames: about 1 GB) stay local; the export keeps one row per attempt and nothing private
(no messages with addresses or session ids, no local paths).
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamkit.stats import wilson  # noqa: E402

EXPORT = ROOT / "docs" / "results" / "dream1-attempts.json"


def _family(task: str) -> str:
    return next((f for f, w in (("apple→bowl", "apple"), ("orange→bowl", "orange"), ("banana→plate", "banana"))
                 if w in task), task)


def _model_error_kind(e: dict) -> str | None:
    if e.get("reason") != "model_error":
        return None
    text = str((e.get("details") or {}).get("message") or e.get("message") or "")
    for key, kind in (("could not reach", "network: could not reach the API"), ("DreamTimeout", "timeout (45 s)"),
                      ("NO_CAPACITY", "no capacity"), ("SessionExpired", "session expired")):
        if key in text:
            return kind
    return "other"


def export(runs: Path) -> list[dict]:
    rows = []
    for f in sorted(runs.glob("*/run.json")):
        d = json.loads(f.read_text())
        if d.get("mock") or "dream" not in str(d.get("model", "")).lower():
            continue  # real Dream-1 runs only
        for e in d.get("episodes", []):
            judge = e.get("monitor-stats") or e.get("vision-judge") or {}
            lat = (e.get("failure-monitor") or {}).get("latency") or {}
            rows.append({
                "run": d["id"], "started": d.get("started"), "model": d.get("model"), "attempt": e["episode"],
                "task": e["task"], "success": e.get("success"), "reason": e.get("reason"),
                "model_error": _model_error_kind(e), "duration_s": e.get("duration_s"), "requests": e.get("requests"),
                "cost_usd": e.get("cost_usd"), "rtt_mean_ms": lat.get("mean_ms"), "rtt_p95_ms": lat.get("p95_ms"),
                "recoveries": (e.get("recovery") or {}).get("count", 0),
                "judge_success": judge.get("success") if isinstance(judge.get("success"), bool) else None,
                "judge_error": bool(judge.get("error")),
            })
    return rows


def pct(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} (95% CI {lo:.0%}–{hi:.0%})" if n else "no attempts"


def report(rows: list[dict]) -> None:
    runs = sorted({r["run"] for r in rows})
    print(f"real Dream-1 runs: {len(runs)} ({', '.join(runs)}), attempts: {len(rows)}")
    ok = sum(1 for r in rows if r["success"])
    print("success, every attempt:", pct(ok, len(rows)))
    ran = [r for r in rows if r["reason"] not in ("model_error", "error")]
    print("success, attempts where the model answered:", pct(sum(1 for r in ran if r["success"]), len(ran)))
    for fam in ("apple→bowl", "orange→bowl", "banana→plate"):
        sub = [r for r in ran if _family(r["task"]) == fam]
        if sub:
            print(f"  {fam}: {pct(sum(1 for r in sub if r['success']), len(sub))}")
    reasons = collections.Counter(r["reason"] for r in rows if r["success"] is False)
    print("failure reasons:", dict(reasons.most_common()))
    print("model errors:", dict(collections.Counter(r["model_error"] for r in rows if r["model_error"]).most_common()))
    rtt = [r["rtt_mean_ms"] for r in rows if r["rtt_mean_ms"]]
    p95 = [r["rtt_p95_ms"] for r in rows if r["rtt_p95_ms"]]
    if rtt:
        print(f"model round trip (per attempt, {len(rtt)} attempts with calls): median of means "
              f"{statistics.median(rtt):.0f} ms, median p95 {statistics.median(p95):.0f} ms")
    cost = sum(r["cost_usd"] or 0 for r in rows)
    print(f"cost: ${cost:.2f} for {sum(r['requests'] or 0 for r in rows)} model calls; "
          f"${cost / ok:.3f} per success" if ok else f"cost: ${cost:.2f}")
    good = [r["duration_s"] for r in rows if r["success"]]
    print(f"a successful attempt took {statistics.mean(good):.1f} s on average")
    rec = [r for r in rows if r["recoveries"]]
    print(f"recovery stepped in on {len(rec)} attempts; {sum(1 for r in rec if r['success'])} of them then succeeded")
    judged = [r for r in rows if r["judge_success"] is not None and r["success"] is not None]
    agree = sum(1 for r in judged if r["judge_success"] == r["success"])
    print(f"camera judge: judged {len(judged)} of {len(rows)} attempts, agreed with the simulator's success-check on "
          f"{agree} of {len(judged)}; not judged: {len(rows) - len(judged)} "
          f"({sum(1 for r in rows if r['judge_error'])} of them because its model call failed: quota, network)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, help="read the run folders here instead of the export in git")
    ap.add_argument("--export", type=Path, help="write the attempts to this JSON file")
    args = ap.parse_args()
    rows = export(args.runs) if args.runs else json.loads(EXPORT.read_text())
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(rows, indent=1) + "\n")
        print(f"wrote {len(rows)} attempts to {args.export}")
    report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
