"""Judge again, later, the attempts that went "not judged" (no key, quota, network): from the frames vision-judge
saved in the run folder, and write the verdicts back into the run.

    dreamkit judge --run 20261122-123523-5c72      # one run
    dreamkit judge --all-unjudged                  # every finished run with unjudged attempts
    python -m vision_judge.rejudge --all-unjudged  # the same without the dreamkit command

A run that is still being written (status "running") is left alone. 503 (the model is overloaded): wait and try
again. 429 (quota): stop at once and say how many were judged and how many are left.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from dreamkit import config as config_mod
from dreamkit import reasons, storage
from dreamkit.llm import Unavailable, load_env

from . import judge as judge_mod

KEY = "vision-judge"
OLD_KEY = "monitor-stats"  # its name before the rename: older runs keep their frames there
WAITS_S = (10, 30, 60)  # after a 503, before each new try
OVERLOADED = re.compile(r"\b50[34]\b|UNAVAILABLE|overloaded", re.IGNORECASE)
NOTE = ("After the last frame the robot was stopped and sent back to try again; "
        "judge what the frames show up to that point.")


class QuotaExhausted(RuntimeError):
    pass


def _ask(images: list[bytes], task: str, note: str, provider: str, say) -> dict[str, Any]:
    """One verdict; a 503 waits and tries again, a 429 stops everything."""
    for i in range(len(WAITS_S) + 1):
        try:
            return judge_mod.judge(images, task, provider, timeout_s=30.0, max_wait_s=90.0, note=note)
        except Unavailable as error:
            if error.quota or "429" in str(error) or "quota" in str(error).lower():
                raise QuotaExhausted(str(error)) from error
            if i == len(WAITS_S) or not OVERLOADED.search(str(error)):
                raise
            say(f"    model overloaded (503), waiting {WAITS_S[i]} s and trying again")
            time.sleep(WAITS_S[i])
    raise AssertionError("unreachable")


def _summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    from dreamkit.kit import _summary as kit_summary  # the core's own counts  # noqa: PLC0415

    return kit_summary(episodes)


def _apply(ep: dict[str, Any], v: dict[str, Any]) -> None:
    """The verdict into the attempt's result, by the core's rule: the camera decides only what nothing else did."""
    tries = ep[KEY].get("tries") if isinstance(ep.get(KEY), dict) else None
    ep[KEY] = {**v, "judge": "vision", "rejudged": True, **({"tries": tries} if tries else {})}
    if ep.get("success") is None and not ep.get("labeled_by"):
        ep["success"], ep["judged_by"] = v["success"], "vision"
        if not v["success"] and not ep.get("reason"):
            code = v.get("reason_code")
            ep["reason"] = code if code and reasons.is_reason(code) else "judged_failed"
            ep["message"] = ep.get("message") or v.get("reason")


def rejudge_run(run_dir: Path, provider: str, say=print) -> tuple[int, int]:
    """(judged now, still not judged) for one run. Raises QuotaExhausted on a 429."""
    info = storage.read_run(run_dir)
    if info is None:
        return 0, 0
    if info.get("status") == "running":
        say(f"{run_dir.name}: still running, left alone")
        return 0, 0
    folder = next((run_dir / k for k in (KEY, OLD_KEY) if (run_dir / k / "attempts.json").is_file()), None)
    if folder is None:
        return 0, 0
    rows = json.loads((folder / "attempts.json").read_text())
    todo = [r for r in rows if r.get("error")]
    if not todo:
        return 0, 0
    say(f"{run_dir.name}: {len(todo)} not judged")
    episodes = {e["episode"]: e for e in info.get("episodes", [])}
    done = left = 0
    try:
        for row in todo:
            files = [f.get("file") for f in row.get("frames", [])]
            if not files or not all(f and (folder / f).is_file() for f in files):
                say(f"  attempt {row['episode']}, try {row.get('try', 1)}: its frames are not saved, skipped")
                left += 1
                continue
            images = [(folder / f).read_bytes() for f in files]
            try:
                v = _ask(images, str(row.get("task") or info.get("task") or ""),
                         "" if row.get("final", True) else NOTE, provider, say)
            except QuotaExhausted:
                raise
            except Exception as error:  # noqa: BLE001 - this attempt stays not judged, the others go on
                row["error"] = " ".join(str(error).split())[:160]
                say(f"  attempt {row['episode']}, try {row.get('try', 1)}: still not judged ({row['error']})")
                left += 1
                continue
            times = [f.get("t_s") for f in row["frames"]]
            row.pop("error", None)
            row.update({k: val for k, val in v.items() if k != "frames"}, rejudged=True)
            row["frames"] = [{**f, "t_s": t, "file": name}
                             for f, t, name in zip(v["frames"], times, files, strict=True)]
            if v["reason_code"]:
                row["reason_label"] = reasons.label(v["reason_code"])
            ep = episodes.get(row["episode"])
            if ep is not None and row.get("final", True):
                _apply(ep, v)
            elif ep is not None and isinstance(ep.get(KEY), dict):  # an earlier try that recovery cut short
                ep[KEY].setdefault("tries", []).append({"try": row.get("try"), **v})
            done += 1
            say(f"  attempt {row['episode']}, try {row.get('try', 1)}: "
                f"{'success' if v['success'] else 'failed: ' + str(v['reason'])} (score {v['score']})")
    finally:  # what was judged before a 429 is kept
        if done:
            (folder / "attempts.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
            info["summary"] = _summary(info.get("episodes", []))
            storage.write_json(run_dir / "run.json", info)
            for e in info.get("episodes", []):
                result = run_dir / f"ep{e['episode']:03d}" / "result.json"
                if result.is_file():
                    storage.write_json(result, e)
    return done, left


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dreamkit judge", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", action="append", metavar="ID", help="a run id (repeatable)")
    which.add_argument("--all-unjudged", action="store_true", help="every finished run with unjudged attempts")
    ap.add_argument("-c", "--config", default="dreamkit.yaml")
    ap.add_argument("--provider", default="gemini", choices=["gemini", "claude"])
    args = ap.parse_args(argv)
    cfg_path = Path(args.config)
    load_env(cfg_path.resolve().parent)
    cfg, base = config_mod.load(cfg_path) if cfg_path.is_file() else (config_mod.Config(), Path.cwd())
    root = cfg.runs_path(base)
    dirs = [root / r for r in args.run] if args.run else sorted(p for p in root.iterdir() if p.is_dir())
    total = left = 0
    try:
        for d in dirs:
            if not (d / "run.json").is_file():
                print(f"{d.name}: no such run in {root}", file=sys.stderr)
                continue
            n, m = rejudge_run(d, args.provider)
            total, left = total + n, left + m
    except QuotaExhausted as error:
        print(f"stopped: the model's quota is used up (429: {' '.join(str(error).split())[:120]}). "
              f"{total} judged before it, kept. Try again tomorrow, or add ANTHROPIC_API_KEY to .env "
              f"(--provider claude).", file=sys.stderr)
        return 2
    print(f"{total} attempts judged now" + (f", {left} still not judged" if left else ""))
    return 0 if not left else 1


if __name__ == "__main__":
    raise SystemExit(main())
