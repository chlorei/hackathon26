"""Compare: the same task and seeds under several variants (models x prompts), then a paired test against the first.

"Did the fine-tune help?" and "which prompt is better?" are the same question here: variant B against
baseline A on identical scenes. McNemar's exact test says whether the difference is more than luck.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import sim
from .arena import format_summary, load_results, run_arena, slug
from .policies import make_policy
from .stats import paired


@dataclass
class Variant:
    policy: str
    prompt: str | None = None

    @property
    def label(self) -> str:
        return self.policy + (f" | {self.prompt}" if self.prompt else "")


def run_compare(
    task,
    variants: list[Variant],
    *,
    n: int,
    out_dir: Path,
    seed0: int = 0,
    max_seconds: float = 60.0,
    action_limit: int = 48,
    gifs: str = "failures",
    log: Callable[[str], None] = print,
) -> dict:
    out_dir = Path(out_dir)
    env = sim.make_env()
    summaries, dirs = [], []
    try:
        for v in variants:
            d = out_dir / slug(v.label)
            dirs.append(d)
            log(f"\n=== {v.label} ===")
            policy = make_policy(v.policy, action_limit=action_limit)
            policy.name = v.label
            summaries.append(
                run_arena(task, policy, n=n, out_dir=d, seed0=seed0, prompt=v.prompt, max_seconds=max_seconds,
                          gifs=gifs, env=env, log=log)
            )
    finally:
        env.close()
    base = [r for r in load_results(dirs[0]) if seed0 <= r["seed"] < seed0 + n]
    tests = []
    for v, d in zip(variants[1:], dirs[1:]):
        other = [r for r in load_results(d) if seed0 <= r["seed"] < seed0 + n]
        tests.append({"a": variants[0].label, "b": v.label, **paired(base, other)})
    report = {"task": task.name, "variants": [v.label for v in variants], "runs": [str(d.name) for d in dirs],
              "summaries": summaries, "paired_vs_first": tests}
    (out_dir / "compare.json").write_text(json.dumps(report, indent=2))
    return report


def format_compare(report: dict) -> str:
    lines = [format_summary(s) for s in report["summaries"]]
    for t in report["paired_vs_first"]:
        verdict = "significant" if t["p_value"] < 0.05 else "could be luck"
        lines.append(
            f"\n{t['b']}  vs  {t['a']}: {t['rate_b']:.0%} vs {t['rate_a']:.0%} on {t['seeds']} shared seeds, "
            f"delta {t['delta']:+.0%}; only-B {t['only_b_succeeded']}, only-A {t['only_a_succeeded']}, "
            f"p = {t['p_value']} ({verdict})"
        )
    return "\n".join(lines)
