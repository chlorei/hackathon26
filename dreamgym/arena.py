"""Arena: N seeded episodes of one task with one policy -> episodes.jsonl, summary.json, GIFs of failures.

Results are appended one line per episode, so a run cut by a dropped network or Ctrl-C resumes where it stopped
(`resume=True` skips seeds already in episodes.jsonl). Seeds are `seed0 .. seed0+n-1`: two policies run on the same
seeds see the same scenes, which is what makes Compare's paired test fair.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from . import sim
from .core import Policy, Task
from .runner import run_episode
from .stats import summarize


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:80] or "run"


def save_gif(frames, path: Path, fps: int = 5) -> None:
    from PIL import Image

    if not frames:
        return
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0, optimize=True)


def load_results(out_dir: Path) -> list[dict]:
    path = Path(out_dir) / "episodes.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_arena(
    task: Task,
    policy: Policy,
    *,
    n: int,
    out_dir: Path,
    seed0: int = 0,
    prompt: str | None = None,
    max_seconds: float = 60.0,
    gifs: str = "failures",  # "none" | "failures" | "all"
    resume: bool = True,
    env=None,
    log: Callable[[str], None] = print,
) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "gifs").mkdir(parents=True, exist_ok=True)
    done = {r["seed"] for r in load_results(out_dir)} if resume else set()
    own_env = env is None
    env = env or sim.make_env()
    log_path = out_dir / "episodes.jsonl"
    if not resume and log_path.exists():
        log_path.unlink()
    try:
        for i, seed in enumerate(range(seed0, seed0 + n)):
            if seed in done:
                continue
            result, frames = run_episode(
                env, task, policy, seed=seed, prompt=prompt, max_seconds=max_seconds, keep_frames=gifs != "none"
            )
            if gifs == "all" or (gifs == "failures" and not result.success):
                gif = out_dir / "gifs" / f"seed{seed:04d}_{result.reason}.gif"
                save_gif(frames, gif)
                result.gif = str(gif.relative_to(out_dir))
            with log_path.open("a") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
            mark = "OK  " if result.success else "FAIL"
            log(
                f"[{i + 1}/{n}] seed {seed}: {mark} {result.reason:<15} "
                f"{result.sim_seconds:5.1f}s sim, {result.calls} calls, ${result.cost_usd:.3f}"
                + (f"  ({result.error})" if result.error else "")
            )
    finally:
        policy.close()
        if own_env:
            env.close()
    rows = [r for r in load_results(out_dir) if seed0 <= r["seed"] < seed0 + n]
    summary = {
        "task": task.name,
        "policy": policy.name,
        "prompt": prompt or task.prompt,
        "seeds": [seed0, seed0 + n - 1],
        "max_seconds": max_seconds,
        **summarize(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def format_summary(s: dict) -> str:
    lo, hi = s["ci95"]
    lines = [
        f"{s['policy']} on {s['task']} ({s['prompt']!r})",
        f"  success {s['successes']}/{s['episodes']} = {s['success_rate']:.0%}  (95% CI {lo:.0%}-{hi:.0%})",
        f"  failures: {s['failures'] or '-'}",
        f"  mean time to success: {s['mean_seconds_to_success']} s sim",
        f"  cost ${s['cost_usd']:.3f}, per success "
        + (f"${s['cost_per_success_usd']:.3f}" if s["cost_per_success_usd"] is not None else "-"),
    ]
    return "\n".join(lines)
