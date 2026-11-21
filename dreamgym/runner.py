"""One episode: reset and randomise the scene, fill the model's frame windows, run the policy, judge it."""

from __future__ import annotations

import time
from collections import deque
from typing import Callable

import numpy as np

from . import scene, sim
from .core import STEP_S, CALLS_PER_HOUR, Context, EpisodeResult, Policy, Task

StepHook = Callable[[object, np.ndarray, np.ndarray, int], None]


def _label_error(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}"
    if "IK failed" in text:
        return "ik_error"
    if type(error).__module__.startswith("dream"):
        return "api_error"
    if isinstance(error, ValueError):
        return "invalid_action"
    return "error"


def run_episode(
    env,
    task: Task,
    policy: Policy,
    *,
    seed: int,
    prompt: str | None = None,
    max_seconds: float = 60.0,
    hold_seconds: float = 1.0,
    keep_frames: bool = False,
    step_hook: StepHook | None = None,
) -> tuple[EpisodeResult, list[np.ndarray]]:
    """Runs one seeded episode and judges it.

    @param max_seconds Simulated seconds the policy gets (physics holds while the model thinks, as in the sim).
    @param hold_seconds The success condition must hold this long, so a fruit bouncing through the bowl is no success.
    @param keep_frames Return the top camera at 5 fps, for a GIF of the episode.
    @param step_hook Called before every commanded step with (env, left_row, right_row, step): the recorder's tap.
    @return The result, and the kept frames (empty unless asked).
    """
    yam_env, dc, _ = sim.load()
    rng = np.random.default_rng(seed)
    wall0 = time.monotonic()
    prompt = prompt or task.prompt

    env.reset()
    task.reset(env, rng)
    home = yam_env.HOME
    for _ in range(20):  # let randomised objects settle for 1 s
        env.step(home, home)
    start = {o: scene.pos(env, o) for o in scene.FREE_JOINTS}

    ctx = Context(windows={name: deque(maxlen=dc.FRAMES) for name in dc.VIDEO_ROLES}, rng=rng)
    frames: list[np.ndarray] = []
    state = env.state()
    while any(len(w) < dc.FRAMES for w in ctx.windows.values()):  # 9 frames, 0.2 s apart, as the sim does
        env.step(state["left"], state["right"])
        ctx.steps += 1
        if ctx.steps % dc.FRAME_EVERY_STEPS == 0:
            dc.capture(env, ctx.windows, keep=True)
    first_step = ctx.steps
    max_steps = first_step + int(round(max_seconds / STEP_S))
    hold_steps = max(1, int(round(hold_seconds / STEP_S)))

    held, outcome, error, success_at = 0, None, None, None
    try:
        policy.begin(env, task, prompt, ctx)
        while outcome is None and ctx.steps < max_steps:
            chunk = policy.next_chunk(env, ctx)
            for left, right in zip(chunk.left, chunk.right):
                if step_hook is not None:
                    step_hook(env, left, right, ctx.steps)
                env.step(left, right)
                ctx.steps += 1
                if ctx.steps % dc.FRAME_EVERY_STEPS == 0:
                    images = dc.capture(env, ctx.windows, keep=True)
                    if keep_frames:
                        frames.append(images["top"][::2, ::2].copy())
                    policy.on_frame(env, ctx)
                held = held + 1 if task.success(env) else 0
                if held >= hold_steps:
                    outcome = "success"
                    success_at = (ctx.steps - hold_steps - first_step) * STEP_S
                    break
                if ctx.steps >= max_steps:
                    break
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # a failed episode, not a failed run: record it and go on
        error = f"{type(exc).__name__}: {exc}"
        outcome = _label_error(exc)

    if outcome is None:
        outcome = task.failure_reason(env, start)
    result = EpisodeResult(
        seed=seed,
        task=task.name,
        policy=policy.name,
        prompt=prompt,
        success=outcome == "success",
        reason=outcome,
        sim_seconds=round((ctx.steps - first_step) * STEP_S, 2),
        seconds_to_success=None if success_at is None else round(success_at, 2),
        calls=ctx.calls,
        cost_usd=round(ctx.calls * policy.price_per_hour / CALLS_PER_HOUR, 5),
        wall_seconds=round(time.monotonic() - wall0, 1),
        error=error,
    )
    return result, frames
