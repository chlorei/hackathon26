"""One episode: reset and randomise the scene, fill the model's frame windows, run the policy through dreamkit, judge.

The policy is asked through dreamkit.wrap (kit.get_action) and every executed row is a kit.step, inside
kit.episode(): the loop plugins of dreamkit.yaml see the episode as they see your own loop, and recovery's rows replace
the policy's when it acts. Every chunk, the policy's or recovery's, goes through dream_control.validate_chunk.
Two judges look at the episode: DreamGym's (the task's success condition held for `hold_seconds`) and dreamkit's
plugins (success-check, failure-monitor). Whoever decides first ends it; both speak dreamkit's shared reason codes.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable

import numpy as np

from . import scene, sim
from .core import CALLS_PER_HOUR, STEP_S, Context, EpisodeResult, Policy, Task
from .kit import PolicyClient, open_kit

StepHook = Callable[..., None]


def _label_error(error: BaseException) -> str:
    """An exception as a shared reason code: the model's (model_error) or anything else (error)."""
    if type(error).__module__.startswith("dream") and not type(error).__module__.startswith("dreamgym"):
        return "model_error"
    return "error"


def _validated(dc: Any, env: Any, reply: dict[str, Any], limit: int, overridden: bool) -> tuple[np.ndarray, np.ndarray]:
    """dream_control.validate_chunk on every chunk. A recovery move may be longer than 48 rows: validated (and all
    executed) 48 rows at a time; a model chunk executes its first `limit` rows, as the sim does."""
    if not overridden:
        return dc.validate_chunk(env, reply, limit, verbose=False)
    left, right = np.asarray(reply["q_l_chunk"], float), np.asarray(reply["q_r_chunk"], float)
    parts = [dc.validate_chunk(env, {"q_l_chunk": left[i:i + 48], "q_r_chunk": right[i:i + 48]}, 48, verbose=False)
             for i in range(0, len(left), 48)]
    return np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])


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
    kit: Any = None,
) -> tuple[EpisodeResult, list[np.ndarray]]:
    """Runs one seeded episode through dreamkit and judges it.

    @param max_seconds Simulated seconds the policy gets (physics holds while the model thinks, as in the sim).
    @param hold_seconds The success condition must hold this long, so a fruit bouncing through the bowl is no success.
    @param keep_frames Return the top camera at 5 fps, for a GIF of the episode.
    @param step_hook Called before every commanded step with (env, left_row, right_row, step): the recorder's tap.
    @param kit The dreamkit run (open_kit) this episode belongs to; None opens and closes one for this episode.
    @return The result, and the kept frames (empty unless asked).
    """
    from dreamkit.plugin import EndEpisode, StopEpisode

    yam_env, dc, _ = sim.load()
    own_kit = kit is None
    if own_kit:
        kit = open_kit(PolicyClient(policy), model=policy.name, info={"mode": "episode", "task": task.name})
    client: PolicyClient = kit.client
    rng = np.random.default_rng(seed)
    wall0 = time.monotonic()
    prompt = prompt or task.prompt
    limit = int(getattr(policy, "action_limit", 48))

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
    client.bind(env, ctx, task, prompt)

    held, outcome, error, success_at = 0, None, None, None
    with kit.episode(prompt, env=env, seed=seed, task_name=task.name, source="gym"):
        kit.ctx.result["gym"] = {"seed": seed, "task": task.name}
        try:
            policy.begin(env, task, prompt, ctx)
            while outcome is None and ctx.steps < max_steps:
                reply = kit.get_action()  # before_model -> the policy (or recovery's rows) -> after_model
                left_rows, right_rows = _validated(dc, env, reply, limit, overridden="overridden_by" in kit.ctx.store)
                for left, right in zip(left_rows, right_rows):
                    if step_hook is not None:
                        step_hook(env, left, right, ctx.steps)
                    env.step(left, right)
                    ctx.steps += 1
                    images = None
                    if ctx.steps % dc.FRAME_EVERY_STEPS == 0:
                        images = dc.capture(env, ctx.windows, keep=True)
                        if keep_frames:
                            frames.append(images["top"][::2, ::2].copy())
                        policy.on_frame(env, ctx)
                    held = held + 1 if task.success(env) else 0
                    if held >= hold_steps:
                        outcome = "success"
                        success_at = (ctx.steps - hold_steps - first_step) * STEP_S
                    kit.step(env, images=images, command=(left, right))  # the plugins' step; may end the attempt
                    if outcome is not None or ctx.steps >= max_steps:
                        break
        except (EndEpisode, StopEpisode, KeyboardInterrupt):
            raise  # dreamkit (or the user) ended it: kit.episode() records how
        except Exception as exc:  # a failed episode, not a failed run: record it and go on
            error = f"{type(exc).__name__}: {exc}"
            outcome = _label_error(exc)
            kit.ctx.result.update(success=False, reason=outcome, at_s=round(kit.t, 2), details={"message": error})
        if outcome is None:
            outcome = task.failure_reason(env, start)
        if outcome == "success":
            kit.label(True)  # DreamGym's judge: the condition held for hold_seconds
        elif not kit.ctx.result.get("reason"):
            kit.ctx.result.setdefault("success", False)
            kit.ctx.result["reason"] = outcome
        kit.ctx.result["gym"]["judge"] = outcome

    ep = kit.episodes[-1] if kit.episodes else {}
    if outcome is None:  # dreamkit ended the attempt before DreamGym's judge did
        if ep.get("success"):
            outcome = "success"
            success_at = ep.get("at_s") or (ctx.steps - first_step) * STEP_S
        else:
            outcome = ep.get("reason") or task.failure_reason(env, start)
    result = EpisodeResult(
        seed=seed,
        task=task.name,
        policy=policy.name,
        prompt=prompt,
        success=outcome == "success",
        reason=outcome,
        sim_seconds=round((ctx.steps - first_step) * STEP_S, 2),
        seconds_to_success=None if success_at is None else round(float(success_at), 2),
        calls=ctx.calls,
        cost_usd=round(ctx.calls * policy.price_per_hour / CALLS_PER_HOUR, 5),
        wall_seconds=round(time.monotonic() - wall0, 1),
        error=error,
        run_id=kit.run_id,
        episode=ep.get("episode"),
        recoveries=int((ep.get("recovery") or {}).get("count", 0)),
    )
    if own_kit:
        kit.close()
    return result, frames
