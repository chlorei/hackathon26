"""Reproducible scene randomisation: only the objects move (never the arms, cameras or their physics)."""

from __future__ import annotations

import mujoco
import numpy as np

from .sim import FREE_JOINT


def randomize(env, objects: list[str], xy_jitter: float, seed: int, settle_steps: int = 20) -> dict[str, list[float]]:
    """Shifts each listed object's free joint by up to `xy_jitter` m in x and y, then lets physics settle.

    The arms hold their reset pose during settling. Returns the objects' positions after settling.
    """
    rng = np.random.default_rng(seed)
    model, data = env.model, env.data
    for name in objects:
        joint = FREE_JOINT.get(name)
        if joint is None:
            continue
        adr = model.joint(joint).qposadr[0]
        dadr = model.joint(joint).dofadr[0]
        if xy_jitter > 0:
            data.qpos[adr : adr + 2] += rng.uniform(-xy_jitter, xy_jitter, size=2)
        data.qvel[dadr : dadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    state = env.state()
    for _ in range(settle_steps):
        env.step(state["left"], state["right"])
    return {name: env.data.body(name).xpos.round(4).tolist() for name in (*FREE_JOINT, "bowl", "plate")}
