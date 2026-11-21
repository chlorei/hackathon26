"""Joint-limited IK for the sim's YAM arms.

Adapted from Dream Labs' examples/yam_mujoco/mock_pickups.py (`solve`), Apache License 2.0, because the
installed `dreamlabs` wheel ships the example without that script. Changed: kinematics only (no collision pass),
and optional looser tolerances.
"""

from __future__ import annotations

import numpy as np


def solve(env, side, target, rotation, seed, accept_pos=None, accept_ang=None):
    """Solve a joint-limited grasp pose using scratch simulation data.
    @param env Simulation environment (YamEnv).
    @param side Arm name, "left" or "right".
    @param target Desired tool-centre-point position, world frame.
    @param rotation Desired tool rotation matrix, 3x3.
    @param seed Initial joint solution.
    @param accept_pos, accept_ang Looser tolerances (m, rad): if no exact solution is found, the best one within
        them is returned instead of raising (e.g. the gripper a little tilted over a bowl).
    @return The six joint angles that reach the pose.
    """
    import mujoco

    m = env.model
    d = mujoco.MjData(m)
    mujoco.mj_copyData(d, m, env.data)
    ids = [m.joint(f"{side}_joint{i}").id for i in range(1, 7)]
    qa = m.jnt_qposadr[ids]
    va = m.jnt_dofadr[ids]
    limits = m.jnt_range[ids]
    site = m.site(f"{side}_tcp").id
    rng = np.random.default_rng(42)
    best = (1e9, None)
    best_fit = (1e9, 1e9)
    for initial in [seed] + [rng.uniform(limits[:, 0], limits[:, 1]) for _ in range(14)]:
        q = np.array(initial).copy()
        for _ in range(180):
            d.qpos[qa] = q
            # Kinematics only: IK needs poses and Jacobians, not contacts. mj_forward would also run collision
            # detection, and the random restarts put arms through the table: thousands of contacts overflow
            # MuJoCo's arena ("mj_stackAlloc: out of memory").
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            current = d.site_xmat[site].reshape(3, 3)
            rot = rotation @ current.T
            angle = np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))
            axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
            error_r = axis * angle / max(2 * np.sin(angle), 1e-8)
            error = target - d.site_xpos[site]
            score = np.linalg.norm(error) + 0.1 * angle
            if score < best[0]:
                best = (score, q.copy())
                best_fit = (float(np.linalg.norm(error)), float(angle))
            if np.linalg.norm(error) < 0.004 and angle < 0.06:
                return q
            jp = np.zeros((3, m.nv))
            jr = jp.copy()
            mujoco.mj_jacSite(m, d, jp, jr, site)
            J = np.vstack([jp[:, va], jr[:, va] * 0.15])
            e = np.r_[error, error_r * 0.15]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.0001 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.12, 0.12), limits[:, 0], limits[:, 1])
    if accept_pos is not None and best_fit[0] < accept_pos and best_fit[1] < (accept_ang or 0.06):
        return best[1]
    raise ValueError(f"IK failed {side} {target}: score {best[0]:.4f} (pos {best_fit[0]:.3f} m, angle {best_fit[1]:.2f} rad)")


def step_towards(env, side, q, target, rotation, *, iters: int = 4, max_step: float = 0.06, scratch=None):
    """A few damped-least-squares steps from `q` toward a tool pose, bounded per call: smooth teleoperation.

    Unlike solve() it never jumps to a far solution: it moves the arm at most `max_step` rad per joint per call,
    so a target the arm cannot reach just leaves the arm as close as it gets.
    @param scratch An MjData to reuse between calls (pass the same one every tick).
    @return The new six joint angles.
    """
    import mujoco

    m = env.model
    d = scratch if scratch is not None else mujoco.MjData(m)
    mujoco.mj_copyData(d, m, env.data)
    ids = [m.joint(f"{side}_joint{i}").id for i in range(1, 7)]
    qa, va, limits = m.jnt_qposadr[ids], m.jnt_dofadr[ids], m.jnt_range[ids]
    site = m.site(f"{side}_tcp").id
    q0 = np.asarray(q, dtype=float).copy()
    q = q0.copy()
    for _ in range(iters):
        d.qpos[qa] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        current = d.site_xmat[site].reshape(3, 3)
        rot = rotation @ current.T
        angle = np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))
        axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
        error_r = axis * angle / max(2 * np.sin(angle), 1e-8)
        error = np.asarray(target) - d.site_xpos[site]
        jp = np.zeros((3, m.nv))
        jr = jp.copy()
        mujoco.mj_jacSite(m, d, jp, jr, site)
        J = np.vstack([jp[:, va], jr[:, va] * 0.15])
        e = np.r_[error, error_r * 0.15]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.0005 * np.eye(6), e)
        q = np.clip(q + np.clip(dq, -0.05, 0.05), limits[:, 0], limits[:, 1])
    return np.clip(q, q0 - max_step, q0 + max_step)
