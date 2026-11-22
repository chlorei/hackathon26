"""A scripted expert for pick-and-place: it knows where the object is and plans joints with IK.

It is the demonstration generator for fine-tuning (no hand teleoperation needed) and a sanity baseline for the
judge: if the scripted expert fails a seed, the seed (or the success check) is suspect, not Dream-1.
Built on mock_pickups.solve(), Dream Labs' own joint-limited IK for the sim.
"""

from __future__ import annotations

import numpy as np

from .. import scene, sim
from ..core import Chunk, Context

## Grasp yaw per object at its stock orientation (mock_pickups grips the banana across).
BASE_YAW = {"banana": np.pi / 2}
## Round fruit can be gripped at any yaw, so each arm uses the yaw its wrist reaches best over the table
## (worked out from the YAM kinematics: the right arm cannot hold a gripper-down pose at yaw 0 near the centre).
ROUND = {"apple", "orange"}
SIDE_YAW = {"left": 0.0, "right": np.pi / 2}
## Grasp height of the tool centre point above the object's centre (mock_pickups' values).
GRASP_DZ = {"orange": -0.009}
## Release height above the target's origin (bowl rim is ~0.058 m high).
RELEASE_DZ = {"bowl": 0.11, "plate": 0.08}
## Heights tried, in order, above the release point for the carry and the retreat. A gripper-down pose over the
## bowl is only reachable up to ~0.14 m above it, so the bowl needs the lower ones.
CARRY_DZ = (0.05, 0.03, 0.015)
RETREAT_DZ = (0.10, 0.05, 0.03)
## Lift heights tried after the grasp.
LIFT_DZ = (0.15, 0.12, 0.10)
## Heights above the object to come down from. 0.14 first (as before); an arm reaching far across the table (the
## apple to the plate on the other side) gets there only lower.
APPROACH_DZ = (0.14, 0.10, 0.07)


def _smooth(n: int) -> np.ndarray:
    s = np.linspace(0, 1, n)
    return s * s * (3 - 2 * s)


class ScriptedPickPlace:
    """Approach, grasp, lift, carry, release, retreat; then hold. Zero cost, no network."""

    name = "scripted"
    price_per_hour = 0.0

    def __init__(self, noise_m: float = 0.0, chunk_rows: int = 48):
        self.noise_m = noise_m
        self.chunk_rows = chunk_rows
        self._rows: list[tuple[np.ndarray, np.ndarray]] = []
        self._i = 0

    def begin(self, env, task, prompt: str, ctx: Context) -> None:
        yam_env, _, mock = sim.load()
        home = yam_env.HOME
        obj, target = task.obj, task.target
        p = scene.pos(env, obj)
        goal = scene.pos(env, target)
        near = "left" if p[0] > 0 else "right"
        shift = [ctx.rng.normal(0, self.noise_m, 2) if self.noise_m else np.zeros(2) for _ in range(2)]
        errors = []
        # The nearer arm first; if it cannot reach every pose (e.g. an object near the middle), the other arm.
        for side in (near, "right" if near == "left" else "left"):
            try:
                rows = self._plan(env, mock, side, obj, target, p, goal, shift)
            except ValueError as error:
                errors.append(str(error))
                continue
            hold = home.copy()
            self._rows = [(r, hold) if side == "left" else (hold, r) for r in rows]
            self._i = 0
            return
        raise ValueError("; ".join(errors))

    def _plan(self, env, mock, side, obj, target, p, goal, shift) -> list[np.ndarray]:
        """Joint rows for one arm: above, down, close, lift, carry, lower, open, retreat. Raises if unreachable."""
        if obj in ROUND:
            yaw = SIDE_YAW[side]
        else:
            yaw = BASE_YAW.get(obj, 0.0) + getattr(env, "dreamgym_yaw", {}).get(obj, 0.0)
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]])
        q = np.asarray(env.state()[side][:6], dtype=float)
        rows: list[np.ndarray] = []

        def reach(point, flexible):
            """Exact pose first; for carrying, fall back to a turned, lower or slightly tilted gripper."""
            nonlocal rotation
            point = np.asarray(point, dtype=float)
            try:
                return mock.solve(env, side, point, rotation, q)
            except ValueError:
                if not flexible:
                    raise
            base = np.arctan2(rotation[1, 0], rotation[0, 0])
            for accept in ((None, None), (0.012, 0.35)):
                for dz in (0.0, -0.02, 0.03, -0.04):
                    for dyaw in (np.pi / 2, -np.pi / 2, np.pi, np.pi / 4, -np.pi / 4, 0.0):
                        c2, s2 = np.cos(base + dyaw), np.sin(base + dyaw)
                        rot = np.array([[c2, s2, 0], [s2, -c2, 0], [0, 0, -1.0]])
                        try:
                            end = mock.solve(env, side, point + [0, 0, dz], rot, q,
                                             accept_pos=accept[0], accept_ang=accept[1])
                        except Exception:  # unreachable (ValueError) or a MuJoCo error on an odd pose
                            continue
                        rotation = rot
                        return end
            raise ValueError(f"IK failed {side} {point}: no turned, lowered or tilted pose either")

        def move(point, opening, steps, flexible=False):
            nonlocal q
            end = reach(point, flexible)
            for a in _smooth(steps):
                rows.append(np.r_[q + (end - q) * a, opening])
            q = end

        def move_first(points, opening, steps, flexible=False):
            """The first of several heights that is reachable without the fallback; else the fallback on the first."""
            nonlocal q
            for point in points:
                try:
                    return move(point, opening, steps)
                except ValueError:
                    continue
            return move(points[0], opening, steps, flexible=True) if flexible else move(points[-1], opening, steps)

        def grip(start, stop, steps):
            for a in _smooth(steps):
                rows.append(np.r_[q, start + (stop - start) * a])

        dz = GRASP_DZ.get(obj, 0.003)
        pick = p + np.r_[shift[0], 0.0]
        # round fruit can be gripped at any yaw, so a far reach may turn the gripper (the flexible fallback)
        round_fruit = obj in ROUND
        move_first([pick + [0, 0, h] for h in APPROACH_DZ], 1.0, 50, flexible=round_fruit)  # above the object
        move(p + [0, 0, dz], 1.0, 40, flexible=round_fruit)  # down
        grip(1.0, 0.0, 25)  # close
        move_first([p + [0, 0, h] for h in LIFT_DZ], 0.0, 40, flexible=round_fruit)  # lift
        drop = goal + [0, 0, RELEASE_DZ[target]] + np.r_[shift[1], 0.0]
        move_first([drop + [0, 0, h] for h in CARRY_DZ], 0.0, 60, flexible=True)  # carry
        move(drop, 0.0, 25, flexible=True)  # lower
        grip(0.0, 1.0, 20)  # release
        move_first([drop + [0, 0, h] for h in RETREAT_DZ], 1.0, 30, flexible=True)  # retreat
        return rows

    def next_chunk(self, env, ctx: Context) -> Chunk:
        rows = self._rows[self._i : self._i + self.chunk_rows]
        self._i += len(rows)
        if not rows:  # plan done: hold the last pose
            rows = [self._rows[-1]] * self.chunk_rows
        return Chunk(np.array([l for l, _ in rows]), np.array([r for _, r in rows]))

    def on_frame(self, env, ctx: Context) -> None:
        pass

    def close(self) -> None:
        pass
