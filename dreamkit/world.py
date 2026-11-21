"""ctx.world (PLUGIN_STANDARD.md §6): every plugin measures the world the same way.

World wraps Dream Labs' simulator (YamEnv: MuJoCo model + data + state()); ReplayWorld plays the same answers back
from a recorded fixture, so plugins run unchanged in tests without the simulator."""

from __future__ import annotations

from typing import Any

import numpy as np

TABLE_TOP_Z = 0.76
SIDES = ("left", "right")


class World:
    """The live world of a YamEnv-like env. `available` is False for anything without MuJoCo (a real robot)."""

    def __init__(self, env: Any):
        self.env = env
        self.available = env is not None and hasattr(env, "data") and hasattr(env, "model")
        self.table_z = TABLE_TOP_Z
        self._bodies: list[str] | None = None
        self._force = np.zeros(6)

    # -- positions (ids cached: name lookups are the slow part) ------------------------------------------------
    def _body_id(self, body: str) -> int:
        cache = self.__dict__.setdefault("_bid", {})
        if body not in cache:
            cache[body] = self.env.model.body(body).id
        return cache[body]

    def _site_id(self, site: str) -> int:
        cache = self.__dict__.setdefault("_sid", {})
        if site not in cache:
            cache[site] = self.env.model.site(site).id
        return cache[site]

    def pos(self, body: str) -> np.ndarray:
        """A body's position, metres, world frame (Z up)."""
        return self.env.data.xpos[self._body_id(body)].copy()

    def tcp(self, side: str) -> np.ndarray:
        """The tool point of a hand ("left" / "right")."""
        return self.env.data.site_xpos[self._site_id(f"{side}_tcp")].copy()

    def grip(self, side: str) -> float:
        """Measured gripper opening, 0 closed … 1 open."""
        return float(self.env.state()[side][6])

    def has(self, body: str) -> bool:
        try:
            self._body_id(body)
            return True
        except Exception:
            return False

    # -- contacts ----------------------------------------------------------------------------------------------------
    def bodies(self) -> list[str]:
        if self._bodies is None:
            m = self.env.model
            self._bodies = [m.body(i).name for i in range(m.nbody)]
        return self._bodies

    def contacts(self, with_force: bool = False, prefixes: tuple[str, ...] | None = None) -> list[tuple[Any, ...]]:
        """Body pairs in contact now: [(a, b)], or [(a, b, force_n)] with the normal force in newtons.

        `prefixes` keeps only contacts where a body name starts with one of them (and computes forces only for
        those): contacts(with_force=True, prefixes=("left_", "right_")) is every robot contact."""
        import mujoco  # noqa: PLC0415

        m, d = self.env.model, self.env.data
        n = d.ncon
        if n == 0:
            return []
        names = self.bodies()
        con = d.contact
        b1 = m.geom_bodyid[con.geom1[:n]]
        b2 = m.geom_bodyid[con.geom2[:n]]
        if prefixes is not None:
            robot = self.__dict__.get("_robot_mask")
            if robot is None or self.__dict__.get("_robot_prefixes") != prefixes:
                robot = np.array([nm.startswith(prefixes) for nm in names])
                self._robot_mask, self._robot_prefixes = robot, prefixes
            keep = np.nonzero(robot[b1] | robot[b2])[0]
        else:
            keep = range(n)
        out: list[tuple[Any, ...]] = []
        for i in keep:
            a, b = names[b1[i]], names[b2[i]]
            if with_force:
                mujoco.mj_contactForce(m, d, int(i), self._force)
                out.append((a, b, abs(float(self._force[0]))))
            else:
                out.append((a, b))
        return out

    # -- kinematics (safety checks) ------------------------------------------------------------------------------
    @property
    def can_fk(self) -> bool:
        return self.available

    def fk_tcp(self, rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Where each tool point would be with these six joints per side (a scratch copy; the sim is untouched)."""
        import mujoco  # noqa: PLC0415

        m = self.env.model
        if not hasattr(self, "_scratch"):
            self._scratch = mujoco.MjData(m)
            self._qadr = {s: [m.joint(f"{s}_joint{i}").qposadr[0] for i in range(1, 7)] for s in SIDES}
            self._site = {s: m.site(f"{s}_tcp").id for s in SIDES}
        mujoco.mj_copyData(self._scratch, m, self.env.data)
        for s, q in rows.items():
            self._scratch.qpos[self._qadr[s]] = q[:6]
        mujoco.mj_kinematics(m, self._scratch)
        return {s: self._scratch.site_xpos[self._site[s]].copy() for s in SIDES}

    def joint_limits(self, side: str) -> np.ndarray:
        """(6, 2) joint ranges in radians."""
        m = self.env.model
        return np.array([m.joint(f"{side}_joint{i}").range for i in range(1, 7)])


class ReplayWorld(World):
    """The world as recorded in a fixture, one step at a time."""

    def __init__(self, frames: dict[str, Any], limits: dict[str, np.ndarray] | None = None):
        super().__init__(None)
        self.available = True
        self._rec = frames
        self._i = 0
        self._limits = limits or {}

    def at(self, i: int) -> None:
        self._i = i

    def pos(self, body: str) -> np.ndarray:
        return np.array(self._rec["pos"][body][self._i], dtype=float)

    def tcp(self, side: str) -> np.ndarray:
        return np.array(self._rec["tcp"][side][self._i], dtype=float)

    def grip(self, side: str) -> float:
        return float(self._rec["grip"][side][self._i])

    def has(self, body: str) -> bool:
        return body in self._rec["pos"]

    def bodies(self) -> list[str]:
        return list(self._rec.get("bodies", []))

    def contacts(self, with_force: bool = False, prefixes: tuple[str, ...] | None = None) -> list[tuple[Any, ...]]:
        rows = self._rec["contacts"][self._i]
        if prefixes is not None:
            rows = [c for c in rows if c[0].startswith(prefixes) or c[1].startswith(prefixes)]
        return [tuple(c) if with_force else (c[0], c[1]) for c in rows]

    @property
    def can_fk(self) -> bool:
        return False

    def joint_limits(self, side: str) -> np.ndarray:
        return np.asarray(self._limits.get(side, [[-np.pi, np.pi]] * 6))
