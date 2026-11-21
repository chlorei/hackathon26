"""safety: tightens (never loosens) each chunk: a joint speed cap and a workspace box for the tool points."""

from __future__ import annotations

import mujoco
import numpy as np

from ..plugin import Context, Plugin

SIDES = ("left", "right")


class Safety(Plugin):
    """Rows that move a joint faster than max_speed are slowed; a row that takes a tool point out of the box ends
    the chunk there (the arm holds until the next model call)."""

    name = "safety"
    group = "safety"
    description = "Caps joint speed and keeps the tool points inside a workspace box"
    defaults = {
        "max_speed_rad_s": 3.0,
        "box_min": [-0.6, -0.65, 0.745],  # x, y, z in metres (world): just above the table top
        "box_max": [0.6, 0.40, 1.30],
        "check_box": True,
        "warn_every_s": 2.0,
    }

    def on_episode_start(self, ctx: Context) -> None:
        self.scratch = mujoco.MjData(ctx.env.model)
        m = ctx.env.model
        self.qadr = {s: [m.joint(f"{s}_joint{i}").qposadr[0] for i in range(1, 7)] for s in SIDES}
        self.site = {s: m.site(f"{s}_tcp").id for s in SIDES}
        self.clipped = 0
        self.cut = 0
        self.last_warn: dict[str, float] = {}
        self.quiet: dict[str, int] = {}

    def _warn(self, ctx: Context, kind: str, message: str, **data) -> None:
        """At most one warning of a kind every warn_every_s; the ones in between are counted into the next."""
        if ctx.t - self.last_warn.get(kind, -1e9) < float(self.params["warn_every_s"]):
            self.quiet[kind] = self.quiet.get(kind, 0) + 1
            return
        self.last_warn[kind] = ctx.t
        extra = self.quiet.pop(kind, 0)
        ctx.emit("warn", message + (f" (+{extra} more since last warning)" if extra else ""), **data)

    def after_model(self, ctx: Context) -> None:
        if ctx.actions is None:
            return
        left, right = (np.array(a, dtype=float, copy=True) for a in ctx.actions)
        rows = {"left": left, "right": right}
        step_cap = float(self.params["max_speed_rad_s"]) * 0.05
        slowed = 0
        for s in SIDES:
            prev = ctx.state[s][:6].copy()
            for i in range(len(rows[s])):
                delta = rows[s][i, :6] - prev
                clipped = np.clip(delta, -step_cap, step_cap)
                if not np.allclose(clipped, delta):
                    slowed += 1
                    rows[s][i, :6] = prev + clipped
                prev = rows[s][i, :6]
        keep = len(left)
        if self.params["check_box"]:
            m = ctx.env.model
            lo, hi = np.asarray(self.params["box_min"]), np.asarray(self.params["box_max"])
            mujoco.mj_copyData(self.scratch, m, ctx.env.data)
            mujoco.mj_kinematics(m, self.scratch)

            def outside_by(pos):
                return float(np.sum(np.maximum(lo - pos, 0) + np.maximum(pos - hi, 0)))

            # A hand already outside may move back in: only rows that go farther out are cut.
            allowed = {s: outside_by(self.scratch.site_xpos[self.site[s]]) + 1e-3 for s in SIDES}
            for i in range(len(left)):
                for s in SIDES:
                    self.scratch.qpos[self.qadr[s]] = rows[s][i, :6]
                mujoco.mj_kinematics(m, self.scratch)
                outside = [s for s in SIDES if outside_by(self.scratch.site_xpos[self.site[s]]) > allowed[s]]
                if outside:
                    keep = i
                    pos = self.scratch.site_xpos[self.site[outside[0]]].round(3).tolist()
                    self._warn(ctx, "box", f"row {i + 1} would take the {outside[0]} hand out of the workspace "
                                           f"{pos}; chunk cut to {i} rows", side=outside[0], position=pos)
                    self.cut += 1
                    break
                for s in SIDES:
                    allowed[s] = min(allowed[s], outside_by(self.scratch.site_xpos[self.site[s]]) + 1e-3)
        if slowed:
            self.clipped += slowed
            self._warn(ctx, "speed", f"slowed {slowed} rows to {self.params['max_speed_rad_s']} rad/s", rows=slowed)
        ctx.actions = (rows["left"][:keep], rows["right"][:keep])

    def on_episode_end(self, ctx: Context) -> None:
        if self.clipped or self.cut:
            ctx.result["safety"] = {"rows_slowed": self.clipped, "chunks_cut": self.cut}
