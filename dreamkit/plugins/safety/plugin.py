"""safety: tightens each chunk, never loosens it — a joint speed cap and a workspace box (PLUGIN_STANDARD.md).

Rows that move a joint faster than the cap are slowed; a row that takes a tool point out of the box ends the chunk
there and the arm holds until the next model call."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from dreamkit.plugin import Plugin, PluginContext
from dreamkit.standard import Manifest

SIDES = ("left", "right")
STEP_S = 0.05


class Params(BaseModel):
    max_speed_rad_s: float = Field(3.0, gt=0, le=20, description="No joint moves faster than this",
                                   json_schema_extra={"unit": "rad/s"})
    box_min_m: list[float] = Field([-0.6, -0.65, 0.745], description="Workspace box corner (x, y, z), world frame",
                                   json_schema_extra={"unit": "m"})
    box_max_m: list[float] = Field([0.6, 0.40, 1.30], description="Opposite workspace box corner (x, y, z)",
                                   json_schema_extra={"unit": "m"})
    check_box: bool = Field(True, description="Cut a chunk where a tool point would leave the box")
    warn_every_s: float = Field(2.0, gt=0, le=60, description="At most one warning of a kind this often",
                                json_schema_extra={"unit": "s"})


class Safety(Plugin):
    """Caps joint speed and keeps both tool points inside a workspace box."""

    name = "safety"
    Params = Params
    manifest = Manifest(group="safety", icon="ShieldCheck", capability="modify_actions",
                        hooks=["on_episode_start", "after_model", "on_episode_end"])

    def on_episode_start(self, ctx: PluginContext) -> None:
        self.slowed = 0
        self.cut = 0
        self.last_warn: dict[str, float] = {}

    def _warn(self, ctx: PluginContext, kind: str, message: str, **data) -> None:
        if ctx.t - self.last_warn.get(kind, -1e9) >= self.params.warn_every_s:
            self.last_warn[kind] = ctx.t
            ctx.emit("warn", message, **data)

    def after_model(self, ctx: PluginContext) -> None:
        if ctx.actions is None or not ctx.state:
            return
        left, right = ctx.actions
        rows = {"left": left, "right": right}
        cap = self.params.max_speed_rad_s * STEP_S
        slowed = 0
        for s in SIDES:
            prev = np.asarray(ctx.state[s][:6], dtype=float)
            for i in range(len(rows[s])):
                delta = rows[s][i, :6] - prev
                clipped = np.clip(delta, -cap, cap)
                if not np.allclose(clipped, delta):
                    slowed += 1
                    rows[s][i, :6] = prev + clipped
                prev = rows[s][i, :6]
        keep = len(left)
        if self.params.check_box and ctx.world.can_fk:
            lo, hi = np.asarray(self.params.box_min_m), np.asarray(self.params.box_max_m)

            def outside(pos):
                return float(np.sum(np.maximum(lo - pos, 0) + np.maximum(pos - hi, 0)))

            now = ctx.world.fk_tcp({s: np.asarray(ctx.state[s][:6]) for s in SIDES})
            allowed = {s: outside(now[s]) + 1e-3 for s in SIDES}  # a hand already outside may move back in
            for i in range(len(left)):
                tcp = ctx.world.fk_tcp({s: rows[s][i] for s in SIDES})
                bad = [s for s in SIDES if outside(tcp[s]) > allowed[s]]
                if bad:
                    keep = i
                    self.cut += 1
                    self._warn(ctx, "box", f"row {i + 1} would take the {bad[0]} hand out of the workspace, chunk cut "
                                           f"to {i} rows", side=bad[0])
                    break
                allowed = {s: min(allowed[s], outside(tcp[s]) + 1e-3) for s in SIDES}
        if slowed:
            self.slowed += slowed
            self._warn(ctx, "speed", f"slowed {slowed} rows to {self.params.max_speed_rad_s} rad/s", rows=slowed)
        if slowed or keep < len(left):
            ctx.actions = (rows["left"][:keep], rows["right"][:keep])

    def on_episode_end(self, ctx: PluginContext) -> None:
        if self.slowed or self.cut:
            ctx.result["safety"] = {"rows_slowed": self.slowed, "chunks_cut": self.cut}
            ctx.ui.badge(f"safety {self.slowed + self.cut}×", "warn")
