"""Bad on purpose: a modify_actions plugin that makes the arm move faster than the model asked."""

import numpy as np
from pydantic import BaseModel

from dreamkit import Manifest, Plugin


class Params(BaseModel):
    """No settings."""


class ModifySpeedsUp(Plugin):
    """Triples every joint step, which the standard forbids."""

    name = "modify-speeds-up"
    Params = Params
    manifest = Manifest(group="safety", icon="Plug", capability="modify_actions", hooks=["after_model"])

    def after_model(self, ctx):
        if ctx.actions is None or not ctx.state:
            return
        out = []
        for i, side in enumerate(("left", "right")):
            rows = ctx.actions[i].copy()
            start = np.asarray(ctx.state[side][:6], dtype=float)
            rows[:, :6] = start + 3.0 * (rows[:, :6] - start)  # faster, maybe past the joint limits
            out.append(rows)
        ctx.actions = (out[0], out[1])
