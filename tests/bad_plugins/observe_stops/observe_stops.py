"""Bad on purpose: an observe plugin that tries to stop the robot (and to write the verdict)."""

from pydantic import BaseModel

from dreamkit import Manifest, Plugin


class Params(BaseModel):
    """No settings."""


class ObserveStops(Plugin):
    """Tries to stop the robot although it may only observe."""

    name = "observe-stops"
    Params = Params
    manifest = Manifest(group="diagnosis", icon="Plug", capability="observe", hooks=["on_step"], step_rate_hz=20)

    def on_step(self, ctx):
        if ctx.t > 2.0:
            ctx.stop("dropped", "observe plugins may not stop the robot")
