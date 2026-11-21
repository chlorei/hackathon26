"""Bad on purpose: settings without descriptions or units, a docstring that is not a sentence."""

from pydantic import BaseModel

from dreamkit import Manifest, Plugin


class Params(BaseModel):
    window: float = 1.0  # no description, no unit, no suffix
    mode: str = "fast"  # no description


class NoDescriptions(Plugin):
    """my plugin"""

    name = "no-descriptions"
    Params = Params
    manifest = Manifest(group="diagnosis", icon="Plug", capability="observe", hooks=["on_step"])

    def on_step(self, ctx):
        pass
