"""Dream-1 (base, flash or your fine-tune) through DreamAPI, speaking exactly what the YAM sim example speaks."""

from __future__ import annotations

from ..core import Chunk, Context
from .. import sim


class DreamPolicy:
    """Calls DreamAPI with the sim's cameras and joints; executes the chunk it answers.

    Uses dream_control's own helpers (observation keys, frame windows, chunk validation), so behaviour matches
    `dreamcli example yam-mujoco`. One DreamClient session is kept for all episodes; `reset_episode()` separates them.
    """

    def __init__(self, checkpoint: str, *, action_limit: int = 48, price_per_hour: float = 3.0, label: str | None = None):
        self.checkpoint = checkpoint
        self.name = label or checkpoint
        self.action_limit = action_limit
        self.price_per_hour = price_per_hour
        self._client = None
        self._station = False
        self._prompt = ""

    # The model id goes to dream_control's helpers as `model`.
    def _open(self, station: bool) -> None:
        _, dc, _ = sim.load()
        key = dc.settings().get("DREAM_API_KEY")
        if not key:
            raise RuntimeError("No DREAM_API_KEY: export it, or write ~/.dreamlabs/credentials (platform Setup API page)")
        if self._client is not None:
            self._client.close()
        self._client = dc.open_policy(dc.policy_config(self.checkpoint, None, station), key)
        self._station = station

    def _feed_windows(self, env, ctx: Context) -> None:
        _, dc, _ = sim.load()
        dc.feed(self._client, env, ctx.windows, self._prompt, self.checkpoint, ctx.steps, every=0, station=self._station)

    def begin(self, env, task, prompt: str, ctx: Context) -> None:
        _, dc, _ = sim.load()
        self._prompt = prompt
        if self._client is None:
            self._open(self.checkpoint in dc.POSE_MODELS)
        else:
            self._client.reset_episode()
        self._feed_windows(env, ctx)

    def next_chunk(self, env, ctx: Context) -> Chunk:
        _, dc, _ = sim.load()
        try:
            actions = self._client.get_action()
        except Exception as error:  # a model that reads hand poses: reopen on the station keys, as the sim does
            if self._station or not dc.needs_poses(error):
                raise
            self._open(True)
            self._feed_windows(env, ctx)
            actions = self._client.get_action()
        ctx.calls += 1
        left, right = dc.validate_chunk(env, actions, self.action_limit, verbose=False)
        return Chunk(left, right)

    def on_frame(self, env, ctx: Context) -> None:
        _, dc, _ = sim.load()
        dc.feed(self._client, env, ctx.windows, self._prompt, self.checkpoint, ctx.steps, station=self._station)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
