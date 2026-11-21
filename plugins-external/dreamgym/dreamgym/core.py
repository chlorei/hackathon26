"""The plugin contracts of DreamGym: a Task says what to do and how success is judged, a Policy says who does it.

Everything else (Arena, Compare, Recorder) is built from these two, so a new task or a new model is one class.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

## One control step of the sim: YamEnv.step() runs 50 ms of physics.
STEP_S = 0.05
## Dream-1 bills per robot-hour of actions; one get_action answers 1.6 s of actions, so an hour is 2250 calls.
CALLS_PER_HOUR = 2250


@dataclass
class Context:
    """What an episode carries between the runner and the policy.

    - windows: per sim camera (`top`, `left`, `right`), the base64 JPEG frames the model sees, oldest first.
    - steps: control steps run so far (50 ms each, simulated time).
    - calls: billed model calls so far.
    """

    windows: dict[str, deque]
    steps: int = 0
    calls: int = 0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)


@dataclass
class Chunk:
    """Rows to execute, one per control step: each `(rows, 7)`, six joints in radians then gripper 0..1."""

    left: np.ndarray
    right: np.ndarray


@runtime_checkable
class Task(Protocol):
    """A task plugin: the scene setup, the prompt and the success judge."""

    name: str
    prompt: str

    def reset(self, env: Any, rng: np.random.Generator) -> None:
        """Arrange the scene for one episode (called after env.reset()); randomise with `rng` only."""

    def success(self, env: Any) -> bool:
        """Whether the goal holds right now; the runner asks every step and needs it to hold for a while."""

    def failure_reason(self, env: Any, start: dict[str, np.ndarray]) -> str:
        """Why an episode that did not succeed failed, one short label (`untouched`, `misplaced`, ...)."""


@runtime_checkable
class Policy(Protocol):
    """A policy plugin: Dream-1, a fine-tune, a scripted expert, ..."""

    name: str
    price_per_hour: float

    def begin(self, env: Any, task: Task, prompt: str, ctx: Context) -> None:
        """Start an episode; the frame windows are already full."""

    def next_chunk(self, env: Any, ctx: Context) -> Chunk:
        """The next rows to execute; may block on the network. Count billed calls in `ctx.calls`."""

    def on_frame(self, env: Any, ctx: Context) -> None:
        """A new camera frame joined the windows (every 0.2 s of sim time)."""

    def close(self) -> None:
        """Release connections."""


@dataclass
class EpisodeResult:
    """One episode, as written to `episodes.jsonl`."""

    seed: int
    task: str
    policy: str
    prompt: str
    success: bool
    reason: str  # "success" or a shared failure code (dreamkit.reasons)
    sim_seconds: float
    seconds_to_success: float | None
    calls: int
    cost_usd: float
    wall_seconds: float
    error: str | None = None
    gif: str | None = None
    run_id: str | None = None  # the dreamkit run (History)
    episode: int | None = None  # the attempt in it
    recoveries: int = 0  # how many times dreamkit's recovery stepped in

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
