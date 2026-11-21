"""Task plugins. A task = scene randomisation + prompt + success judge + failure labels.

To add your own: subclass or copy PickPlace (or write any class with the Task methods of core.py) and add an
instance to TASKS. `dreamgym tasks` lists them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from . import scene


@dataclass
class PickPlace:
    """Put one object into the bowl or onto the plate.

    Randomisation, per episode: the object (and, with `shuffle_others`, the other fruit) moved up to `jitter_m`
    from its start and turned up to `yaw_rad`; lights scaled by 1 +- `lighting`. jitter 0, yaw 0, lighting 0 is
    the stock scene.
    """

    name: str
    obj: str
    target: str  # "bowl" or "plate"
    prompt: str
    jitter_m: float = 0.03
    yaw_rad: float = 0.4
    shuffle_others: bool = True
    lighting: float = 0.15

    # --- scene ---------------------------------------------------------------------------------------------------

    def reset(self, env, rng: np.random.Generator) -> None:
        movable = [self.obj] + ([o for o in scene.FRUITS if o != self.obj] if self.shuffle_others else [])
        for _ in range(100):
            layout = {
                o: scene.start_xy(env, o) + rng.uniform(-self.jitter_m, self.jitter_m, size=2) for o in movable
            }
            if self._layout_ok(env, layout):
                break
        else:  # no layout found: keep the stock one rather than overlap objects
            layout = {o: scene.start_xy(env, o) for o in movable}
        for o, xy in layout.items():
            scene.place(env, o, xy, yaw=float(rng.uniform(-self.yaw_rad, self.yaw_rad)))
        scene.scale_lights(env, 1.0 + float(rng.uniform(-self.lighting, self.lighting)))

    def _layout_ok(self, env, layout: dict[str, np.ndarray]) -> bool:
        fixed = {o: scene.pos(env, o)[:2] for o in scene.FREE_JOINTS if o not in layout}
        points = list(layout.values()) + list(fixed.values())
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if np.linalg.norm(points[i] - points[j]) < 0.07:
                    return False
        bowl, plate = scene.pos(env, "bowl")[:2], scene.pos(env, "plate")[:2]
        return all(np.linalg.norm(xy - bowl) > 0.11 and np.linalg.norm(xy - plate) > 0.10 for xy in layout.values())

    # --- judge ---------------------------------------------------------------------------------------------------

    def placed(self, env) -> bool:
        return scene.in_bowl(env, self.obj) if self.target == "bowl" else scene.on_plate(env, self.obj)

    def success(self, env) -> bool:
        return self.placed(env) and scene.at_rest(env, self.obj) and scene.released(env, self.obj)

    def failure_reason(self, env, start: dict[str, np.ndarray]) -> str:
        """Why it failed, in dreamkit's shared codes (dreamkit.reasons): the same words as History and the monitor."""
        if scene.off_table(env, self.obj):
            return "fell_off_table"
        if self.placed(env):
            return "still_held" if not scene.released(env, self.obj) else "timeout"  # in place, still rolling
        if not scene.released(env, self.obj):
            return "still_held"
        moved = float(np.linalg.norm(scene.pos(env, self.obj) - start[self.obj]))
        if moved < 0.015:
            return "not_reached"
        return "released_off_target"


## How each scene thing may be named in a prompt (the sim's plate is blue-glazed).
NAMES = {"apple": ["apple", "red apple"], "banana": ["banana", "yellow banana"], "orange": ["orange", "orange fruit"],
         "bowl": ["bowl", "ceramic bowl"], "plate": ["plate", "blue plate"]}


def prompt_variants(task: "PickPlace", limit: int = 8) -> list[str]:
    """Paraphrases of a task's prompt for the Prompt Tuner; the task's own prompt comes first (the baseline)."""
    prep = "into" if task.target == "bowl" else "onto"
    o, t = NAMES.get(task.obj, [task.obj]), NAMES.get(task.target, [task.target])
    candidates = [
        task.prompt,
        f"pick up the {o[0]} and put it {prep} the {t[0]}",
        f"place the {o[-1]} {prep} the {t[-1]}",
        f"move the {o[0]} {prep} the {t[0]}",
        f"grasp the {o[-1]}, lift it, and release it {prep} the {t[0]}",
        f"put the {o[-1]} {prep} the {t[-1]}",
        f"{task.obj} {prep} {task.target}",
        f"carefully pick up the {o[0]} and gently place it {prep} the {t[0]}",
    ]
    seen, out = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:limit]


TASKS: dict[str, PickPlace] = {
    t.name: t
    for t in (
        PickPlace("apple_bowl", "apple", "bowl", "put apple into the bowl"),
        PickPlace("orange_bowl", "orange", "bowl", "put orange into the bowl"),
        PickPlace("banana_plate", "banana", "plate", "put banana on the plate"),
    )
}


def get_task(name: str, **overrides) -> PickPlace:
    """A registered task, with fields overridden (`jitter_m=0` for the stock layout, `prompt=...`)."""
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {', '.join(TASKS)}")
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return dataclasses.replace(TASKS[name], **overrides)
