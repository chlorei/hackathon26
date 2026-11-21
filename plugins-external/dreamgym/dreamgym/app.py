"""DreamGym as a dreamkit app plugin (PLUGIN_STANDARD.md §13): the Gym tab, its API and `dreamkit gym ...`.

Arena, Compare and Prompt Tuner run seeded episodes in Dream Labs' simulator through dreamkit.wrap, so the loop
plugins of dreamkit.yaml (failure-monitor, recovery, harvester, ...) run in every episode, and every run lands in
History with a Gym badge. It needs the simulator's world (scene randomisation, the success judge): "sim only".
"""

from __future__ import annotations

from pathlib import Path

from dreamkit.sdk import App, AppContext, Manifest


def router(ctx: AppContext):
    """/apps/dreamgym/api: runs, datasets, teleop. DreamGym's own folders sit beside dreamkit.yaml in gym/."""
    from .web.server import make_router

    return make_router(ctx.base / "gym" / "runs", ctx.config_path)


def commands(argv: list[str]) -> int:
    """`dreamkit gym arena ...` (or `dreamkit dreamgym ...`): DreamGym's own command line."""
    from .cli import main

    return main(argv)


app = App(
    name="dreamgym",
    manifest=Manifest(kind="app", title="Gym", icon="Dumbbell", needs=["robot", "world"]),
    router=router,
    page=Path(__file__).parent / "web" / "page",
    commands=commands,
    description="Arena, Compare, Prompt Tuner, datasets and Teleop: seeded episodes in the simulator, "
                "every one through dreamkit's plugins",
)
