"""DreamGym as a dreamkit app plugin (PLUGIN_STANDARD.md §13): the Data tab, its API and `dreamkit dreamgym ...`.

The tab is for fine-tuning data only: collect demonstrations (scripted expert, Dream-1, teleop), review them, train,
and compare the fine-tune with the base model. Live runs, History and the loop plugins stay in dreamkit's own tabs.
Arena, Compare and Prompt Tuner (the command line) run seeded episodes in Dream Labs' simulator through dreamkit.wrap, so the loop
plugins of dreamkit.yaml (failure-monitor, recovery, harvester, ...) run in every episode, and every run lands in
History with a Gym badge. It needs the simulator's world (scene randomisation, the success judge): "sim only".
"""

from __future__ import annotations

from pathlib import Path

from dreamkit.sdk import App, AppContext, Manifest


def router(ctx: AppContext):
    """/apps/dreamgym/api: datasets, evaluations, teleop. Everything in dreamkit's data folder (runs/, datasets/)."""
    from .web.server import make_router

    return make_router(ctx.runs_dir, ctx.config_path, getattr(ctx, "datasets_dir", None))


def commands(argv: list[str]) -> int:
    """`dreamkit dreamgym arena ...` (or `dreamkit data ...`): DreamGym's own command line."""
    from .cli import main

    return main(argv)


app = App(
    name="dreamgym",
    manifest=Manifest(kind="app", title="Data", icon="Database", needs=["robot", "world"]),
    router=router,
    page=Path(__file__).parent / "web" / "page",
    commands=commands,
    description="Fine-tuning data: collect demonstrations (scripted, Dream-1, teleop), review them, train Dream-1, "
                "compare the fine-tune with the base model",
)
