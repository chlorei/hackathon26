"""No simulator needed: config parsing, task parsing, plugin isolation."""

import numpy as np

from dreamkit import Context, Plugin, StopEpisode
from dreamkit.config import Config, PluginSpec, dump_plugins
from dreamkit.events import Emitter
from dreamkit.manager import PluginManager
from dreamkit.task import parse


def test_plugin_entries_all_forms():
    cfg = Config.model_validate({"plugins": ["a", {"b": {"x": 1}}, {"c": {"enabled": False}}, {True: None} and "d",
                                             {"retry": {True: ["dropped"], "attempts": 1}}]})
    assert [p.name for p in cfg.plugins] == ["a", "b", "c", "d", "retry"]
    assert cfg.plugins[1].params == {"x": 1}
    assert cfg.plugins[2].enabled is False
    assert cfg.plugins[4].params == {"on": ["dropped"], "attempts": 1}  # YAML's `on:` key read as True


def test_dump_roundtrip():
    specs = [PluginSpec(name="a"), PluginSpec(name="b", params={"k": 2}), PluginSpec(name="c", enabled=False)]
    again = Config.model_validate({"plugins": dump_plugins(specs)}).plugins
    assert again == specs


def test_task_parse():
    assert parse("put apple into the bowl") == ("apple", "bowl")
    assert parse("Put the banana on the plate.") == ("banana", "plate")
    assert parse("wave hello") == (None, None)


class Boom(Plugin):
    name = "boom"

    def on_step(self, ctx):
        raise RuntimeError("bad plugin")


class Stopper(Plugin):
    name = "stopper"

    def on_step(self, ctx):
        ctx.stop("enough")


def _ctx():
    return Context(env=None, task="t", model="m", run_id="r", emitter=Emitter(console=False))


def test_crashing_plugin_becomes_an_error_event(monkeypatch):
    manager = PluginManager([])
    manager.plugins = [Boom(), Stopper()]
    ctx = _ctx()
    try:
        manager.call("on_step", ctx)
    except StopEpisode as stop:
        assert stop.plugin == "stopper" and stop.reason == "enough"
    else:
        raise AssertionError("stop() must propagate")
    errors = [r for r in ctx.emitter.records if r["type"] == "event" and r["level"] == "error"]
    assert errors and errors[0]["plugin"] == "boom" and "bad plugin" in errors[0]["message"]


def test_builtins_are_discovered():
    manager = PluginManager([PluginSpec(name=n) for n in ("success-check", "failure-monitor", "retry", "nope")])
    assert manager.names == ["success-check", "failure-monitor", "retry"]
    assert manager.missing == ["nope"]


def test_camera_check_flags_black_frame():
    from dreamkit.plugins.camera_check import CameraCheck

    plugin = CameraCheck()
    ctx = _ctx()
    ctx.frames = {"top": np.zeros((36, 64, 3), np.uint8), "left": np.random.randint(0, 255, (36, 64, 3), np.uint8)}
    plugin.on_episode_start(ctx)
    try:
        plugin.before_model(ctx)
    except StopEpisode as stop:
        assert "camera top is black" in stop.reason
    else:
        raise AssertionError("a black camera must stop the attempt")
