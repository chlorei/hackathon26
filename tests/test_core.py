"""No simulator needed: config parsing, task parsing, discovery, plugin isolation."""

from pydantic import BaseModel

from dreamkit import Manifest, Plugin
from dreamkit.config import Config, PluginSpec, dump_plugins
from dreamkit.events import Emitter
from dreamkit.manager import PluginManager
from dreamkit.plugin import Context
from dreamkit.task import parse


def test_plugin_entries_all_forms():
    cfg = Config.model_validate({"plugins": ["a", {"b": {"x": 1}}, {"c": {"enabled": False}},
                                             {"recovery": {True: ["dropped"]}}]})
    assert [p.name for p in cfg.plugins] == ["a", "b", "c", "recovery"]
    assert cfg.plugins[1].params == {"x": 1}
    assert cfg.plugins[2].enabled is False
    assert cfg.plugins[3].params == {"on": ["dropped"]}  # YAML's bare `on:` key read as True


def test_dump_roundtrip():
    specs = [PluginSpec(name="a"), PluginSpec(name="b", params={"k": 2}), PluginSpec(name="c", enabled=False)]
    assert Config.model_validate({"plugins": dump_plugins(specs)}).plugins == specs


def test_task_parse():
    assert parse("put apple into the bowl") == ("apple", "bowl")
    assert parse("Put the banana on the plate.") == ("banana", "plate")
    assert parse("wave hello") == (None, None)


class Boom(Plugin):
    """Raises on every step, to show a crash never reaches the loop."""

    name = "boom"
    manifest = Manifest(group="diagnosis", icon="Plug", capability="observe", hooks=["on_step"])

    def on_step(self, ctx):
        raise RuntimeError("bad plugin")


def test_a_crashing_plugin_becomes_an_error_event_then_is_switched_off():
    manager = PluginManager([])
    manager.plugins = [Boom()]
    manager.stats = {"boom": __import__("dreamkit.manager", fromlist=["Stats"]).Stats()}
    core = Context(env=None, task="t", model="m", run_id="r", emitter=Emitter(console=False))
    for step in range(6):
        core.step = step
        manager.call("on_step", core)
    errors = [r for r in core.emitter.records if r["type"] == "event" and r["level"] == "error"]
    assert errors[0]["plugin"] == "boom" and "bad plugin" in errors[0]["message"]
    assert manager.stats["boom"].disabled.startswith("switched off after 5 errors in a row")
    assert manager.stats["boom"].errors == 5  # the sixth step was not called


def test_builtins_are_discovered_and_retry_is_merged():
    manager = PluginManager([PluginSpec(name=n) for n in ("success-check", "failure-monitor", "retry", "nope")])
    assert manager.names == ["failure-monitor"]
    assert manager.missing == ["nope"] and "part of recovery" in manager.refused["retry"]
    assert "evaluation tool" in manager.refused["success-check"]  # §6: the world is for evaluation only
    assert PluginManager([PluginSpec(name="success-check")], evaluation=True).names == ["success-check"]


def test_requires_orders_and_refuses():
    class Needs(Plugin):
        """Needs failure-monitor to run first."""

        name = "needs-fm"
        manifest = Manifest(group="diagnosis", icon="Plug", capability="observe", hooks=["on_step"],
                            requires=["failure-monitor"])

        def on_step(self, ctx):
            pass

    m = PluginManager([])
    fm = m.available["failure-monitor"]["cls"]()
    assert [p.name for p in m._resolve([Needs(), fm])] == ["failure-monitor", "needs-fm"]
    assert m._resolve([Needs()]) == [] and "requires failure-monitor" in m.refused["needs-fm"]


def test_params_are_validated_models():
    p = PluginManager([PluginSpec(name="failure-monitor", params={"track_err_rad": 0.2})]).plugins[0]
    assert isinstance(p.params, BaseModel) and p.params.track_err_rad == 0.2
    bad = PluginManager([PluginSpec(name="failure-monitor", params={"track_err_rad": -1})])
    assert not bad.plugins and "invalid settings" in bad.failed[0]
