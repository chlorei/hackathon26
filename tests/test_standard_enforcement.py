"""Proof that the standard is enforced (PLUGIN_STANDARD.md §11): three plugins written to break it, all refused with
an error that names the rule, the place and the fix — and the robot never feels them."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dreamkit import Manifest, Plugin, PluginStandardError
from dreamkit.manager import PluginManager, discover
from dreamkit.config import PluginSpec
from dreamkit.testing import fake_ctx, replay

BAD = Path(__file__).resolve().parent / "bad_plugins"


def test_a_plugin_without_descriptions_does_not_even_import():
    spec = importlib.util.spec_from_file_location("no_descriptions", BAD / "no_descriptions" / "no_descriptions.py")
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(PluginStandardError) as error:
        spec.loader.exec_module(module)
    text = str(error.value)
    assert "Params.window has no description" in text and "Params.mode has no description" in text
    assert "Params.window is a number without a unit" in text
    assert "docstring" in text and "§4" in text
    assert "no_descriptions.py:" in text  # file:line
    assert 'Add description="..."' in text  # and how to fix it
    found, problems = discover([BAD / "no_descriptions"])
    assert "no-descriptions" not in found and any("refused by the plugin standard" in p for p in problems)


def test_an_observe_plugin_cannot_stop_the_robot():
    r = replay("success_apple_bowl", plugins=["success-check", "observe-stops"],
               extra_dirs=[BAD / "observe_stops"])
    assert r.result["success"] is True  # the attempt went on to the end
    assert r.result.get("stopped_by") is None
    errors = r.errors("observe-stops")
    assert errors and "may not stop" in errors[0]["message"] and "§3" in errors[0]["message"]
    assert r.stats["observe-stops"]["disabled"].startswith("switched off after 5 errors in a row")


def test_a_modify_plugin_cannot_speed_the_arm_up():
    r = replay("success_apple_bowl", plugins=["modify-speeds-up", "success-check"],
               extra_dirs=[BAD / "modify_speeds_up"])
    errors = r.errors("modify-speeds-up")
    assert errors and "change rejected" in errors[0]["message"]
    assert "faster than the model asked" in errors[0]["message"] or "joint limits" in errors[0]["message"]
    assert r.result["success"] is True  # the model's own chunks reached the robot unchanged


# -- the scoped context, rule by rule --------------------------------------------------------------------------------

class Probe(Plugin):
    """Probes what an observe plugin may do."""

    name = "probe"
    manifest = Manifest(group="diagnosis", icon="Plug", capability="observe", hooks=["on_step"])

    def on_step(self, ctx):
        pass


def test_observe_actions_are_read_only():
    ctx = fake_ctx(Probe(), actions=(np.zeros((4, 7)), np.zeros((4, 7))))
    with pytest.raises(ValueError):
        ctx.actions[0][0, 0] = 1.0  # writeable=False
    with pytest.raises(PluginStandardError, match="may not change actions"):
        ctx.actions = (np.zeros((4, 7)), np.zeros((4, 7)))


@pytest.mark.parametrize("call, rule", [
    (lambda c: c.emit("loud", "x"), "level"),
    (lambda c: c.emit("info", "Capitalised."), "lowercase fragment"),
    (lambda c: c.emit("info", "x", distance=1.5), "without a unit suffix"),
    (lambda c: c.emit("info", "x", reason="made_up"), "not in dreamkit/reasons.py"),
    (lambda c: c.request_retry(), "may not request retry"),
    (lambda c: c.override([], []), "may not override"),
    (lambda c: c.files.path("../escape.txt"), "inside run_dir/<plugin>"),
    (lambda c: c.result.__setitem__("success", True), "only success-check writes it"),
    (lambda c: c.result.__setitem__("reason", "dropped"), "only failure-monitor writes it"),
    (lambda c: c.frames, "needs_frames"),
])
def test_scoped_context_refuses(call, rule):
    with pytest.raises(PluginStandardError, match=rule.replace("(", r"\(").replace(")", r"\)")):
        call(fake_ctx(Probe()))


def test_own_result_key_and_good_event_are_fine():
    ctx = fake_ctx(Probe())
    ctx.result["probe"] = {"ok": True}
    ctx.emit("info", "all good at 1.0 s", dist_m=0.1, side="left")
    assert ctx.files.path("ep001_trace.json").parent.name == "probe"




def test_undeclared_hook_is_refused_at_definition():
    with pytest.raises(PluginStandardError, match="implemented but not declared"):
        class Hooky(Plugin):  # noqa: F841
            """Implements a hook it does not declare."""

            name = "hooky"
            manifest = Manifest(group="diagnosis", icon="Plug", capability="observe", hooks=[])

            def on_step(self, ctx):
                pass


def test_a_second_control_plugin_is_not_loaded():
    class Guard(Plugin):
        """Stops on everything."""

        name = "table-guard"
        manifest = Manifest(group="core", icon="Plug", capability="control", hooks=["on_step"])

        def on_step(self, ctx):
            pass

    m = PluginManager([PluginSpec(name="recovery")])
    m.available["table-guard"] = {"cls": Guard, "source": "local", "origin": "test"}
    resolved = m._resolve([m.available["recovery"]["cls"](), Guard()])
    assert [p.name for p in resolved] == ["recovery"]
    assert m.refused["table-guard"].startswith("conflicts with recovery")
