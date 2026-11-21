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


# -- §7a: a plugin's own page -----------------------------------------------------------------------------------------

def test_a_plugin_page_in_the_theme_passes(tmp_path):
    from dreamkit.standard import check_ui

    (tmp_path / "ui").mkdir()
    (tmp_path / "ui" / "index.html").write_text(
        '<style>.x { color: var(--dk-ok); border: 1px solid var(--dk-line); font-family: var(--dk-font-mono) }'
        ' #grid:hover { fill: var(--dk-bad) }</style><div class="dk-card" style="background: var(--dk-panel)"></div>'
        '<script>fetch(`/media/${run}/monitor-stats/results.json`); el.style.color = "var(--dk-bad)"</script>'
        '<a href="/media/r/x/results.csv" download>csv</a>')
    assert check_ui(tmp_path, "ui/index.html") == []


def test_a_plugin_page_outside_the_theme_is_refused(tmp_path):
    from dreamkit.standard import check_ui

    (tmp_path / "ui").mkdir()
    (tmp_path / "ui" / "index.html").write_text(
        '<link rel="stylesheet" href="https://cdn.example.com/bootstrap.css">\n'
        '<style>@font-face { font-family: X; src: url(x.woff) }\n.a { color: #ff0000 }\n.b { background: white }\n'
        '.c { font-family: "Comic Sans MS" }</style>\n<div style="border-color: rgb(1,2,3)"></div>\n'
        '<script src="//unpkg.com/react"></script><script>el.style.fill = "#abc"</script>')
    found = " | ".join(str(p) for p in check_ui(tmp_path, "ui/index.html"))
    for what in ("external URL 'https://", "external URL '//unpkg.com", "@font-face", "'#ff0000'", "'white'",
                 "own font", "'rgb('", "'#abc' in a script"):
        assert what in found, what
    assert "§7a" in found and "index.html:" in found  # the rule, the file and the line
    assert "missing file" in str(check_ui(tmp_path, "ui/other.html")[0])


def test_manifest_ui_must_stay_inside_the_package():
    assert Manifest(ui="ui/index.html").ui == "ui/index.html"
    for bad in ("../x.html", "/abs/index.html", "ui/index.js"):
        with pytest.raises(ValueError):
            Manifest(ui=bad)


def test_the_heartbeat_keeps_going_while_the_core_waits_for_deferred_jobs():
    import threading
    import time

    from dreamkit.kit import Deferred

    d = Deferred(lambda plugin, error: None)
    release = threading.Event()
    d("slow-plugin", release.wait, (5,), {})
    beats = []
    threading.Timer(0.35, release.set).start()
    assert d.flush(timeout=5, heartbeat=lambda: beats.append(time.monotonic()), every_s=0.1)
    assert len(beats) >= 2  # Live keeps hearing from the loop while a plugin waits for its model
    d("slow-plugin", time.sleep, (1,), {})
    assert d.flush(timeout=0.2, every_s=0.05) is False  # the timeout still holds
