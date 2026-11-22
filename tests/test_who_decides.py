"""Who decides success: the world (success-check) -> a vision judge -> a person. Disagreement is a warning."""

import textwrap

import pytest

from dreamkit.testing import replay

VISION = textwrap.dedent('''
    from dreamkit.plugin import Plugin, PluginContext
    from dreamkit.standard import Manifest


    class VisionStub(Plugin):
        """Says what a vision judge would say, from a setting."""

        name = "vision-stub"
        manifest = Manifest(group="diagnosis", icon="Eye", capability="observe", hooks=["on_episode_end"],
                            step_rate_hz=1)

        def on_episode_end(self, ctx: PluginContext) -> None:
            ctx.result["vision-stub"] = {"judge": "vision", "success": SAYS, "reason_code": "dropped"}
''')


@pytest.fixture()
def vision(tmp_path):
    def make(says: bool):
        d = tmp_path / ("yes" if says else "no")
        d.mkdir()
        (d / f"vision_stub_{'yes' if says else 'no'}.py").write_text(VISION.replace("SAYS", str(says)))
        return d
    return make


def test_the_world_decides_and_a_disagreeing_vision_is_a_warning(vision):
    r = replay("success_apple_bowl", plugins=["success-check", "vision-stub"], extra_dirs=[vision(False)], keep=True)
    assert r.result["success"] is True and r.result["judged_by"] == "world"
    assert r.result["disagreement"] == {"world": True, "vision": False, "by": "vision-stub"}
    assert r.events.any(plugin="dreamkit", level="warn")


def test_without_a_world_the_vision_judge_decides(vision):
    r = replay("success_apple_bowl", plugins=["success-check", "vision-stub"], extra_dirs=[vision(False)],
               world=False, keep=True)
    assert r.result["success"] is False and r.result["judged_by"] == "vision"
    assert r.result["reason"] == "dropped" and "disagreement" not in r.result


def test_agreeing_judges_say_nothing_more(vision):
    r = replay("success_apple_bowl", plugins=["success-check", "vision-stub"], extra_dirs=[vision(True)], keep=True)
    assert r.result["judged_by"] == "world" and "disagreement" not in r.result
