"""dreamkit.wrap around a fake DreamClient: hooks fire inside the user's calls, no simulator needed."""

import json

import numpy as np
import yaml

import dreamkit

SPIES = """
from dreamkit import Manifest, Plugin


def _seen(ctx):
    return ctx.store.setdefault("seen", [])


class SpyStop(Plugin):
    \"\"\"Records every hook and stops the attempt on the third step.\"\"\"

    name = "spy-stop"
    manifest = Manifest(group="core", icon="Plug", capability="control",
                        hooks=["on_run_start", "on_episode_start", "before_model", "on_step", "on_episode_end",
                               "on_run_end"])

    def on_run_start(self, ctx): _seen(ctx).append("run_start")
    def on_episode_start(self, ctx): _seen(ctx).append(f"episode_start:{ctx.task}")
    def before_model(self, ctx): _seen(ctx).append("before_model")

    def on_step(self, ctx):
        _seen(ctx).append("step")
        if ctx.step == 3:
            ctx.stop("stalled", "enough")

    def on_episode_end(self, ctx): _seen(ctx).append(f"episode_end:{ctx.result.get('reason')}")
    def on_run_end(self, ctx): _seen(ctx).append("run_end")


class SpyTrim(Plugin):
    \"\"\"Keeps only the first ten rows of each chunk.\"\"\"

    name = "spy-trim"
    manifest = Manifest(group="safety", icon="Plug", capability="modify_actions", hooks=["after_model"])

    def after_model(self, ctx):
        ctx.store.setdefault("seen", []).append(f"after_model:{ctx.timing['rtt_ms']:.0f}")
        left, right = ctx.actions
        ctx.actions = (left[:10], right[:10])
"""


class FakeClient:
    """The slice of dream.DreamClient a loop uses."""

    def __init__(self):
        self.calls = []
        self.last_timing = None

    def policy(self, config):
        self.calls.append("policy")

    def set_observation(self, frames, observation, **kw):
        self.calls.append("set_observation")

    def get_action(self, **kw):
        self.calls.append("get_action")
        self.last_timing = {"rtt_ms": 420.0}
        return {"q_l_chunk": np.zeros((48, 7)).tolist(), "q_r_chunk": np.ones((48, 7)).tolist()}

    def reset_episode(self):
        self.calls.append("reset_episode")

    def execution_horizon(self):
        return 48


def _config(tmp_path):
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins" / "spies.py").write_text(SPIES)
    path = tmp_path / "dreamkit.yaml"
    path.write_text(yaml.safe_dump({"runs_dir": str(tmp_path / "runs"),
                                    "plugins": ["spy-stop", "spy-trim", "failure-monitor"]}))
    return path


def test_hooks_fire_inside_the_users_calls(tmp_path):
    fake = FakeClient()
    client = dreamkit.wrap(fake, config=_config(tmp_path), console=False)
    client.policy({"checkpoint": "dreamlabs/dream-1-beta"})
    assert client.execution_horizon() == 48  # anything else passes straight through

    with client.episode("put apple into the bowl"):
        client.set_observation({}, {"prompt": "put apple into the bowl"})
        reply = client.get_action()
        assert len(reply["q_l_chunk"]) == 10  # the modify plugin's edit reached the user's code
        for _ in range(10):
            client.step(command=(np.zeros(7), np.ones(7)))
        raise AssertionError("the control plugin's stop should have left the with-block")
    client.close()
    seen = client.ctx.store["seen"]
    assert seen[:5] == ["run_start", "episode_start:put apple into the bowl", "before_model", "after_model:420", "step"]
    assert seen.count("step") == 3
    assert "episode_end:stalled" in seen and seen[-1] == "run_end"
    assert fake.calls == ["policy", "set_observation", "get_action", "reset_episode"]

    run = json.loads(next((tmp_path / "runs").glob("*/run.json")).read_text())
    assert run["status"] == "finished" and run["model"] == "dreamlabs/dream-1-beta" and run["control"] == "spy-stop"
    (ep,) = run["episodes"]
    assert ep["reason"] == "stalled" and ep["stopped_by"] == "spy-stop" and ep["at_s"] == 0.15
    assert ep["failure-monitor"]["latency"]["mean_ms"] == 420.0
    assert run["plugin_stats"]["spy-trim"]["errors"] == 0
    assert (tmp_path / "runs" / run["id"] / "live.json").is_file()


def test_retry_keeps_the_scene(tmp_path):
    client = dreamkit.wrap(FakeClient(), config=None, console=False, runs_dir=str(tmp_path / "runs"))
    with client.episode("t"):
        client.ctx._retry = True
    assert client.retry_requested
    with client.episode("t"):
        pass
    with client.episode("t"):
        pass
    client.close()
    assert [(e["scene"], e["attempt"]) for e in client.episodes] == [(0, 1), (0, 2), (1, 1)]
