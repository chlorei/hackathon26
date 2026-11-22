"""failure-monitor judges from what a real arm reports; world-judge (an evaluation tool) gives the right answer."""

import numpy as np
import pytest

from dreamkit.testing import fake_ctx, replay

WORKING = ["camera-check", "gripper-check", "failure-monitor"]
## What the arm's own signals can tell, per recorded failure (None: they cannot, vision-judge or a person decides).
FROM_THE_ARM = {
    "camera_black": "camera_failure", "camera_frozen": "camera_failure", "collision_table": "collision",
    "dropped_mid_carry": "dropped", "grasped_air": "grasped_air", "stalled": "stalled",
    "fell_off_table": "dropped",  # it fell while held: the arm feels the drop, not where it landed
    "not_reached": "grasped_air",  # the hand closed short of the apple: the fingers met
    "released_off_target": None, "released_near_target": None,  # a calm release looks like a placement
}


@pytest.mark.parametrize("fixture", sorted(FROM_THE_ARM))
def test_names_what_the_arm_can_tell(fixture):
    r = replay(fixture, plugins=WORKING, world=False)
    assert r.result.get("reason") == FROM_THE_ARM[fixture], r.result.get("candidates")
    assert r.budget_ms("failure-monitor") < 1.0
    assert not r.errors("failure-monitor")
    if FROM_THE_ARM[fixture]:
        assert r.result["details"]["source"] == "robot"


@pytest.mark.parametrize("fixture", [f for f in FROM_THE_ARM])
def test_world_judge_names_the_recorded_failure(fixture):
    """The evaluation tool, with the world: the answer the monitor is measured against (dreamkit doctor --sim)."""
    r = replay(fixture, plugins=["camera-check", "success-check", "world-judge"])
    assert r.result["world-judge"]["reason"] == r.expected["reason"]


def test_quiet_on_success_with_a_timeline():
    r = replay("success_apple_bowl", plugins=WORKING, world=False)
    assert r.result.get("reason") is None and r.result.get("success") is None  # unknown until a judge says
    kinds = [m["kind"] for m in r.result["timeline"]]
    assert "grasp" in kinds and "release" in kinds


def test_model_latency_is_part_of_failure_monitor():
    r = replay("slow_model", plugins=["failure-monitor"], world=False)
    assert r.events.any(plugin="failure-monitor", level="warn")
    assert r.result["failure-monitor"]["latency"]["mean_ms"] > 700
    assert r.result["failure-monitor"]["if_failed"]["reason"] == "slow_model"  # the cause if a judge says it failed


def test_live_verdict_mid_attempt():
    r = replay("dropped_mid_carry", plugins=["failure-monitor"], world=False)
    v = [e for e in r.events if e["data"].get("kind") == "verdict"]
    assert v and v[0]["data"]["reason"] == "dropped" and v[0]["t_s"] < 12


def test_released_early_when_the_hand_opens_while_moving():
    from dreamkit.plugins.failure_monitor.plugin import FailureMonitor

    plugin = FailureMonitor()
    ctx = fake_ctx(plugin, hook="on_episode_start", state={"left": np.zeros(7), "right": np.zeros(7)})
    core = ctx._core
    core.robot.reset({"left": np.r_[np.zeros(6), 1.0], "right": np.r_[np.zeros(6), 1.0]})
    plugin.on_episode_start(ctx)
    q = np.zeros(6)
    for step in range(60):  # close on an object (fingers stop at 0.5), carry it, open while still swinging
        t = step * 0.05
        q = q + (0.06 if step > 20 else 0.0)  # 1.2 rad/s once the carry starts
        cmd = 0.0 if step < 50 else 1.0
        grip = 0.5 if step < 50 else 1.0
        core.t, core.step = t, step
        core.robot.update({"left": np.r_[q, grip], "right": np.r_[np.zeros(6), 1.0]},
                          (np.r_[q, cmd], np.r_[np.zeros(6), 1.0]))
        plugin.on_step(ctx)
    assert plugin.verdicts and plugin.verdicts[0]["reason"] == "released_early"
