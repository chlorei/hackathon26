"""Replay tests: one attempt where the gripper closes on nothing, one clean grasp."""

from dreamkit.testing import replay


def test_flags_a_gripper_closed_on_nothing():
    result = replay("grasped_air", plugins=["gripper-check"])
    assert result.events.any(plugin="gripper-check", level="warn")
    assert result.result["gripper-check"]["events"][0]["kind"] == "empty"
    assert result.budget_ms("gripper-check") < 1.0


def test_quiet_on_a_clean_grasp():
    result = replay("success_apple_bowl", plugins=["gripper-check"])
    assert not result.events.of("gripper-check")
    assert "gripper-check" not in result.result


def test_works_without_the_simulator_world():
    result = replay("grasped_air", plugins=["gripper-check"], world=False)
    assert result.result["gripper-check"]["events"][0]["kind"] == "empty"
